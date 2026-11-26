"""Batch pick-and-place data of one photo: domain-randomized configurations of its twin and cousins (calibrated_scene_reconstruction.dr), each
with the environment exported, grasps synthesized and N episodes recorded, all with the one goal marker.

    envs/recon/bin/python tools/collect_place.py sessions/<name> [--robot vega_1p_sharpa] [--scenes twin cousin_00 ...]
        [--per-scene 8] [--episodes 6] [--batch 6] [--stages configs obstacles grasps episodes] [--prefix "<gpu launcher>"]

Stages (each skips what is done, so a run resumes where it stopped):
  configs    calibrated_scene_reconstruction.dr: sessions/<name>/dr/<scene>__d<k>.json (kept if present; --fresh draws them again)
  obstacles  apps/scene_obstacles.py over every configuration missing grasps/_env/<tag>.json (one Isaac process)
  grasps     calibrated_scene_reconstruction.grasping.synth --pool-rounds R --variants <tags> (robots with a BODex hand): one BODex pool per
             target mesh, R GPU batches of --seeds seeds all around it, then each configuration's grasps selected
             from the pool (facing its arm, clear of its environment); a finished round is kept
  episodes   apps/episode.py --task-type place --receptacles marker (or --receptacles) --batch B per configuration until it holds
             --episodes successful episodes (episodes/<tag>__place[__e<k>]): plans validated B at a time in scene
             copies, the successes recorded B at a time in the copies too (each copy renders only itself); a
             configuration whose run recorded fewer moves on
Writes sessions/<name>/logs/collect/<stage|tag>.log and outputs/collect/<session>__<robot>.json (per configuration:
episodes recorded, physics trials, why plans failed).
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import threading
import time
from collections import Counter
from pathlib import Path

from calibrated_scene_reconstruction.core.paths import ROOT
from calibrated_scene_reconstruction.robots.spec import load_spec  # noqa: E402

DEFAULT_ROBOT = "vega_1p_sharpa"


LOG_MAX = 64 * 2**20  # bytes: a job's log past this is Kit stuck repeating one error (a normal log: a few MB)
LOG_KEEP = (8 * 2**20, 2**20)  # its first and last bytes kept for the post-mortem


def start_gate(path, gap):
    """Wait until `gap` s have passed since the last Isaac start recorded in `path` (by any collect process on these
    configurations), then record this one: Kits started in the same second sometimes crash at once in their extension
    registry (carb.variant), shared by every run, or repeat its error until the log cap kills them."""
    import fcntl

    with open(path, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)  # held while waiting: the next process waits for this start
        f.seek(0)
        last = float(f.read().strip() or 0)
        time.sleep(max(0.0, last + gap - time.time()))
        f.seek(0)
        f.truncate()
        f.write(f"{time.time():.1f}\n")


from calibrated_scene_reconstruction.agent.jobs import run as run  # noqa: E402,F401 -- legacy tool API


def mem_available_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return float("inf")


_start = threading.Lock()


def wait_memory(min_gb, settle=90):
    """Hold a new job back until `min_gb` of memory is available; starts are serialized and spaced by `settle` s, the
    time an Isaac process takes to reach most of its peak, so that parallel starts do not all pass the check at once."""
    with _start:
        while mem_available_gb() < min_gb:
            time.sleep(30)
        time.sleep(settle)


def episode_dirs(s, tag, robot):
    base = tag if robot == DEFAULT_ROBOT else f"{tag}__{robot}"
    pat = re.compile(re.escape(base) + r"__place(__e\d+)?")  # the run's first episode, then __e1, __e2, ...
    return sorted(d for d in (s / "episodes").glob(f"{base}__place*")
                  if pat.fullmatch(d.name) and (d / "result.json").exists())


def recorded(s, tag, robot):
    return [d for d in episode_dirs(s, tag, robot) if json.loads((d / "result.json").read_text()).get("success")
            and (d / "raw" / "meta.json").exists()]


def next_k(s, tag, robot):
    """The episode index after a configuration's last recorded one (0 when it has none)."""
    ks = [int(m.group(1) or 0) for d in recorded(s, tag, robot) if (m := re.search(r"__place(?:__e(\d+))?$", d.name))]
    return max(ks) + 1 if ks else 0


def summary(s, tag, robot):
    eps = recorded(s, tag, robot)
    trials, fails = 0, Counter()
    for d in episode_dirs(s, tag, robot)[:1]:  # every recorded episode of a run carries the run's trials
        r = json.loads((d / "result.json").read_text())
        for t in r.get("trials", []):
            if "lift_m" in t:
                trials += 1
                if not t.get("success"):
                    fails["physics"] += 1
            else:
                fails[(t.get("plan") or {}).get("fail", "no IK").split(" at ")[0].split("(")[0].strip()] += 1
        if r.get("reason"):
            fails[r["reason"]] += 1
    return {"episodes": len(eps), "trials": trials, "fails": dict(fails.most_common(6))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--robot", default=DEFAULT_ROBOT)
    ap.add_argument("--scenes", nargs="*", help="default: the twin and every cousin")
    ap.add_argument("--per-scene", type=int, default=8, help="randomized configurations per scene")
    ap.add_argument("--episodes", type=int, default=6, help="successful episodes per configuration")
    ap.add_argument("--batch", type=int, default=12, help="scene copies per process: plans, validations and "
                                                           "recordings at once")
    ap.add_argument("--action-noise", type=float, default=0.0,
                    help="rad: apps/episode.py --action-noise (corrective labels); a configuration that already holds "
                         "episodes adds the rest after them, so clean and perturbed episodes can share it")
    ap.add_argument("--max-trials", type=int, default=48)
    ap.add_argument("--variations", type=int, default=6, help="random trajectories planned per grasp (planning is "
                                                              "cheap; most are rejected in a cramped scene)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=256, help="BODex seeds per pool round (one GPU batch)")
    ap.add_argument("--pool-rounds", type=int, default=8,
                    help='BODex pool rounds per target mesh')
    ap.add_argument("--top-k", type=int, default=32, help="grasps selected from the pool per configuration")
    ap.add_argument("--wrist-min-cos", type=float, default=None,
                    help="the selection's wrist-facing-arm cosine (synth default 0.25; from one base, as --photo-base, "
                         "the arm reaches most grasps above 0.5 and few below)")
    ap.add_argument("--fresh", action="store_true", help="draw the configurations again")
    ap.add_argument("--stages", nargs="*", default=["configs", "obstacles", "grasps", "episodes"])
    ap.add_argument("--prefix", default="", help="launcher for GPU jobs (a shared-GPU slot wrapper)")
    ap.add_argument("--episode-prefix", default=None,
                    help="launcher for the episode runs (default --prefix): with --jobs > 1, one that takes a single "
                         "GPU slot rather than the whole GPU")
    ap.add_argument("--jobs", type=int, default=1, help="configurations recorded at once (Isaac processes)")
    ap.add_argument("--stagger", type=float, default=30,
                    help="episodes: s between two Isaac starts, over every collect process on the session (0: off)")
    ap.add_argument("--letter", default="d", help="configuration tags <scene>__<letter><k> (calibrated_scene_reconstruction.dr --letter)")
    ap.add_argument("--receptacles", default="marker",
                    help="apps/episode.py --receptacles: marker (one orange mat everywhere), scene (the scene graph's "
                         "task.place kinds, the photo's own target built to task.place.match) or kinds")
    ap.add_argument("--same-target", action="store_true", help="calibrated_scene_reconstruction.dr --same-target: the twin's target everywhere")
    ap.add_argument("--base-jitter", default="0,0", help="m,deg: apps/episode.py --base-jitter")
    ap.add_argument("--photo-base", type=float, default=0.0,
                    help="m: the robot where the photo shows it, within this distance (apps/episode.py --photo-base; "
                         "grasps facing the arm from there)")
    ap.add_argument("--start-states", nargs="*", default=None, help="apps/episode.py --start-states (policy states)")
    ap.add_argument("--start-states-p", type=float, default=0.5)
    ap.add_argument("--noise-p", type=float, default=0.7, help="apps/episode.py --noise-p")
    ap.add_argument("--start-on", type=float, default=0.0, help="calibrated_scene_reconstruction.dr --start-on (target on a board / plate / tray)")
    ap.add_argument("--tray", type=float, default=0.0, help="calibrated_scene_reconstruction.dr --tray (a tray the goal marker may lie on)")
    ap.add_argument("--target-right", default=None, help="calibrated_scene_reconstruction.dr --target-right lo,hi")
    ap.add_argument("--target-forward", default=None, help="calibrated_scene_reconstruction.dr --target-forward lo,hi")
    ap.add_argument("--min-grasps", type=int, default=0,
                    help="episodes: skip a configuration whose grasp selection kept fewer (logs/collect/grasps__<letter>.log)")
    ap.add_argument("--episode-timeout", type=float, default=3600, help="s: a configuration's run killed past this")
    ap.add_argument("--stall-timeout", type=float, default=900, help="s: a run whose log stops growing this long is killed")
    ap.add_argument("--min-mem-gb", type=float, default=0.0,
                    help="episodes: start a configuration only while this much memory is available (one start at a "
                         "time, spaced); the running ones go on")
    ap.add_argument("--retry-failed", action="store_true",
                    help="episodes: run configurations again whose last run added nothing (logs/collect/<tag>.failed)")
    ap.add_argument("--max-configs", type=int, default=0, help="episodes: at most this many configurations (0: all)")
    ap.add_argument("--tag-regex", default=None,
                    help="only the configurations whose tag matches (re.search), every stage after configs: e.g. "
                         "'__v[5-9][0-9]$' adds v50-v99 without touching v00-v49's grasps or topping up their episodes")
    ap.add_argument("--episode-args", default="",
                    help="more apps/episode.py options, quoted (v4: --style fixed --pre-release-s 0,0.5 --done-hold-s "
                         "1,2 --goal-direction base --object-window 0.30,0.60,-0.45,0.25 --video-only)")
    args = ap.parse_args()
    s = Path(args.session)
    name = s.name
    logs = s / "logs" / "collect"
    prefix = shlex.split(args.prefix)
    ep_prefix = shlex.split(args.prefix if args.episode_prefix is None else args.episode_prefix)
    spec = load_spec(args.robot)
    scenes = args.scenes or ["twin"] + sorted(f.stem for f in (s / "cousins").glob("cousin_*.json"))
    py = [str(ROOT / "envs" / "recon" / "bin" / "python")]
    isaac = [str(ROOT / "tools" / "run_isaac.sh")]

    if "configs" in args.stages:
        have = sorted((s / "dr").glob(f"*__{args.letter}[0-9]*.json")) if (s / "dr").exists() else []
        if args.fresh:  # what was built on the old draws: their environments and episodes
            import shutil

            for f in have:
                (s / "grasps" / "_env" / f"{f.stem}.json").unlink(missing_ok=True)
                base = f.stem if args.robot == DEFAULT_ROBOT else f"{f.stem}__{args.robot}"
                for d in (s / "episodes").glob(f"{base}__place*"):  # a recording cut short too (no result.json)
                    if re.fullmatch(re.escape(base) + r"__place(__e\d+)?", d.name):
                        shutil.rmtree(d)
                f.unlink()
        if args.fresh or not have:
            rc, dt = run([*py, "-m", "calibrated_scene_reconstruction.dr", str(s), "--scenes", *scenes, "--per-scene", str(args.per_scene),
                          "--seed", str(args.seed), "--letter", args.letter, "--start-on", str(args.start_on),
                          "--tray", str(args.tray)]
                         + ([f"--target-right={args.target_right}"] if args.target_right else [])  # "=": a window
                         + ([f"--target-forward={args.target_forward}"] if args.target_forward else [])  # may start "-"
                         + (["--same-target"] if args.same_target else []), logs / "configs.log")
            print(f"[collect] configurations: {(logs / 'configs.log').read_text().strip().splitlines()[-1]}", flush=True)
    tags = sorted((f.stem for f in (s / "dr").glob(f"*__{args.letter}[0-9]*.json")
                   if f.stem.rsplit(f"__{args.letter}", 1)[0] in scenes),
                  key=lambda t: (scenes.index(t.rsplit(f"__{args.letter}", 1)[0]), t))  # the scenes' order: first first
    if args.tag_regex:
        tags = [t for t in tags if re.search(args.tag_regex, t)]
    print(f"[collect] {len(tags)} configurations over {len(scenes)} scenes", flush=True)

    if "obstacles" in args.stages:
        todo = [t for t in tags if not (s / "grasps" / "_env" / f"{t}.json").exists()]
        if todo:
            rc, dt = run([*isaac, "apps/scene_obstacles.py", str(s), "--tags", *todo], logs / "obstacles.log", prefix)
            print(f"[collect] obstacles for {len(todo)} configurations: exit {rc}, {dt:.0f} s", flush=True)

    if "grasps" in args.stages and any("bodex" in a["ee"] for a in spec["arms"].values()):
        env = {"PYTHONPATH": str(ROOT / "src"), "MEM": "16G"}
        cmd = [str(ROOT / "tools" / "run_capped.sh"), str(ROOT / "envs" / "grasp" / "bin" / "python"), "-m",
               "calibrated_scene_reconstruction.grasping.synth", str(s), "--robot", args.robot, "--seeds", str(args.seeds),
               "--pool-rounds", str(args.pool_rounds), "--top-k", str(args.top_k), "--variants", *tags,
               *(["--photo-base"] if args.photo_base else []),
               *(["--wrist-min-cos", str(args.wrist_min_cos)] if args.wrist_min_cos is not None else [])]
        t0 = time.time()
        # per letter (another collection's keeps its own), appended: a later run on more configurations keeps the
        # earlier selections' record
        with open(logs / f"grasps__{args.letter}.log", "a") as f:
            rc = subprocess.run(["env", *(f"{k}={v}" for k, v in env.items()), *prefix, *cmd], cwd=ROOT, stdout=f,
                                stderr=subprocess.STDOUT).returncode
        print(f"[collect] grasps: exit {rc}, {time.time() - t0:.0f} s", flush=True)

    if "episodes" in args.stages:
        if args.min_grasps:  # the selection's count per configuration, as the grasps stage logged it
            kept = {}
            gls = [logs / "grasps.log", logs / f"grasps__{args.letter}.log"]  # the shared log of older runs first
            for line in (ln for gl in gls if gl.exists() for ln in gl.read_text().splitlines()):
                if m := re.match(r"\[(\S+)\] .*: (\d+) grasps selected", line):
                    kept[m.group(1)] = int(m.group(2))
            few = [t_ for t_ in tags if kept.get(t_, 0) < args.min_grasps]
            tags = [t_ for t_ in tags if t_ not in few]
            print(f"[collect] {len(few)} configurations with fewer than {args.min_grasps} grasps left out: "
                  f"{' '.join(few)}", flush=True)
        if args.max_configs:
            tags = tags[:args.max_configs]

        def one(i, tag):
            have = len(recorded(s, tag, args.robot))
            if have >= args.episodes:
                return
            if (logs / f"{tag}.failed").exists() and not args.retry_failed:  # a run that recorded nothing
                return
            lock = logs / f"{tag}.lock"  # another collect process on the same configurations skips this one
            try:
                os.close(os.open(lock, os.O_CREAT | os.O_EXCL))
            except FileExistsError:
                return
            try:
                if args.min_mem_gb:
                    wait_memory(args.min_mem_gb)
                record(i, tag, have)
            finally:
                lock.unlink(missing_ok=True)

        def record(i, tag, have):
            log = logs / f"{tag}.log"
            if args.stagger:
                start_gate(logs / ".isaac_start", args.stagger)
            rc, dt = run([*isaac, "apps/episode.py", str(s), "--tag", tag, "--variant", str(s / "dr" / f"{tag}.json"),
                          "--robot", args.robot, "--task-type", "place", "--receptacles", args.receptacles,
                          "--episodes", str(args.episodes - have), "--first-k", str(next_k(s, tag, args.robot)),
                          "--action-noise", str(args.action_noise), "--batch", str(args.batch),
                          "--base-jitter", args.base_jitter, "--start-states-p", str(args.start_states_p),
                          "--noise-p", str(args.noise_p),
                          *(["--start-states", *args.start_states] if args.start_states else []),
                          "--max-trials", str(args.max_trials), "--max-grasps", str(args.top_k),
                          "--goal-tries", "1500",
                          "--variations", str(args.variations), *shlex.split(args.episode_args),
                          *(["--photo-base", str(args.photo_base)] if args.photo_base else []),
                          "--seed", str(args.seed + i + 1000 * have)], log, ep_prefix,
                         timeout=args.episode_timeout or None, stall=args.stall_timeout or None)
            sm = summary(s, tag, args.robot)
            if sm["episodes"] == have and rc != -9:  # nothing added: marked, so no process runs it again
                (logs / f"{tag}.failed").write_text(json.dumps(sm) + "\n")
            print(f"[collect] {tag}: {sm['episodes']} episodes, {sm['trials']} physics trials, {dt:.0f} s, exit {rc}"
                  f" | {sm['fails']}", flush=True)

        # configurations in parallel: one Isaac process each (a batch of copies uses < 5 GB of GPU memory; their
        # CPU physics is what a larger batch does not speed up)
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max(1, args.jobs)) as pool:
            list(pool.map(lambda it: one(*it), enumerate(tags)))

    rep = {t: summary(s, t, args.robot) for t in tags}
    out = ROOT / "outputs" / "collect" / f"{name}__{args.robot}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"session": name, "robot": args.robot, "configurations": rep,
                               "episodes": sum(r["episodes"] for r in rep.values())}, indent=1))
    by_scene = Counter()
    for t, r in rep.items():
        by_scene[t.rsplit(f"__{args.letter}", 1)[0]] += r["episodes"]
    print(f"[collect] {sum(by_scene.values())} episodes: {dict(by_scene)} -> {out}", flush=True)


if __name__ == "__main__":
    main()
