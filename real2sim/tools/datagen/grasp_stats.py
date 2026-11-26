"""Grasp success statistics of one robot on a session's scenes: many physics trials over several seeds instead of the
one episode a pipeline run records, with a confidence interval and the reasons trials fail.

    envs/recon/bin/python tools/grasp_stats.py sessions/<name> --robot R [--tags twin cousin_00] [--seeds 0 1 2]
        [--trials 12] [--batch 6] [--synth-seeds 0 1] [--prefix "<gpu launcher>"] [--aggregate-only]

Per synthesis seed (robots with a BODex hand; --synth-seeds, default the synthesis' own seed 0) the grasps are
synthesized again (calibrated_scene_reconstruction.grasping.synth --seed k --force), then per scene tag and episode seed apps/episode.py validates
up to --trials physics rollouts (--no-record: random trajectories over the ranked grasps, nothing recorded) and writes
episodes/_stats/<tag>[__<robot>]__s<seed>.json. The aggregate (printed, and written to
outputs/stats/<session>__<robot>.json):

  trials       physics rollouts: successes / n, Wilson 95 % interval
  runs         seeds with at least one successful rollout (what one pipeline run with that seed would record)
  physics      failed rollouts by outcome: not lifted (the hand came up without the object), dropped (lifted, then
               out of the hand), tilted (in hand but turned past 45 deg), in-hand rotation over the lift (median, max)
  planning     grasps out of IK reach, plans rejected by reason (support, environment, start IK, ...)
  grasps       synthesized grasps that pass the quality gate per synthesis seed (the spread of a seed change)

--prefix runs every GPU job through a launcher (a shared-GPU slot wrapper, for example).
"""
import argparse
import json
import re
import shlex
import subprocess
import time
from collections import Counter
from math import sqrt
from pathlib import Path

from calibrated_scene_reconstruction.core.paths import ROOT
from calibrated_scene_reconstruction.core.paths import rel, resolve  # noqa: E402
from calibrated_scene_reconstruction.robots.spec import load_spec  # noqa: E402

DEFAULT_ROBOT = "vega_1p_sharpa"


def wilson(k, n, z=1.96):
    """Wilson score interval of a binomial proportion (k successes of n)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def physics_outcome(t):
    """Why a validated rollout failed (episode.py's success: lift > 6 cm, tilt < 45 deg, within 0.2 m of the TCP)."""
    lift, tilt, dist = t.get("lift_m", 0.0), t.get("tilt_deg", 0.0), t.get("tcp_dist_m", 1.0)
    if t.get("success"):
        return "success"
    if lift <= 0.06:
        return "not lifted"
    if dist >= 0.2:
        return "dropped"
    return "tilted" if tilt >= 45 else "other"


def plan_reason(t):
    """A planner rejection's reason without its numbers (frame indices, millimetres, box ids)."""
    fail = t.get("plan", {}).get("fail", "")
    fail = re.sub(r"at frame \d+", "", fail)
    fail = re.sub(r"#\d+", "", fail)
    fail = re.sub(r"\([^()]*\d[^()]*\)", "", fail)  # (err 7.1 mm, 3.0 deg), (target gap 13 mm there), ...
    fail = re.sub(r"-?\d+(\.\d+)? ?(mm|m|deg|rad/frame)\b", "", fail)
    fail = re.sub(r"^\S+ in the support\s*$", "a link other than the hand in the support (idle arm, torso)", fail)
    return re.sub(r"\s+", " ", fail).strip()


def aggregate(files):
    trials, runs, plans, rot = [], [], Counter(), []
    for f in files:
        d = json.loads(resolve(f).read_text())
        phys = [t for t in d["trials"] if "lift_m" in t]
        trials += phys
        runs.append(any(t.get("success") for t in phys))
        if d.get("reason"):  # no grasp passed the gate, no accepted placement
            plans[d["reason"]] += 1
        for t in d["trials"]:
            if not t.get("ik_ok", True):
                plans["no IK"] += 1
            elif "lift_m" not in t:
                plans[plan_reason(t) or "planned, not validated"] += 1
        rot += [t["inhand_rot_deg"] for t in phys if t.get("inhand_rot_deg") is not None and t.get("lift_m", 0) > 0.03]
    k = sum(bool(t.get("success")) for t in trials)
    lo, hi = wilson(k, len(trials))
    rk = sum(runs)
    rlo, rhi = wilson(rk, len(runs))
    rot.sort()
    return {"trials": {"n": len(trials), "success": k, "rate": k / len(trials) if trials else None, "ci95": [lo, hi]},
            "runs": {"n": len(runs), "with_success": rk, "rate": rk / len(runs) if runs else None, "ci95": [rlo, rhi]},
            "physics": dict(Counter(physics_outcome(t) for t in trials)),
            "inhand_rot_deg": {"n": len(rot), "median": rot[len(rot) // 2] if rot else None,
                               "max": rot[-1] if rot else None},
            "planning": dict(plans.most_common()), "files": [rel(f) for f in files]}


def fmt(name, a):
    t, r = a["trials"], a["runs"]
    if not t["n"]:
        return f"{name}: no physics trial ({a['planning']})"
    return (f"{name}: trials {t['success']}/{t['n']} = {100 * t['rate']:.0f} % [95 % CI {100 * t['ci95'][0]:.0f}-"
            f"{100 * t['ci95'][1]:.0f} %], runs with a success {r['with_success']}/{r['n']} | physics {a['physics']} | "
            f"in-hand rotation median {a['inhand_rot_deg']['median'] or 0:.0f} max {a['inhand_rot_deg']['max'] or 0:.0f} "
            f"deg | planning {a['planning']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--robot", default=DEFAULT_ROBOT)
    ap.add_argument("--tags", nargs="*", default=["twin"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--trials", type=int, default=12, help="physics rollouts per scene and seed")
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--synth-seeds", nargs="*", type=int, default=None,
                    help="re-synthesize the grasps with each seed (BODex hands); default: the existing grasps")
    ap.add_argument("--prefix", default="", help="launcher prepended to every GPU job")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()
    s = Path(args.session).resolve()
    spec = load_spec(args.robot)
    bodex = any(a["ee"].get("bodex") for a in spec["arms"].values())
    pre = shlex.split(args.prefix)
    stats = s / "episodes" / "_stats"
    synth_seeds = args.synth_seeds if (bodex and args.synth_seeds and not args.aggregate_only) else [None]
    report = {"session": s.name, "robot": args.robot, "by_synth_seed": {}, "grasps_kept": {}}
    for ks in synth_seeds:
        if ks is not None and not args.aggregate_only:
            log = subprocess.run(pre + ["env", f"PYTHONPATH={ROOT / 'src'}", "MEM=16G", str(ROOT / "tools/run_capped.sh"),
                                        str(ROOT / "envs/grasp/bin/python"), "-m", "calibrated_scene_reconstruction.grasping.synth", str(s),
                                        "--robot", args.robot, "--variants", *args.tags, "--seed", str(ks), "--force"],
                                 cwd=ROOT, capture_output=True, text=True, check=False).stdout
            report["grasps_kept"][ks] = [int(m) for m in re.findall(r"(\d+) pass the quality gate", log)]
            print(f"synthesis seed {ks}: {report['grasps_kept'][ks]} grasps pass the quality gate", flush=True)
        files = []
        for tag in args.tags:
            etag = tag if args.robot == DEFAULT_ROBOT else f"{tag}__{args.robot}"
            if args.aggregate_only:
                files += sorted(stats.glob(f"{etag}__s*.json"))
                continue
            for seed in args.seeds:
                f = stats / f"{etag}__s{seed}.json"
                f.unlink(missing_ok=True)
                t_run = time.time()
                var = [] if tag == "twin" else ["--variant", str(s / "cousins" / f"{tag}.json")]
                subprocess.run(pre + [str(ROOT / "tools/run_isaac.sh"), "apps/episode.py", str(s), "--tag", tag,
                                      *var, "--robot", args.robot, "--seed", str(seed), "--episodes", "999",
                                      "--max-trials", str(args.trials), "--batch", str(args.batch), "--no-record"],
                               cwd=ROOT, capture_output=True, text=True, check=False)
                res = s / "episodes" / etag / "result.json"
                if not f.exists() and res.exists() and res.stat().st_mtime > t_run:  # stopped before any trial
                    stats.mkdir(parents=True, exist_ok=True)
                    f.write_text(json.dumps({"robot": args.robot, "tag": tag, "seed": seed, "trials": [],
                                             "reason": json.loads(res.read_text()).get("reason", "no trial")}))
                if f.exists():
                    files.append(f)
                    if ks is not None:  # keep this synthesis seed's runs apart from the next one's
                        f.rename(f.with_name(f"{f.stem}__k{ks}.json"))
                        files[-1] = f.with_name(f"{f.stem}__k{ks}.json")
                    print(fmt(f"  {tag} seed {seed}", aggregate([files[-1]])), flush=True)
        report["by_synth_seed"][str(ks)] = aggregate(files)
    allf = [f for a in report["by_synth_seed"].values() for f in a["files"]]
    report["all"] = aggregate(allf)
    print(fmt(f"{s.name} {args.robot}", report["all"]), flush=True)
    out = ROOT / "outputs" / "stats" / f"{s.name}__{args.robot}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
