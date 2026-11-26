"""The states a policy failed in, from an evaluation's per-frame traces: plan starts for a DAgger collection that
resumes where the policy went wrong (apps/episode.py --resume-states) instead of from where it began.

    envs/recon/bin/python tools/fail_states.py sessions/<name> <run> --out-letter b [--kinds miss drop]

For each rollout trace (eval/<run>/<tag>__<robot>__place_<k>_trace.npy), identify the first failure event:
  miss   a grasp that closed (hand closure >= 0.5 of the recorded grasp's, from the rollout's open start) without
         lifting the cup: the state 1 s later
  drop   the cup lifted (> 3 cm), then back on the support (< 1 cm): the state 0.5 s later
  freeze the cup held up (> 2 cm) while the hand's net motion over 2 s stays under 4 cm, for 3 s: the state 1 s in
         (the cup in the hand: kept for a planner that starts from a held cup, not used by --resume-states)
  push   the cup slid >= 1 cm on the support before the hand ever closed, in a rollout that never lifted it (the hand
         nudging the cup away and stopping short of it): the state 1 s later
  knock  the cup tipped over (> 45 deg) before any lift: the state 1 s later, the cup lying on the support
Only states with the cup resting upright on the support (miss, drop, push) are kept for those kinds; knock keeps the
cup lying (within 6 cm of its start height). A rollout counts as failed by the evaluation's criterion
(calibrated_scene_reconstruction.tasks.rigid.success.on_mat). The configurations are
copied under --out-letter (tools/copy_configs.py, the copy of the run's tag), and each copy's states go to
eval/<run>/fail/<copy tag>__<robot>__place_fail_states.npz: q (N, joints), base (N, 3), cup (N, 7: xyz + wxyz),
kind (N,), t (N,), k (N,).

The trace columns: TCP xyz, cup xyz, the joint positions, TCP rotation (9), cup quaternion wxyz. A rollout's base is
its own "base" (recorded since the cup-reset fix) or, for older new-layout runs, the k-th distinct base in the run's
<tag>_states.npz (every rollout's first policy state carries its base).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from calibrated_scene_reconstruction.tasks.rigid.success import on_mat  # noqa: E402

from calibrated_scene_reconstruction.core.paths import ROOT
FPS = 30


def hand_postures(s, src, robot):
    """(right-hand joint indices, the hand's joints over the recording) of a recording of the configuration."""
    d = s / "episodes" / src
    names = json.loads((d / "raw" / "meta.json").read_text())["joint_names"]
    hi = np.array([i for i, n in enumerate(names) if n.startswith("right_")])
    return hi, np.load(d / "raw" / "steps.npz")["q"][:, hi]


def first_recording(s, tag, robot):
    """The configuration's first recording, else its source's (a copy that recorded nothing)."""
    reg = s / "dr" / "_copies.json"
    for t_ in (tag, json.loads(reg.read_text()).get(tag) if reg.exists() else None):
        for d in sorted((s / "episodes").glob(f"{t_}__{robot}__place*")) if t_ else ():
            if (d / "raw" / "steps.npz").exists() and json.loads((d / "result.json").read_text()).get("base"):
                return d.name
    return None


def events(tr, hi, rec, nj, every=False):
    """[(kind, frame, delay)] of the failure events of a rollout's trace: the event's frame and the kind's own delay to
    the state taken (miss, push, knock 1 s; drop 0.5 s; freeze 1 s into the stillness); the first miss only, or with
    `every` each grasp attempt that closed without lifting. Closure: the hand's joint travel from the rollout's first
    frame (the open hand every rollout starts from) over the travel to the recording's frame farthest from it (a
    recording may itself start half closed, from a policy state)."""
    q = tr[:, 6:6 + nj][:, hi]
    clo = np.abs(q - q[0]).sum(1) / max(np.abs(rec - q[0]).sum(1).max(), 1e-6)
    cup, tcp = tr[:, 3:6], tr[:, :3]
    dz = cup[:, 2] - cup[0, 2]
    n, out = len(tr), []
    i = 0
    while i < n:  # grasp attempts: closure up through 0.5, until it drops below 0.3
        if clo[i] >= 0.5:
            j0 = i
            while i < n and clo[i] > 0.3:
                i += 1
            if (dz[j0:i] > 0.03).any():
                break  # lifted: no miss (a later fall is a drop)
            out.append(("miss", j0, FPS))
            if not every:
                break
        i += 1
    up = np.flatnonzero(dz > 0.03)
    if len(up):
        down = np.flatnonzero((np.arange(n) > up[0]) & (dz < 0.01))
        if len(down):
            out.append(("drop", int(down[0]), FPS // 2))
    if not (dz > 0.03).any():  # never lifted: a slide before the hand closed
        slid = np.flatnonzero(np.linalg.norm(cup[:, :2] - cup[0, :2], axis=1) >= 0.01)
        if len(slid) and clo[:slid[0] + 1].max() < 0.5:
            out.append(("push", int(slid[0]), FPS))
    up_ = np.r_[tr[:, 15 + nj:19 + nj]]
    tilt = np.degrees(np.arccos(np.clip(Rotation.from_quat(up_, scalar_first=True).apply([0, 0, 1])[:, 2], -1, 1)))
    tipped = np.flatnonzero(tilt > 45)
    if len(tipped) and (not len(up) or tipped[0] < up[0]):
        out.append(("knock", int(tipped[0]), FPS))
    w = 2 * FPS
    net = np.r_[np.linalg.norm(tcp[w:] - tcp[:-w], axis=1), np.full(w, np.inf)]
    still = (dz > 0.02) & (net < 0.04)
    run = 0
    for t_ in range(n):
        run = run + 1 if still[t_] else 0
        if run == 3 * FPS:
            out.append(("freeze", t_ - 3 * FPS, FPS))
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("run")
    ap.add_argument("--robot", default="vega_1p_sharpa_arm")
    ap.add_argument("--out-letter", required=True, help="the configurations' copies (tools/copy_configs.py)")
    ap.add_argument("--kinds", nargs="*", default=["miss", "drop", "freeze"])
    ap.add_argument("--offsets", nargs="*", type=float, default=None,
                    help="s after each event to take a state at (more plan starts per failure), in place of the kind's "
                         "own delay")
    ap.add_argument("--every", action="store_true",
                    help="every usable event of a rollout (each missed grasp, then a drop, ...), not the first only")
    a = ap.parse_args()
    s = Path(a.session)
    run_dir = s / "eval" / a.run
    out_dir = run_dir / "fail"
    out_dir.mkdir(exist_ok=True)
    per_tag, counts = {}, {}
    for f in sorted(run_dir.glob(f"*__{a.robot}__place.json")):
        tag = f.name.split("__")[0] + "__" + f.name.split("__")[1]
        src0 = first_recording(s, tag, a.robot)
        if src0 is None:
            continue
        hi, rec = hand_postures(s, src0, a.robot)
        sf = run_dir / f"{tag}__{a.robot}__place_states.npz"
        known = np.load(sf)["base"] if sf.exists() else None
        rolls = json.loads(f.read_text())["rollouts"]
        # the bases the rollouts stood on, in order: the policy states (<tag>_states.npz) start every rollout with its
        # own (one rollout per base with --new-layouts)
        seq = []
        for b_ in (known if known is not None else []):
            if not seq or np.abs(np.asarray(b_) - seq[-1]).max() > 1e-4:
                seq.append(np.asarray(b_, float))
        for r in rolls:
            tf_ = run_dir / f"{tag}__{a.robot}__place_{r['k']}_trace.npy"
            if on_mat(r) or not tf_.exists():
                continue
            tr = np.load(tf_)
            nj = tr.shape[1] - 6 - 9 - 4
            if np.linalg.norm(tr[min(2 * FPS, len(tr) - 1), 3:6] - tr[0, 3:6]) > 0.01:
                continue  # the cup knocked away before the policy acted: an evaluation fault
            if r.get("base"):
                base = np.asarray(r["base"], float)
            elif len(seq) == len(rolls):
                base = seq[r["k"]]
            else:
                continue  # no base to stand on
            for kind, ev, delay in sorted(events(tr, hi, rec, nj, a.every), key=lambda e: e[1]):
                if kind not in a.kinds:
                    continue
                took = 0
                for off in ([round(o * FPS) for o in a.offsets] if a.offsets and kind != "freeze" else [delay]):
                    j = min(ev + off, len(tr) - 1)
                    cup = np.r_[tr[j, 3:6], tr[j, 15 + nj:19 + nj]]
                    tilt = np.degrees(np.arccos(np.clip(Rotation.from_quat(cup[3:7], scalar_first=True).apply([0, 0, 1])[2], -1, 1)))
                    if kind == "knock" and abs(cup[2] - tr[0, 5]) > 0.06:
                        continue  # not lying on the support
                    if kind not in ("freeze", "knock") and (abs(cup[2] - tr[0, 5]) > 0.01 or tilt > 15):
                        continue  # not resting upright on the support
                    per_tag.setdefault(tag, []).append((tr[j, 6:6 + nj], base, cup, kind, j / FPS, r["k"]))
                    counts[kind] = counts.get(kind, 0) + 1
                    took += 1
                if took and not a.every:
                    break  # the first usable event of the rollout
    if not per_tag:
        sys.exit("no failure states")
    cp = subprocess.run([sys.executable, str(ROOT / "tools" / "copy_configs.py"), str(s), "--letter", a.out_letter,
                         "--sources", *sorted(per_tag)], capture_output=True, text=True, check=True)
    copies = json.loads((s / "dr" / "_copies.json").read_text())
    for copy, src in copies.items():
        if copy.split("__")[1][0] != a.out_letter or src not in per_tag:
            continue
        rows = per_tag[src]
        np.savez_compressed(out_dir / f"{copy}__{a.robot}__place_fail_states.npz",
                            q=np.stack([x[0] for x in rows]).astype(np.float32),
                            base=np.stack([x[1] for x in rows]).astype(np.float32),
                            cup=np.stack([x[2] for x in rows]).astype(np.float32),
                            kind=np.array([x[3] for x in rows]), t=np.array([x[4] for x in rows], np.float32),
                            k=np.array([x[5] for x in rows]))
    print(cp.stdout.strip().splitlines()[-1] if cp.stdout.strip() else "", flush=True)
    print(f"{sum(counts.values())} states over {len(per_tag)} configurations: {counts} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
