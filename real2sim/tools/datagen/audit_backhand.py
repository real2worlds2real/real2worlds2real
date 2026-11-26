"""Grasps a right hand would not take, out of a session's recorded episodes: each episode's hand as recorded when the
lift starts (steps.npz tcp_pose: the hand's base link) against calibrated_scene_reconstruction.grasping.synth.natural_grasp in the frame of the
base it stood on (result.json base yaw): palm not facing the robot's right, thumb neither down nor right. An episode
that breaks it moves to episodes/_rejected/ (kept for a look, out of every export).

    PYTHONPATH=src envs/recon/bin/python tools/audit_backhand.py sessions/<name> [--robot vega_1p_sharpa] [--dry-run]
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from calibrated_scene_reconstruction.core.paths import robot_asset_path
from calibrated_scene_reconstruction.grasping.synth import hand_axes, natural_grasp  # noqa: E402
from calibrated_scene_reconstruction.grasping.trajgen import SEG  # noqa: E402
from calibrated_scene_reconstruction.robots.spec import load_spec  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--robot", default="vega_1p_sharpa")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    s = Path(a.session)
    spec = load_spec(a.robot)
    side = next(iter(spec["arms"]))
    axes = hand_axes(robot_asset_path(spec["arms"][side]["ee"]["bodex"]["asset"], required=True))
    rej = s / "episodes" / "_rejected"
    n = moved = 0
    for res in sorted(Path(s / "episodes").glob("*/result.json")):
        r = json.loads(res.read_text())
        steps = res.parent / "raw" / "steps.npz"
        if not r.get("success") or not steps.exists() or not r.get("base") or r.get("robot") != spec["name"]:
            continue
        z = np.load(steps)
        if "segment" not in z.files or "tcp_pose" not in z.files:  # an early recording without them
            continue
        lift = np.flatnonzero(z["segment"] == SEG["lift"])
        if not len(lift):
            continue
        n += 1
        R = Rotation.from_quat(z["tcp_pose"][int(lift[0]), 3:7], scalar_first=True).as_matrix()
        yaw = np.radians(float(r["base"][2]))
        robot = (np.array([np.cos(yaw), np.sin(yaw), 0.0]), np.array([-np.sin(yaw), np.cos(yaw), 0.0]),
                 np.array([0.0, 0.0, 1.0]))
        ok, m = natural_grasp(R, axes, robot, side)
        if not ok:
            moved += 1
            print(f"{res.parent.name}: " + ", ".join(f"{k} {v:+.2f}" for k, v in m.items()) + " -> _rejected")
            if not a.dry_run:
                rej.mkdir(exist_ok=True)
                shutil.move(str(res.parent), str(rej / res.parent.name))
    print(f"{n} episodes checked, {moved} with a grasp a right hand would not take"
          + (" (dry run)" if a.dry_run else ", moved to episodes/_rejected"))


if __name__ == "__main__":
    main()
