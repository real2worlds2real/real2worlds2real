"""One DROID episode (LeRobot v2.1 port, e.g. the Hugging Face dataset cadene/droid_1.0.1) -> real/<demo>/.

    python -m calibrated_scene_reconstruction.datagen.demo.adapters.droid data/chunk-000/episode_000000.parquet real/<demo>
           [--video videos/chunk-000/observation.images.exterior_1_left/episode_000000.mp4] [--camera exterior_1_left]
           [--task TEXT] [--check]

Columns used: observation.state.cartesian_position (the arm's flange in the robot base frame: xyz + extrinsic xyz
Euler, DROID's convention; the TCP between the Robotiq 2F-85's fingertips is `tcp_offset` further along its z: projected
into the exterior camera, the flange at a close lands on the wrist, flange + 0.15 m on the grasped object), observation.state.gripper_position (0 open .. 1 closed; it stops at the object's width), timestamp,
observation.state.joint_position, language_instruction, camera_extrinsics.<camera> (the camera's pose in the base
frame, same convention; constant per episode for the exterior cameras). The photo is the camera's first frame (AV1:
decoded by ffmpeg's libdav1d on the CPU). `--check` runs the CPU half of demo re-targeting with objects inferred from
the gripper events (segment.event_objects): segments, success spec and re-targeted plans under object jitter.
"""
from calibrated_scene_reconstruction.core.adapters.droid import _col as _col, pose6_to_T as pose6_to_T  # noqa: F401
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .. import io

FPS = 15
TCP_OFFSET = 0.15  # flange -> fingertip centre of DROID's Robotiq 2F-85 along the flange's z (m)






def convert(parquet, out, camera="exterior_1_left", video=None, task=None, tcp_offset=TCP_OFFSET):
    import pandas as pd
    from calibrated_scene_reconstruction.core.adapters.droid import read_dataframe
    df = read_dataframe(parquet)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    T_base_ee = pose6_to_T(_col(df, "observation.state.cartesian_position")) @ io.make_T(p=[0.0, 0.0, tcp_offset])
    T_base_cam = pose6_to_T(_col(df, f"camera_extrinsics.{camera}")[0])
    traj = {"t": _col(df, "timestamp")[:, 0] if "timestamp" in df else np.arange(len(df)) / FPS,
            "ee_pose": io.T_to_pose7(T_base_ee), "gripper": _col(df, "observation.state.gripper_position")[:, 0]}
    if "observation.state.joint_position" in df:
        traj["q"] = _col(df, "observation.state.joint_position")
    np.savez(out / "traj.npz", **traj)
    text = task or (str(df["language_instruction"].iloc[0]).strip() if "language_instruction" in df else "")
    # gripper_max_width: the 2F-85's stroke; its position (0 open .. 1 closed) stops at what it holds, so the
    # opening while holding is (1 - position) * stroke (demo.generate: the held width, functional.classify)
    meta = {"robot": "franka_panda", "gripper": "robotiq_2f85", "gripper_max_width": 0.085, "side": "right",
            "task": text or "droid episode",
            "fps": FPS, "T_cam_base": io.inv(T_base_cam).tolist(),
            "tcp_offset": tcp_offset, "source": {"dataset": "droid", "episode": Path(parquet).stem, "camera": camera,
                       "success": bool(df["is_episode_successful"].iloc[0]) if "is_episode_successful" in df else None}}
    if video:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-c:v", "libdav1d", "-i", str(video), "-frames:v", "1",
                        str(out / "photo.png")], check=True)
        meta["photo"] = "photo.png"
    (out / "demo.json").write_text(json.dumps(meta, indent=1))
    return out


def check(demo_dir, n=50, seed=0, jitter=0.08, thr=0.3):
    """CPU re-targeting on a converted episode, in the robot base frame (world = base): no twin, so the objects are
    inferred from the gripper events. Returns a summary dict."""
    from .. import retarget, sampler, segment
    meta = json.loads((Path(demo_dir) / "demo.json").read_text())
    demo = io.resample(io.load(demo_dir, io.inv(np.asarray(meta["T_cam_base"]))), 30)
    p_cam = (io.inv(demo.T_world_base) @ demo.ee)[:, :3, 3]  # TCP in the camera: in front of it if the extrinsics hold
    objects = segment.event_objects(demo, thr)
    segs, final = segment.segment(demo, objects, thr=thr)
    spec = retarget.success_spec(segs, final, objects)
    cfg = sampler.merge(sampler.DEFAULT, {"objects": {k: {"xy": [jitter, jitter], "yaw_deg": 30} for k in objects}})
    rng, plans, err = np.random.default_rng(seed), 0, []
    for _ in range(n):
        poses = sampler.perturb_objects(objects, cfg, rng)
        p = retarget.plan(segs, poses, demo.ee[0], 30)
        if p is None:
            continue
        plans += 1
        for sg in segs:  # the TCP at each event, in its reference's new frame, equals the demo's
            if sg.ref and sg.event != "end":
                k = np.flatnonzero((p["segment"] == sg.index) & (p["phase"] == retarget.PHASE["skill"]))[-1]
                err.append(np.abs(io.inv(poses[sg.ref]) @ p["ee"][k] - sg.ee_rel[-1]).max())
    return {"frames": len(demo.t), "events": segment.gripper_events(demo.gripper, thr),
            "tcp_in_front_of_camera": float(np.mean(p_cam[:, 2] > 0)),
            "segments": [s.to_dict() for s in segs], "success": spec,
            "success_on_demo": None if spec is None else bool(retarget.check_success(spec, final)),
            "plans": f"{plans}/{n}", "event_pose_err_max": float(max(err)) if err else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("out")
    ap.add_argument("--video", default=None)
    ap.add_argument("--camera", default="exterior_1_left")
    ap.add_argument("--task", default=None)
    ap.add_argument("--tcp-offset", type=float, default=TCP_OFFSET, help="flange -> TCP along the flange's z (m)")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    out = convert(a.parquet, a.out, a.camera, a.video, a.task, a.tcp_offset)
    print(f"-> {out}")
    if a.check:
        print(json.dumps(check(out), indent=1, default=str))


if __name__ == "__main__":
    main()
