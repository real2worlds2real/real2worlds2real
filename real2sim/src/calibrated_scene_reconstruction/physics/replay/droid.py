"""One DROID episode (LeRobot v2.1 port, Hugging Face `cadene/droid_1.0.1`, Apache-2.0) -> a real2sim recording
real/<name>/ (calibrated_scene_reconstruction.physics.replay.data), in the robot base frame (world = base, so meta.base = [0, 0, 0]).

    python -m calibrated_scene_reconstruction.physics.replay.droid <episode.parquet> <camera.mp4> real/<name> --object "orange can"
           [--camera exterior_2_left] [--shape cylinder] [--support table] [--fx 132.5] [--t0 S --t1 S]

DROID records a Franka Panda with a Robotiq 2F-85 at 15 Hz; images and state share the frame clock, so
video_offset_s = 0. Columns used: observation.state.joint_position (7 joints, rad), observation.state.gripper_position
(0 open .. 1 closed), observation.state.cartesian_position (DROID's end-effector pose in the base frame: xyz + extrinsic
xyz Euler), camera_extrinsics.<camera> (the camera's pose in the base frame, same convention, OpenCV axes; constant for
the exterior cameras), timestamp. The port stores 320 x 180 AV1 video without intrinsics: K is the ZED 2's nominal
HD720 pinhole (fx = fy = 530 px at 1280 x 720, principal point at the centre) scaled to the stored size unless
--fx/--cx/--cy are given (check it with the TCP projection overlay, `overlay_tcp`). The video is transcoded to H.264
(lossless) so OpenCV reads it.
"""
from calibrated_scene_reconstruction.physics.common.schema import write_document
import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np

from calibrated_scene_reconstruction.core.adapters.droid import _col, pose6_to_T
from .data import REAL_DIR

JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
ZED2_HD720_FX = 530.0  # px at 1280 x 720 (2.12 mm lens): nominal, DROID's released data carries no intrinsics


def nominal_K(width, height, fx=None, cx=None, cy=None):
    f = fx if fx is not None else ZED2_HD720_FX * width / 1280.0
    return np.array([[f, 0, width / 2 if cx is None else cx], [0, f, height / 2 if cy is None else cy], [0, 0, 1.0]])


def wxyz(R):
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(R).as_quat(scalar_first=True)


def decode(video):
    """Frames (F, H, W, 3) uint8 of an AV1 (or any) video through ffmpeg (libdav1d when the stream is AV1)."""
    info = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(video)], capture_output=True, text=True, check=False).stderr
    m = re.search(r"Video: (\w+).*?, (\d{2,5})x(\d{2,5})", info)
    if not m:
        raise ValueError(f"{video}: no video stream")
    codec, w, h = m.group(1), int(m.group(2)), int(m.group(3))
    dec = ["-c:v", "libdav1d"] if codec == "av1" else []
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", *dec, "-i", str(video), "-f", "rawvideo", "-pix_fmt",
                          "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def encode(frames, out, fps):
    """Lossless H.264 (yuv444, qp 0) of RGB frames."""
    h, w = frames.shape[1:3]
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                    "-r", str(fps), "-i", "-", "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p", str(out)],
                   input=np.ascontiguousarray(frames).tobytes(), check=True)


def convert(parquet, video, out, object_prompt, camera="exterior_2_left", shape="cylinder", support="table",
            fx=None, cx=None, cy=None, t0=None, t1=None, fps=15.0, robot="franka_panda__robotiq_2f85"):
    """Write real/<name>/{meta.json, trajectory.npz, video.mp4, camera.json}; frames outside [t0, t1] (seconds from
    the episode start) are dropped from both. Returns the recording directory."""
    import pandas as pd

    from calibrated_scene_reconstruction.core.adapters.droid import read_dataframe
    df = read_dataframe(parquet)
    out = Path(out)
    if not out.is_absolute() and not out.parent.exists():
        out = REAL_DIR / out
    out.mkdir(parents=True, exist_ok=True)
    t = _col(df, "timestamp")[:, 0] if "timestamp" in df else np.arange(len(df)) / fps
    t = t - t[0]
    frames = decode(video)
    if len(frames) != len(df):
        raise ValueError(f"{video}: {len(frames)} frames, {parquet}: {len(df)} rows")
    keep = (t >= (t0 if t0 is not None else -np.inf)) & (t <= (t1 if t1 is not None else np.inf))
    T_ee = pose6_to_T(_col(df, "observation.state.cartesian_position"))[keep]
    ee = np.concatenate([T_ee[:, :3, 3], np.stack([wxyz(R) for R in T_ee[:, :3, :3]])], 1)
    np.savez(out / "trajectory.npz", t=t[keep] - t[keep][0], q=_col(df, "observation.state.joint_position")[keep],
             gripper=_col(df, "observation.state.gripper_position")[keep, 0], ee_pose=ee)
    encode(frames[keep], out / "video.mp4", fps)
    h, w = frames.shape[1:3]
    T_base_cam = pose6_to_T(_col(df, f"camera_extrinsics.{camera}")[0])
    K = nominal_K(w, h, fx, cx, cy)
    write_document(out / "camera.json", {"K": K.tolist(), "width": w, "height": h,
                                                 "T_world_cam": T_base_cam.tolist(),
                                                 "K_source": "given" if fx else "ZED 2 nominal (HD720) scaled"},
                                                indent=1)
    text = str(df["language_instruction"].iloc[0]).strip() if "language_instruction" in df else ""
    meta = {"robot": robot, "joints": JOINTS, "gripper": "command", "base": [0.0, 0.0, 0.0], "ee_link": "panda_link8",
            "object": object_prompt, "video_offset_s": 0.0,
            "twin": {"kind": "tabletop", "support": support, "shape": shape},
            "source": {"dataset": "droid_1.0.1", "camera": camera, "task": text,
                       "episode": int(df["episode_index"].iloc[0]) if "episode_index" in df else Path(parquet).stem,
                       "t0": float(t[keep][0]), "frames": int(keep.sum()),
                       "real_gripper": "robotiq_2f85", "success": bool(df["is_episode_successful"].iloc[0])
                       if "is_episode_successful" in df else None}}
    write_document(out / "meta.json", meta, indent=1)
    return out


def overlay_tcp(rec, frames, every=1):
    """Frames with the recorded TCP (trajectory ee_pose) projected through camera.json: a red dot per frame, the
    whole path in yellow. The check of the extrinsics and K before anything is fitted."""
    from .observe import project

    uv = project(rec.K, rec.T_world_cam, rec.ee_pose[:, :3])[0]
    vis = np.array(frames[::every])
    h, w = vis.shape[1:3]
    for k, f in enumerate(vis):
        for j, p in enumerate(uv):
            if np.all(np.isfinite(p)) and 0 <= p[0] < w and 0 <= p[1] < h:
                r = 2 if j == min(k * every, len(uv) - 1) else 0
                f[max(int(p[1]) - r, 0):int(p[1]) + r + 1, max(int(p[0]) - r, 0):int(p[0]) + r + 1] = \
                    (255, 0, 0) if r else (255, 255, 0)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--object", required=True, help="SAM 3 prompt of the moved object")
    ap.add_argument("--camera", default="exterior_2_left")
    ap.add_argument("--shape", default="cylinder", choices=("cylinder", "box"))
    ap.add_argument("--support", default="table", help="SAM 3 prompt of the support surface")
    ap.add_argument("--fx", type=float)
    ap.add_argument("--cx", type=float)
    ap.add_argument("--cy", type=float)
    ap.add_argument("--t0", type=float)
    ap.add_argument("--t1", type=float)
    ap.add_argument("--robot", default="franka_panda__robotiq_2f85", help="configs/robots spec (DROID: Robotiq hand)")
    a = ap.parse_args()
    out = convert(a.parquet, a.video, a.out, a.object, a.camera, a.shape, a.support, a.fx, a.cx, a.cy, a.t0, a.t1,
                  robot=a.robot)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
