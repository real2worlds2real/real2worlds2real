"""One demonstration of a task mapped into the twin's world frame (the photo camera's floor frame: Session.world_frame).

real/<demo>/
  demo.json   {"robot": "franka_panda", "side": "right", "task": "put the cup in the basket", "fps": 30,
               "photo": "photo.png", "T_cam_base": 4x4 robot base in the photo camera (OpenCV axes)
               | "calibration": "calib.json"}
  traj.npz    t (T,) seconds; ee_pose (T, 7: xyz + quat wxyz, TCP in the robot base frame) and/or q (T, D);
              gripper (T,) command, 0 open .. 1 closed (a hand: its closure, any monotone scalar)
  calib.json  {"p_base": (N, 3), "p_cam": (N, 3)}: N >= 3 non-collinear TCP positions from the robot's FK and the same
              points in the photo camera (a marker on the gripper photographed from the photo's viewpoint, depth from
              parsing/geometry.npz); fit_rigid gives T_cam_base

The photo is the one the twin is reconstructed from (or taken from the same viewpoint), so p_world = T_world_cam @
T_cam_base @ p_base. A joint-only demo needs the robot's FK (the live articulation in Isaac Sim: pass `fk`).
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def pose7_to_T(p):
    """xyz + quat wxyz (the exporter's tcp_pose / object_pose layout) -> 4x4; works on (..., 7)."""
    p = np.asarray(p, float)
    T = np.zeros(p.shape[:-1] + (4, 4))
    T[..., :3, :3] = Rotation.from_quat(p[..., [4, 5, 6, 3]].reshape(-1, 4)).as_matrix().reshape(p.shape[:-1] + (3, 3))
    T[..., :3, 3] = p[..., :3]
    T[..., 3, 3] = 1.0
    return T


def T_to_pose7(T):
    T = np.asarray(T, float)
    q = Rotation.from_matrix(T[..., :3, :3].reshape(-1, 3, 3)).as_quat().reshape(T.shape[:-2] + (4,))
    return np.concatenate([T[..., :3, 3], q[..., [3, 0, 1, 2]]], -1)


def inv(T):
    T = np.asarray(T, float)
    out = np.zeros_like(T)
    Rt = np.swapaxes(T[..., :3, :3], -1, -2)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -(Rt @ T[..., :3, 3:])[..., 0]
    out[..., 3, 3] = 1.0
    return out


def make_T(R=None, p=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = p
    return T


def fit_rigid(P_src, P_dst):
    """Kabsch: the rigid T (no scale) with P_dst ~ T @ P_src, and the rms residual. Calibration: P_src the TCP in the
    base frame, P_dst the same points in the photo camera -> T_cam_base."""
    P_src, P_dst = np.asarray(P_src, float), np.asarray(P_dst, float)
    if len(P_src) < 3:
        raise ValueError("fit_rigid needs >= 3 correspondences")
    cs, cd = P_src.mean(0), P_dst.mean(0)
    U, _, Vt = np.linalg.svd((P_src - cs).T @ (P_dst - cd))  # 3x3
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    T = make_T(R, cd - R @ cs)
    res = P_dst - (P_src @ R.T + T[:3, 3])
    return T, float(np.sqrt((res ** 2).sum(1).mean()))


@dataclass
class Demo:
    robot: str
    task: str
    fps: float
    t: np.ndarray            # (T,)
    ee: np.ndarray           # (T, 4, 4) TCP in the world frame
    gripper: np.ndarray      # (T,) 0 open .. 1 closed
    T_world_base: np.ndarray
    side: str = "right"
    q: np.ndarray = None     # (T, D) when recorded
    meta: dict = field(default_factory=dict)


def load(path, T_world_cam, fk=None):
    """real/<demo> -> Demo in the world frame. T_world_cam: Session.world_frame()[2] (OpenCV camera in the world)."""
    d = Path(path)
    meta = json.loads((d / "demo.json").read_text())
    z = np.load(d / "traj.npz")
    if "T_cam_base" in meta:
        T_cam_base = np.asarray(meta["T_cam_base"], float)
    elif "calibration" in meta:
        c = json.loads((d / meta["calibration"]).read_text())
        T_cam_base, rms = fit_rigid(c["p_base"], c["p_cam"])
        meta["calibration_rms_m"] = rms
    else:
        raise ValueError(f"{d}/demo.json: needs T_cam_base or calibration")
    T_world_base = np.asarray(T_world_cam, float) @ T_cam_base
    q = z["q"] if "q" in z.files else None
    if "ee_pose" in z.files:
        ee_base = pose7_to_T(z["ee_pose"])
    elif q is not None and fk is not None:
        ee_base = np.stack([fk(qi) for qi in q])
    else:
        raise ValueError(f"{d}/traj.npz: needs ee_pose, or q with an fk")
    fps = float(meta.get("fps", 30))
    t = z["t"] if "t" in z.files else np.arange(len(ee_base)) / fps
    return Demo(robot=meta["robot"], task=meta["task"], fps=fps, t=np.asarray(t, float),
                ee=T_world_base @ ee_base, gripper=np.asarray(z["gripper"], float), T_world_base=T_world_base,
                side=meta.get("side", "right"), q=q, meta=meta)


def resample(demo, fps):
    """The demo at a fixed rate (the sim records at 30 fps): linear positions, slerped orientations, nearest gripper."""
    from scipy.spatial.transform import Slerp
    t_new = np.arange(demo.t[0], demo.t[-1] + 1e-9, 1.0 / fps)
    ee = np.zeros((len(t_new), 4, 4))
    ee[:, 3, 3] = 1.0
    for k in range(3):
        ee[:, k, 3] = np.interp(t_new, demo.t, demo.ee[:, k, 3])
    ee[:, :3, :3] = Slerp(demo.t, Rotation.from_matrix(demo.ee[:, :3, :3]))(t_new).as_matrix()
    idx = np.clip(np.searchsorted(demo.t, t_new), 0, len(demo.t) - 1)
    return Demo(robot=demo.robot, task=demo.task, fps=fps, t=t_new, ee=ee, gripper=demo.gripper[idx],
                T_world_base=demo.T_world_base, side=demo.side, q=None, meta=demo.meta)
