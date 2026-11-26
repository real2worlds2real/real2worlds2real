"""A real robot recording and its time alignment.

real/<name>/
  meta.json        {"robot": configs/robots/<robot>, "session": "sessions/<scene>", "object": SAM 3 prompt / twin name,
                    "joints": [dof names, the column order of q], "gripper": "command" | null,
                    "base": [x, y, yaw_deg] of the robot in the world frame, optional "base_z" (no session),
                    "video_offset_s": optional, video time of trajectory t = 0 (estimated when absent),
                    "ee_link": optional, the robot link whose pose ee_pose records (kinematic check of the replay),
                    "twin": {"support", "shape", "mass"?} without a session (real2sim.tabletop -> twin.json)}
  trajectory.csv   header `t,<joint>...[,gripper][,ee_x,ee_y,ee_z,ee_qw,ee_qx,ee_qy,ee_qz]`, t in seconds
  trajectory.npz   or arrays t (T,), q (T, J), optional gripper (T,), ee_pose (T, 7) [xyz + wxyz]
  video.mp4        the fixed camera
  camera.json      {"K": 3x3, "width", "height", "T_world_cam": 4x4 camera-to-world, OpenCV axes (x right, y down,
                    z forward), in the twin's world frame; optional "dist": OpenCV distortion coefficients}
  events.json      optional hand labels in video seconds: {"lift": t, "slip": t, "drop": t, "settle": t}
"""
from calibrated_scene_reconstruction.physics.common.schema import read_document
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from calibrated_scene_reconstruction.core.paths import ROOT

REAL_DIR = ROOT / "real"
EE_COLS = ("ee_x", "ee_y", "ee_z", "ee_qw", "ee_qx", "ee_qy", "ee_qz")


@dataclass
class Recording:
    name: str
    meta: dict
    t: np.ndarray                     # (T,) seconds, strictly increasing, t[0] = 0
    q: np.ndarray                     # (T, J) joint positions in meta["joints"] order
    gripper: np.ndarray = None        # (T,) command, 0 open .. 1 closed
    ee_pose: np.ndarray = None        # (T, 7)
    camera: dict = field(default_factory=dict)
    events: dict = field(default_factory=dict)
    video: Path = None
    path: Path = None                 # the recording directory

    @property
    def joints(self):
        return list(self.meta["joints"])

    @property
    def K(self):
        return np.asarray(self.camera["K"], float)

    @property
    def T_world_cam(self):
        return np.asarray(self.camera["T_world_cam"], float)


def read_csv(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    head, body = rows[0], np.array([[float(v) for v in r] for r in rows[1:] if r], float)
    return {h.strip(): body[:, i] for i, h in enumerate(head)}


def load(path):
    """Load and validate real/<name>/ (or any directory of that layout); raises ValueError on a bad recording."""
    d = Path(path)
    if not d.is_absolute() and not d.exists():
        d = REAL_DIR / d
    meta = read_document(d / "meta.json")
    for k in ("robot", "joints"):
        if k not in meta:
            raise ValueError(f"{d / 'meta.json'}: missing '{k}'")
    joints = list(meta["joints"])
    if (d / "trajectory.npz").exists():
        z = np.load(d / "trajectory.npz")
        t, q = np.asarray(z["t"], float), np.asarray(z["q"], float)
        grip = np.asarray(z["gripper"], float) if "gripper" in z else None
        ee = np.asarray(z["ee_pose"], float) if "ee_pose" in z else None
    elif (d / "trajectory.csv").exists():
        cols = read_csv(d / "trajectory.csv")
        missing = [c for c in ["t"] + joints if c not in cols]
        if missing:
            raise ValueError(f"{d / 'trajectory.csv'}: missing columns {missing}")
        t, q = cols["t"], np.stack([cols[j] for j in joints], 1)
        grip = cols.get("gripper")
        ee = np.stack([cols[c] for c in EE_COLS], 1) if all(c in cols for c in EE_COLS) else None
    else:
        raise ValueError(f"{d}: no trajectory.csv or trajectory.npz")
    if q.ndim != 2 or q.shape != (len(t), len(joints)):
        raise ValueError(f"q has shape {q.shape}, expected ({len(t)}, {len(joints)}) (meta joints)")
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(t)):
        raise ValueError("non-finite samples in the trajectory")
    if np.any(np.diff(t) <= 0):
        raise ValueError("trajectory timestamps must be strictly increasing")
    for a, n in ((grip, "gripper"), (ee, "ee_pose")):
        if a is not None and len(a) != len(t):
            raise ValueError(f"{n} has {len(a)} samples, t has {len(t)}")
    cam = read_document(d / "camera.json") if (d / "camera.json").exists() else {}
    if cam:
        validate_camera(cam)
    ev = read_document(d / "events.json") if (d / "events.json").exists() else {}
    video = d / meta.get("video", "video.mp4")
    return Recording(d.name, meta, t - t[0], q, grip, ee, cam, ev, video if video.exists() else None, d)


def validate_camera(cam):
    K, T = np.asarray(cam.get("K"), float), np.asarray(cam.get("T_world_cam"), float)
    if K.shape != (3, 3) or T.shape != (4, 4):
        raise ValueError("camera.json: K must be 3x3 and T_world_cam 4x4")
    R = T[:3, :3]
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-4) or np.linalg.det(R) < 0:
        raise ValueError("camera.json: T_world_cam rotation is not a proper rotation")
    if not (cam.get("width") and cam.get("height")):
        raise ValueError("camera.json: width / height required")


def resample(t, x, rate, hold=False, t_end=None):
    """Samples at `rate` Hz from 0 to t_end (default t[-1]): linear interpolation per column, or zero-order hold
    (`hold`, for discrete commands such as the gripper). Returns (ts, xs)."""
    t_end = t[-1] if t_end is None else t_end
    ts = np.arange(0.0, t_end + 1e-9, 1.0 / rate)
    x = np.asarray(x, float)
    x2 = x.reshape(len(t), -1)
    if hold:
        idx = np.clip(np.searchsorted(t, ts, side="right") - 1, 0, len(t) - 1)
        xs = x2[idx]
    else:
        xs = np.stack([np.interp(ts, t, c) for c in x2.T], 1)
    return ts, xs.reshape((len(ts),) + x.shape[1:])


def speed(t, q):
    """Joint-space speed |dq/dt| (T,), the trajectory's motion signal for offset estimation."""
    return np.linalg.norm(np.gradient(np.asarray(q, float), t, axis=0), axis=1)


def estimate_offset(t_a, a, t_b, b, max_lag=2.0, rate=100.0):
    """Lag (s) maximising the normalised cross-correlation of a(t) and b(t - lag): an event at time s on `b`'s clock
    shows up at s + lag on `a`'s. Typical use: a = video motion energy (video clock), b = joint speed
    (trajectory clock) -> the video time of trajectory t = 0. Returns (lag, peak correlation)."""
    t0, t1 = min(t_a[0], t_b[0]) - max_lag, max(t_a[-1], t_b[-1]) + max_lag
    g = np.arange(t0, t1, 1.0 / rate)

    def norm(t, x):
        y = np.interp(g, t, x, left=np.nan, right=np.nan)
        m = np.isfinite(y)
        y[m] = (y[m] - y[m].mean()) / (y[m].std() + 1e-12)
        return np.where(m, y, 0.0), m

    ya, ma = norm(t_a, a)
    yb, mb = norm(t_b, b)
    n = round(max_lag * rate)
    best = (0.0, -np.inf)
    for k in range(-n, n + 1):
        sb, sm = np.roll(yb, k), np.roll(mb, k)  # b shifted later by k samples
        if k > 0:
            sm[:k] = False
        elif k < 0:
            sm[k:] = False
        both = ma & sm
        if both.sum() < rate * 0.5:  # need half a second of overlap
            continue
        c = float((ya[both] * sb[both]).mean())
        if c > best[1]:
            best = (k / rate, c)
    return best


def video_motion(frames, mask=None):
    """Mean absolute frame difference per frame (T,) (first = second), optionally inside a region `mask` (H, W)."""
    f = np.asarray(frames, np.float32)
    if f.ndim == 4:
        f = f.mean(-1)
    d = np.abs(np.diff(f, axis=0))
    if mask is not None:
        d = d[:, mask]
    e = d.reshape(len(d), -1).mean(1)
    return np.concatenate([e[:1], e])


def align(rec, rate, video_offset=None, time_scale=1.0):
    """The recording on the sim clock: dict ts (sim seconds from trajectory t = 0), q (N, J), gripper (N,) or None,
    and video_t (N,), the video time of each sim sample (offset + ts; the video clock is the wall clock)."""
    off = rec.meta.get("video_offset_s", 0.0) if video_offset is None else video_offset
    t = rec.t * time_scale  # time_scale > 1: the controller ran slower than its timestamps claim
    ts, q = resample(t, rec.q, rate)
    g = resample(t, rec.gripper, rate, hold=True)[1] if rec.gripper is not None else None
    return {"ts": ts, "q": q, "gripper": g, "video_t": off + ts}
