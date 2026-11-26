"""The twin's metric scale and vertical from a calibrated demo (run after perceive, before the scene graph).

    python -m calibrated_scene_reconstruction.demo.frame sessions/<name> --demo real/<demo> [--no-scale] [--no-gravity] [--window 4]

A public dataset episode gives one low-resolution frame and a calibrated camera; the photo's monocular point map
(MoGe-2) is metric only up to its own estimate, and a floor fit needs a large visible floor. The demo knows both:
* scale: the TCP at the demo's first close sits on the object it grasps, so its range from the camera over the point
  map's range at the TCP's pixel (median over a small window) is the point map's scale error; parsing/geometry.npz
  (points, depth) is multiplied by it, the original kept as geometry_moge.npz.
* vertical: a fixed-base arm stands upright, its base z is the world's z (instead of the floor fit); x is the camera's
  right on the floor, the origin under the camera, the floor at the median height of the `floor` instances (else the
  1st percentile of all heights). parsing/world_frame.json is written; recon.autograph keeps an existing frame.
Both are printed with the residual they correct (scale factor, angle between the base z and the floor fit).
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from calibrated_scene_reconstruction.scene.session import Session
from calibrated_scene_reconstruction.demo import io, segment


def tcp_scale(points, valid, K, p_tcp, window=4):
    """Range of the TCP (camera frame) over the point map's range at its pixel (median in a (2w+1)^2 window)."""
    H, W = points.shape[:2]
    Kp = K * np.array([[W], [H], [1.0]]) if K[0, 2] < 2 else K  # normalized intrinsics
    uv = Kp @ p_tcp
    u, v = (int(x) for x in np.rint(uv[:2] / uv[2]))
    if not (0 <= u < W and 0 <= v < H):
        raise ValueError(f"TCP projects outside the photo ({u}, {v})")
    sl = (slice(max(0, v - window), v + window + 1), slice(max(0, u - window), u + window + 1))
    r = np.linalg.norm(points[sl][valid[sl]], axis=1)
    if not len(r):
        raise ValueError("no valid depth at the TCP's pixel")
    return float(np.linalg.norm(p_tcp) / np.median(r)), (u, v)


def base_frame(n, floor_h):
    """World frame with z = n (camera frame, up), x = camera right projected, origin on the floor under the camera."""
    n = np.asarray(n, float) / np.linalg.norm(n)
    x = np.array([1.0, 0, 0]) - n[0] * n
    x /= np.linalg.norm(x)
    R_wc = np.stack([x, np.cross(n, x), n])
    o = floor_h * n
    T = io.make_T(R_wc, -R_wc @ o)
    return R_wc, o, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--demo", required=True)
    ap.add_argument("--no-scale", action="store_true")
    ap.add_argument("--no-gravity", action="store_true")
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--thr", type=float, default=0.3, help="gripper threshold (DROID: it stops at the object)")
    a = ap.parse_args()
    s = Session(a.session)
    p = Path(a.demo)
    meta = json.loads((p / "demo.json").read_text())
    T_cam_base = np.asarray(meta["T_cam_base"], float)
    z = np.load(p / "traj.npz")
    ee_cam = T_cam_base @ io.pose7_to_T(z["ee_pose"])
    f_close = next(f for f, c in segment.gripper_events(z["gripper"], a.thr) if c)
    gpath = s / "parsing" / "geometry.npz"
    g = dict(np.load(gpath))
    out = {"close_frame": int(f_close)}
    if not a.no_scale:
        k, uv = tcp_scale(g["points"], g["mask"], g["K"], ee_cam[f_close][:3, 3], a.window)
        if not (s / "parsing" / "geometry_moge.npz").exists():
            shutil.copy(gpath, s / "parsing" / "geometry_moge.npz")
        g0 = dict(np.load(s / "parsing" / "geometry_moge.npz"))
        g["points"], g["depth"] = g0["points"] * k, g0["depth"] * k
        np.savez_compressed(gpath, **g)
        out.update(scale=k, tcp_pixel=uv)
        print(f"point map scale x{k:.3f} (TCP at the close, pixel {uv})")
    if not a.no_gravity:
        n = T_cam_base[:3, 2]
        P, M = g["points"], g["mask"]
        inst = json.loads((s / "parsing" / "instances.json").read_text())
        masks = s.masks()
        fm = np.zeros(M.shape, bool)
        for i, x in enumerate(inst):
            if x["label"] == "floor":
                fm |= masks[i]
        h = P[fm & M] @ n if (fm & M).sum() > 500 else None
        floor_h = float(np.median(h)) if h is not None else float(np.percentile(P[M] @ n, 1))
        if h is not None:  # how far the floor fit's normal is from the base's z
            c = P[fm & M].mean(0)
            _, V = np.linalg.eigh(np.cov((P[fm & M] - c).T))
            out["floor_fit_vs_base_z_deg"] = float(np.degrees(np.arccos(abs(V[:, 0] @ n))))
        R_wc, o, T = base_frame(n, floor_h)
        (s / "parsing" / "world_frame.json").write_text(json.dumps(
            {"R_wc": R_wc.tolist(), "o": o.tolist(), "T_world_cam_cv": T.tolist(), "camera_height_m": -floor_h,
             "source": f"robot base z of {p.name}"}, indent=1))
        out.update(camera_height_m=-floor_h)
        print(f"world z = robot base z, camera {-floor_h:.3f} m above the floor"
              + (f"; floor fit {out['floor_fit_vs_base_z_deg']:.1f} deg off" if "floor_fit_vs_base_z_deg" in out else ""))
    s.write_json("parsing/demo_frame.json", out)


if __name__ == "__main__":
    main()
