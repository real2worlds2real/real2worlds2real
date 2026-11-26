"""The real robot's base in the twin's world: its surface at the joint readout of the capture fitted to the splat.

This fixed-articulation ICP is a legacy diagnostic. Default robot_joints video
calibration uses visual_render_pipeline --joint-prior and independently checks
images, joint-command uncertainty and scale ambiguity. Frozen deliveries cannot
be rewritten by this command.

    tools/run_splat.sh -m calibrated_scene_reconstruction.recon.robot_fit sessions/<name> [--apply] [--anchor] [--points auto|splat|sparse]
    (after apps/robot_cloud.py and recon.splat_capture; with --points splat after splat_export)

apps/robot_cloud.py poses the spec's USD at the joints the robot held while the scene was filmed (configs/<name>/
robot_state.json) and writes its visible surface in the base frame. Here it is fitted to the trained splat before the
crop (export/full.ply at the board's scale: opaque, small Gaussians within `--radius` of the recorded base and above
the floor's layer): trimmed ICP (the closest `--keep` of the model's points, the hidden ones -- the base's underside,
the far side -- find nothing near), over the base's x, y, yaw and height, from a grid of starts round the recorded base
(recon.multiview's, from the photo's depth); then once more with a free scale, which the board's scale should leave
near 1 -- an independent check of it (or, with no board, the scale itself, as the robot's own size). Writes
splat/robot_fit.json and qa/splat/robot_fit.png (top and side views: grey the splat, orange the fitted robot, blue
the robot where the scene graph had it); `--apply` writes the base into the scene graph's photo_robot.base
(`_robot_fit` note), which placements' real base and the splat's robot column take.

`--points sparse`: COLMAP's sparse points (splat/data/sparse/full, tracks of 3 frames or more, reprojection under
1.5 px) in place of the splat's Gaussians -- no splat needed, so a mesh or completion twin takes the robot's scale
too (docs/twin_pipeline.md S1). `auto`:
the splat when it is exported, else the sparse points. The anchor composes with the placement before any earlier
anchor (align.json sim3_before_robot), so it can run again (the splat's Gaussians stand in the current placement)."""
import argparse
import json

import cv2
import numpy as np
from scipy.spatial import cKDTree

from calibrated_scene_reconstruction.scene.session import Session
from calibrated_scene_reconstruction.scene.background.splat.config import folder
from calibrated_scene_reconstruction.scene.review.zone_check import world_gaussians


def sparse_points(out, before_anchor=False):
    """COLMAP's sparse points in the twin's world (tracks of 3 frames or more, under 1.5 px), as (X, opacity 1, size 0):
    with `before_anchor` placed by the similarity before any robot anchor (the one the anchor corrects)."""
    import pycolmap

    al = json.loads((out / "align.json").read_text())
    sim = al.get("sim3_before_robot", al["sim3"]) if before_anchor else al["sim3"]
    rec = pycolmap.Reconstruction(str(out / "data" / "sparse" / "full"))
    pts = [p for p in rec.points3D.values() if p.track.length() >= 3 and p.error < 1.5]
    P = np.array([p.xyz for p in pts])
    X = float(sim["scale"]) * P @ np.asarray(sim["R"]).T + np.asarray(sim["t"])
    return X, np.ones(len(X)), np.zeros(len(X))


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def trimmed_icp(M, tree, S, x, y, z, yaw, keep, iters=60, scale=False, k0=1.0, floor=None, workers=1):
    """4-DOF (or 4-DOF + scale) trimmed ICP of model M (base frame) to scene S; returns pose, scale, residuals."""
    k = k0
    mz = float(M[:, 2].min())
    for _ in range(iters):
        T = k * M @ rot_z(yaw).T + [x, y, z]
        d, j = tree.query(T, workers=workers)
        sel = d <= np.quantile(d, keep)
        A, B = M[sel], S[j[sel]]
        # yaw + translation (+ scale) by least squares on the xy plane, z by the mean offset
        a0, b0 = A.mean(0), B.mean(0)
        Ac, Bc = A - a0, B - b0
        H = Ac[:, :2].T @ Bc[:, :2]
        yaw = float(np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1]))
        R = rot_z(yaw)
        if floor is not None:  # heights from the floor up: B_z - floor = k (A_z - lowest), xy as before
            Az, Bz = A[:, 2] - mz, B[:, 2] - floor
            if scale:
                k = float((np.sum((Ac[:, :2] @ R[:2, :2].T) * Bc[:, :2]) + np.sum(Az * Bz))
                          / (np.sum(Ac[:, :2] ** 2) + np.sum(Az ** 2)))
            t = b0 - k * a0 @ R.T
            x, y, z = float(t[0]), float(t[1]), floor - k * mz
            continue
        if scale:
            k = float(np.sum((Ac @ R.T) * Bc) / np.sum(Ac * Ac))
        t = b0 - k * a0 @ R.T
        x, y, z = float(t[0]), float(t[1]), float(t[2])
    T = k * M @ rot_z(yaw).T + [x, y, z]
    d, _ = tree.query(T, workers=workers)
    return x, y, z, yaw, k, d


def robot_start(s, out, sim):
    """None without the photo's frame in the model."""
    import pycolmap

    kf = s / "capture" / "keyframes.json"
    ids = (s.scene_graph.get("photo_robot") or {}).get("ids") or []
    if not kf.exists() or not ids:
        return None
    name = f"0_{json.loads(kf.read_text())['photo']:05d}.jpg"
    rec = pycolmap.Reconstruction(str(out / "data" / "sparse" / "full"))
    img = next((im for im in rec.images.values() if im.name == name), None)
    if img is None:
        return None
    cam = rec.cameras[img.camera_id]
    fx, fy, cx, cy = (float(v) for v in cam.params[:4])
    P = np.array([p.xyz for p in rec.points3D.values() if p.track.length() >= 3])
    Mc = np.asarray(img.cam_from_world().matrix())
    Xc = P @ Mc[:, :3].T + Mc[:, 3]
    ok = Xc[:, 2] > 0.05
    u, v = fx * Xc[:, 0] / Xc[:, 2] + cx, fy * Xc[:, 1] / Xc[:, 2] + cy
    masks = s.masks()
    m = np.any(masks[ids], 0)
    m = cv2.erode(m.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    H, W = m.shape
    ui, vi = (u * W / cam.width).astype(int), (v * H / cam.height).astype(int)
    ok &= (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ok[ok] &= m[vi[ok], ui[ok]]
    if ok.sum() < 30:
        return None
    d = Xc[ok, 2]
    near = d < np.percentile(d, 30) * 1.25  # the robot, not the wall behind it through its gaps
    X = float(sim["scale"]) * P[ok][near] @ np.asarray(sim["R"]).T + np.asarray(sim["t"])
    return np.median(X[:, :2], 0)


def floor_near(X, xy, r0=0.4, r1=1.5):
    """The floor's height round a place: the mode of the points below half a metre between r0 and r1 m from it."""
    d = np.hypot(X[:, 0] - xy[0], X[:, 1] - xy[1])
    m = (d > r0) & (d < r1) & (np.abs(X[:, 2]) < 0.5)
    if m.sum() < 50:
        return 0.0
    h, e = np.histogram(X[m, 2], bins=np.arange(-0.5, 0.5, 0.005))
    return float(e[np.argmax(h)] + 0.0025)


def anchor(s, out, a, M, tree, S, base, y0):
    """The splat placed by the robot (`--anchor`, a session without a board): the readout's robot fitted to the splat
    with a free scale, from a grid of places, headings and scales round the twin's robot; splat/align.json's similarity is then rescaled about the floor under the task's support (the
    floor to z = 0) to the robot's own size -- the scene's metres are the robot's -- and the twin's robot set where it
    was fitted."""
    fl = base.get("floor", 0.0)  # the wheels on the floor round the robot (main: floor_near)
    # coarse to fine: every start of the grid a short ICP on a twentieth of the model, the best eight then the full
    # model
    Ms = M[np.random.default_rng(0).permutation(len(M))[: max(2000, len(M) // 20)]]
    starts = []
    for dx in (-0.4, -0.2, 0.0, 0.2, 0.4):
        for dy in (-0.4, -0.2, 0.0, 0.2, 0.4):
            for dyaw in (np.radians(np.arange(0, 360, 30)) if base.get("from_frames") else np.radians([-30, -15, 0, 15, 30])):
                for k0 in (0.75, 0.9, 1.0, 1.15, 1.35):
                    r = trimmed_icp(Ms, tree, S, base["xy"][0] + dx, base["xy"][1] + dy, 0.0, y0 + dyaw, a.keep,
                                    iters=6, scale=True, k0=k0, floor=fl, workers=a.workers)
                    starts.append((np.sort(r[5])[: int(a.keep * len(Ms))].mean() / r[4], r))
    best = None
    for _, r0 in sorted(starts, key=lambda q: q[0])[:8]:
        r = trimmed_icp(M, tree, S, r0[0], r0[1], r0[2], r0[3], a.keep, iters=12, scale=True, k0=r0[4], floor=fl, workers=a.workers)
        kept = np.sort(r[5])[: int(a.keep * len(M))].mean() / r[4]
        if best is None or kept < best[0]:
            best = (kept, r)
    x, y, z, yaw, k, _ = best[1]
    x, y, z, yaw, k, d = trimmed_icp(M, tree, S, x, y, z, yaw, a.keep, iters=80, scale=True, k0=k, floor=fl, workers=a.workers)
    dm = d / k
    kept = np.sort(dm)[: int(a.keep * len(M))]
    al = json.loads((out / "align.json").read_text())
    # the placement the points were put in: the sparse points by the one before any anchor, the splat by the current
    s0 = al.get("sim3_before_robot", al["sim3"]) if a.points == "sparse" else al["sim3"]
    c, R, t = float(s0["scale"]), np.asarray(s0["R"]), np.asarray(s0["t"])
    # the frames stay where the key frames put them (on the twin's support: recon.splat_capture), rescaled about the
    # floor under the task's support -- the robot's place is fitted, not taken from the photo's; the twin's robot then goes where it stands
    if not (np.isfinite(k) and 0.5 < 1 / k < 2.0) or (dm < 0.02).mean() < 0.2:
        raise SystemExit(f"[anchor] no fit: scale x{1 / k if k else float('nan'):.4f}, {100 * (dm < 0.02).mean():.0f} % "
                         "within 2 cm -- align.json left as it was")
    sg = s.scene_graph
    sb = np.asarray(sg["furniture"][sg["task"]["support"]]["box"], float)
    O = np.array([(sb[0, 0] + sb[1, 0]) / 2, (sb[0, 1] + sb[1, 1]) / 2, fl])
    O2 = np.array([O[0], O[1], 0.0])
    new = {"scale": c / k, "R": R.tolist(), "t": ((t - O) / k + O2).tolist()}
    T0 = (np.array([x, y, z]) - O) / k + O2  # the fitted robot's base frame in the new placement
    al["sim3_before_robot"] = s0
    al["sim3"] = new
    al["scale_from"] = f"robot (recon.robot_fit --anchor, {a.points})"
    al["robot_anchor"] = {"scale_vs_before": round(1 / k, 4), "about": O2[:2].round(4).tolist(),
                          "base": [round(float(T0[0]), 4), round(float(T0[1]), 4)],
                          "yaw_deg": round(float(np.degrees(yaw)), 2), "floor_before": round(float(fl), 4),
                          "residual_median_m": round(float(np.median(kept)), 4),
                          "within_2cm_share": round(float((dm < 0.02).mean()), 3)}
    (out / "align.json").write_text(json.dumps(al, indent=1) + "\n")
    g = s.config("scene_graph.json")
    g["photo_robot"]["base"] = {"xy": [round(float(T0[0]), 4), round(float(T0[1]), 4)],
                                "yaw_deg": round(float(np.degrees(yaw)), 2)}
    g["photo_robot"]["_base_from"] = "recon.robot_fit --anchor: the readout's robot fitted where the frames see it"
    (s.config_dir / "scene_graph.json").write_text(json.dumps(g, indent=1, ensure_ascii=False) + "\n")
    print(f"[anchor] the robot fitted at x{k:.4f} the old placement's units, {1000 * np.median(kept):.1f} mm median "
          f"(the closest {100 * a.keep:.0f} %), {100 * (dm < 0.02).mean():.0f} % within 2 cm; the frames rescaled "
          f"x{1 / k:.4f} about the floor under the task's support, the robot at ({T0[0]:.3f}, {T0[1]:.3f}) yaw "
          f"{np.degrees(yaw):.1f} -> splat/align.json, photo_robot.base", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--anchor", action="store_true", help="place the splat by the robot (no board): see anchor()")
    ap.add_argument("--radius", type=float, default=1.2, help="the splat's points within this of the recorded base, m")
    ap.add_argument("--keep", type=float, default=0.6, help="the share of the model's points that drive the fit")
    ap.add_argument("--points", default="auto", choices=["auto", "splat", "sparse"])
    ap.add_argument("--cloud", default="readout", choices=["readout", "home"], help="home: apps/robot_cloud.py --state home")
    ap.add_argument('--workers', type=int, default=1, help='CPU nearest-neighbour workers; -1 uses all cores (default: 1)')
    ap.add_argument("--links", default="", help="comma-separated link-name prefixes fitted alone (no readout: the "
                    "base, the wheels, the lowest torso link -- what the joints do not move)")
    a = ap.parse_args()
    if a.workers == 0 or a.workers < -1:
        ap.error('--workers must be -1 or a positive integer')
    s = Session(a.session)
    if (a.anchor or a.apply) and (s/'qa/independent_freeze.json').exists():
        ap.error('Frozen reconstruction: create a new calibration session')
    out = folder(s)
    rc_files = (s / "parsing" / "robot_cloud_home.npz",) if a.cloud == "home" else (s / "parsing" / "robot_cloud.npz", out / "robot_cloud.npz")
    rc = np.load(next(f for f in rc_files if f.exists()))
    M = rc["points"].astype(np.float64)
    vox = np.unique(np.floor(M / 0.01).astype(np.int64), axis=0, return_index=True)[1]  # 1 cm voxels
    if a.links:
        names_ = [str(n) for n in rc["links"]]
        pre = tuple(p_.strip() for p_ in a.links.split(","))
        sel = np.isin(rc["link"], [i for i, n in enumerate(names_) if n.startswith(pre)])
        vox = vox[sel[vox]]
    M = M[vox]
    sg = s.scene_graph
    base = (sg.get("photo_robot") or {}).get("base")
    wide = base is None
    if wide:  # no base from the photo (recon.multiview found no base face): the robot where the frames see it, else
        al0 = json.loads((out / "align.json").read_text())  # round the support -- a wide grid of starts
        st = robot_start(s, out, al0.get("sim3_before_robot", al0["sim3"]))
        lo_, hi_ = (np.asarray(b_, float) for b_ in sg["furniture"][sg["task"]["support"]]["box"])
        c_ = st[:2] if st is not None else (lo_[:2] + hi_[:2]) / 2
        base = {"xy": [float(c_[0]), float(c_[1])], "yaw_deg": 0.0, "from": "frames" if st is not None else "support"}
        a.radius = max(a.radius, 2.0)
        print(f"[robot] no recorded base: starts round ({c_[0]:.2f}, {c_[1]:.2f}) ({base['from']})", flush=True)
    if a.points == "auto":
        a.points = "splat" if any((out / "export").glob("*.ply")) else "sparse"
    X, op, size = world_gaussians(s, out) if a.points == "splat" else sparse_points(out, before_anchor=a.anchor)
    radius = max(a.radius, 1.8) if a.anchor else a.radius  # the placement to correct may be half a metre off
    if a.anchor:  # the photo's base far from the robot the walk's frames see: start (and stay) where they see it
        al0 = json.loads((out / "align.json").read_text())
        st = robot_start(s, out, al0.get("sim3_before_robot", al0["sim3"]))
        if st is not None and np.hypot(st[0] - base["xy"][0], st[1] - base["xy"][1]) > 0.5:
            print(f"[anchor] the robot in the photo frame's points at ({st[0]:.2f}, {st[1]:.2f}), "
                  f"{np.hypot(st[0] - base['xy'][0], st[1] - base['xy'][1]):.2f} m from the photo's base: "
                  "the fit starts there and keeps the walk's placement", flush=True)
            base = {**base, "xy": [float(st[0]), float(st[1])], "from_frames": True}
        base = {**base, "floor": floor_near(X, base["xy"])}
        print(f"[anchor] the floor round the robot at z {base['floor']:+.3f} (the wheels held on it)", flush=True)
    near = ((np.hypot(X[:, 0] - base["xy"][0], X[:, 1] - base["xy"][1]) < radius) & (X[:, 2] > 0.05)
            & (X[:, 2] < M[:, 2].max() * (1.5 if a.anchor else 1.0) + 0.3) & (op > 0.3) & (size < 0.05))
    S = X[near]
    tree = cKDTree(S)
    best = None
    y0 = np.radians(base["yaw_deg"])
    if a.anchor:
        anchor(s, out, a, M, tree, S, base, y0)
        return
    steps = np.arange(-1.2, 1.21, 0.2) if wide else (-0.15, 0.0, 0.15)
    turns = (np.radians(np.arange(0, 360, 30)) if wide or base.get('from_frames')
             else np.radians([-20, -10, 0, 10, 20]))
    for dx in steps:
        for dy in steps:
            for dyaw in turns:
                r = trimmed_icp(M, tree, S, base["xy"][0] + dx, base["xy"][1] + dy, 0.0, y0 + dyaw, a.keep, iters=25, workers=a.workers)
                kept = np.sort(r[5])[: int(a.keep * len(M))]
                if best is None or kept.mean() < best[0]:
                    best = (kept.mean(), r)
    x, y, z, yaw, _, _ = best[1]
    x, y, z, yaw, _, d = trimmed_icp(M, tree, S, x, y, z, yaw, a.keep, workers=a.workers)
    xs, ys, zs, yaws, k, ds = trimmed_icp(M, tree, S, x, y, z, yaw, a.keep, scale=True, workers=a.workers)
    kept = np.sort(d)[: int(a.keep * len(M))]
    res = {"base": {"xy": [round(x, 4), round(y, 4)], "yaw_deg": round(float(np.degrees(yaw)), 2)},
           "height_offset_m": round(z, 4), "recorded": base,
           "moved_m": round(float(np.hypot(x - base["xy"][0], y - base["xy"][1])), 4),
           "turned_deg": round(float(np.degrees(yaw) - base["yaw_deg"]), 2),
           "residual_median_m": round(float(np.median(kept)), 4),
           "within_2cm_share": round(float((d < 0.02).mean()), 3), "within_5cm_share": round(float((d < 0.05).mean()), 3),
           "free_scale": round(k, 4), "free_scale_residual_median_m": round(float(np.median(np.sort(ds)[: int(a.keep * len(M))])), 4),
           "model_points": int(len(M)), "scene_points": int(len(S))}
    # per link: the share of its points within 2 cm (a link the readout put elsewhere -- an arm that moved between the
    # filming and the readout -- fits worse than the rest) and the lowest point (the wheels against the floor)
    L = rc["link"][vox]
    names = [str(n) for n in rc["links"]]
    res["links"] = {names[li]: round(float((d[L == li] < 0.02).mean()), 3) for li in np.unique(L) if (L == li).sum() >= 200}
    res["lowest_point_m"] = round(float(z + M[:, 2].min()), 4)
    (out / "robot_fit.json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"[robot] base ({x:.3f}, {y:.3f}) yaw {np.degrees(yaw):.1f} deg, height {z:+.3f} m: moved "
          f"{res['moved_m']:.3f} m, turned {res['turned_deg']:+.1f} deg from the recorded base; the closest "
          f"{100 * a.keep:.0f} % at {1000 * res['residual_median_m']:.1f} mm median, {100 * res['within_2cm_share']:.0f} % of "
          f"the model within 2 cm; a free scale fits at {k:.4f}", flush=True)

    # QA: top (x, y) and side (along the robot's facing, z) views
    fitted = M @ rot_z(yaw).T + [x, y, z]
    old = M @ rot_z(y0).T + [base["xy"][0], base["xy"][1], 0.0]
    fwd = np.array([np.cos(yaw), np.sin(yaw)])
    side = np.array([-fwd[1], fwd[0]])
    cell = 0.01
    c0 = np.array([x, y])

    def draw(proj, pts_list, cols, lo, hi):
        W, H = int((hi[0] - lo[0]) / cell), int((hi[1] - lo[1]) / cell)
        img = np.full((H, W, 3), 255, np.uint8)
        for P, col in zip(pts_list, cols):
            uv = proj(P)
            i, j = ((uv[:, 0] - lo[0]) / cell).astype(int), ((uv[:, 1] - lo[1]) / cell).astype(int)
            ok = (i >= 0) & (i < W) & (j >= 0) & (j < H)
            img[j[ok], i[ok]] = col
        return cv2.flip(img, 0)
    r_ = a.radius
    top = draw(lambda P: P[:, :2], [S, old, fitted], [(160, 160, 160), (220, 120, 40), (30, 140, 255)],
               c0 - r_, c0 + r_)
    sidev = draw(lambda P: np.c_[(P[:, :2] - c0) @ side, P[:, 2]], [S, old, fitted],
                 [(160, 160, 160), (220, 120, 40), (30, 140, 255)], (-r_, 0.0), (r_, 2.0))
    h = max(top.shape[0], sidev.shape[0])
    pad = lambda im: np.pad(im, ((0, h - im.shape[0]), (0, 10), (0, 0)), constant_values=255)  # noqa: E731
    qa = s / "qa" / "splat"
    qa.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(qa / "robot_fit.png"), np.hstack([pad(top), pad(sidev)]))
    if a.apply:
        g = s.config("scene_graph.json")  # as reviewed (Session.scene_graph adds the derived closure walls)
        g["_robot_fit"] = {"from": g["photo_robot"]["base"], "fit": res,
                           "note": "the real robot's base from its surface at the capture's joint readout fitted to the "
                                   "splat (recon.robot_fit)"}
        g["photo_robot"]["base"] = res["base"]
        (s.config_dir / "scene_graph.json").write_text(json.dumps(g, indent=1, ensure_ascii=False) + "\n")
        print(f"applied to {s.config_dir / 'scene_graph.json'}: rerun from placements")


if __name__ == "__main__":
    main()
