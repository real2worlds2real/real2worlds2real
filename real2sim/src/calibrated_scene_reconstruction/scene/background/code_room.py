"""The room written as code from the walk (twin.json `background: code`; built by sim.code_scene): no learned
reconstruction of the room -- its shell and each piece of furniture measured from the capture and written as a
procedural kit (assets.office_kits) at its place and size, in the colours the walk saw.

    envs/recon/bin/python -m calibrated_scene_reconstruction.scene.background.splat.inventory sessions/<name> things   # the inventory (SAM 3 + the splat)
    envs/recon/bin/python -m calibrated_scene_reconstruction.recon.code_scene sessions/<name> measure     # -> configs/<name>/code_scene.json
    tools/run_isaac.sh apps/scene/code_compare.py sessions/<name>                      # renders beside the walk's frames

measure: within `REACH` m of the task's support, outside the task zone (the twin's own: support, rig, task objects):
  - the floor (a carpet at the level of the points round the support, tinted to them), the ceiling (the level the
    points top out at), the walls (vertical lines through the points between 1.7 and 2.4 m -- over the furniture --,
    found one after another, each a panel along its points in their colour);
  - the light panels: the brightest Gaussians at the ceiling, clustered into rectangles;
  - every thing of the inventory (splat/cousins/things) of a class with a kit: its Gaussians' oriented box (their
    main horizontal axis; 2-98 % extents), standing on the floor when its bottom is within 8 cm of it, its colour the
    median of its Gaussians' (linear).
The agent reviews the result (qa/code/plan.png, apps/scene/code_compare.py) and corrects code_scene.json by hand where the
inventory erred (a class, a merged pair, a missing piece) -- the file is the scene.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from calibrated_scene_reconstruction.scene.session import Session
from calibrated_scene_reconstruction.scene.background.splat.config import settings, task_zone

REACH = 6.0
CEILING = 2.75  # m over the floor where the walk never saw the ceiling (an inference to confirm)
C0 = 0.28209479177387814  # the SH DC term's constant
KIT = {"desk": "desk", "table": "desk", "cabinet": "cabinet", "shelving unit": "shelving_unit",
       "bookcase": "shelving_unit", "cubicle partition": "partition", "cardboard box": "cardboard_box",
       "computer monitor": "monitor", "computer tower": "pc_tower", "whiteboard": "whiteboard",
       "office chair": "office_chair", "stool": "office_chair", "printer": "printer", "sofa": "sofa",
       "trash bin": "trash_bin"}


# (length, depth, height) ranges per kit, m: outside by over a third a piece is dropped (two things the inventory
# merged: a 'partition' a metre thick, a desk three deep), within that it is clamped
SIZES = {"desk": ((0.6, 2.6), (0.4, 1.2), (0.6, 1.1)), "cabinet": ((0.3, 3.0), (0.3, 0.8), (0.4, 2.2)),
         "shelving_unit": ((0.4, 3.0), (0.25, 0.8), (0.4, 2.2)), "partition": ((0.4, 3.0), (0.02, 0.12), (0.8, 2.0)),
         "cardboard_box": ((0.12, 1.2), (0.12, 1.0), (0.08, 1.0)), "monitor": ((0.3, 1.1), (0.05, 0.4), (0.25, 0.8)),
         "pc_tower": ((0.15, 0.6), (0.15, 0.6), (0.25, 0.7)), "whiteboard": ((0.6, 3.0), (0.0, 0.1), (0.4, 1.6)),
         "office_chair": ((0.4, 0.8), (0.4, 0.8), (0.7, 1.4)), "sofa": ((1.2, 3.0), (0.7, 1.2), (0.65, 1.1)),
         "trash_bin": ((0.2, 0.7), (0.2, 0.7), (0.2, 1.0)), "printer": ((0.4, 1.2), (0.4, 1.0), (0.3, 1.5))}


def sane(kit, p):
    """The piece clamped into its class's sizes, or None when it is far outside them."""
    r = SIZES.get(kit)
    if r is None:
        return p
    p = dict(p)
    for key, (lo, hi) in zip(("length", "depth", "height"), r):
        v = p.get(key)
        if v is None:
            continue
        if v > hi * 1.33 or v < lo * 0.5:
            return None
        p[key] = round(float(np.clip(v, lo, hi)), 3)
    return p


def tex_tint(tex, target):
    """The tint that takes a Poly Haven texture's mean colour (linear) to a measured one: a tint scales the texture
    (the twin's carpet came out brown, the measured grey times the grey texture's own colour)."""
    from PIL import Image

    from calibrated_scene_reconstruction.core.paths import POLYHAVEN_DIR

    im = np.asarray(Image.open(POLYHAVEN_DIR / "textures" / tex / "diff.jpg").convert("RGB").resize((128, 128)))
    mean = linear(im.reshape(-1, 3) / 255.0).mean(0)
    return np.clip(np.asarray(target) / np.maximum(mean, 1e-3), 0.0, 4.0).round(3).tolist()


def frame_depths(s, names, ds=4, checkpoint=None, pool="avg", rgb=None, seen=None):
    """The twin splat's depth (world metres, 1/ds resolution) at walk frames {name: z} (splat env); `checkpoint`
    another of the session's, `pool` "min" keeps each block's nearest surface."""
    import shutil

    import torch
    from threedgrut.datasets.colmap_gsplat import transform_cameras

    from calibrated_scene_reconstruction.scene.background.splat.config import folder, train_data
    from calibrated_scene_reconstruction.scene.background.splat.geometry import world_of_model
    from calibrated_scene_reconstruction.scene.background.splat.fix import model_transform, write_model

    sp = folder(s)
    T, rec = model_transform(sp)
    W_m, _ = world_of_model(sp, s)
    k = float(np.cbrt(abs(np.linalg.det(W_m[:3, :3]))))
    gd = s / "qa" / "code" / "depth_data"
    shutil.rmtree(gd, ignore_errors=True)
    (gd / "images").mkdir(parents=True)
    src = train_data(sp) / "images"
    images = []
    for img in sorted(rec.images.values(), key=lambda i: i.name):
        if img.name in names:
            M = np.eye(4)
            M[:3] = np.asarray(img.cam_from_world().inverse().matrix())
            (gd / "images" / img.name).symlink_to((src / img.name).resolve())
            images.append((img.name, img.camera_id, transform_cameras(T, M[None])[0]))
    write_model(gd, rec, dict(rec.cameras), images, (np.zeros((1, 3)), np.zeros((1, 3), np.uint8)))
    orig = torch.load

    def load(*a, **kw):
        c_ = orig(*a, **kw)
        if isinstance(c_, dict) and "config" in c_:
            c_["config"].dataset.test_split_interval = -1
            c_["config"].dataset.normalize_world_space = False
        return c_
    torch.load = load
    from threedgrut.render import Renderer

    tj = json.loads((sp / "train.json").read_text())
    r = Renderer.from_checkpoint(checkpoint_path=str(checkpoint or s / tj["checkpoint"]), path=str(gd), out_dir=str(gd / "r_out"),
                                 save_gt=False, computes_extra_metrics=True)
    torch.load = orig
    out = {}
    with torch.no_grad():
        for batch in r.dataloader:
            i = int(batch["frame_idx"][0])
            g = r.dataset.get_gpu_batch_with_intrinsics(batch)
            o = r.model(g)
            zz = (o["pred_dist"][0, ..., 0] * g.rays_dir[0, ..., 2]).float() * k
            zp = torch.nn.functional.avg_pool2d(zz[None, None], ds) if pool == "avg" else\
                -torch.nn.functional.max_pool2d(-zz[None, None], ds)
            out[Path(str(r.dataset.image_paths[i])).name] = zp[0, 0].cpu().numpy()
            if seen is not None:  # (a dict: "count", each Gaussian's number of these frames it drew in)
                vis = (o["mog_visibility"].reshape(-1) > 0).int()
                seen["count"] = vis if "count" not in seen else seen["count"] + vis
            if rgb is not None:  # (a dict to fill: the splat's own render of the frame, 0-1)
                rgb[Path(str(r.dataset.image_paths[i])).name] = o["pred_features"][0, ..., :3].clamp(0, 1).float().cpu().numpy()
    shutil.rmtree(gd, ignore_errors=True)
    return out, ds


def frame_color(s, P, n_frames=40, blockers=None):
    """The median colour (linear) the walk's frames show at world points P: each projected into frames spread over
    the walk where it falls inside the image in front of the camera and is the surface the splat's depth there shows
    (the floor behind a desk or a partition showed the desk's top, the partition's face)."""
    from PIL import Image

    z = np.load(s / "parsing" / "posed_frames" / "cameras.npz")
    names = [str(n) for n in z["names"]]
    img_dir = s.dir.parent.parent / "sessions" / s.dir.name / "splat" / "data" / "images"
    img_dir = img_dir if img_dir.exists() else s / "splat" / "data" / "images"
    vals = []
    pick = np.linspace(0, len(names) - 1, n_frames).astype(int)
    try:
        depths, ds = frame_depths(s, {names[i] for i in pick})
    except ImportError:  # (outside the splat env: no occlusion test)
        depths, ds = None, 1
    for i in pick:
        Xc = (P - z["centre"][i]) @ z["R_world_cam"][i]
        ok = Xc[:, 2] > 0.5
        K = z["K"][i]
        u = K[0, 0] * Xc[ok, 0] / Xc[ok, 2] + K[0, 2]
        v = K[1, 1] * Xc[ok, 1] / Xc[ok, 2] + K[1, 2]
        W, H = z["size"][i]
        m = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if depths is not None and names[i] in depths:
            dz = depths[names[i]]
            zi = dz[np.clip((v / ds).astype(int), 0, dz.shape[0] - 1), np.clip((u / ds).astype(int), 0, dz.shape[1] - 1)]
            m &= np.abs(Xc[ok, 2] - zi) < 0.06 * Xc[ok, 2]  # the point is what the camera sees
        if m.sum() < 5:  # (a frame sees little open floor: the desks hide most of it)
            continue
        im = np.asarray(Image.open(img_dir / names[i]).convert("RGB"))
        vals.append(im[v[m].astype(int), u[m].astype(int)] / 255.0)
    if not vals:
        return None
    return linear(np.median(np.concatenate(vals), 0)).round(3).tolist()


def linear(c):
    c = np.clip(np.asarray(c, float), 0, 1)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def gaussians(s):
    """The exported background's Gaussians (world): positions, display colours (the SH DC term), opacities."""
    from plyfile import PlyData

    from calibrated_scene_reconstruction.scene.background.splat.config import sim3

    v = PlyData.read(str(s / "splat" / "export" / "background.ply"))["vertex"].data
    c, R, t = sim3(s)
    X = c * np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64) @ R.T + t
    rgb = np.clip(0.5 + C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1), 0, 1)
    op = 1 / (1 + np.exp(-v["opacity"].astype(np.float64)))
    return X, rgb, op


def oriented(P):
    """(centre xy, yaw deg, length, depth, z0, z1) of points: their main horizontal axis, 2-98 % extents."""
    xy = P[:, :2]
    m = np.median(xy, 0)
    w, V = np.linalg.eigh(np.cov((xy - m).T))
    a = V[:, np.argmax(w)]
    yaw = float(np.degrees(np.arctan2(a[1], a[0])))
    R = np.array([[np.cos(np.radians(yaw)), -np.sin(np.radians(yaw))], [np.sin(np.radians(yaw)), np.cos(np.radians(yaw))]])
    q = (xy - m) @ R
    lo, hi = np.percentile(q, 2, 0), np.percentile(q, 98, 0)
    c = m + R @ ((lo + hi) / 2)
    z0, z1 = np.percentile(P[:, 2], 2), np.percentile(P[:, 2], 98)
    return c, yaw, float(hi[0] - lo[0]), float(hi[1] - lo[1]), float(z0), float(z1)


def walls(P, rgb, zc, n_max=8, tol=0.04, min_pts=150):
    """Vertical planes through points (xy) one after another (RANSAC), each a panel along its inliers."""
    rng = np.random.default_rng(0)
    out, left = [], np.ones(len(P), bool)
    for _ in range(n_max):
        Q = P[left]
        if len(Q) < min_pts:
            break
        best = None
        for _ in range(800):
            a, b = Q[rng.choice(len(Q), 2, replace=False)]
            t = b - a
            if np.linalg.norm(t) < 0.5:
                continue
            nrm = np.array([-t[1], t[0]]) / np.linalg.norm(t)
            inl = np.abs((Q - a) @ nrm) < tol
            if best is None or inl.sum() > best[0].sum():
                best = (inl, a, nrm)
        inl, a, nrm = best
        if inl.sum() < min_pts:
            break
        tang = np.array([nrm[1], -nrm[0]])
        s_ = (Q[inl] - a) @ tang
        s0, s1 = np.percentile(s_, 1), np.percentile(s_, 99)
        if s1 - s0 < 0.6:
            idx = np.flatnonzero(left)[inl]
            left[idx] = False
            continue
        off = float(np.median((Q[inl] - a) @ nrm))
        p0, p1 = a + off * nrm + s0 * tang, a + off * nrm + s1 * tang
        col = np.median(rgb[np.flatnonzero(left)[inl]], 0)
        out.append({"p0": p0.tolist(), "p1": p1.tolist(), "normal": nrm.tolist(), "points": int(inl.sum()),
                    "color": linear(col).round(3).tolist()})
        left[np.flatnonzero(left)[inl]] = False
    return out


def panels(X, rgb, op, zc, region):
    """The ceiling's light panels: the brightest opaque Gaussians within 15 cm of the ceiling, on a 10 cm grid, their
    connected clusters' boxes ([x, y, w, h])."""
    import cv2

    lo, hi = region
    lum = rgb.mean(1)
    m = (np.abs(X[:, 2] - zc) < 0.15) & (op > 0.3) & (lum > 0.93) & np.all(X[:, :2] > lo, 1) & np.all(X[:, :2] < hi, 1)
    cell = 0.1
    n = np.ceil((hi - lo) / cell).astype(int)
    g = np.zeros((n[1], n[0]), np.uint8)
    ij = np.floor((X[m, :2] - lo) / cell).astype(int)
    g[np.clip(ij[:, 1], 0, n[1] - 1), np.clip(ij[:, 0], 0, n[0] - 1)] = 1
    g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    k, lab, st, _ = cv2.connectedComponentsWithStats(g)
    out = []
    for i in range(1, k):
        x, y, w, h, a = st[i]
        if a < 6 or max(w, h) * cell > 3.0:  # a stray bright speck, or a white wall's top
            continue
        out.append([round(float(lo[0] + (x + w / 2) * cell), 2), round(float(lo[1] + (y + h / 2) * cell), 2),
                    round(float(w * cell), 2), round(float(h * cell), 2)])
    return out


def measure(s):
    from calibrated_scene_reconstruction.scene.background.splat.inventory import cdir

    sg = s.scene_graph
    lo_s, hi_s = (np.asarray(b, float) for b in sg["furniture"][sg["task"]["support"]]["box"])
    c = (lo_s[:2] + hi_s[:2]) / 2
    inside, _ = task_zone(s, settings(s)["crop"])
    X, rgb, op = gaussians(s)
    near = np.all(np.abs(X[:, :2] - c) < REACH, 1)
    dense = near & (op > 0.3)
    # floor and ceiling
    floor_pts = dense & (np.abs(X[:, 2]) < 0.06)
    zf = float(np.median(X[floor_pts, 2])) if floor_pts.sum() > 100 else 0.0
    up = dense & (X[:, 2] > zf + 2.3) & (X[:, 2] < zf + 4.5)
    if up.sum() > 200:
        h, e = np.histogram(X[up, 2], bins=np.arange(zf + 2.3, zf + 4.5, 0.02))
        zc, zc_from = float(e[np.argmax(h)] + 0.01), "the level the points top out at"
    else:  # a walk filmed level or down never sees the ceiling
        zc, zc_from = zf + CEILING, f"inferred: no point over {zf + 2.3:.1f} m, an office's suspended ceiling"
    # the floor's colour: its least saturated half (box bottoms and table feet stand in the band too)
    sat = rgb.max(1) - rgb.min(1)
    fp = floor_pts & (sat < np.percentile(sat[floor_pts], 50)) if floor_pts.sum() else floor_pts
    fcol = linear(np.median(rgb[fp], 0)).round(3).tolist() if fp.sum() else [0.3, 0.3, 0.3]
    # better, the frames' own pixels on open floor (nothing over it within 0.3 m: not the shade under a desk) --
    # the Gaussians' base colour of the floor read 0.07 where the frames show 0.26
    from scipy.spatial import cKDTree

    tall = X[dense & (X[:, 2] > zf + 0.1), :2]
    g = np.stack(np.meshgrid(np.arange(-REACH, REACH, 0.15), np.arange(-REACH, REACH, 0.15)), -1).reshape(-1, 2) + c
    open_ = cKDTree(tall).query(g, distance_upper_bound=0.3)[0] == np.inf
    # what blocks a line of sight: the room's Gaussians and the task zone (cut out of the exported splat: the desk's
    # light top and the white robot stood in for the floor behind them)
    gz = np.stack(np.meshgrid(np.arange(-REACH, REACH, 0.05), np.arange(-REACH, REACH, 0.05),
                              np.arange(zf + 0.05, zf + 2.0, 0.05)), -1).reshape(-1, 3) + [c[0], c[1], 0.0]
    zone_pts = gz[inside(gz)]
    open_ &= ~inside(np.c_[g, np.full(len(g), zf + 0.1)])
    fc = frame_color(s, np.c_[g[open_], np.full(open_.sum(), zf)],
                      blockers=cKDTree(np.r_[X[dense & (X[:, 2] > zf + 0.05)], zone_pts])) if open_.sum() > 20 else None
    fcol = fc or fcol
    pieces = {
        "floor": {"kit": "panel", "lying": True, "center": c.round(3).tolist(), "z": zf, "length": 2 * REACH + 2,
                  "depth": 2 * REACH + 2, "face": {"tex": "proc_carpet_tile_grey",
                                                   "tint": tex_tint("proc_carpet_tile_grey", fcol)}, "tile": 2.0,
                  "_from": "the points' level round the support, their colour"},
        "ceiling": {"kit": "panel", "lying": True, "center": c.round(3).tolist(), "z": zc + 0.05,
                    "length": 2 * REACH + 2, "depth": 2 * REACH + 2,
                    "face": {"color": [0.8, 0.8, 0.78], "roughness": 0.9}, "_from": zc_from},
    }
    band = dense & (X[:, 2] > zf + 1.7) & (X[:, 2] < min(zc - 0.1, zf + 2.4))
    mid_pts = X[dense & (X[:, 2] > zf + 0.2) & (X[:, 2] < zf + 1.5), :2]
    k = 0
    for w in walls(X[band, :2], rgb[band], zc):
        p0, p1 = np.asarray(w["p0"]), np.asarray(w["p1"])
        mid, d = (p0 + p1) / 2, p1 - p0
        # a wall bounds the room: the support on its one side, (almost) nothing behind it -- a line over the tops of
        # partitions and shelves crossed the open office
        nrm = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-6)
        if (c - mid) @ nrm < 0:
            nrm = -nrm
        along = np.abs((mid_pts - mid) @ (d / max(np.linalg.norm(d), 1e-6))) < np.linalg.norm(d) / 2
        behind = ((mid_pts - mid) @ nrm < -0.15) & along
        if behind.sum() > 0.05 * max(along.sum(), 1):
            continue
        k += 1
        pieces[f"wall_{k}"] = {"kit": "panel", "center": mid.round(3).tolist(), "z": zf,
                               "yaw_deg": round(float(np.degrees(np.arctan2(d[1], d[0]))), 2),
                               "length": round(float(np.linalg.norm(d)), 3), "height": round(zc - zf, 3),
                               "face": {"color": w["color"], "roughness": 0.85}, "skirting": [0.2, 0.2, 0.2],
                               "_from": f"a vertical line through {w['points']} points over the furniture"}
    if k < 3:  # an open office's walls beyond the reach: the room closed at the points' extent, off-white (inferred)
        P2 = X[dense & (X[:, 2] > zf + 0.2), :2]
        lo2, hi2 = np.percentile(P2, 1, 0) - 0.3, np.percentile(P2, 99, 0) + 0.3
        for j, (a, b) in enumerate([((lo2[0], lo2[1]), (hi2[0], lo2[1])), ((hi2[0], lo2[1]), (hi2[0], hi2[1])),
                                    ((hi2[0], hi2[1]), (lo2[0], hi2[1])), ((lo2[0], hi2[1]), (lo2[0], lo2[1]))]):
            a, b = np.asarray(a), np.asarray(b)
            d = b - a
            dvec = d / max(np.linalg.norm(d), 1e-6)
            inward = np.array([-dvec[1], dvec[0]]) * (1 if (c - a) @ np.array([-dvec[1], dvec[0]]) > 0 else -1)
            rel = X[:, :2] - a
            nearw = dense & (X[:, 2] > zf + 1.0) & (X[:, 2] < zf + 2.2) & (rel @ inward > 0) & (rel @ inward < 0.4) &\
                (rel @ dvec > 0) & (rel @ dvec < np.linalg.norm(d))
            # the frames' pixels on the wall's own plane where it is the surface they see (the Gaussians in front
            # of it were the partitions' yellow), else an off-white
            sw = np.arange(0.15, np.linalg.norm(d) - 0.1, 0.3)
            wz = np.arange(zf + 0.8, zf + 2.0, 0.3)
            Pw = np.array([[*(a + dvec * u), zz] for u in sw for zz in wz])
            wcol = frame_color(s, Pw) or [0.78, 0.77, 0.74]
            pieces[f"wall_closing_{j}"] = {
                "kit": "panel", "center": ((a + b) / 2).round(3).tolist(), "z": zf,
                "yaw_deg": round(float(np.degrees(np.arctan2(d[1], d[0]))), 2),
                "length": round(float(np.linalg.norm(d)), 3), "height": round(zc - zf, 3),
                "face": {"color": wcol, "roughness": 0.85}, "skirting": [0.2, 0.2, 0.2],
                "_from": "inferred: the room closed at the points' extent (its walls not within the reach), its colour "
                         "the points' within 0.4 m of it"}
    lights = panels(X, rgb, op, zc, (c - REACH, c + REACH))
    lights_from = "the brightest Gaussians at the ceiling"
    if not lights:  # an unseen ceiling: an office's 1.2 x 0.6 m panels every 2.4 m (an inference to confirm)
        g = np.arange(-REACH + 1.2, REACH, 2.4)
        lights = [[round(float(c[0] + x), 2), round(float(c[1] + y), 2), 1.2, 0.6] for x in g for y in g]
        lights_from = "inferred: the ceiling unseen, an office's panel grid"
    # the inventory's things
    inv = json.loads((cdir(s, "things") / "inventory.json").read_text())
    lab = np.load(cdir(s, "things") / "instances.npz")["thing"]
    Xg = np.load(cdir(s, "things") / "gauss.npz")["X"].astype(np.float64)
    kept = []
    for t in inv["things"]:
        kit = KIT.get(t["class"])
        if kit is None:
            continue
        m = lab == t["id"]
        if m.sum() < 50:
            continue
        P = Xg[m]
        cxy, yaw, L, D, z0, z1 = oriented(P)
        if np.linalg.norm(cxy - c) > REACH or inside(np.array([[*cxy, (z0 + z1) / 2]]))[0]:
            continue
        on_floor = z0 - zf < 0.08
        z = zf if on_floor else z0
        from scipy.spatial import cKDTree

        _, j = cKDTree(X[dense]).query(P[:: max(1, len(P) // 400)])
        col = linear(np.median(rgb[dense][j], 0)).round(3).tolist()
        p = {"kit": kit, "center": cxy.round(3).tolist(), "yaw_deg": round(yaw, 1), "z": round(z, 3),
             "length": round(max(L, 0.05), 3), "depth": round(max(D, 0.03), 3), "height": round(max(z1 - z, 0.05), 3),
             "_from": f"thing {t['id']} ({t['class']}): {int(m.sum())} Gaussians"}
        key = {"desk": "top", "cabinet": "body", "shelving_unit": "frame", "partition": "face",
               "cardboard_box": "color", "pc_tower": "color", "office_chair": "color", "sofa": "color",
               "trash_bin": "color", "printer": "color"}.get(kit)
        if key:
            p[key] = {"color": col, "roughness": 0.6}
        if kit == "desk":  # a top: a wood where the frames see a warm one, else a white laminate, at their colour
            tex = "ash_veneer" if col[0] > 1.15 * col[2] else "proc_laminate_white"
            p["top"] = {"tex": tex, "tint": tex_tint(tex, col)}
        if kit == "monitor":
            p["length"] = round(max(L, D), 3)
        if kit == "whiteboard":
            p.update({"bottom": round(z0 - zf, 3), "height": round(z1 - z0, 3)})
            p["z"] = zf
        p = sane(kit, p)
        if p is None:
            continue
        # the same class over the same floor (the inventory's duplicates): the larger kept
        a = np.array(p["center"])
        dup = [n for n, q in pieces.items() if q.get("kit") == kit and not q.get("lying") and
               np.linalg.norm(np.array(q["center"]) - a) < 0.5 * max(p["length"], q["length"])]
        if dup and max(pieces[n]["length"] * pieces[n].get("depth", 0.05) for n in dup) >= p["length"] * p["depth"]:
            continue
        for n in dup:
            del pieces[n]
        pieces[f"{kit}_{t['id']}"] = p
        kept.append(t["id"])
    old = s.config_dir / "code_scene.json"
    manual = json.loads(old.read_text()).get("manual", {}) if old.exists() else {}
    zone_boxes = [(k, np.asarray(v["box"], float)) for k, v in sg["furniture"].items()
                  if k == sg["task"]["support"] or k.startswith("frame_")]
    pieces, notes = settle(pieces, manual, zf, zone=zone_boxes)
    for n_ in notes:
        print(f"[settle] {n_}", flush=True)
    spec = {"_note": "recon.code_scene measure: the room written as code; `manual` (the agent's corrections against "
                     "the frames: a piece of the same name replaced, null removes it) is kept across measures",
            "floor_z": zf, "ceiling_z": zc, "pieces": pieces, "manual": manual,
            "lights": {"panels": lights, "z": zc, "intensity": 6000.0, "temp": 4500.0, "_from": lights_from}}
    old.write_text(json.dumps(spec, indent=1) + "\n")
    plan(s, spec, c)
    print(f"done: floor {zf:.3f}, ceiling {zc:.2f} m, {sum(k.startswith('wall') for k in pieces)} walls, "
          f"{len(lights)} light panels, {len(kept)} things -> {s.config_dir / 'code_scene.json'}", flush=True)


BIG = ("desk", "cabinet", "shelving_unit", "partition", "sofa", "printer", "whiteboard")
ON_TOPS = ("monitor",)  # never on the floor: on a top or gone
TOL = 0.02


class Box:
    """A piece's world box: its axis-aligned bounds (lo, hi; unpacks as them) and its true footprint -- the kit's
    local rectangle turned by its yaw (a turned desk's axis-aligned box took in floor beyond its top: monitors
    placed there fell through under PhysX)."""

    def __init__(self, p):
        from calibrated_scene_reconstruction.assets import kits

        params = {k: v for k, v in p.items() if k not in ("kit", "center", "yaw_deg", "z", "seed", "_from")}
        V = np.concatenate([q["V"] for q in kits.build(p["kit"], params)["parts"]])
        self.a = np.radians(p.get("yaw_deg", 0.0))
        c, s_ = np.cos(self.a), np.sin(self.a)
        self.R = np.array([[c, -s_], [s_, c]])
        self.c = np.asarray(p["center"], float)
        z = p.get("z", 0.0)
        self.llo, self.lhi = V[:, :2].min(0), V[:, :2].max(0)
        self.z0, self.z1 = V[:, 2].min() + z, V[:, 2].max() + z
        W = V[:, :2] @ self.R.T + self.c
        self.lo = np.r_[W.min(0), self.z0]
        self.hi = np.r_[W.max(0), self.z1]

    def __iter__(self):
        return iter((self.lo, self.hi))

    def samples(self, n=9):
        u, v = np.meshgrid(np.linspace(self.llo[0], self.lhi[0], n), np.linspace(self.llo[1], self.lhi[1], n))
        return np.c_[u.ravel(), v.ravel()] @ self.R.T + self.c

    def inside(self, P, margin=0.0):
        q = (np.asarray(P) - self.c) @ self.R
        return np.all((q > self.llo + margin) & (q < self.lhi - margin), axis=1)


def borne(A, B, share=0.75):
    """A over B's top stands on it: A's middle inside B's footprint and `share` of A's footprint over it (a monitor
    overhanging a desk's edge, a printer on a cabinet's corner toppled under PhysX)."""
    mid = A.c + ((A.llo + A.lhi) / 2) @ A.R.T
    return bool(B.inside(mid[None])[0] and B.inside(A.samples()).mean() >= share)


def overlap(A, B):
    """Two pieces interpenetrating by over TOL: their heights overlap, and one's footprint reaches over TOL into
    the other's."""
    if min(A.z1, B.z1) - max(A.z0, B.z0) <= TOL:
        return False
    return bool(B.inside(A.samples(), TOL).any() or A.inside(B.samples(), TOL).any())


def world_box(p):
    return Box(p)


def problems(pieces, zf, zone=()):
    """[(kind, name, other, amount)]: a piece floating (its bottom over the floor with no top under it), two pieces
    overlapping by over TOL in all three axes, a piece in a task-zone box."""
    boxes = {n: world_box(p) for n, p in pieces.items()
             if p is not None and not n.startswith("_") and not p.get("lying")}  # (walls too: a box in a wall flew)
    kit = {n: pieces[n]["kit"] for n in boxes}
    out = []
    for n, (lo, hi) in boxes.items():
        if kit[n] != "panel" and lo[2] > zf + TOL and not any(m != n and abs(boxes[m].z1 - lo[2]) < 0.03 and
                                                               borne(boxes[n], boxes[m]) for m in boxes):
            out.append(("floating", n, None, round(float(lo[2] - zf), 3)))
        for m, (l2, h2) in boxes.items():
            if m > n and not (kit[n] == kit[m] == "panel") and overlap(boxes[n], boxes[m]):
                out.append(("overlap", n, m, np.round(np.minimum(hi, h2) - np.maximum(lo, l2), 3).tolist()))
        for k, zb in zone:
            if np.all(np.minimum(hi, zb[1]) - np.maximum(lo, zb[0]) > 0):
                out.append(("task zone", n, k, None))
    return out


def settle(pieces, manual, zf, passes=8, zone=()):
    """The measured pieces made physical round the agent's (manual, fixed): duplicates of a class merged (the
    smaller dropped), each on the floor or on the top of a big piece under it (a monitor with no top under it
    dropped: it does not stand on a floor), a small piece pushed out of a piece it overlaps (dropped when that takes
    over 0.5 m), of two big ones overlapping the smaller dropped. Returns (pieces, notes)."""
    pieces = {n: dict(p) for n, p in pieces.items() if n not in manual}
    fixed = {n: p for n, p in manual.items() if p is not None and not n.startswith("_")}
    for k, zb in zone:  # the task zone's boxes stand as fixed obstacles (a box flew out of the rig under PhysX)
        fixed[f"_zone_{k}"] = {"kit": "cardboard_box", "center": ((zb[0][:2] + zb[1][:2]) / 2).tolist(),
                               "z": float(zb[0][2]), "length": float(zb[1][0] - zb[0][0]),
                               "depth": float(zb[1][1] - zb[0][1]), "height": float(zb[1][2] - zb[0][2]), "_zone": True}
    notes = []

    def every():
        return {**pieces, **fixed}

    for _ in range(passes):
        boxes = {n: world_box(p) for n, p in every().items() if not p.get("lying")}
        # rest each movable piece on the highest big top under its middle, else the floor
        for n, p in pieces.items():
            if p.get("lying") or p["kit"] == "panel" or n not in boxes:
                continue
            lo, hi = boxes[n]
            mid = (lo[:2] + hi[:2]) / 2
            tops = [boxes[m].z1 for m in boxes if m != n and every()[m]["kit"] in BIG and
                    borne(boxes[n], boxes[m]) and boxes[m].z1 < lo[2] + 0.3]
            z_new = max(tops) if tops else zf
            if not tops and p["kit"] in ON_TOPS:
                p["_drop"] = "a monitor with no top under it"
                continue
            if abs((lo[2] - z_new)) > TOL:
                hmax = SIZES.get(p["kit"], (None, None, (0, 9)))[2][1]
                if p["kit"] in BIG and not tops and "height" in p and p["height"] + lo[2] - z_new <= hmax:
                    # a desk or a cabinet seen from its top down: its legs reach the floor, its
                    p["height"] = round(float(p["height"] + lo[2] - z_new), 3)
                    # top stays where the frames put it (dropped whole, the desks' tops fell to 0.6 m and with them
                    # what stood on them)
                p["z"] = round(float(p.get("z", 0.0) + z_new - lo[2]), 3)
                notes.append(f"{n}: rests at {z_new:.2f} (was {lo[2]:.2f})")
        for n in [n for n, p in pieces.items() if p.get("_drop")]:
            notes.append(f"{n} dropped: {pieces.pop(n)['_drop']}")
        boxes = {n: world_box(p) for n, p in every().items() if not p.get("lying")}
        for kind, a, b, amt in problems(every(), zf):
            if kind != "overlap" or (a in fixed and b in fixed) or a not in boxes or b not in boxes or\
                    a not in every() or b not in every():  # (one dropped earlier this pass)
                continue
            pa, pb = every()[a], every()[b]
            la, ha = boxes[a]
            lb, hb = boxes[b]
            va, vb = np.prod(ha - la), np.prod(hb - lb)
            mover = a if b in fixed else b if (a in fixed or (pa["kit"] in BIG and pb["kit"] not in BIG) or
                          (pa["kit"] == pb["kit"] and va >= vb) or
                          (pa["kit"] in BIG) == (pb["kit"] in BIG) and va >= vb) else a
            if mover in fixed or mover not in pieces:
                continue
            other = a if mover == b else b
            lm, hm = boxes[mover]
            lo_, ho_ = boxes[other]
            same = pieces[mover]["kit"] == every()[other]["kit"]
            both_big = pieces[mover]["kit"] in BIG and every()[other]["kit"] in BIG
            if same or (both_big and other not in fixed):  # (a big piece against the agent's own: pushed clear)
                pieces.pop(mover)
                notes.append(f"{mover} dropped: overlaps {other} ({'a duplicate' if same else 'both large'})")
                continue
            # the shortest horizontal push out of the other's box
            pushes = [(ho_[0] - lm[0], 0), (lo_[0] - hm[0], 0), (ho_[1] - lm[1], 1), (lo_[1] - hm[1], 1)]
            d, ax = min(pushes, key=lambda q: abs(q[0]))
            if abs(d) > 0.5:
                pieces.pop(mover)
                notes.append(f"{mover} dropped: inside {other}")
                continue
            c = list(pieces[mover]["center"])
            c[ax] = round(c[ax] + d + np.sign(d) * 0.01, 3)
            pieces[mover]["center"] = c
            notes.append(f"{mover}: pushed {d:+.2f} m off {other}")
    for kind, a, b, amt in problems(every(), zf):  # still caught between two pieces after the passes: dropped
        if kind == "overlap":
            for n in (a, b):
                if n in pieces and pieces[n]["kit"] not in BIG:
                    pieces.pop(n)
                    notes.append(f"{n} dropped: still overlapping after {passes} passes")
                    break
    return pieces, notes


def check(s):
    """The code room's physical check (qa/code/check.json): what still floats or overlaps, the task zone."""
    spec = json.loads((s.config_dir / "code_scene.json").read_text())
    sg = s.scene_graph
    zone = [(k, np.asarray(v["box"], float)) for k, v in sg["furniture"].items()
            if k == sg["task"]["support"] or k.startswith("frame_")]
    out = problems({**spec["pieces"], **spec.get("manual", {})}, spec["floor_z"], zone)
    (s / "qa" / "code").mkdir(parents=True, exist_ok=True)
    (s / "qa" / "code" / "check.json").write_text(json.dumps(out, indent=1) + "\n")
    for kind, a, b, amt in out:
        print(f"[check] {kind}: {a}" + (f" / {b}" if b else "") + (f" {amt}" if amt is not None else ""))
    print(f"done: {len(out)} problems -> {s / 'qa' / 'code' / 'check.json'}", flush=True)


def plan(s, spec, c, R=REACH, cell=0.02):
    """qa/code/plan.png: the pieces' footprints over the floor plan (1 m grid)."""
    import cv2

    from calibrated_scene_reconstruction.scene.background.splat.inventory import cdir

    n = int(2 * R / cell)
    img = np.zeros((n, n, 3), np.uint8)

    def px(x, y):
        return int((x - c[0] + R) / cell), int(n - 1 - (y - c[1] + R) / cell)

    Xg = np.load(cdir(s, "things") / "gauss.npz")["X"]
    m = np.all(np.abs(Xg[:, :2] - c) < R, 1) & (Xg[:, 2] > 0.1)
    for x, y in Xg[m, :2][::5]:
        u, v = px(x, y)
        if 0 <= u < n and 0 <= v < n:
            img[v, u] = (60, 60, 60)
    for k in range(-int(R), int(R) + 1):
        cv2.line(img, px(np.floor(c[0]) + k, c[1] - R), px(np.floor(c[0]) + k, c[1] + R), (40, 40, 40), 1)
        cv2.line(img, px(c[0] - R, np.floor(c[1]) + k), px(c[0] + R, np.floor(c[1]) + k), (40, 40, 40), 1)
    for name, p in spec["pieces"].items():
        if p.get("lying"):
            continue
        L, D = p["length"], p.get("depth", 0.05)
        a = np.radians(p.get("yaw_deg", 0.0))
        Rm = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        pts = np.array([[-L / 2, -D / 2], [L / 2, -D / 2], [L / 2, D / 2], [-L / 2, D / 2]]) @ Rm.T + p["center"]
        m = p.get("face") or p.get("top") or p.get("body") or p.get("frame") or p.get("color") or {}
        col = tuple(int(255 * v ** (1 / 2.2)) for v in m["color"][::-1]) if "color" in m else (200, 200, 200)
        cv2.polylines(img, [np.array([px(*q) for q in pts], np.int32)], True, col, 2)
        cv2.putText(img, name, px(*p["center"]), 0, 0.35, (255, 255, 255), 1)
    for x, y, w, h in spec["lights"]["panels"]:
        cv2.rectangle(img, px(x - w / 2, y - h / 2), px(x + w / 2, y + h / 2), (0, 255, 255), 1)
    out = s / "qa" / "code"
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "plan.png"), img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("step", choices=["measure", "check"])
    a = ap.parse_args()
    {"measure": measure, "check": check}[a.step](Session(a.session))


if __name__ == "__main__":
    main()


# Public legacy spelling.
frame_colour = frame_color
