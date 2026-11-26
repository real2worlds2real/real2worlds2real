"""The task support's footprint carved from the posed frames: no tape measure, no texture needed.

    tools/run_splat.sh -m calibrated_scene_reconstruction.recon.support_carve sessions/<name> [--apply]     (envs/splat: pycolmap)

On the support top's plane (the twin's height) a 1 cm grid over the twin's box widened by 0.4 m; each point is
projected into every posed frame that has it in view (splat/data/sparse/0, recon.splat_capture: COLMAP at the
board's scale). What the pixel there shows is one of three things, by SAM 3 silhouettes (recon.splat_masks, the
zone's projection as the gate): the support (<image>_carve.png, prompts table / desk) -- evidence for the point;
an occluder, the robot, the rig, the cups, a board (<image>_occ.png) -- no evidence either way, something stood
between; or the background -- evidence against: nothing there blocked the view of the floor. A point is carved
when the background shows in `bg_frac` of its evidence frames (at least `min_bg`). The frames must see the support from round it (the walks do); a side the frames never passed keeps the
widened box there, which the `seen` fraction reports.

Writes
splat/support_carve.{json,png}; `--apply` writes the carved footprint into the scene graph's support box (the
height kept) with a `_support_carve` note, for the twin to be rebuilt from `layout`."""
import argparse
import json

import cv2
import numpy as np
import pycolmap

from calibrated_scene_reconstruction.scene.session import Session


def top_height(out, c, R, t, lo, hi, reach=0.3):
    """The support top's height in the frames: the mode of COLMAP's points over the twin's footprint, within `reach`
    of the twin's height (the twin's comes from the photo, before the frames set the scale)."""
    # the full model: sparse/0 is the training's, its task zone's points taken out
    P = np.array([p.xyz for p in pycolmap.Reconstruction(str(out / "data" / "sparse" / "full")).points3D.values()])
    X = c * P @ R.T + t
    m = np.all(X[:, :2] > lo[:2], 1) & np.all(X[:, :2] < hi[:2], 1) & (np.abs(X[:, 2] - hi[2]) < reach)
    if m.sum() < 100:
        return float(hi[2])
    h, e = np.histogram(X[m, 2], bins=np.arange(hi[2] - reach, hi[2] + reach, 0.005))
    # a second level holding a share of the points: the top mapped twice -- poses that drifted between the frames
    # that saw it
    k = int(np.argmax(h))
    far = np.abs(np.arange(len(h)) - k) > 6  # 3 cm apart
    if far.any() and h[far].max() > 0.15 * h[k]:
        k2 = int(np.flatnonzero(far)[np.argmax(h[far])])
        print(f"[carve] WARNING the top mapped twice: {h[k]} points at z {e[k] + 0.0025:.3f}, {h[k2]} at "
              f"{e[k2] + 0.0025:.3f} -- the frames' poses drifted (splat_capture's matching)", flush=True)
    return float(e[k] + 0.0025)


def top_evidence(out, c, R, t, z, xs, ys, cell, band=0.015, block=0.05):
    """Where COLMAP's points lie at the top's height (within `band`): their largest connected cluster on `block` m
    cells, its convex hull filled, on the carve's grid. The silhouettes are of "a desk": another desk beside the task's
    at its own height, seen through the grid's parallax, joined the top."""
    P = np.array([p.xyz for p in pycolmap.Reconstruction(str(out / "data" / "sparse" / "full")).points3D.values()])
    X = c * P @ R.T + t
    Y = X[np.abs(X[:, 2] - z) < band, :2]
    nb = (int(np.ceil((xs[-1] - xs[0]) / block)) + 1, int(np.ceil((ys[-1] - ys[0]) / block)) + 1)
    ij = np.floor((Y - [xs[0], ys[0]]) / block).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < nb[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < nb[1])
    occ = np.zeros((nb[1], nb[0]), np.uint8)
    occ[ij[ok, 1], ij[ok, 0]] = 1
    occ = cv2.dilate(occ, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(occ, connectivity=8)
    full = np.ones((len(ys), len(xs)), bool)
    if n < 2:
        return full
    big = lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    pts = np.argwhere(big)[:, ::-1].astype(np.int32)  # (x, y) block indices
    hull = np.zeros_like(occ)
    cv2.fillConvexPoly(hull, cv2.convexHull(pts), 1)
    gx = np.clip(((xs - xs[0]) / block).astype(int), 0, nb[0] - 1)
    gy = np.clip(((ys - ys[0]) / block).astype(int), 0, nb[1] - 1)
    return hull[np.ix_(gy, gx)].astype(bool)


def refine_edges(rec, img_dir, suffix, occ_suffix, c, R, t, fp, z, reach=0.06, step=0.01, probe=0.02, min_votes=200,
                 min_score=0.3, evidence="mask"):
    """Each side of the carved rectangle moved to where the frames see the top end: for an offset d along its outward
    normal (within `reach`), points along the side `probe` m inside and outside the edge at d are projected into
    every frame; where neither falls on an occluder the frame votes -- the inside on the support's mask, the outside
    off it. The side goes to the offset whose score (the inside's support share minus the outside's) is highest, when
    at least `min_votes` votes back it and it reaches `min_score`; else it stays (a band lying on the edge shows as a
    trough, a table beyond as a far peak: 12 cm out at the cups' desk's end, the neighbour's top). The cell carve stops short of, or overshoots, a top's edges under what lies on
    them (a frame's bands): its rectangle on three captures of one desk 1.52-1.92 m long. Returns (fp, {side: d})."""
    (x0, y0), (x1, y1) = fp
    sides = {"x0": (np.array([-1.0, 0.0]), x0, (y0, y1)), "x1": (np.array([1.0, 0.0]), x1, (y0, y1)),
             "y0": (np.array([0.0, -1.0]), y0, (x0, x1)), "y1": (np.array([0.0, 1.0]), y1, (x0, x1))}
    offs = np.arange(-reach, reach + 1e-9, step)
    pts, meta = [], []  # every probe point: (side, offset index, inside?)
    for name, (n, e, (a, b)) in sides.items():
        along = np.arange(a + 0.1, b - 0.1, 0.02)
        for k, d in enumerate(offs):
            for inside, sgn in ((True, -1.0), (False, 1.0)):
                w = e * np.abs(n).sum() + d + sgn * probe  # the coordinate across the side
                P = (np.c_[np.full_like(along, w), along] if n[0] else np.c_[along, np.full_like(along, w)])
                pts.append(np.c_[P, np.full(len(P), z)])
                meta += [(name, k, inside)] * len(P)
    P = np.concatenate(pts)
    Pc = (P - t) @ R / c
    side_i = np.array([list(sides).index(m[0]) for m in meta])
    off_i = np.array([m[1] for m in meta])
    ins = np.array([m[2] for m in meta])
    on = np.zeros(len(P))
    cnt = np.zeros(len(P))
    for img in rec.images.values():
        mp, op = img_dir / (img.name[:-4] + suffix + ".png"), img_dir / (img.name[:-4] + occ_suffix + ".png")
        if not mp.exists() or not op.exists():
            continue
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) < 128
        oc = cv2.imread(str(op), cv2.IMREAD_GRAYSCALE) < 128
        if evidence == "color":  # the top's own colour, measured on its mask in this frame (a strip of desk beyond a
            # band lying on it, which the masks stop short of, is desk-coloured all the same)
            lab_ = cv2.cvtColor(cv2.imread(str(img_dir / img.name)), cv2.COLOR_BGR2LAB).astype(np.float32)
            core = cv2.erode((m & ~oc).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
            if core.sum() < 500:
                continue
            ref = np.median(lab_[core], 0)
            spread = np.percentile(np.linalg.norm(lab_[core] - ref, axis=1), 90)
            m = np.linalg.norm(lab_ - ref, axis=2) <= max(spread, 8.0)
        if m.mean() < 0.005:
            continue
        cam = rec.cameras[img.camera_id]
        fx, fy, cx, cy = (float(v) for v in cam.params[:4])
        M = np.asarray(img.cam_from_world().matrix())
        X = Pc @ M[:, :3].T + M[:, 3]
        ok = X[:, 2] > 0.05
        u = np.where(ok, fx * X[:, 0] / np.where(ok, X[:, 2], 1) + cx, -1)
        v = np.where(ok, fy * X[:, 1] / np.where(ok, X[:, 2], 1) + cy, -1)
        ok &= (u >= 0) & (u < cam.width - 1) & (v >= 0) & (v < cam.height - 1)
        idx = np.flatnonzero(ok)
        ui, vi = u[idx].astype(int), v[idx].astype(int)
        free = ~oc[vi, ui]
        idx, ui, vi = idx[free], ui[free], vi[free]
        on[idx] += m[vi, ui]
        cnt[idx] += 1
    out = [[x0, y0], [x1, y1]]
    moved = {}
    for si, (name, (n, e, _)) in enumerate(sides.items()):
        best, prof = None, []
        for k, d in enumerate(offs):
            sel_i, sel_o = (side_i == si) & (off_i == k) & ins, (side_i == si) & (off_i == k) & ~ins
            ni, no = cnt[sel_i].sum(), cnt[sel_o].sum()
            if ni < min_votes or no < min_votes:
                continue
            score = on[sel_i].sum() / ni - on[sel_o].sum() / no
            prof.append(f"{100 * d:+.0f}:{score:.2f}")
            if best is None or score > best[0]:
                best = (score, d)
        print(f"[carve] {name} score by offset (cm): {' '.join(prof)}", flush=True)
        if best is None or best[0] < min_score:  # no clear edge in reach (a band on it, another table beyond)
            continue
        moved[name] = (round(float(best[1]), 3), round(float(best[0]), 2))
        j = 0 if name.startswith("x") else 1
        out[0 if name.endswith("0") else 1][j] = float(e + best[1] * (n[j]))
    return out, moved


def overlay(s, rec, img_dir, c, R, t, fp, z, n=8):
    """qa/splat/support_box.jpg: the carved top's rectangle (green) drawn into n frames spread over the walk that see
    all of it or most -- the review's check that the box lies on the top's edges (S2: its box against its edges)."""
    corners = np.array([[fp[0][0], fp[0][1], z], [fp[1][0], fp[0][1], z], [fp[1][0], fp[1][1], z],
                        [fp[0][0], fp[1][1], z]])
    Pc = (corners - t) @ R / c
    shots = []
    for img in sorted(rec.images.values(), key=lambda i: i.name):
        cam = rec.cameras[img.camera_id]
        fx, fy, cx, cy = (float(x) for x in cam.params[:4])
        M = np.asarray(img.cam_from_world().matrix())
        X = Pc @ M[:, :3].T + M[:, 3]
        if (X[:, 2] <= 0.05).any():
            continue
        uv = np.c_[fx * X[:, 0] / X[:, 2] + cx, fy * X[:, 1] / X[:, 2] + cy]
        inside = ((uv[:, 0] > -0.2 * cam.width) & (uv[:, 0] < 1.2 * cam.width) &
                  (uv[:, 1] > -0.2 * cam.height) & (uv[:, 1] < 1.2 * cam.height)).sum()
        if inside >= 3:
            shots.append((img.name, uv))
    if not shots:
        return
    pick = [shots[int(k)] for k in np.linspace(0, len(shots) - 1, min(n, len(shots)))]
    tiles = []
    for name, uv in pick:
        im = cv2.imread(str(img_dir / name))
        cv2.polylines(im, [uv.astype(np.int32).reshape(-1, 1, 2)], True, (0, 230, 0), 4)
        cv2.putText(im, name, (20, 50), 0, 1.4, (0, 255, 255), 3)
        tiles.append(cv2.resize(im, (640, 360)))
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    qa = s / "qa" / "splat"
    qa.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(qa / "support_box.jpg"), np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--suffix", default=None)
    ap.add_argument("--occluder-suffix", default=None)
    ap.add_argument("--margin", type=float, default=0.8, help='Search margin around the estimated support bounds, in metres.')
    ap.add_argument("--cell", type=float, default=0.01)
    ap.add_argument("--min-frames", type=int, default=5, help="a point carved only when this many frames saw it")
    ap.add_argument("--bg-frac", type=float, default=0.5, help="carved when this share of the frames that saw it (at "
                    "least --min-bg) show background there: a missed silhouette must not carve the top")
    ap.add_argument("--min-bg", type=int, default=3)
    ap.add_argument("--refine-edges", action="store_true", help="each side moved to where the frames' masks see the "
                    "top end (refine_edges; prints each side's score by offset)")
    ap.add_argument("--edge-evidence", default="mask", choices=["mask", "color"],
                    help="refine_edges: the support's masks, or the top's own colour in each frame")
    ap.add_argument("--edge-reach", type=float, default=0.06)
    ap.add_argument("--under", type=float, default=0.06, help="m the top is followed under occluders on its edge (a "
                    "rig's band, a lip: a few cm; at 0.12 it ran 10 cm on under the robot standing at the edge)")
    a = ap.parse_args()
    s = Session(a.session)
    from .mask_inputs import settings as mask_settings
    masks = mask_settings(s)
    a.suffix = a.suffix or masks['support_suffix']
    a.occluder_suffix = a.occluder_suffix or masks['occluder_suffix']
    out = s / "splat"
    al = json.loads((out / "align.json").read_text())
    c, R, t = al["sim3"]["scale"], np.asarray(al["sim3"]["R"]), np.asarray(al["sim3"]["t"])
    sg = s.scene_graph
    key = sg["task"]["support"]
    lo, hi = (np.asarray(b, float) for b in sg["furniture"][key]["box"])
    plane_evidence = None
    from .observed_plane import use_labelled_tracks
    if use_labelled_tracks(s.config('splat.json', {}), al):
        from calibrated_scene_reconstruction.scene.capture.placement_planes import from_photo
        from .observed_plane import top_height as observed_top_height
        reconstruction = pycolmap.Reconstruction(str(out/'data/sparse/full'))
        _, support_points, _ = from_photo(s, reconstruction)
        z, plane_evidence = observed_top_height(c*support_points@R.T+t)
        print(f'[carve] observed support plane: {plane_evidence["inliers"]}/'
              f'{plane_evidence["tracks"]} actual labelled tracks at z {z:.4f}',flush=True)
    else:
        z = top_height(out, c, R, t, lo, hi)
    dz = z - float(hi[2])  # the frames' top against the twin's
    xs = np.arange(lo[0] - a.margin, hi[0] + a.margin, a.cell)
    ys = np.arange(lo[1] - a.margin, hi[1] + a.margin, a.cell)
    gx, gy = np.meshgrid(xs, ys)
    P = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)], 1)
    Pc = (P - t) @ R / c  # world -> COLMAP (R orthonormal: inverse is the transpose)
    rec = pycolmap.Reconstruction(str(out / "data" / "sparse" / "0"))
    img_dir = out / "data" / "images"
    seen = np.zeros(len(P), np.int32)
    occd = np.zeros(len(P), np.int32)  # frames whose pixel there shows an occluder
    bg = np.zeros(len(P), np.int32)
    n_img = 0
    for img in rec.images.values():
        mpath = img_dir / (img.name[:-4] + a.suffix + ".png")
        opath = img_dir / (img.name[:-4] + a.occluder_suffix + ".png")
        if not mpath.exists() or not opath.exists():
            continue
        m = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
        occ = cv2.imread(str(opath), cv2.IMREAD_GRAYSCALE) < 128
        if (m < 128).mean() < 0.005 and occ.mean() < 0.005:  # nothing found: no evidence (a missed frame)
            continue
        cam = rec.cameras[img.camera_id]  # each video its own camera (splat_capture)
        fx, fy, cx, cy = (float(x) for x in cam.params[:4])
        M = np.asarray(img.cam_from_world().matrix())
        X = Pc @ M[:, :3].T + M[:, 3]
        zc = X[:, 2]
        ok = zc > 0.05
        u = np.where(ok, fx * X[:, 0] / np.where(ok, zc, 1) + cx, -1)
        v = np.where(ok, fy * X[:, 1] / np.where(ok, zc, 1) + cy, -1)
        ok &= (u >= 0) & (u < cam.width - 1) & (v >= 0) & (v < cam.height - 1)
        if not ok.any():
            continue
        ui, vi = u[ok].astype(int), v[ok].astype(int)
        evidence = ~occ[vi, ui]  # an occluder in the pixel: no evidence either way
        idx = np.flatnonzero(ok)
        seen[idx[evidence]] += 1
        occd[idx[~evidence]] += 1
        bg[idx[evidence & (m[vi, ui] > 127)]] += 1
        n_img += 1
    carved = (bg >= np.maximum(a.min_bg, a.bg_frac * seen)) & (seen >= a.min_frames)
    keep = ~carved & top_evidence(out, c, R, t, z, xs, ys, a.cell).ravel()
    K = keep.reshape(gy.shape).astype(np.uint8)
    # Close narrow gaps before removing isolated specks.
    K = cv2.morphologyEx(K, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    K = cv2.morphologyEx(K, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(K)
    if n < 2:
        raise SystemExit(f"support_carve: everything carved ({n_img} frames): the silhouettes miss the support")
    ci = (int((((lo[1] + hi[1]) / 2) - ys[0]) / a.cell), int((((lo[0] + hi[0]) / 2) - xs[0]) / a.cell))
    big = lab[ci] if lab[ci] > 0 else 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    mk = lab == big
    # the top runs on under what lies on its edge: a rig's bands along the desk's edges hide the top's last centimetres
    # in every frame -- no evidence either way, so the carve stopped at their inner edge. Grown 1 cm a step, up to `a.under` m,
    # through cells occluded in most of the frames that see them.
    occ_g, seen_g = occd.reshape(gy.shape), seen.reshape(gy.shape)
    under = (occ_g >= a.min_bg) & (occ_g >= 0.6 * (occ_g + seen_g))
    k3 = np.ones((3, 3), np.uint8)
    grown = mk.copy()
    for _ in range(int(round(a.under / a.cell))):
        nxt = (cv2.dilate(grown.astype(np.uint8), k3) > 0) & (under | grown)
        if (nxt == grown).all():
            break
        grown = nxt
    added = int(grown.sum() - mk.sum())
    ring = (cv2.dilate(mk.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0) & ~mk  # 5 cm round the region
    sg_, bg_, oc_ = (v.reshape(gy.shape)[ring] for v in (seen, bg, occd))
    print(f"[carve] 5 cm round the region: frames with evidence {np.median(sg_):.0f}, background {np.median(bg_):.0f}, "
          f"occluded {np.median(oc_):.0f} (medians)", flush=True)
    mk = grown
    # the edges where the region fills at least half its fullest column / row: a top is a filled rectangle, and a
    # piece joined at one corner by the closing fills only a few
    cols, rows = mk.sum(0), mk.sum(1)
    ii = np.flatnonzero(cols >= 0.5 * cols.max())
    jj = np.flatnonzero(rows >= 0.5 * rows.max())
    fp = [[float(xs[ii.min()]), float(ys[jj.min()])], [float(xs[ii.max()] + a.cell), float(ys[jj.max()] + a.cell)]]
    if a.refine_edges:  # (opt-in: on three captures of one desk it helped two and pulled the third further in)
        fp_cells = fp
        fp, moved = refine_edges(rec, img_dir, a.suffix, a.occluder_suffix, c, R, t, fp, z, reach=a.edge_reach,
                                 evidence=a.edge_evidence)
        print(f"[carve] the sides moved to where the frames see the top end (m, score): {moved}; from "
              f"{fp_cells[1][0] - fp_cells[0][0]:.3f} x {fp_cells[1][1] - fp_cells[0][1]:.3f}", flush=True)
    unseen = float((seen.reshape(gy.shape)[mk] < a.min_frames).mean())
    edge = bool(mk[0].any() or mk[-1].any() or mk[:, 0].any() or mk[:, -1].any())  # cut by the search's border
    if edge:
        print(f"[carve] the top reaches the search's border ({a.margin} m round the twin's box): rerun with a wider --margin", flush=True)
    res = {"observed_plane": plane_evidence, 'mask_revisions': {'support_suffix': a.suffix, 'occluder_suffix': a.occluder_suffix},
           "support": key, "frames": n_img, "footprint": fp, "size": [fp[1][0] - fp[0][0], fp[1][1] - fp[0][1]],
           "top_z": round(z, 4), "twin_top_z": round(float(hi[2]), 4),
           "at_search_border": edge,
           "twin_box": [lo[:2].tolist(), hi[:2].tolist()], "twin_size": [float(hi[0] - lo[0]), float(hi[1] - lo[1])],
           "unseen_share_of_footprint": unseen, "cell": a.cell}
    (out / "support_carve.json").write_text(json.dumps(res, indent=1) + "\n")
    vis = np.zeros((*gy.shape, 3), np.uint8)
    vis[keep.reshape(gy.shape)] = (90, 90, 90)
    vis[mk] = (60, 180, 60)
    vis[(seen.reshape(gy.shape) < a.min_frames)] = (40, 40, 120)
    px = lambda x, y: (int((x - xs[0]) / a.cell), int((y - ys[0]) / a.cell))  # noqa: E731
    cv2.rectangle(vis, px(lo[0], lo[1]), px(hi[0], hi[1]), (255, 255, 255), 1)
    cv2.imwrite(str(out / "support_carve.png"), cv2.flip(vis, 0)[:, :, ::-1])
    print(f"[carve] {n_img} frames: the {key} top {res['size'][0]:.3f} x {res['size'][1]:.3f} m at z {z:.3f} (the "
          f"twin's {hi[2]:.3f}), "
          f"x {fp[0][0]:.3f}..{fp[1][0]:.3f}, y {fp[0][1]:.3f}..{fp[1][1]:.3f} (the twin's box {res['twin_size'][0]:.3f} x "
          f"{res['twin_size'][1]:.3f}); {100 * unseen:.0f} % of it seen by fewer than {a.min_frames} frames; "
          f"{added * a.cell ** 2:.3f} m2 followed under occluders on its edge")
    overlay(s, rec, img_dir, c, R, t, fp, z)
    # the top's own yaw against the world's axes (its carved region's least-area rectangle): the world is the room's
    # (autograph: the walls' yaw), the boxes are level with it -- a support standing askew in the room is carved as a
    # level box round part of it
    pts = np.argwhere(mk)[:, ::-1].astype(np.float32)  # (x cell, y cell)
    (cx_, cy_), (w_, h_), ang = cv2.minAreaRect(pts)
    res['oriented_footprint'] = {
        'xy': [float(xs[0]+cx_*a.cell), float(ys[0]+cy_*a.cell)],
        'size': [float(w_*a.cell), float(h_*a.cell)], 'yaw_deg': float(ang),
        'source': 'Minimum-area rectangle of the observed carved region; initialization only, not support approval'}
    yaw = float(((ang + 45) % 90) - 45)  # the rectangle's edge nearest the x axis, degrees
    res["top_yaw_deg"] = round(yaw, 2)
    (out / "support_carve.json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"[carve] the top's edges at {yaw:+.1f} deg to the world's axes", flush=True)
    from calibrated_scene_reconstruction.scene.twin.gates import options as twin_options

    topt = twin_options(s) or {}
    if (abs(yaw) > 3 and topt.get("background") == "zone" and a.apply
            and topt.get('support_route') != 'agent_blender'):
        # the task zone alone: no wall needs the room's yaw -- the world turned to the support's, the stages from the
        # scene graph rerun in it (the photo's world frame kept by autograph, the frames placed again)
        fp_path = s / "parsing" / "world_frame.json"
        wf = json.loads(fp_path.read_text())
        th = np.radians(-yaw)
        Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
        R_wc = Rz @ np.asarray(wf["R_wc"], float)
        o = np.asarray(wf["o"], float)
        T = np.eye(4)
        T[:3, :3] = R_wc
        T[:3, 3] = -R_wc @ o
        wf.update({"R_wc": R_wc.tolist(), "T_world_cam_cv": T.tolist(),
                   "support_yaw_deg": round(float(wf.get("support_yaw_deg", 0.0)) + yaw, 2),
                   "_support_yaw": "recon.support_carve: background zone, the world turned to the support's edges"})
        fp_path.write_text(json.dumps(wf, indent=1) + "\n")
        print(f"[carve] background zone: the world turned {-yaw:+.1f} deg to the support's edges "
              f"({fp_path}); rerun from graph (the scene graph, the frames' placement and this carve in it)",
              flush=True)
        raise SystemExit(3)
    if a.apply:
        sg = s.config("scene_graph.json")  # as reviewed: Session.scene_graph adds the derived closure walls
        box = sg["furniture"][key]["box"]
        sg["furniture"][key]["box"] = [[fp[0][0], fp[0][1], box[0][2] + dz], [fp[1][0], fp[1][1], box[1][2] + dz]]
        sg["_support_carve"] = {"from": box, "to": sg["furniture"][key]["box"], "frames": n_img,
                                "note": "the support's footprint carved from the posed frames (recon.support_carve)"}
        (s.config_dir / "scene_graph.json").write_text(json.dumps(sg, indent=1, ensure_ascii=False) + "\n")
        print(f"applied to {s.config_dir / 'scene_graph.json'}: rebuild the twin from layout")


if __name__ == "__main__":
    main()
