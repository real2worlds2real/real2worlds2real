"""Posed frames: walk frames with their COLMAP poses (at the board's scale, in the twin's world) and a metric depth
each, as views for the mesh pipeline -- the room's objects, its walls, their textures -- beside the photo and its
DA3 views. No Gaussian splat: only splat_capture's posed frames (or another session's, `posed_frames.json` "source").

    tools/run_splat.sh -m calibrated_scene_reconstruction.recon.posed_frames sessions/<name> select [--n 120]     (envs/splat: pycolmap)
    envs/recon/bin/python -m calibrated_scene_reconstruction.recon.posed_frames sessions/<name> depth              (envs/recon: MoGe-2)

select  `n` frames spread over the walks: farthest-point sampling on the cameras' positions and viewing directions
        (a direction counts as `dir_weight` m), frames with fewer than `min_points` triangulated points left out (a
        blurred or featureless frame); per frame its K, its camera's pose in the world (R_world_cam OpenCV axes,
        the centre) and its observations of the sparse points (pixel, metric depth along the optical axis), from
        sparse/full (the task zone's points kept). Writes parsing/posed_frames/select.npz, and every frame's camera
        in parsing/posed_frames/cameras.npz.
depth   MoGe-2 per frame with the frame's own field of view, its depth scaled to the frame's sparse points (median of
        COLMAP / MoGe depth at the observed pixels; frames whose ratios scatter over `max_spread` -- the MAD over the
        median -- are dropped: MoGe bent that frame's geometry); the pixels MoGe masks out or whose scaled depth
        disagrees with a sparse point within 3 px by over 10 % or that lies at a depth edge (a jump of over `edge` of the depth per pixel: lifted, such a pixel hangs in the air between the two surfaces) are not `ok`. Writes parsing/posed_frames/frames.npz
        (images at `width` px, K at that size, planar depth, ok, R_world_cam, centre, names) and frames.json (per
        frame: scale, spread, residual), and qa/posed_frames.jpg (frames with their depth).
configs/<name>/posed_frames.json (optional): {"source": "sessions/<other>"} takes another session's splat/data and
splat/align.json; "n", "width", "dir_weight", "min_points", "max_spread"."""
import argparse
import json
from pathlib import Path

import numpy as np

from calibrated_scene_reconstruction.scene.session import Session

DEFAULTS = {"n": 120, "width": 960, "dir_weight": 1.0, "min_points": 300, "max_spread": 0.12, "edge": 0.03,
            "seg": 0.5, "wall_band": [1.6, 2.5], "wall_reach": 0.4, "merge": 0.25, "close": 2,
            "wall_fill": 0.6}


def settings(s):
    cfg = {**DEFAULTS, **s.config("posed_frames.json", {})}
    src = Path(cfg["source"]) if cfg.get("source") else s.dir
    if not src.is_absolute():
        src = s.dir.parent.parent / src if not src.exists() else src
    cfg["source_dir"] = src
    return cfg


def select(s, cfg):
    import pycolmap

    src = cfg["source_dir"]
    a = json.loads((src / "splat" / "align.json").read_text())["sim3"]
    c, Rs, ts = float(a["scale"]), np.asarray(a["R"], float), np.asarray(a["t"], float)
    rec = pycolmap.Reconstruction(str(src / "splat" / "data" / "sparse" / "full"))
    imgs = sorted(rec.images.values(), key=lambda i: i.name)
    cams, dirs, nobs = [], [], []
    for img in imgs:
        T = np.asarray(img.cam_from_world().matrix())
        Rcw, t = T[:, :3], T[:, 3]
        cams.append(c * Rs @ (-Rcw.T @ t) + ts)
        dirs.append(Rs @ Rcw.T[:, 2])
        nobs.append(sum(p.has_point3D() for p in img.points2D))
    cams, dirs, nobs = np.array(cams), np.array(dirs), np.array(nobs)
    cand = np.flatnonzero(nobs >= cfg["min_points"])
    F = np.c_[cams[cand], cfg["dir_weight"] * dirs[cand]]
    pick = [int(np.argmax(nobs[cand]))]
    d = np.linalg.norm(F - F[pick[0]], axis=1)
    while len(pick) < min(cfg["n"], len(cand)):
        k = int(np.argmax(d))
        pick.append(k)
        d = np.minimum(d, np.linalg.norm(F - F[k], axis=1))
    sel = sorted(cand[pick])
    out = {"names": [], "K": [], "size": [], "R_world_cam": [], "centre": [], "obs": []}
    for i in sel:
        img = imgs[i]
        cam = rec.cameras[img.camera_id]
        fx, fy, cx, cy = (float(x) for x in cam.params[:4])
        T = np.asarray(img.cam_from_world().matrix())
        uv, z = [], []
        for p in img.points2D:
            if p.has_point3D():
                X = rec.points3D[p.point3D_id].xyz
                uv.append(p.xy)
                z.append(c * float(T[2, :3] @ X + T[2, 3]))  # metric depth along the optical axis
        out["names"].append(img.name)
        out["K"].append([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        out["size"].append([cam.width, cam.height])
        out["R_world_cam"].append(Rs @ T[:, :3].T)
        out["centre"].append(cams[i])
        out["obs"].append(np.c_[np.array(uv), np.array(z)].astype(np.float32))
    dst = s / "parsing" / "posed_frames"
    dst.mkdir(parents=True, exist_ok=True)
    allK, allsize, allR = [], [], []
    for img in imgs:  # every frame's camera (the close-ups of a small object: recon.object_fit, apps/object_closeups)
        cam = rec.cameras[img.camera_id]
        fx, fy, cx, cy = (float(x) for x in cam.params[:4])
        allK.append([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        allsize.append([cam.width, cam.height])
        allR.append(Rs @ np.asarray(img.cam_from_world().matrix())[:, :3].T)
    np.savez_compressed(dst / "cameras.npz", names=np.array([i.name for i in imgs]), K=np.array(allK),
                        size=np.array(allsize), camera_model=np.array([rec.cameras[i.camera_id].model.name for i in imgs]),
                        R_world_cam=np.array(allR), centre=cams, source=str(src))
    np.savez_compressed(dst / "select.npz", names=np.array(out["names"]), K=np.array(out["K"]),
                        size=np.array(out["size"]), R_world_cam=np.array(out["R_world_cam"]),
                        centre=np.array(out["centre"]), obs=np.array(out["obs"], dtype=object),
                        source=str(src))
    print(f"done: {len(sel)} of {len(imgs)} frames ({len(cand)} with >= {cfg['min_points']} points), "
          f"{np.mean([len(o) for o in out['obs']]):.0f} observations each -> {dst / 'select.npz'}")


def depth(s, cfg):
    import cv2
    import torch
    from moge.model.v2 import MoGeModel

    dst = s / "parsing" / "posed_frames"
    z = np.load(dst / "select.npz", allow_pickle=True)
    src = Path(str(z["source"]))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(dev).eval()
    W = int(cfg["width"])
    keep, rows = [], []
    images, Ks, depths, oks = [], [], [], []
    for i, name in enumerate(z["names"]):
        img = cv2.imread(str(src / "splat" / "data" / "images" / str(name)))[:, :, ::-1]
        h0, w0 = img.shape[:2]
        f = W / w0
        img = cv2.resize(img, (W, int(round(h0 * f))), interpolation=cv2.INTER_AREA)
        K = z["K"][i].copy()
        K[:2] *= f
        fov = float(np.degrees(2 * np.arctan(W / 2 / K[0, 0])))
        x = torch.tensor(img / 255.0, dtype=torch.float32, device=dev).permute(2, 0, 1)
        with torch.no_grad():
            o = model.infer(x, fov_x=fov, use_fp16=dev == "cuda")
        D = o["depth"].float().cpu().numpy()
        M = o["mask"].cpu().numpy().astype(bool) & np.isfinite(D)
        ob = z["obs"][i].astype(np.float64)
        u, v = (ob[:, 0] * f).astype(int), (ob[:, 1] * f).astype(int)
        inb = (u >= 0) & (u < W) & (v >= 0) & (v < img.shape[0])
        u, v, zc = u[inb], v[inb], ob[inb, 2]
        good = M[v, u] & (zc > 0.1)
        r = zc[good] / D[v[good], u[good]]
        if len(r) < 50:
            rows.append({"name": str(name), "dropped": "too few observations on MoGe's mask"})
            continue
        k = float(np.median(r))
        spread = float(np.median(np.abs(r / k - 1)))
        Dm = D * k
        res = np.abs(Dm[v[good], u[good]] / zc[good] - 1)
        row = {"name": str(name), "scale": round(k, 4), "spread": round(spread, 4),
               "residual_median": round(float(np.median(res)), 4), "observations": int(good.sum())}
        if spread > cfg["max_spread"]:
            row["dropped"] = f"ratios scatter {spread:.2f} > {cfg['max_spread']}"
            rows.append(row)
            continue
        # not ok: off MoGe's mask, or near a sparse point it disagrees with by over 10 %
        bad = np.zeros(img.shape[:2], np.uint8)
        far = good.copy()
        far[good] = res > 0.10
        for uu, vv in zip(u[far], v[far]):
            cv2.circle(bad, (int(uu), int(vv)), 3, 1, -1)
        # and not at a depth edge: a pixel between a near and a far surface is lifted into the air between them (the
        # fused room grew rays from every camera); 2 px either side of a jump of over `edge` of the depth
        gy, gx = np.gradient(np.where(M, Dm, np.nan))
        jump = ~(np.hypot(gx, gy) / np.maximum(Dm, 0.1) < cfg["edge"])
        jump = cv2.dilate(jump.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        ok = M & (bad == 0) & ~jump
        rows.append(row)
        keep.append(i)
        images.append(img)
        Ks.append(K)
        depths.append(np.where(ok, Dm, 0).astype(np.float16))
        oks.append(ok)
        if i % 20 == 0:
            print(f"[depth] {i + 1}/{len(z['names'])} {name}: scale {k:.3f}, spread {spread:.3f}", flush=True)
    del model
    H = max(im.shape[0] for im in images)
    pad = lambda a: np.pad(a, ((0, H - a.shape[0]), (0, 0)) + ((0, 0),) * (a.ndim - 2))  # noqa: E731
    np.savez_compressed(dst / "frames.npz", names=z["names"][keep], images=np.stack([pad(a) for a in images]),
                        K=np.stack(Ks), depth=np.stack([pad(a) for a in depths]), ok=np.stack([pad(a) for a in oks]),
                        R_world_cam=z["R_world_cam"][keep], centre=z["centre"][keep])
    (dst / "frames.json").write_text(json.dumps({"kept": len(keep), "of": len(z["names"]), "frames": rows}, indent=1) + "\n")
    # QA: a dozen frames beside their depth
    import cv2 as _cv

    tiles = []
    for j in np.linspace(0, len(keep) - 1, min(12, len(keep))).astype(int):
        im = images[j]
        d = depths[j].astype(np.float32)
        dv = np.clip(d / 8.0, 0, 1)
        col = _cv.applyColorMap((255 * (1 - dv)).astype(np.uint8), _cv.COLORMAP_TURBO)
        col[~oks[j]] = 0
        t = np.hstack([im[:, :, ::-1], col])
        tiles.append(_cv.resize(t, (960, int(960 * t.shape[0] / t.shape[1]))))
    hh = max(t.shape[0] for t in tiles)
    tiles = [np.pad(t, ((0, hh - t.shape[0]), (0, 0), (0, 0))) for t in tiles]
    rows_img = [np.hstack(tiles[k:k + 2]) for k in range(0, len(tiles) - 1, 2)]
    (s / "qa").mkdir(exist_ok=True)
    _cv.imwrite(str(s / "qa" / "posed_frames.jpg"), np.vstack(rows_img))
    sc = np.array([r["scale"] for r in rows if "dropped" not in r])
    sp = np.array([r["spread"] for r in rows if "dropped" not in r])
    print(f"done: {len(keep)} of {len(z['names'])} frames, MoGe scale {np.median(sc):.3f} (x{sc.min():.2f}-{sc.max():.2f}), "
          f"ratio spread median {np.median(sp):.3f} -> {dst / 'frames.npz'}, {s / 'qa' / 'posed_frames.jpg'}")


def lift(s, per_frame=30000, seed=0):
    """The frames' ok pixels in the world: (points, frame index)."""
    z = np.load(s / "parsing" / "posed_frames" / "frames.npz")
    rng = np.random.default_rng(seed)
    P, F = [], []
    for i in range(len(z["names"])):
        D = z["depth"][i].astype(np.float32)
        v, u = np.nonzero(z["ok"][i] & (D > 0.1) & (D < 15))
        k = rng.choice(len(u), min(per_frame, len(u)), replace=False)
        u, v = u[k], v[k]
        K = z["K"][i]
        X = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones(len(u))], 1) * D[v, u][:, None]
        P.append(X @ z["R_world_cam"][i].T + z["centre"][i])
        F.append(np.full(len(u), i))
    return np.concatenate(P), np.concatenate(F)


def structure(s, cfg, apply=False):
    """The room's walls, floor and ceiling from the fused frames. On a 10 cm grid in the twin's (room-aligned)
    frame: the floor the frames saw (points within 8 cm of z = 0, its largest region round the task support) and the
    tall structure (points `wall_band` high: above desks and partitions, below the ceiling). Per side of the floor's
    bounding box, strips `seg` wide across it: the floor's outermost cell, and the outermost tall cell within
    `wall_reach` past it, is the wall there (the floor's edge + 5 cm where no tall cell stands); strips whose walls
    lie within `merge` of each other are one piece (the median), a piece under 1 m joins its deeper neighbour. Steps
    between pieces are closed by scenegraph.close_room. The ceiling: the densest 5 cm band of points 2.3-3.2 m up.
    Writes parsing/posed_frames/structure.json and qa/posed_frames_structure.png; `--apply` replaces the scene
    graph's walls, floor and ceiling (their looks: the old floor's and ceiling's, the walls' painted drywall matched
    to what is seen), `_frames_structure` noted."""
    import cv2
    from scipy import ndimage

    P, _ = lift(s)
    cell, seg = 0.10, cfg["seg"]
    lo = np.percentile(P[:, :2], 0.2, 0) - 1.0
    hi = np.percentile(P[:, :2], 99.8, 0) + 1.0
    n = np.ceil((hi - lo) / cell).astype(int)

    def grid(m, k):
        g = np.zeros(n[::-1], np.int32)
        ij = np.floor((P[m, :2] - lo) / cell).astype(int)
        ok = np.all((ij >= 0) & (ij < n), 1)
        np.add.at(g, (ij[ok, 1], ij[ok, 0]), 1)
        return g >= k
    zc = P[(P[:, 2] > 2.3) & (P[:, 2] < 3.2), 2]
    h, e = np.histogram(zc, bins=np.arange(2.3, 3.2001, 0.05))
    ceiling = float(e[h.argmax()] + 0.025) if len(zc) else 2.6
    floor = grid(np.abs(P[:, 2]) < 0.08, 3)
    floor = ndimage.binary_closing(floor, np.ones((5, 5)))
    lab, _ = ndimage.label(floor)
    sg = s.config("scene_graph.json")
    t = np.asarray(sg["furniture"][sg["task"]["support"]]["box"], float)
    c = np.floor(((t[0][:2] + t[1][:2]) / 2 - lo) / cell).astype(int)
    # the task support's own region; under the support the floor is hidden: the region nearest its centre
    li = lab[c[1], c[0]]
    if li == 0:
        jj, ii = np.nonzero(lab)
        k = np.argmin((ii - c[0]) ** 2 + (jj - c[1]) ** 2)
        li = lab[jj[k], ii[k]]
    floor = lab == li
    # tall structure: a cell whose points fill most of the band's 10 cm layers (a vertical surface: a wall, a
    # partition, a tall shelf); a cell with points in the band at all also took the lights, the ducts and the
    # depth's streaks, and closed the whole office in
    z0, z1 = cfg["wall_band"][0], min(cfg["wall_band"][1], ceiling - 0.15)
    layers = np.arange(z0, z1, 0.1)
    m = (P[:, 2] > z0) & (P[:, 2] < z1)
    ij = np.floor((P[m, :2] - lo) / cell).astype(int)
    lz = np.floor((P[m, 2] - z0) / 0.1).astype(int)
    okk = np.all((ij >= 0) & (ij < n), 1) & (lz < len(layers))
    occ = np.zeros((n[1], n[0], len(layers)), bool)
    occ[ij[okk, 1], ij[okk, 0], lz[okk]] = True
    tall = occ.mean(2) >= cfg["wall_fill"]
    # Estimate room extent from observed cells connected to the task support. Close wall gaps
    # and bound the fill by observed space to prevent leakage through unobserved doorways.
    seen = ndimage.binary_dilation(grid((P[:, 2] > -0.1) & (P[:, 2] < ceiling), 2), iterations=3)
    seen = ndimage.binary_fill_holes(seen)
    barrier = ndimage.binary_dilation(tall, iterations=cfg["close"]) | ~seen
    lab, _ = ndimage.label(~barrier)
    cams = np.load(s / "parsing" / "posed_frames" / "frames.npz")["centre"]
    cij = np.clip(np.floor((cams[:, :2] - lo) / cell).astype(int), 0, n - 1)
    seeds = {int(lab[j, i]) for i, j in cij} - {0}
    room = ndimage.binary_fill_holes(ndimage.binary_dilation(np.isin(lab, list(seeds)), iterations=cfg["close"]))
    floor = room
    jj, ii = np.nonzero(floor)
    xs = lo[0] + (np.arange(n[0]) + 0.5) * cell
    ys = lo[1] + (np.arange(n[1]) + 0.5) * cell
    bx = (xs[ii.min()] - cell / 2, xs[ii.max()] + cell / 2)
    by = (ys[jj.min()] - cell / 2, ys[jj.max()] + cell / 2)
    reach = int(round(cfg["wall_reach"] / cell))
    walls = {}
    for side, ax, sign in (("left", 0, -1), ("right", 0, 1), ("front", 1, -1), ("back", 1, 1)):
        o = 1 - ax  # the axis along the side
        span = by if o == 1 else bx
        strips = []
        for a0 in np.arange(span[0], span[1], seg):
            a1 = min(a0 + seg, span[1])
            if o == 1:
                rows = (ys >= a0) & (ys < a1)
                f, w, coord = floor[rows].any(0), tall[rows].any(0), xs
            else:
                cols = (xs >= a0) & (xs < a1)
                f, w, coord = floor[:, cols].any(1), tall[:, cols].any(1), ys
            if not f.any():
                continue
            idx = np.flatnonzero(f)
            edge = idx.max() if sign > 0 else idx.min()
            band = np.arange(edge, edge + sign * (reach + 1), sign)
            band = band[(band >= 0) & (band < len(coord))]
            hit = band[w[band]]
            at = coord[hit[-1]] if len(hit) else coord[edge] + sign * 0.05
            strips.append([a0, a1, float(at)])
        pieces = []
        for a0, a1, at in strips:
            if pieces and abs(np.median(pieces[-1][2]) - at) < cfg["merge"] and abs(pieces[-1][1] - a0) < 1e-6:
                pieces[-1][1] = a1
                pieces[-1][2].append(at)
            else:
                pieces.append([a0, a1, [at]])
        pieces = [[a0, a1, float(np.median(ats))] for a0, a1, ats in pieces]
        k = 0
        while k < len(pieces):  # a short piece joins its deeper neighbour
            if pieces[k][1] - pieces[k][0] < 1.0 and len(pieces) > 1:
                nb = [j for j in (k - 1, k + 1) if 0 <= j < len(pieces)]
                j = max(nb, key=lambda j: sign * pieces[j][2])
                pieces[j][0], pieces[j][1] = min(pieces[j][0], pieces[k][0]), max(pieces[j][1], pieces[k][1])
                pieces.pop(k)
                k = 0
                continue
            k += 1
        for k, (a0, a1, at) in enumerate(pieces):
            walls[f"wall_{side}_{k}"] = {"role": "wall", "plane": "xy"[ax], "at": round(at, 3),
                                         "range": [[round(a0, 3), round(a1, 3)], [0.0, round(ceiling, 3)]], "ids": [],
                                         "normal": "-" if sign > 0 else "+"}
    ats = {sd: [w["at"] for k, w in walls.items() if f"_{sd}_" in k] for sd in ("left", "right", "front", "back")}
    rng = [[min(ats["left"]), max(ats["right"])], [min(ats["front"]), max(ats["back"])]]
    res = {"walls": walls, "floor_range": rng, "ceiling": ceiling,
           "floor_seen_m2": round(float(floor.sum() * cell * cell), 1)}
    dst = s / "parsing" / "posed_frames"
    (dst / "structure.json").write_text(json.dumps(res, indent=1) + "\n")
    img = np.full((*n[::-1], 3), 255, np.uint8)
    img[floor] = (215, 235, 215)
    img[tall] = (90, 90, 90)
    px = lambda x, y: (int((x - lo[0]) / cell), int((y - lo[1]) / cell))  # noqa: E731
    for w in walls.values():
        (a0, a1), at = w["range"][0], w["at"]
        p0, p1 = (px(at, a0), px(at, a1)) if w["plane"] == "x" else (px(a0, at), px(a1, at))
        cv2.line(img, p0, p1, (30, 30, 220), 2)
    for w in sg["structure"].values():
        if w.get("role") == "wall" and w.get("plane") in ("x", "y"):
            (a0, a1), at = w["range"][0], w["at"]
            p0, p1 = (px(at, a0), px(at, a1)) if w["plane"] == "x" else (px(a0, at), px(a1, at))
            cv2.line(img, p0, p1, (220, 120, 40), 1)
    cv2.rectangle(img, px(*t[0][:2]), px(*t[1][:2]), (0, 140, 255), 2)
    img = cv2.resize(cv2.flip(img, 0), None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    (s / "qa").mkdir(exist_ok=True)
    cv2.imwrite(str(s / "qa" / "posed_frames_structure.png"), img)
    print(f"[structure] {len(walls)} wall pieces, floor {rng[0][1] - rng[0][0]:.1f} x {rng[1][1] - rng[1][0]:.1f} m "
          f"({res['floor_seen_m2']} m2 seen), ceiling {ceiling:.2f} m", flush=True)
    for k, w in walls.items():
        print(f"  {k}: {w['plane']} = {w['at']:.2f}, {w['range'][0][0]:.2f}..{w['range'][0][1]:.2f}")
    if apply:
        old = sg["structure"]
        fl = next(v for k, v in old.items() if v.get("role") == "floor" or k == "floor")
        ce = next((v for k, v in old.items() if v.get("role") == "ceiling" or k == "ceiling"), {})
        new = {k: {**w, "look": {"pbr": "proc_painted_drywall", "tint": "match",
                                 "_why": "recon.posed_frames structure: a wall the walks saw"}} for k, w in walls.items()}
        new["floor"] = {**fl, "range": [[round(v, 3) for v in r] for r in rng]}
        new["ceiling"] = {**ce, "role": "ceiling", "plane": "z", "at": round(ceiling, 3),
                          "range": [[round(v, 3) for v in r] for r in rng], "ids": [], "normal": "-"}
        sg["_frames_structure"] = {"from": old, "note": "the room's walls, floor and ceiling from the posed frames "
                                                        "(recon.posed_frames structure)"}
        sg["structure"] = new
        (s.config_dir / "scene_graph.json").write_text(json.dumps(sg, indent=1, ensure_ascii=False) + "\n")
        print(f"applied to {s.config_dir / 'scene_graph.json'}: rebuild from layout")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("step", choices=["select", "depth", "structure"])
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--apply", action="store_true", help="structure: write it into the scene graph")
    a = ap.parse_args()
    s = Session(a.session)
    cfg = settings(s)
    if a.n:
        cfg["n"] = a.n
    if a.step == "structure":
        structure(s, cfg, a.apply)
    else:
        (select if a.step == "select" else depth)(s, cfg)


if __name__ == "__main__":
    main()
