"""A task object's shape, refined against the frames round by round with the agent in the loop.

    envs/recon/bin/python -m calibrated_scene_reconstruction.recon.object_refine sessions/<name> run <obj> [--apply]
    envs/recon/bin/python -m calibrated_scene_reconstruction.recon.object_refine sessions/<name> status

A kit (recon.object_fit) is a family of shapes: what lies outside it -- a post's corners cut back for the screws
below them, a handle, a tab -- the fit averages away, and its IoU and unexplained share barely move. `run` takes the object's fitted kit and the silhouettes the fit
used (splat/object_fit/<obj>.npz) and

1. carves the kit's volume by them: 1 mm cells (`--voxel`), a cell is dropped when it falls outside the silhouette
   (grown by `--slack` px) in `--agree` of the frames that see it (a part SAM 3 missed in a few frames stays) -- what the frames show missing goes; what no
   silhouette can show (the inside of a hole) stays the kit's;
2. checks the kit and the carved shape against every silhouette, locally: the pixels where they differ, in pieces
   (at least 2 px thick: a pose a pixel off is no piece); a piece counts at `--piece` of the silhouette's area; per
   frame all its pieces together -- `local_kit` / `local_carved` (their median share; the share of frames with one);
3. draws the evidence: qa/object_refine/<obj>.jpg -- per frame the frame, the silhouette (red: the frame shows it,
   the shape lacks it; green: the shape has it, the frame does not; white: both) for the kit and the carved shape.

Writes splat/object_refine.json, stamps qa/review/OBJ.<obj>.ran; `--apply` sets dynamic_objects.<obj>.fix.mesh (the
carved mesh, in the kit's frame: sim.scene builds the object's look and collider from it; the kit's parameters stay
for its opening and the task check).

The loop (docs/twin_agent.md): read the sheet and the numbers, judge the shape, and record the round --
`python -m calibrated_scene_reconstruction.twin sessions/<name> record <json>` with "stage": "OBJ.<obj>" (checks, review, findings with
`action`, pass). Act (the kit or its parameters in dynamic_objects.<obj>.fit, the prompt, this module's arguments),
`run` again, record again -- until no finding is open. The S2 gate holds until every task object's latest round
passes a run no older than it."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy import ndimage

from calibrated_scene_reconstruction.assets import kits
from calibrated_scene_reconstruction.scene.session import Session
from calibrated_scene_reconstruction.scene.objects.object_hull import project


def world_kit(r):
    """The fitted kit as one mesh in the world (its xy, yaw, on its support)."""
    k = kits.build(r["kit"], r["params"])
    V = np.concatenate([np.asarray(p["V"], float) for p in k["parts"]])
    F, o = [], 0
    for p in k["parts"]:
        F.append(np.asarray(p["F"], int) + o)
        o += len(p["V"])
    a = np.radians(r["yaw_deg"])
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return trimesh.Trimesh(V @ R.T + [r["xy"][0], r["xy"][1], r["support_z"]], np.concatenate(F), process=True), R


def raster(mesh, K, Rw, c, win, shape):
    """A mesh's silhouette in a window: every triangle filled (exact: a cell grid filled a plate's through holes)."""
    from calibrated_scene_reconstruction.core.geometry.raster import camera_triangle_mask

    x0, y0, _, _, f = win
    camera = (np.asarray(mesh.vertices)-c) @ Rw
    crop = np.array([[f, 0., -x0*f], [0., f, -y0*f], [0., 0., 1.]])
    return camera_triangle_mask(camera, mesh.faces, crop @ K, shape)


def pieces(a, b, share):
    """The pieces where a and b differ (at least 2 px thick), largest first, each as (its share of a's area, mask,
    which: 'lacks' -- a has it, b does not -- or 'adds')."""
    out = []
    area = max(int(a.sum()), 1)
    for which, diff in (("lacks", a & ~b), ("adds", b & ~a)):
        thick = cv2.morphologyEx(diff.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
        lab, n = ndimage.label(thick)
        # Count all labels once. Building a full-image mask for every tiny
        # component made complex source comparisons quadratic in component
        # count, even though almost all components failed the same area gate.
        counts = np.bincount(lab.ravel(), minlength=n + 1)
        for j in np.flatnonzero(counts[1:] / area >= share) + 1:
            out.append((float(counts[j] / area), lab == j, which))
    return sorted(out, key=lambda q: -q[0])


def run(s, obj, a):
    fit = json.loads((s / "splat" / "object_fit.json").read_text())
    if obj not in fit:  # the report flags it (a task object with no fit); the stage goes on to show the rest
        print(f"[object_refine] {obj}: not fitted (recon.object_fit) -- nothing to check", flush=True)
        return
    r = fit[obj]
    d = np.load(s / "splat" / "object_fit" / f"{obj}.npz", allow_pickle=True)
    from .mask_cache import read_masks
    frames, masks = [int(i) for i in d["frames"]], read_masks(d)
    windows = [tuple(w) for w in d["windows"]]
    z = dict(np.load(s / "parsing" / "posed_frames" / "cameras.npz"))
    mesh, R = world_kit(r)
    vg = mesh.voxelized(a.voxel).fill(method="orthographic")  # a through hole stays open ("holes" filled them)
    cells = np.asarray(vg.points)
    seen, inn = np.zeros(len(cells), int), np.zeros(len(cells), int)
    ker = np.ones((2 * a.slack + 1, 2 * a.slack + 1), np.uint8)
    for i, m, w in zip(frames, masks, windows):
        x0, y0, _, _, f = w
        u, v, dd = project(cells, z["K"][i], z["R_world_cam"][i], z["centre"][i])
        u, v = (u - x0) * f, (v - y0) * f
        H, W = m.shape
        ok = (dd > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        M = cv2.dilate(m.astype(np.uint8), ker) > 0
        seen += ok
        hit = np.zeros(len(cells), bool)
        hit[ok] = M[v[ok].astype(int), u[ok].astype(int)]
        inn += hit
    # a cell goes only when the frames that see it agree it is not there (a part SAM 3 missed in a few frames, a
    # hand in front: the cell stays)
    keep = (seen < a.min_seen) | (seen - inn < a.agree * np.maximum(seen, 1))
    kept = cells[keep]
    # the carved shape as a mesh in the kit's frame
    idx = np.round((kept - cells.min(0)) / a.voxel).astype(int)
    grid = np.zeros(idx.max(0) + 3, bool)
    grid[tuple((idx + 1).T)] = True
    grid = ndimage.binary_opening(grid, np.ones((2, 2, 2)))  # single-cell crumbs off
    lab, n = ndimage.label(grid)
    if n > 1:  # its largest piece
        grid = lab == (1 + int(np.argmax(ndimage.sum(grid, lab, range(1, n + 1)))))
    from skimage import measure
    V, F, _, _ = measure.marching_cubes(ndimage.gaussian_filter(grid.astype(float), 0.5), 0.5,
                                        spacing=(a.voxel,) * 3)
    Vw = V - a.voxel + cells.min(0)
    carved = trimesh.Trimesh(Vw, F[:, ::-1], process=True)
    trimesh.smoothing.filter_taubin(carved, iterations=6)
    carved.fix_normals()
    org = np.array([r["xy"][0], r["xy"][1], r["support_z"]])
    local = trimesh.Trimesh((np.asarray(carved.vertices) - org) @ R, carved.faces, process=False)
    out_dir = s / "objects" / "task" / obj
    out_dir.mkdir(parents=True, exist_ok=True)
    local.export(out_dir / "mesh.glb")
    # the local check, kit and carved, and the evidence
    rows, stats = [], {"kit": [], "carved": []}
    for k, (i, m, w) in enumerate(zip(frames, masks, windows)):
        rk = raster(mesh, z["K"][i], z["R_world_cam"][i], z["centre"][i], w, m.shape)
        rc = raster(carved, z["K"][i], z["R_world_cam"][i], z["centre"][i], w, m.shape)
        pk, pc = pieces(m, rk, a.piece), pieces(m, rc, a.piece)
        stats["kit"].append(sum(q[0] for q in pk))  # all pieces together: a post's four cut corners are four
        stats["carved"].append(sum(q[0] for q in pc))  # small ones
        if len(rows) < 8:
            img = _frame(s, z, i, w, m.shape)
            tiles = [img]
            for ren in (rk, rc):
                vis = (0.45 * img).astype(np.uint8)
                vis[m & ~ren] = (230, 40, 40)
                vis[ren & ~m] = (40, 210, 40)
                vis[m & ren] = (235, 235, 235)
                tiles.append(vis)
            cv2.putText(tiles[1], f"kit {100 * stats['kit'][-1]:.1f}%", (6, 20), 0, 0.55, (255, 255, 0), 2)
            cv2.putText(tiles[2], f"carved {100 * stats['carved'][-1]:.1f}%", (6, 20), 0, 0.55, (255, 255, 0), 2)
            rows.append(np.vstack([cv2.resize(t, (260, 260)) for t in tiles]))
    qa = s / "qa" / "object_refine"
    qa.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(qa / f"{obj}.jpg"), np.hstack(rows)[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
    res_path = s / "splat" / "object_refine.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}

    def summ(v):
        v = np.asarray(v)
        return {"pieces_share_median": round(float(np.median(v)), 4), "frames_with_piece": round(float((v > 0).mean()), 2)}
    res[obj] = {"kit": r["kit"], "frames": len(frames), "voxel_m": a.voxel, "cells": int(len(cells)),
                "carved_share": round(float(1 - keep.mean()), 4), "local_kit": summ(stats["kit"]),
                "local_carved": summ(stats["carved"]), "mesh": str(out_dir / "mesh.glb"),
                "sheet": f"qa/object_refine/{obj}.jpg",
                "args": {"voxel": a.voxel, "slack": a.slack, "agree": a.agree, "piece": a.piece}}
    res_path.write_text(json.dumps(res, indent=1) + "\n")
    (s / "qa" / "review").mkdir(parents=True, exist_ok=True)
    (s / "qa" / "review" / f"OBJ.{obj}.ran").touch()
    print(f"[object_refine] {obj}: {100 * res[obj]['carved_share']:.1f} % of the kit's cells carved; local differences "
          f"kit {res[obj]['local_kit']}, carved {res[obj]['local_carved']} over {len(frames)} frames -> "
          f"{res[obj]['sheet']}", flush=True)
    if a.apply:
        cfg = s.config_dir / "scene_graph.json"
        g = json.loads(cfg.read_text())
        fx = g["dynamic_objects"][obj].setdefault("fix", {})
        fx["mesh"] = f"sessions/{s.dir.name}/objects/task/{obj}/mesh.glb"
        fx["_mesh"] = (f"recon.object_refine: the kit carved by its silhouettes ({100 * res[obj]['carved_share']:.1f} % "
                       "of its cells)")
        cfg.write_text(json.dumps(g, indent=1, ensure_ascii=False) + "\n")
        print(f"applied -> {cfg}", flush=True)


def _frame(s, z, i, w, shape):
    x0, y0, cw, ch, f = w
    img = cv2.imread(str(Path(str(z["source"])) / "splat" / "data" / "images" / str(z["names"][i])))[:, :, ::-1]
    win = img[int(y0):int(y0 + ch), int(x0):int(x0 + cw)]
    return cv2.resize(win, shape[::-1], interpolation=cv2.INTER_AREA)


def _set(p, path, v):
    """p with the dotted parameter `path` (plate_holes.inset) set to v."""
    p = json.loads(json.dumps(p))
    keys = path.split(".")
    q = p
    for k in keys[:-1]:
        q = q.setdefault(k, {})
    q[keys[-1]] = v
    return p


def tune(s, obj, a):
    """The agent's calibration of parameters the fit does not measure (a plate's holes, a post's cut corners): every
    combination of `--param path=lo:hi:n`, the kit rendered exactly into every silhouette, scored by the local
    differences (all pieces' share, the worst fifth of the frames left out: a hand, a missed part); the best, and
    with --apply written into dynamic_objects.<obj>.fit.params (the fit keeps them)."""
    import itertools

    fit = json.loads((s / "splat" / "object_fit.json").read_text())
    r = fit[obj]
    d = np.load(s / "splat" / "object_fit" / f"{obj}.npz", allow_pickle=True)
    from .mask_cache import read_masks
    frames, masks = [int(i) for i in d["frames"]], read_masks(d)
    windows = [tuple(w) for w in d["windows"]]
    z = dict(np.load(s / "parsing" / "posed_frames" / "cameras.npz"))
    grids = []
    for spec in a.param:
        path, rng = spec.split("=")
        lo, hi, n = rng.split(":")
        grids.append((path, np.linspace(float(lo), float(hi), int(n))))

    def score(params):
        mesh, _ = world_kit({**r, "params": params})
        v = []
        for i, m, w in zip(frames, masks, windows):
            ren = raster(mesh, z["K"][i], z["R_world_cam"][i], z["centre"][i], w, m.shape)
            v.append(sum(q[0] for q in pieces(m, ren, a.piece)))
        v = np.sort(v)[: max(1, int(np.ceil(0.8 * len(v))))]
        return float(np.mean(v))
    base = score(r["params"])
    best = (base, r["params"], None)
    for combo in itertools.product(*[g[1] for g in grids]):
        p = r["params"]
        for (path, _), v in zip(grids, combo):
            p = _set(p, path, round(float(v), 5))
        sc = score(p)
        if sc < best[0]:
            best = (sc, p, combo)
    vals = {g[0]: round(float(v), 5) for g, v in zip(grids, best[2])} if best[2] is not None else {}
    print(f"[object_refine] {obj}: local differences {100 * base:.2f} % as fitted, {100 * best[0]:.2f} % at "
          f"{vals or 'the fitted values'}", flush=True)
    if a.apply and vals:
        cfg = s.config_dir / "scene_graph.json"
        g = json.loads(cfg.read_text())
        f_ = g["dynamic_objects"][obj].setdefault("fit", {})
        prm = f_.get("params") or {}
        for path, v in vals.items():
            prm = _set(prm, path, v)
        f_["params"] = prm
        f_["_tuned"] = (f"recon.object_refine tune: {vals}, local differences {100 * base:.2f} -> "
                        f"{100 * best[0]:.2f} % of the silhouettes")
        cfg.write_text(json.dumps(g, indent=1, ensure_ascii=False) + "\n")
        print(f"applied -> {cfg} (rerun object_fit to fit the rest with them)", flush=True)


def status(s):
    from calibrated_scene_reconstruction.scene.twin.gates import review

    sg = s.scene_graph
    for n, o in (sg.get("dynamic_objects") or {}).items():
        if not o.get("fit"):
            continue
        r = review(s, f"OBJ.{n}")
        print(f"{n}: " + (f"round {r['round']} {'pass' if r['pass'] else 'open'} ({len(r['findings'])} finding(s))"
                          if r else "no round recorded"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("cmd", choices=["run", "tune", "status"])
    ap.add_argument("--param", action="append", default=[], help="tune: path=lo:hi:n (plate_holes.inset=0.01:0.02:5)")
    ap.add_argument("obj", nargs="?")
    ap.add_argument("--voxel", type=float, default=0.001)
    ap.add_argument("--slack", type=int, default=1)
    ap.add_argument("--agree", type=float, default=0.8, help="a cell goes when this share of its frames lack it")
    ap.add_argument("--min-seen", type=int, default=4)
    ap.add_argument("--piece", type=float, default=0.003, help="a local difference counts at this share of the area")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    s = Session(a.session)
    kits.use_session(s.config_dir)
    if a.cmd == "status":
        status(s)
        return
    (tune if a.cmd == "tune" else run)(s, a.obj, a)


if __name__ == "__main__":
    main()
