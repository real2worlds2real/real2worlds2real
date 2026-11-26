"""The work zone's structures -- whatever stands on or round the task's support and is not a task object (a camera
frame, a fixture, a shelf, a jig) -- as meshes carved from the posed frames' silhouettes: no kit, no shape assumed.

    envs/recon/bin/python -m calibrated_scene_reconstruction.recon.structure_hull sessions/<name> [--apply]
    (after recon.posed_frames select; the scene graph's `structures`)

The scene graph names them: `structures.<name>` {"prompt": "black aluminium frame", "on": <support>, "margin": 0.25,
"height": 2.0} -- the prompt is what SAM 3 is asked in the frames, the region searched is the support's footprint
grown by `margin` (m), from its top up `height` (m). Per structure:

1. up to `--frames` posed frames that see the region, from all round it; in each the union of SAM 3's instances of
   the prompt (the whole frame, at half size).
2. a grid of `--voxel` cells over the region, each projected into every frame: it is *seen* by the frames it falls
   inside (in front of the camera), and *in* by those whose silhouette (grown by `--slack` px: a pose a few pixels
   off) holds it. A cell stays when seen by `--min-seen` frames and in `--agree` of them (a robot's arm in front of
   a bar in a few frames); a cell seen by fewer is unknown, never kept.
3. the kept cells' pieces of at least `--min-cells`, standing on the support (their bottom within 4 cm of its top);
   marching cubes over them -> a mesh in the support's frame (origin: the region's centre on the top).
4. the check, frame by frame: the mesh's cells projected against the silhouette -- `precision` (the share of the
   structure's pixels on the silhouette: nothing added) and `recall` (the share of the silhouette the structure
   covers: nothing missing), their medians; and where the structure meets cells no frame saw (it goes on where the
   capture did not look: the top of a frame's posts) -- reported, the twin ends there.

Writes objects/structures/<name>/mesh.glb, splat/structures.json, qa/structures/<name>.jpg (frames: silhouette red,
structure green, both white). `--apply`: scene graph set_pieces.<name> {"kit": "mesh", "path", "color", "center",
"on"} (sim.setpieces builds it, a triangle-mesh collider), `_structure_hull` a note.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image
from scipy import ndimage
from skimage import measure

from calibrated_scene_reconstruction.scene.session import Session
from calibrated_scene_reconstruction.scene.objects import frame_color
from calibrated_scene_reconstruction.scene.objects.object_fit import pick_frames
from calibrated_scene_reconstruction.scene.objects.object_hull import CropSam3, project

SCALE = 0.5  # the silhouettes at half the frame's size
OCCLUDERS = ["humanoid robot", "robot arm", "robot hand"]  # structures.<name>.occluders: what may stand in front


def frames_for(z, lo, hi, n, margin=40):
    """Up to n frames of the whole walk that look at the region (its middle or the middles of its two halves, so a
    long one is seen end to end), evenly along the walk: the nearest frames of each direction were a run of
    neighbours that all looked from the same side."""
    c = (lo + hi) / 2
    long_ = int(np.argmax(hi[:2] - lo[:2]))
    pts = [c]
    for f_ in (0.25, 0.75):
        e = c.copy()
        e[long_] = lo[long_] + f_ * (hi[long_] - lo[long_])
        pts.append(e)
    W, H = (int(v) for v in z["size"][0])
    see = []
    for i in range(len(z["names"])):
        for p in pts:
            u, v, d = project(p[None], z["K"][i], z["R_world_cam"][i], z["centre"][i])
            if d[0] > 0.2 and margin < u[0] < W - margin and margin < v[0] < H - margin:
                see.append(i)
                break
    if len(see) <= n:
        return see
    return [see[int(k)] for k in np.linspace(0, len(see) - 1, n).round()]


def carve(z, frames, masks, occl, lo, hi, vox, slack, min_seen, agree, dev):
    """(kept, seen) boolean grids over the region (cells of `vox`), and the grid's axes. A frame whose occluder
    silhouette (the robot) covers a cell's pixel off the structure's silhouette does not vote on it: the thing may
    stand behind."""
    axes = [np.arange(lo[k] + vox / 2, hi[k], vox) for k in range(3)]
    G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    P = torch.tensor(G, dtype=torch.float32, device=dev)
    seen = torch.zeros(len(G), dtype=torch.int32, device=dev)
    inn = torch.zeros(len(G), dtype=torch.int32, device=dev)
    ker = np.ones((2 * slack + 1, 2 * slack + 1), np.uint8)
    for i, m, o in zip(frames, masks, occl):
        M = torch.tensor(cv2.dilate(m.astype(np.uint8), ker) > 0, device=dev)
        O = torch.tensor(cv2.dilate(o.astype(np.uint8), ker) > 0, device=dev) & ~M
        H, W = M.shape
        K = torch.tensor(z["K"][i], dtype=torch.float32, device=dev)
        R = torch.tensor(z["R_world_cam"][i], dtype=torch.float32, device=dev)
        c = torch.tensor(z["centre"][i], dtype=torch.float32, device=dev)
        Xc = (P - c) @ R
        d = Xc[:, 2]
        u = (K[0, 0] * Xc[:, 0] / d + K[0, 2]) * SCALE
        v = (K[1, 1] * Xc[:, 1] / d + K[1, 2]) * SCALE
        ok = (d > 0.1) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
        ui, vi = u[ok].round().long(), v[ok].round().long()
        hid = torch.zeros_like(ok)
        hid[ok] = O[vi, ui]
        ok &= ~hid
        ui, vi = u[ok].round().long(), v[ok].round().long()
        seen += ok.int()
        hit = torch.zeros_like(ok)
        hit[ok] = M[vi, ui]
        inn += hit.int()
    seen, inn = seen.cpu().numpy(), inn.cpu().numpy()
    kept = (seen >= min_seen) & (inn >= agree * np.maximum(seen, 1))
    shape = tuple(len(a) for a in axes)
    return kept.reshape(shape), (seen >= min_seen).reshape(shape), axes


def splat_cells(cells, vox, K, R, c, shape):
    """The cells (world centres) drawn into a frame at SCALE: each a square of its projected size."""
    H, W = shape
    Xc = (cells - c) @ R
    d = Xc[:, 2]
    ok = d > 0.1
    u = (K[0, 0] * Xc[ok, 0] / d[ok] + K[0, 2]) * SCALE
    v = (K[1, 1] * Xc[ok, 1] / d[ok] + K[1, 2]) * SCALE
    r = np.maximum(1, np.ceil(0.5 * vox * K[0, 0] * SCALE / d[ok])).astype(int)
    img = np.zeros((H, W), bool)
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    for rr in np.unique(r[inside]):
        sel = inside & (r == rr)
        canvas = np.zeros((H, W), np.uint8)
        canvas[v[sel].astype(int), u[sel].astype(int)] = 1
        img |= cv2.dilate(canvas, np.ones((2 * rr - 1, 2 * rr - 1), np.uint8)) > 0
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--voxel", type=float, default=0.01)
    ap.add_argument("--slack", type=int, default=1, help="px at half size the silhouettes are grown by")
    ap.add_argument("--min-seen", type=int, default=6)
    ap.add_argument("--agree", type=float, default=0.7)
    ap.add_argument("--min-cells", type=int, default=300)
    ap.add_argument("--reuse", action="store_true", help="the silhouettes of the last run (same prompt)")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    s = Session(a.session)
    sg = s.scene_graph
    from calibrated_scene_reconstruction.scene.twin.gates import meshed_structures

    todo = sg.get("structures") or {}
    if not meshed_structures(s):
        print("done: background splat or zone -- only what the task touches is a mesh (the structures: the splat's, "
              "or left out)")
        return
    if not todo:
        print("done: no structures in the scene graph")
        return
    z = dict(np.load(s / "parsing" / "posed_frames" / "cameras.npz"))
    names = [str(n) for n in z["names"]]
    img_dir = Path(str(z["source"])) / "splat" / "data" / "images"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sam = None
    out_json, res = s / "splat" / "structures.json", {}
    qa = s / "qa" / "structures"
    qa.mkdir(parents=True, exist_ok=True)
    for name, st in todo.items():
        sup = st.get("on") or sg["task"]["support"]
        box = np.asarray(sg["furniture"][sup]["box"], float)
        top = float(box[1, 2])
        mg, ht = float(st.get("margin", 0.25)), float(st.get("height", 2.0))
        lo = np.array([box[0, 0] - mg, box[0, 1] - mg, top + 0.5 * a.voxel])
        hi = np.array([box[1, 0] + mg, box[1, 1] + mg, top + ht])
        prompt = st["prompt"]
        cache = s / "splat" / "structures" / f"{name}.npz"
        occ_prompts = st.get("occluders", OCCLUDERS)
        if a.reuse and cache.exists() and str(np.load(cache)["prompt"]) == prompt and "occl" in np.load(cache) and\
                list(np.load(cache)["occ_prompts"]) == list(occ_prompts):
            d = np.load(cache, allow_pickle=True)
            frames, masks = [int(i) for i in d["frames"]], [np.asarray(m, bool) for m in d["masks"]]
            occl = [np.asarray(m, bool) for m in d["occl"]]
        else:
            sam = sam or CropSam3(None)
            frames, masks, occl = [], [], []
            c = (lo + hi) / 2
            for i in frames_for(z, lo, np.array([hi[0], hi[1], top + 0.3]), a.frames):
                im = np.asarray(Image.open(img_dir / names[i]).convert("RGB"))
                u, v, _ = project(c[None], z["K"][i], z["R_world_cam"][i], z["centre"][i])
                m = sam(im, prompt, u[0], v[0], union=1e9)
                if m is None:
                    continue
                H, W = m.shape
                o = np.zeros_like(m)
                for op in occ_prompts:  # what stands in front of it in some frames: the robot
                    mo = sam(im, op, u[0], v[0], union=1e9)
                    if mo is not None:
                        o |= mo
                frames.append(i)
                masks.append(cv2.resize(m.astype(np.uint8), (int(W * SCALE), int(H * SCALE)),
                                        interpolation=cv2.INTER_NEAREST).astype(bool))
                occl.append(cv2.resize(o.astype(np.uint8), (int(W * SCALE), int(H * SCALE)),
                                       interpolation=cv2.INTER_NEAREST).astype(bool))
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, frames=np.array(frames, int), masks=np.array(masks, dtype=object),
                                occl=np.array(occl, dtype=object), occ_prompts=np.array(occ_prompts), prompt=prompt)
        if len(frames) < a.min_seen:
            print(f"[structure] {name}: '{prompt}' in {len(frames)} frames, not carved", flush=True)
            continue
        kept, seen, axes = carve(z, frames, masks, occl, lo, hi, a.voxel, a.slack, a.min_seen, a.agree, dev)
        lab, n = ndimage.label(kept, np.ones((3, 3, 3)))
        sizes = ndimage.sum(kept, lab, range(1, n + 1))
        zmin = ndimage.minimum(np.broadcast_to(np.arange(kept.shape[2]), kept.shape), lab, range(1, n + 1))
        keep_ids = [j + 1 for j in range(n) if sizes[j] >= a.min_cells and axes[2][int(zmin[j])] <= top + 0.04]
        body = np.isin(lab, keep_ids)
        if not body.any():
            print(f"[structure] {name}: nothing kept ({n} pieces, none of {a.min_cells} cells on the support)",
                  flush=True)
            continue
        # where it meets cells no frame saw: it goes on beyond the capture
        unseen_next = ndimage.binary_dilation(body) & ~seen
        top_seen = float(axes[2][np.nonzero(body.any((0, 1)))[0].max()]) - top
        # the mesh in the support's frame (origin: the region's middle on its top)
        org = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, top])
        pad = np.pad(body, 1).astype(float)
        pad = ndimage.gaussian_filter(pad, 0.6)
        V, F, _, _ = measure.marching_cubes(pad, 0.5, spacing=(a.voxel,) * 3)
        V = V - a.voxel + np.array([axes[0][0], axes[1][0], axes[2][0]]) - org
        mesh = trimesh.Trimesh(V, F[:, ::-1], process=True)
        trimesh.smoothing.filter_taubin(mesh, iterations=8)
        d = s / "objects" / "structures" / name
        d.mkdir(parents=True, exist_ok=True)
        mesh.export(d / "mesh.glb")
        # the check and the colour, frame by frame
        cells = np.stack(np.meshgrid(*axes, indexing="ij"), -1)[body]
        prec, rec, px, tiles = [], [], [], []
        for k, (i, m, o) in enumerate(zip(frames, masks, occl)):
            r = splat_cells(cells, a.voxel, z["K"][i], z["R_world_cam"][i], z["centre"][i], m.shape)
            r &= ~(o & ~m)  # behind an occluder: not judged
            if r.sum() < 50 or m.sum() < 50:
                continue
            prec.append((r & m).sum() / r.sum())
            rec.append((r & m).sum() / m.sum())
            im = np.asarray(Image.open(img_dir / names[i]).convert("RGB"))
            im = cv2.resize(im, m.shape[::-1], interpolation=cv2.INTER_AREA)
            both = cv2.erode((r & m).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            px.append(im[both])
            if k % max(1, len(frames) // 8) == 0 and len(tiles) < 8:
                vis = (0.5 * im).astype(np.uint8)
                vis[m & ~r] = (220, 0, 0)
                vis[r & ~m] = (0, 200, 0)
                vis[r & m] = (235, 235, 235)
                cv2.putText(vis, f"{names[i]} p {prec[-1]:.2f} r {rec[-1]:.2f}", (6, 22), 0, 0.6, (255, 255, 0), 2)
                tiles.append(cv2.resize(vis, (640, 360)))
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(qa / f"{name}.jpg"), np.vstack([np.hstack(tiles[j:j + 4]) for j in range(0, len(tiles), 4)])[:, :, ::-1],
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        alb, _ = frame_color.albedo(np.concatenate(px)) if px else (np.array([0.5, 0.5, 0.5]), None)
        ext = cells.max(0) - cells.min(0) + a.voxel
        res[name] = {"prompt": prompt, "on": sup, "frames": len(frames), "cells": int(body.sum()),
                     "extent_m": ext.round(3).tolist(), "top_seen_over_support_m": round(top_seen, 3),
                     "meets_unseen_cells": int(unseen_next.sum()),
                     "precision_median": round(float(np.median(prec)), 3) if prec else None,
                     "recall_median": round(float(np.median(rec)), 3) if rec else None,
                     "color": alb.round(3).tolist(), "origin": org.round(4).tolist(),
                     "pieces": len(keep_ids)}
        print(f"[structure] {name}: {len(keep_ids)} piece(s), {int(body.sum())} cells of {100 * a.voxel:.0f} cm, extent "
              f"{np.round(100 * ext, 1).tolist()} cm, up to {100 * top_seen:.0f} cm over the top; precision "
              f"{res[name]['precision_median']}, recall {res[name]['recall_median']} over {len(prec)} frames; "
              f"{int(unseen_next.sum())} cells where it meets what no frame saw", flush=True)
    out_json.write_text(json.dumps(res, indent=1) + "\n")
    if a.apply and res:
        cfg = s.config_dir / "scene_graph.json"
        g = json.loads(cfg.read_text())
        for name, r in res.items():
            rel = f"sessions/{s.dir.name}/objects/structures/{name}/mesh.glb"  # the repository's: no machine's path
            g.setdefault("set_pieces", {})[name] = {"kit": "mesh", "path": rel, "color": r["color"],
                                                    "center": r["origin"][:2], "on": r["on"]}
            g["set_pieces"][name]["_structure_hull"] = (
                f"recon.structure_hull: carved from SAM 3's '{r['prompt']}' in {r['frames']} frames; precision "
                f"{r['precision_median']}, recall {r['recall_median']}; seen up to {r['top_seen_over_support_m']} m "
                "over the support")
        cfg.write_text(json.dumps(g, indent=1, ensure_ascii=False) + "\n")
        print(f"applied -> {cfg}", flush=True)
    print(f"done: {len(res)} structure(s) -> {out_json}, {qa}", flush=True)


if __name__ == "__main__":
    main()
