"""The task's support top as the walk frames see it: flat marks on it -- tape, a label, a mat -- carved from their
silhouettes, and its own colour.

    envs/recon/bin/python -m calibrated_scene_reconstruction.recon.support_marks sessions/<name> [--mark "tape=blue tape@-0.25,1.58"]
        [--look] [--reach 0.15] [--frames 40] [--apply]
    envs/recon/bin/python -m calibrated_scene_reconstruction.recon.support_marks sessions/<name> --from-config --apply   (the task_zone stage)
    (after recon.posed_frames select: parsing/posed_frames/cameras.npz)

A mark a few tenths of a millimetre thick is in no reconstruction: SAM 3D and the furniture boxes leave the support's
top bare, and the support's texture is a library wood. Per mark
(`name=prompt@x,y`: what SAM 3 is asked for, about where it is): up to `--frames` frames that see the place from
every side (recon.object_fit.pick_frames), SAM 3's instances of the prompt within 60 px of it in a crop round it,
together; a 2 mm grid on the support's top within `--reach` of the place keeps a cell that lies in the silhouette in
`--agree` of the frames that see it; each piece over 0.5 cm2 is outlined (1.5 mm). Its colour: the frames' median
inside the silhouettes, away from their edges (recon.frame_colour: the frames' clip read as a white of 0.8).

`--look`: the support top's own colour -- a 2 cm grid over its top, 8 cm in from its edges and 12 cm clear of every
object and mark, projected into every frame that sees it, the median over all (an arm or a cable over a few cells
leaves it); recon.frame_colour.match_texture picks the library texture of the look's kind (a wood for a wood) whose
unclipped channels' ratios are nearest and tints it to them. Writes splat/support_marks.json, qa/support_marks/
<mark>.jpg (crops: the silhouette red, the carved outline green); `--apply` writes the scene graph's
set_pieces.<mark> (kit "decal": a 0.4 mm sheet on the support's top, no collider) with `_support_marks`, and with
`--look` the support's look.top {"pbr", "tint"} with `_frame_colour`. `--from-config`: the agent's requests in the
scene graph, `support_marks` {"marks": {name: {"prompt", "near": [x, y]}}, "look": true}."""
import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from calibrated_scene_reconstruction.core.paths import POLYHAVEN_DIR
from calibrated_scene_reconstruction.scene.session import Session
from calibrated_scene_reconstruction.scene.objects import frame_color
from calibrated_scene_reconstruction.scene.objects.object_fit import frames_support_z, pick_frames
from calibrated_scene_reconstruction.scene.objects.object_hull import CropSam3, project

CELL, CROP, TILE = 0.002, 640, 300  # (a 320 px crop held a tape L 5-8 cm beside a cup out at 0.5 m)
WOODS = ("veneer", "wood", "plywood", "bamboo", "oak", "walnut", "cherry", "ash")


def support_look(s, sg, z, img_dir, clear):
    """(albedo, clipped, frames) of the support top's clear middle over every frame that sees it."""
    (x0, y0, _), (x1, y1, top) = sg["furniture"][sg["task"]["support"]]["box"]
    top = frames_support_z(s) or top
    gx, gy = np.arange(x0 + 0.08, x1 - 0.08, 0.02), np.arange(y0 + 0.08, y1 - 0.08, 0.02)
    G = np.stack(np.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)
    for c in clear:
        G = G[np.linalg.norm(G - np.asarray(c)[None, :2], axis=1) > 0.12]
    G3 = np.c_[G, np.full(len(G), top)]
    W, H = (int(v) for v in z["size"][0])
    px, n = [], 0
    for i in range(0, len(z["names"]), 3):
        u, v, d = project(G3, z["K"][i], z["R_world_cam"][i], z["centre"][i])
        ok = (d > 0.3) & (u >= 2) & (u < W - 2) & (v >= 2) & (v < H - 2)
        if ok.sum() < 0.3 * len(G):
            continue
        img = np.asarray(Image.open(img_dir / str(z["names"][i])).convert("RGB"))
        px.append(img[v[ok].astype(int), u[ok].astype(int)])
        n += 1
    alb, clipped = frame_color.albedo(np.concatenate(px))
    return alb, clipped, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--mark", action="append", default=[], help="name=prompt@x,y")
    ap.add_argument("--look", action="store_true", help="the support top's own colour")
    ap.add_argument("--reach", type=float, default=0.15)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--agree", type=float, default=0.6)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--from-config", action="store_true",
                    help="the scene graph's support_marks {marks: {name: {prompt, near}}, look}")
    a = ap.parse_args()
    s = Session(a.session)
    sg = s.scene_graph
    if a.from_config:  # the agent's requests from the review (task_zone stage)
        req = sg.get("support_marks") or {}
        a.mark += [f"{n}={m['prompt']}@{m['near'][0]},{m['near'][1]}" for n, m in (req.get("marks") or {}).items()
                   if not any(x.split("=", 1)[0] == n for x in a.mark)]
        a.look = a.look or bool(req.get("look"))
    if not a.mark and not a.look:
        print("done: nothing asked (no --mark, no --look, no support_marks in the scene graph)")
        return
    z = np.load(s / "parsing" / "posed_frames" / "cameras.npz")
    img_dir = Path(str(z["source"])) / "splat" / "data" / "images"
    names = [str(n) for n in z["names"]]
    W, H = (int(v) for v in z["size"][0])
    top_twin = float(sg["furniture"][sg["task"]["support"]]["box"][1][2])
    top = frames_support_z(s) or top_twin  # measured where the frames see the top; the decal goes on the twin's
    seg = CropSam3(CROP) if a.mark else None
    out_qa = s / "qa" / "support_marks"
    out_qa.mkdir(parents=True, exist_ok=True)
    res_path = s / "splat" / "support_marks.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    for spec in a.mark:
        name, rest = spec.split("=", 1)
        prompt, at = rest.rsplit("@", 1)
        c = np.array([*(float(v) for v in at.split(",")), top])
        gx = np.arange(c[0] - a.reach, c[0] + a.reach, CELL)
        gy = np.arange(c[1] - a.reach, c[1] + a.reach, CELL)
        G = np.stack(np.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)
        G3 = np.c_[G, np.full(len(G), top + 0.0003)]
        n_seen, n_in = np.zeros(len(G), np.int32), np.zeros(len(G), np.int32)
        crops, cols = [], []
        for i in pick_frames(z, c, a.frames):
            img = np.asarray(Image.open(img_dir / names[i]).convert("RGB"))
            u, v, dc = project(c[None], z["K"][i], z["R_world_cam"][i], z["centre"][i])
            # the instances within the reach round the place, in this frame's pixels
            reach_px = float(z["K"][i][0, 0]) * a.reach / max(float(dc[0]), 0.1)
            m = seg(img, prompt, u[0], v[0], union=max(60, int(0.7 * reach_px)))
            if m is None:
                continue
            gu, gv, d = project(G3, z["K"][i], z["R_world_cam"][i], z["centre"][i])
            ok = (d > 0.1) & (gu >= 0) & (gu < W - 1) & (gv >= 0) & (gv < H - 1)
            n_seen[ok] += 1
            n_in[np.flatnonzero(ok)[m[gv[ok].astype(int), gu[ok].astype(int)]]] += 1
            core = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            cols.append(img[core if core.sum() >= 20 else m])  # its middle: the edges blur into the support
            crops.append((i, img, m, u[0], v[0]))
        keep = (n_seen >= 6) & (n_in >= a.agree * np.maximum(n_seen, 1))
        occ = keep.reshape(len(gx), len(gy)).T.astype(np.uint8)  # rows y, columns x
        cnt, _ = cv2.findContours(occ, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        outlines = []
        for k in cnt:
            if cv2.contourArea(k) * CELL * CELL < 0.5e-4:
                continue
            k = cv2.approxPolyDP(k, 0.0015 / CELL, True)[:, 0, :]
            outlines.append(np.c_[gx[0] + k[:, 0] * CELL, gy[0] + k[:, 1] * CELL])
        if not outlines:
            print(f"[support_marks] {name}: nothing carved ({len(crops)} silhouettes)", flush=True)
            continue
        allP = np.concatenate(outlines)
        mid = (allP.min(0) + allP.max(0)) / 2
        lin, clipped = frame_color.albedo(np.concatenate(cols))
        area = sum(cv2.contourArea(o.astype(np.float32)) for o in outlines)
        res[name] = {"prompt": prompt, "center": mid.round(4).tolist(), "z": round(top_twin, 4),
                     "outlines": [(o - mid).round(4).tolist() for o in outlines], "color": lin.round(3).tolist(),
                     "clipped": clipped.tolist(), "area_cm2": round(area * 1e4, 1), "frames": len(crops)}
        print(f"[support_marks] {name}: {len(outlines)} piece(s), {area * 1e4:.1f} cm2 over "
              f"{np.ptp(allP[:, 0]) * 100:.1f} x {np.ptp(allP[:, 1]) * 100:.1f} cm at ({mid[0]:.3f}, {mid[1]:.3f}); "
              f"{len(crops)} silhouettes; colour (linear) {lin.round(3).tolist()}"
              + (f", clipped {clipped.tolist()}" if clipped.any() else ""), flush=True)
        tiles = []
        for i, img, m, uc, vc in crops[:8]:
            h = 70
            x0, y0 = int(np.clip(uc - h, 0, W - 2 * h)), int(np.clip(vc - h, 0, H - 2 * h))
            crop = cv2.resize(img[y0:y0 + 2 * h, x0:x0 + 2 * h], (TILE, TILE), interpolation=cv2.INTER_CUBIC)
            f = TILE / (2 * h)
            mm = cv2.resize(m[y0:y0 + 2 * h, x0:x0 + 2 * h].astype(np.uint8), (TILE, TILE),
                            interpolation=cv2.INTER_NEAREST)
            cm, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(crop, cm, -1, (255, 0, 0), 1)
            for o in outlines:
                ou, ov, _ = project(np.c_[o, np.full(len(o), top)], z["K"][i], z["R_world_cam"][i], z["centre"][i])
                pts = np.round(np.c_[(ou - x0) * f, (ov - y0) * f]).astype(np.int32)
                cv2.polylines(crop, [pts], True, (0, 255, 0), 1)
            cv2.putText(crop, names[i], (4, 14), 0, 0.42, (255, 255, 0), 1)
            tiles.append(crop)
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[j:j + 4]) for j in range(0, len(tiles), 4)])
        cv2.imwrite(str(out_qa / f"{name}.jpg"), sheet[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
    look = None
    if a.look:
        clear = [(o.get("fix") or {}).get("xy") for o in sg.get("dynamic_objects", {}).values()]
        clear = [c for c in clear if c] + [r["center"] for r in res.values() if "center" in r]
        alb, clipped, n = support_look(s, sg, z, img_dir, clear)
        sup = sg["furniture"][sg["task"]["support"]]
        cur = ((sup.get("look") or {}).get("top") or {}).get("pbr", "")
        texs = sorted(p.name for p in (POLYHAVEN_DIR / "textures").iterdir() if (p / "diff.jpg").exists())
        # a wood for a wood: the current look's kind or the support's class (a desk's top is wood; with no look yet
        # the match took a plaster, and that plaster then stood as the current look)
        if any(w in cur for w in WOODS) or re.search(r"table|desk|bench|workbench", sup.get("_label", "")):
            texs = [t for t in texs if any(w in t for w in WOODS)]
        pick = frame_color.match_texture(alb, clipped, texs)
        if pick:
            look = {"pbr": pick[0], "tint": pick[1]}
            res["_look"] = {"albedo": alb.round(3).tolist(), "clipped": clipped.tolist(), "frames": n, **look,
                            "mean_after_tint": pick[2].round(3).tolist()}
            print(f"[support_marks] support top: albedo {alb.round(3).tolist()} over {n} frames (clipped "
                  f"{clipped.tolist()}) -> {pick[0]} tinted {pick[1][0]} (mean {pick[2].round(3).tolist()}; "
                  f"was {cur or 'none'})", flush=True)
    res_path.parent.mkdir(parents=True, exist_ok=True)
    res_path.write_text(json.dumps(res, indent=1) + "\n")
    if a.apply:
        cfg = s.config_dir / "scene_graph.json"
        g = json.loads(cfg.read_text())
        for spec in a.mark:
            name = spec.split("=", 1)[0]
            if name not in res:
                continue
            r = res[name]
            g.setdefault("set_pieces", {})[name] = {
                "kit": "decal", "center": r["center"], "z": r["z"], "outlines": r["outlines"], "color": r["color"],
                "_support_marks": f"recon.support_marks: '{r['prompt']}' carved on the support's top from "
                                  f"{r['frames']} frames' silhouettes, {r['area_cm2']} cm2"}
        if look:
            f = g["furniture"][g["task"]["support"]]
            f.setdefault("look", {})["top"] = {**look, "_frame_colour": (
                f"recon.support_marks --look: the top's albedo {res['_look']['albedo']} over {res['_look']['frames']} "
                f"frames, channels clipped {res['_look']['clipped']} (recon.frame_colour: the clip a white of "
                f"{frame_color.WHITE})")}
        cfg.write_text(json.dumps(g, indent=1, ensure_ascii=False) + "\n")
        print(f"applied -> {cfg}", flush=True)
    print(f"done: {len([k for k in res if not k.startswith('_')])} marks -> {res_path}, {out_qa}", flush=True)


if __name__ == "__main__":
    main()
