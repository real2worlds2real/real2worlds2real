"""S4a'' furniture the completion invented: pieces in the completed views placed as synthetic furniture in the twin.

    python -m calibrated_scene_reconstruction.recon.pano_furniture sessions/<name> [--tag sdxl] [--apply]

The panorama completion (recon.panorama) paints whole furniture into what the photo did not see (a dining table
behind the camera, stools along a counter); the bake then only saw it as paint on the wall and floor. Here
  1. SAM 3 segments a generic furniture vocabulary (FURNITURE: what stands on the floor and is in a robot's way; a
     wall shelf reads well enough as paint) plus the agent's completion.json "furniture" list in the ring views,
     outside the photo's own pixels; per view the duplicate answers of one piece are merged,
  2. every mask is lifted to world points with the view's metric depth (the joint depth scales of recon.panorama, so
     a piece stands where the completed walls are), and fitted with a floor-standing, room-axis box: robust extents,
     top at the 97th height percentile, and the side facing away from the camera (never seen) pushed to a per-class
     minimum depth; boxes clipped to the room,
  3. pieces that are the photo's own furniture continued by the completion are dropped: boxes holding the photo's
     reconstruction or overlapping a scene-graph furniture box; so are pieces within TASK_CLEAR of the task support
     (robot bases and arm motions are planned around what the photo shows), pieces cut by the bottom of the frame
     and small frame-cut slivers; the same piece seen by several views is kept once (the least frame-cut, largest
     view of it),
  4. each piece gets a model: the session's own reconstruction of that class (recon.furniture: a completed bentwood
     chair is the photo's chair), else the best converted library model of the class (DINOv2 on the view crop: the
     views carry ~100-300 px per piece, too little for SAM 3D Objects), else a proxy coloured like the crop (a table
     as top + legs, anything else a box),
  5. the placed pieces' masks join the view's removed-object mask and the view plate is inpainted again (LaMa), so
     the bake (bake.PanoSource) paints the wall behind a piece instead of the piece (what is not placed stays paint).
Pieces are marked synthetic; the simulator gives them box colliders only (sim.scene).

Writes scene/pano[/<tag>]/{furniture.json, furniture_masks.npz (--reuse), qa_furniture.jpg}, updates
views/<v>_{mask,plate}.png; --apply writes
"synthetic_furniture" into configs/<name>/scene_graph.json (recon.materials carries it into scene/twin.json).
"""
import argparse
import json
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw

from calibrated_scene_reconstruction.scene.graph import dumps, structure_role
from calibrated_scene_reconstruction.scene.session import Session
from calibrated_scene_reconstruction.scene.background.panorama import N, load_views, plates

RING = ["front", "left60", "right60", "left120", "right120", "back"]
# label -> (class, minimum depth in m): the far side of a piece is never seen from the panorama camera
FURNITURE = {"chair": ("chair", 0.45), "armchair": ("chair", 0.7), "stool": ("stool", 0.35),
             "bar stool": ("stool", 0.35), "bench": ("bench", 0.35), "table": ("table", 0.6), "desk": ("table", 0.6),
             "sofa": ("sofa", 0.8), "cabinet": ("cabinet", 0.4), "sideboard": ("cabinet", 0.4),
             "bookshelf": ("cabinet", 0.3), "counter": ("counter", 0.6)}  # wall shelves stay paint
MIN_HEIGHT = {"chair": 0.6, "stool": 0.4, "bench": 0.35, "table": 0.45, "sofa": 0.5, "cabinet": 0.5, "counter": 0.7,
              "shelf": 0.1}  # lower: only its top was seen (frame-cut), its height is unknown
HANGS = {"shelf", "cabinet"}  # may hang on a wall
TASK_CLEAR = 0.7  # m around the task support kept as the photo shows it
MIN_PX = 1500  # mask pixels (of 1024 x 1024) worth a piece


def furniture_vocab(cfg):
    extra = {p: (p.split()[-1], 0.4) for p in cfg.get("furniture", []) if p not in FURNITURE}
    return {**FURNITURE, **extra}


def segment(views, vocab, thr=0.45):
    """SAM 3 instances of the vocabulary per ring view (one model load), duplicates of one piece merged."""
    import torch
    from transformers import Sam3Model, Sam3Processor

    model = Sam3Model.from_pretrained("facebook/sam3").to("cuda").eval()
    proc = Sam3Processor.from_pretrained("facebook/sam3")
    out = {}
    for name in RING:
        img = views[name][0]
        inst = []
        for label in vocab:
            inputs = proc(images=Image.fromarray(img), text=label, return_tensors="pt").to("cuda")
            with torch.no_grad():
                o = model(**inputs)
            r = proc.post_process_instance_segmentation(o, threshold=thr, mask_threshold=0.5,
                                                        target_sizes=inputs.get("original_sizes").tolist())[0]
            for m, sc in zip(r["masks"], r["scores"]):
                m = m.cpu().numpy().astype(bool)
                if 0.002 < m.mean() < 0.5:  # a "wall" answer covers the view
                    inst.append({"label": label, "score": float(sc), "mask": m})
        inst.sort(key=lambda i: -i["score"])
        keep = []
        for i in inst:  # one piece answering several labels (table / desk): the most confident label
            if all((i["mask"] & k["mask"]).sum() < 0.6 * min(i["mask"].sum(), k["mask"].sum()) for k in keep):
                keep.append(i)
        out[name] = keep
        print(f"[pano_furniture] {name}: {sorted(i['label'] for i in keep)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return out


class Room:
    """Room bounds, the scene graph's furniture boxes and the photo's own reconstruction, in world coordinates."""

    def __init__(self, s: Session):
        sg = s.scene_graph
        st = sg["structure"]
        self.floor = np.asarray(next(v for k, v in st.items() if structure_role(k, v) == "floor")["range"], float)
        self.ceil = float(next((v["at"] for k, v in st.items() if structure_role(k, v) == "ceiling"), 2.6))
        self.boxes = [np.asarray(f["box"], float) for f in sg.get("furniture", {}).values()]
        sup = sg.get("furniture", {}).get(sg.get("task", {}).get("support"))
        self.task = np.asarray(sup["box"], float)[:, :2] + [[-TASK_CLEAR] * 2, [TASK_CLEAR] * 2] if sup else None
        geo = s.geometry()
        R_wc, o, T = s.world_frame()
        P = (geo["points"][geo["mask"]] - o) @ np.asarray(R_wc).T
        self.photo = P[np.random.default_rng(0).choice(len(P), min(len(P), 200000), replace=False)]
        self.cam = np.asarray(T)[:3, 3]

    def clip(self, box, margin=0.02):
        lo, hi = box
        lo[:2] = np.maximum(lo[:2], self.floor[:, 0] + margin)
        hi[:2] = np.minimum(hi[:2], self.floor[:, 1] - margin)
        hi[2] = min(hi[2], self.ceil - 0.05)
        return np.array([lo, hi])

    def near_task(self, box):
        """Within TASK_CLEAR of the task support: robot bases and arm sweeps are planned around what the photo shows."""
        return self.task is not None and bool(np.all(box[0, :2] < self.task[1]) and np.all(box[1, :2] > self.task[0]))

    def is_photo_furniture(self, box, frac=0.3):
        """The photo reconstructed something solid in it (the completion continued a real piece), or it overlaps a
        scene-graph furniture box."""
        lo, hi = box + np.array([[0.05, 0.05, 0.05], [-0.05, -0.05, -0.05]])
        inside = np.all((self.photo > lo) & (self.photo < hi), 1).sum()
        if inside > 300:
            return True
        area = np.prod(box[1, :2] - box[0, :2])
        for b in self.boxes:
            ov = np.prod(np.clip(np.minimum(box[1, :2], b[1, :2]) - np.maximum(box[0, :2], b[0, :2]), 0, None))
            if ov > frac * area and box[0, 2] < b[1, 2]:
                return True
        return False


def rays(R, K, v, u):
    d = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0], (v + 0.5 - K[1, 2]) / K[1, 1], np.ones(len(u))], -1)
    return (d / np.linalg.norm(d, axis=-1, keepdims=True)) @ R.T


def lift(depth, ok, R, K, cam, mask, taken, anchored):
    """World points of a mask's interior (eroded: depth at a silhouette mixes in the wall behind). In a view the photo
    does not anchor, rescaled about the camera so that the floor the view shows around the piece's foot lies at z = 0
    (with neither photo pixels nor much floor, a view's joint depth scale can be far off; a piece stands on its own
    floor)."""
    m = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ok
    v, u = np.nonzero(m)
    if len(v) < 300:
        return np.zeros((0, 3)), 1.0
    z = depth[v, u]
    near = z < np.median(z) + 0.6  # the piece is the front-most surface under its mask
    P = cam + rays(R, K, v[near], u[near]) * z[near, None]
    ring = cv2.dilate(mask.astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool) & ~taken & ok
    ring[: int(np.percentile(v, 75))] = False  # beside / below the foot of the piece
    fv, fu = np.nonzero(ring)
    k = 1.0
    if not anchored and len(fv) > 200:
        fz = (rays(R, K, fv, fu) * depth[fv, fu, None])[:, 2]  # camera-relative height
        floor = np.percentile(fz, 25)  # the floor: the lowest of what surrounds the foot
        if floor < -0.3:
            k = float(np.clip(-cam[2] / floor, 0.3, 3.0))
    return cam + (P - cam) * k, k


def fit_box(P, cam, min_depth, room, hangs=False):
    """Room-axis box: robust extents, standing on the floor (legs hidden behind a table do not lift it; hangs=True: a
    cabinet seen only above 0.5 m hangs where it is seen; "always": a shelf), the hidden far side pushed to the
    class's minimum depth."""
    lo = np.percentile(P, 3, axis=0)
    hi = np.percentile(P, 97, axis=0)
    if lo[2] < {False: 10.0, True: 0.5}.get(hangs, -1.0):  # a cabinet seen only high up hangs, a shelf always
        lo[2] = 0.0
    c = (lo + hi) / 2
    ax = int(np.argmax(np.abs(c[:2] - cam[:2])))  # the horizontal axis the camera looks along
    if hi[ax] - lo[ax] < min_depth:
        if c[ax] > cam[ax]:
            hi[ax] = lo[ax] + min_depth
        else:
            lo[ax] = hi[ax] - min_depth
    o = 1 - ax  # seen edge-on along the other axis: widen about the centre
    if hi[o] - lo[o] < min_depth:
        lo[o], hi[o] = c[o] - min_depth / 2, c[o] + min_depth / 2
    return room.clip(np.array([lo, hi]))


def footprint_overlap(a, b):
    """Intersection of two boxes' footprints over the smaller footprint."""
    i = np.prod(np.clip(np.minimum(a[1, :2], b[1, :2]) - np.maximum(a[0, :2], b[0, :2]), 0, None))
    return i / max(min(np.prod(a[1, :2] - a[0, :2]), np.prod(b[1, :2] - b[0, :2])), 1e-9)


def pieces(s, views, deps, inst, vocab, room):
    """Candidate pieces of every view (rejected ones keep their reason), and the kept ones."""
    cands = []
    for name in RING:
        img, kv, R = views[name]
        k = deps["names"].index(name)
        for i in inst[name]:
            if i["label"] not in vocab:  # cached masks of a label no longer asked for
                continue
            m = i["mask"] & ~kv
            cls, min_depth = vocab[i["label"]]
            c = {"label": i["label"], "class": cls, "view": name, "score": round(i["score"], 3),
                 "pixels": int(m.sum()), "mask": m, "why": ""}
            cands.append(c)
            if m.sum() < MIN_PX or (i["mask"] & kv).sum() > 0.3 * i["mask"].sum():
                c["why"] = "photo pixels" if m.sum() >= MIN_PX else "small"
                continue
            taken = np.any([j["mask"] for j in inst[name]], 0)
            P, c["floor_scale"] = lift(deps["depth"][k], deps["ok"][k], R, deps["K"][k], deps["cam"], m, taken,
                                       anchored=kv.mean() > 0.02)
            if len(P) < 300:
                c["why"] = "no depth"
                continue
            ys, xs = np.nonzero(m)
            if m[N - 5:].sum() > 0.3 * 5 * (xs.max() - xs.min() + 1) and cls not in HANGS:
                c["why"] = "foot out of the view"  # a near piece cut by the frame bottom: height, place unknown
                continue
            box = fit_box(P, deps["cam"], min_depth, room, "always" if cls == "shelf" else cls in HANGS)
            size = box[1] - box[0]
            c["box"] = box
            if not (MIN_HEIGHT.get(cls, 0.25) < size[2] < 2.4 and size[:2].max() < 3.5 and size[:2].min() > 0.1):
                c["why"] = f"size {np.round(size, 2).tolist()}"
                continue
            if room.is_photo_furniture(box):
                c["why"] = "photo furniture"
                continue
            if room.near_task(box):
                c["why"] = "task area"
                continue
            c["frame_cut"] = bool(min(ys.min(), xs.min()) < 4 or max(ys.max(), xs.max()) > N - 5)
            if c["frame_cut"] and m.sum() < 4 * MIN_PX:
                c["why"] = "frame-cut sliver"
                continue
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            c["solid"] = float(i["mask"][(y0 + y1) // 2:y1, x0:x1].mean())  # lower half filled: a block, not legs
            c["color"] = np.median(img[cv2.erode(m.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)], 0)
            c["rank"] = i["score"] * np.sqrt(m.sum()) * (0.4 if c["frame_cut"] else 1.0)
    ok = sorted((c for c in cands if not c["why"]), key=lambda c: -c["rank"])
    keep = []
    def same(a, b):  # footprints overlapping; labelled apart: boxes alike (a stool at a counter is not the counter)
        if a["class"] == b["class"]:
            return footprint_overlap(a["box"], b["box"]) > 0.25
        A, B = a["box"], b["box"]
        inter = np.prod(np.clip(np.minimum(A[1], B[1]) - np.maximum(A[0], B[0]), 0, None))
        return inter > 0.4 * max(np.prod(A[1] - A[0]), np.prod(B[1] - B[0]))

    for c in ok:  # one piece seen from several views: its best view
        dup = next((k for k in keep if same(c, k)), None)
        if dup is None:
            keep.append(c)
        else:
            c["why"] = f"seen better in {dup['view']}"
    return keep, cands


def own_models(s: Session):
    """class -> the session's own furniture model: shared reconstruction (recon.furniture) or retrieved library."""
    out = {}
    for group, spec in s.scene_graph.get("movable_furniture", {}).items():
        d = s / "objects" / group
        cat = spec.get("category", group)
        if (d / "retrieved.json").exists():
            out.setdefault(cat, {"model": s.read_json(f"objects/{group}/retrieved.json")["model"]})
        elif (d / "furniture.json").exists() and s.read_json(f"objects/{group}/furniture.json").get("ok"):
            out.setdefault(cat, {"recon": group})
    return out


def choose_models(s, keep, views, cam):
    """Own reconstruction > converted library model (DINOv2 on the crop) > a coloured proxy."""
    from calibrated_scene_reconstruction.assets.library import Embedder, crop_on_white, models, rank

    own, emb = own_models(s), None
    for c in keep:
        box = c["box"]
        c["center"] = ((box[0, :2] + box[1, :2]) / 2).round(3).tolist()
        c["height"] = round(float(box[1, 2] - box[0, 2]), 3)
        to_cam = cam[:2] - np.asarray(c["center"])
        c["yaw"] = round(float(np.arctan2(to_cam[0], -to_cam[1])), 3)  # library fronts face -y after the up fix
        if c["class"] in own:
            c["source"] = own[c["class"]]
            continue
        lib = set(models(c["class"]))
        if lib:
            emb = emb or Embedder()
            ranked = [(n, sim) for n, sim in rank(crop_on_white(views[c["view"]][0], c["mask"]), c["class"], k=400,
                                                   embedder=emb) if n in lib]
            if ranked:
                c["source"] = {"model": ranked[0][0], "sim": round(ranked[0][1], 3)}
                continue
        c["source"] = {"proxy": "table" if c["class"] in ("table", "chair", "stool", "bench") and c["solid"] < 0.6
                       else "box",
                       "color": ((c["color"] / 255.0) ** 2.2 * 0.85).round(3).tolist()}  # linear albedo


def replate(s, out, views, cands):
    """The masks of the placed pieces (from every view that sees them, outside the photo) join the view's
    removed-object mask of recon.panorama
    (kept as <v>_mask_pano.png, so a re-run starts from it); plates inpainted again."""
    masks = {}
    for name in views:
        f, base = out / "views" / f"{name}_mask.png", out / "views" / f"{name}_mask_pano.png"
        if f.exists() and not base.exists():
            shutil.copy(f, base)
        m = np.asarray(Image.open(base).convert("L")) > 127 if base.exists() else np.zeros((N, N), bool)
        for c in cands:  # placed pieces in every view that sees them; what was not placed stays painted
            if c["view"] == name and (c.get("synthetic") or c["why"].startswith("seen better")):
                m |= c["mask"]
        masks[name] = (m, [])
    plates(views, masks, out)


def qa(out, views, keep, cands, room):
    tiles = []
    for name in RING:
        im = views[name][0].copy()
        for c in cands:
            if c["view"] == name:
                tint = np.array([0, 200, 0] if c.get("synthetic") else [200, 0, 0] if c.get("box") is not None else
                                [0, 0, 200])
                im[c["mask"]] = (0.5 * im[c["mask"]] + 0.5 * tint).astype(np.uint8)
        t = Image.fromarray(im).resize((384, 384))
        d = ImageDraw.Draw(t)
        d.text((4, 2), name, fill=(255, 255, 0))
        for c in cands:
            if c["view"] == name:
                ys, xs = np.nonzero(c["mask"])
                if len(xs):
                    d.text((xs.mean() * 384 / N, ys.mean() * 384 / N), f"{c['label']} {c['why']}"[:40],
                           fill=(255, 255, 255))
        tiles.append(t)
    # top view: room, photo points, kept boxes
    f = room.floor
    sc = 380 / max(f[:, 1] - f[:, 0])
    top = Image.new("RGB", (384, 384), (30, 30, 30))
    d = ImageDraw.Draw(top)
    xy = lambda p: (2 + (p[0] - f[0, 0]) * sc, 382 - (p[1] - f[1, 0]) * sc)
    for p in room.photo[::40]:
        d.point(xy(p), fill=(120, 120, 120))
    for b in room.boxes:
        d.rectangle([*xy((b[0, 0], b[1, 1])), *xy((b[1, 0], b[0, 1]))], outline=(80, 80, 255))
    for c in keep:
        b = c["box"]
        d.rectangle([*xy((b[0, 0], b[1, 1])), *xy((b[1, 0], b[0, 1]))], outline=(0, 255, 0))
        d.text(xy((b[0, 0], b[1, 1])), c["name"], fill=(0, 255, 0))
    d.ellipse([*(np.array(xy(room.cam)) - 4), *(np.array(xy(room.cam)) + 4)], fill=(255, 255, 0))
    tiles.append(top)
    S = Image.new("RGB", (4 * 384, 2 * 384))
    for k, t in enumerate(tiles):
        S.paste(t, ((k % 4) * 384, (k // 4) * 384))
    S.save(out / "qa_furniture.jpg", quality=88)


def apply(s: Session, keep):
    path = s.config_dir / "scene_graph.json"
    sg = json.loads(path.read_text())
    keys = ("label", "class", "view", "box", "center", "height", "yaw", "source", "synthetic")
    sg["synthetic_furniture"] = {c["name"]: {k: c[k] for k in keys} for c in keep}
    path.write_text(dumps(sg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--tag", default="", help="scene/pano/<tag> (the twin's completion)")
    ap.add_argument("--apply", action="store_true", help="write synthetic_furniture into scene_graph.json")
    ap.add_argument("--reuse", action="store_true", help="keep the SAM 3 masks of an earlier run")
    args = ap.parse_args()
    s = Session(args.session)
    out = s / "scene" / "pano" / args.tag if args.tag else s / "scene" / "pano"
    cfg = s.config("completion.json", {})
    vocab = furniture_vocab(cfg)
    views = load_views(s, out)
    z = np.load(out / "views.npz")
    deps = {"names": [str(n) for n in z["names"]], "depth": z["depth"], "ok": z["ok"], "K": z["K"], "cam": z["cam"]}
    room = Room(s)
    cache = out / "furniture_masks.npz"
    if args.reuse and cache.exists():
        z = np.load(cache)
        inst = {n: [{"label": str(lab), "score": float(sc), "mask": np.unpackbits(m)[:N * N].reshape(N, N).astype(bool)}
                    for lab, sc, m in zip(z[f"{n}_labels"], z[f"{n}_scores"], z[f"{n}_masks"])] for n in RING}
    else:
        inst = segment(views, vocab)
        np.savez_compressed(cache, **{f"{n}_{k}": np.array([i[f] for i in inst[n]]) if inst[n] else np.zeros(0)
                                      for n in RING for k, f in (("labels", "label"), ("scores", "score"))},
                            **{f"{n}_masks": np.array([np.packbits(i["mask"]) for i in inst[n]], np.uint8).reshape(
                                len(inst[n]), N * N // 8) for n in RING})
    keep, cands = pieces(s, views, deps, inst, vocab, room)
    for k, c in enumerate(keep):
        c["name"] = f"synthetic_{c['class']}_{k}"
        c["synthetic"] = True
    choose_models(s, keep, views, room.cam)
    replate(s, out, views, cands)
    qa(out, views, keep, cands, room)
    for c in keep:
        c["box"] = c["box"].round(3).tolist()
    rec = [{k: v for k, v in c.items() if k not in ("mask", "color", "rank")} for c in keep]
    (out / "furniture.json").write_text(json.dumps({"pieces": rec, "candidates": len(cands)}, indent=1))
    for c in cands:
        if c["why"]:
            print(f"[pano_furniture]   {c['view']} {c['label']} ({c['pixels']} px, "
                  f"floor x{c.get('floor_scale', 1):.2f}): {c['why']}")
    for c in rec:
        print(f"[pano_furniture] {c['name']}: {c['label']} from {c['view']} (score {c['score']}, {c['pixels']} px"
              f"{', frame-cut' if c['frame_cut'] else ''}) box {c['box']} -> {c['source']}")
    if args.apply:
        apply(s, rec)
        print("[pano_furniture] scene_graph.json updated")


if __name__ == "__main__":
    main()
