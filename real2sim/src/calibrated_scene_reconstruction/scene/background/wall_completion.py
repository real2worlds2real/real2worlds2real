"""S6b' wall completion: furniture painted against each free wall of the room, seen straight on, for recon.room_objects.

    python -m calibrated_scene_reconstruction.recon.wall_completion sessions/<name> [--tags twin,cousin_00] [--model sdxl] [--candidates 4]

The panorama completion (recon.panorama) is painted around the photo camera: near the camera the work zone keeps
furniture out, the far walls come out small and cut by the frame, and the room is left nearly empty. Here every bay
of apps/scene/wall_views.py (a stretch of free wall rendered straight on from inside the room, with its depth) is painted
where furniture may stand: the wall up to 2.1 m and the strip of floor in front of it (the rest of the view, the
room's own structure, stays as rendered). The prompt is the room's type, one piece of furniture of the room's kind
per bay (completion.json "wall_items": the agent's list, cycled over the bays; "walls": {tag: {bay: prompt}} pins
one), the tag's style (the twin's: completion.json "style", a cousin's: its theme's prompt) and the negatives of
recon.panorama. Several candidates per bay; kept: the least text / window / featureless, with furniture in it
(SAM 3). MoGe-2 depth of the painted view, scaled to the rendered depth of the bare wall and floor around it (the
pose and the room are known: what the completion paints stands at a known place and size), never behind the wall.

The agent reviews qa/walls/<tag>.jpg with the completion review (plausible, realistic, standing on the floor
against the wall, nothing floating or glued on); recon.room_objects reconstructs what the bays show (SAM 3D).
Writes scene/walls/<tag>/<bay>_completed.png, <bay>_depth.npy, <bay>_paint.png (the painted region),
candidates.json.
"""
import argparse
import json

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from calibrated_scene_reconstruction.assets.styles import themes
from calibrated_scene_reconstruction.scene.session import Session
from calibrated_scene_reconstruction.scene.background.panorama import NEGATIVE, Painter

ITEMS = {  # room type -> furniture against a wall, one per bay (the agent's completion.json "wall_items" first)
    "lab": ["a long white desk with two computer monitors and a black office chair",
            "a tall grey metal storage cabinet next to a metal shelving unit with cardboard boxes",
            "a workbench with tools and a rolling tool cart",
            "a low filing cabinet with a printer on it and a waste bin",
            "a bookshelf with binders and a potted plant beside it",
            "a metal rack with plastic storage bins and a step ladder"],
    "office": ["a white desk with a computer monitor and an office chair",
               "a tall bookshelf with binders and a potted plant",
               "a low sideboard cabinet with a printer",
               "two filing cabinets and a coat stand",
               "a small meeting table with two chairs",
               "a storage cabinet and a water cooler"],
    "kitchen": ["a tall pantry cabinet and a refrigerator", "a sideboard with a coffee machine and plants",
                "a dining table with chairs", "an open shelf with jars over a low cabinet"],
}
WALL_TOP = 2.1  # m: what may be painted on the wall
MIN_COVER = 0.08  # share of the painted region furniture must cover for a candidate to count as furnished


def bay_files(s: Session, tag):
    return sorted(p for p in (s / "scene" / "walls" / tag).glob("*.npz"))


def world_points(depth, K, T):
    H, W = depth.shape
    uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    X = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu)], -1) * depth[..., None]
    return X @ T[:3, :3].T + T[:3, 3], X


def paint_region(depth, K, T, wall):
    """The pixels where furniture may stand: the wall of the bay up to WALL_TOP and the floor strip in front."""
    P, _ = world_points(np.nan_to_num(depth, nan=1e3, posinf=1e3), K, T)
    ax, o, at, sign = wall["axis"], 1 - wall["axis"], wall["at"], wall["normal"]
    a0, a1 = wall["along"]
    along = (P[..., o] > a0 - 0.05) & (P[..., o] < a1 + 0.05)
    on_wall = (np.abs(P[..., ax] - at) < 0.05) & (P[..., 2] < WALL_TOP)
    d = (P[..., ax] - at) * sign
    on_floor = (np.abs(P[..., 2]) < 0.03) & (d > -0.02) & (d < wall["depth"])
    m = (along & (on_wall | on_floor)).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)).astype(bool)


def style_of(s: Session, tag, cfg):
    if tag == "twin":
        return cfg.get("style", "")
    c = s.read_json(f"cousins/{tag}.json")
    theme = c["theme"]["theme"]
    return themes().get(theme, {}).get("prompt", f"{theme} style interior")


class Judge:
    """SAM 3 shares of text / window / furniture in the painted region, and its featureless share."""

    LABELS = ["desk", "table", "chair", "cabinet", "shelf", "bookshelf", "box", "workbench", "cart", "plant", "printer",
              "sofa", "refrigerator", "rack"]

    def __init__(self):
        from transformers import Sam3Model, Sam3Processor

        self.sam = Sam3Model.from_pretrained("facebook/sam3").to("cuda").eval()
        self.proc = Sam3Processor.from_pretrained("facebook/sam3")

    def seg(self, pil, prompts, shape):
        m = np.zeros(shape, bool)
        for p in prompts:
            inputs = self.proc(images=pil, text=p, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = self.sam(**inputs)
            r = self.proc.post_process_instance_segmentation(out, threshold=0.45, mask_threshold=0.5,
                                                             target_sizes=inputs.get("original_sizes").tolist())[0]
            for k in r["masks"]:
                k = k.cpu().numpy().astype(bool)
                if k.mean() < 0.6:
                    m |= k
        return m

    def __call__(self, img, region):
        pil = Image.fromarray(img)
        n = max(int(region.sum()), 1)
        text = float((self.seg(pil, ("text", "sign"), region.shape) & region).sum() / n)
        window = float((self.seg(pil, ("window",), region.shape) & region).sum() / n)
        furn = float((self.seg(pil, self.LABELS, region.shape) & region).sum() / n)
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        mu = cv2.boxFilter(g, -1, (21, 21))
        sd = np.sqrt(np.maximum(cv2.boxFilter(g * g, -1, (21, 21)) - mu * mu, 0))
        blank = float(((sd < 4.0) & region).sum() / n)
        score = 20 * text + 5 * window + 4 * max(0.0, blank - 0.5) + (3.0 if furn < MIN_COVER else 0.0) \
            - 0.5 * min(furn, 0.5)
        return score, {"text": round(text, 3), "window": round(window, 3), "furniture": round(furn, 3),
                       "blank": round(blank, 3), "score": round(score, 3)}


def metric_depth(moge, img, depth, region, K, T, wall):
    """MoGe-2 depth of the painted view in the rendered metres: one scale from the unpainted wall / floor pixels
    (rendered depth known), clipped to the wall plane (nothing stands behind it)."""
    x = torch.tensor(img / 255.0, dtype=torch.float32, device="cuda").permute(2, 0, 1)
    fov = float(np.degrees(2 * np.arctan(img.shape[1] / 2 / K[0, 0])))
    with torch.no_grad():
        o = moge.infer(x, fov_x=fov)
    z = o["depth"].cpu().numpy() if "depth" in o else o["points"].cpu().numpy()[..., 2]
    ok = np.isfinite(z) & np.isfinite(depth) & (z > 0)
    ring = cv2.dilate(region.astype(np.uint8), np.ones((61, 61), np.uint8)).astype(bool) & ~region & ok
    if ring.sum() < 500:
        ring = ~region & ok
    k = float(np.median(depth[ring] / z[ring]))
    zm = np.where(ok, z * k, depth)
    out = np.where(region, zm, depth)
    P, X = world_points(out, K, T)  # nothing behind the wall: pulled onto it along the ray
    ax, at, sign = wall["axis"], wall["at"], wall["normal"]
    behind = (P[..., ax] - at) * sign < 0
    if behind.any():
        cam = T[:3, 3]
        dirs = P - cam
        t = (at - cam[ax]) / np.where(np.abs(dirs[..., ax]) > 1e-6, dirs[..., ax], 1e-6)
        out = np.where(behind, out * np.clip(t, 0, 1), out)
    return out.astype(np.float32), k


def run_tag(s, tag, args, painter, judge, moge):
    cfg = s.config("completion.json", {})
    room = ((s.scene_graph.get("task") or {}).get("place") or {}).get("room") or "lab"
    items = cfg.get("wall_items") or ITEMS.get(room, ITEMS["lab"])
    pinned = cfg.get("walls", {}).get(tag, {})
    style = style_of(s, tag, cfg)
    neg = cfg.get("negative", NEGATIVE) + ", window, windows, doorway, corridor, mirror"
    files = bay_files(s, tag)
    if not files:
        print(f"[walls] {tag}: no bays (apps/scene/wall_views.py)")
        return
    seed0 = 5000 + (0 if tag == "twin" else 100 * (1 + int(tag.split("_")[-1])))
    info = {}
    for i, f in enumerate(files):
        z = np.load(f)
        wall = json.loads(str(z["wall"]))
        img, depth, K, T = z["rgb"], z["depth"].astype(np.float32), z["K"], z["T_world_cam"]
        region = paint_region(depth, K, T, wall)
        item = pinned.get(wall["name"]) or items[(i + seed0) % len(items)]
        prompt = (f"interior photo of {item} standing on the floor against the wall, {style}, "
                  "realistic interior photography, eye level")
        best = None
        for j in range(args.candidates):
            seed = seed0 + 97 * i + 7919 * j
            gen = painter(img, region, prompt, neg, seed)
            a = cv2.GaussianBlur(region.astype(np.float32), (0, 0), 2)[..., None]
            a[region] = 1.0
            out = (a * gen + (1 - a) * img).clip(0, 255).astype(np.uint8)
            score, sc = judge(out, region)
            sc["seed"] = seed
            info.setdefault(wall["name"], {"prompt": prompt, "candidates": []})["candidates"].append(sc)
            if best is None or score < best[0]:
                best = (score, out, sc)
        out = best[1]
        d, k = metric_depth(moge, out, depth, region, K, T, wall)
        info[wall["name"]].update(kept=best[2]["seed"], depth_scale=round(k, 3))
        base = f.with_suffix("")
        Image.fromarray(out).save(f"{base}_completed.png")
        np.save(f"{base}_depth.npy", d.astype(np.float16))
        Image.fromarray(region.astype(np.uint8) * 255).save(f"{base}_paint.png")
        print(f"[walls] {tag} {wall['name']}: {item} -> seed {best[2]['seed']} {best[2]}, depth x{k:.2f}", flush=True)
    (s / "scene" / "walls" / tag / "candidates.json").write_text(json.dumps(info, indent=1))
    sheet(s, tag)


def sheet(s, tag):
    files = bay_files(s, tag)
    T_ = 384
    S = Image.new("RGB", (4 * T_, ((len(files) + 3) // 4) * (T_ + 18)), (18, 18, 18))
    d = ImageDraw.Draw(S)
    info = json.loads((s / "scene" / "walls" / tag / "candidates.json").read_text()) \
        if (s / "scene" / "walls" / tag / "candidates.json").exists() else {}
    for k, f in enumerate(files):
        p = f.with_name(f.stem + "_completed.png")
        if not p.exists():
            continue
        im = np.asarray(Image.open(p).convert("RGB")).copy()
        reg = np.asarray(Image.open(f.with_name(f.stem + "_paint.png"))) > 127
        edge = cv2.morphologyEx(reg.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)) > 0
        im[edge] = (255, 220, 0)
        x, y = (k % 4) * T_, (k // 4) * (T_ + 18)
        S.paste(Image.fromarray(im).resize((T_, T_)), (x, y + 18))
        sc = next((c for c in info.get(f.stem, {}).get("candidates", []) if c["seed"] == info[f.stem].get("kept")), {})
        d.text((x + 3, y + 3), f"{tag} {f.stem}  furn {sc.get('furniture', '-')} text {sc.get('text', '-')}",
               fill=(255, 255, 0))
    out = s / "qa" / "walls"
    out.mkdir(parents=True, exist_ok=True)
    S.save(out / f"{tag}.jpg", quality=88)


def tags(s: Session):
    return ["twin"] + sorted(p.stem for p in (s / "cousins").glob("cousin_*.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--tags", default="")
    ap.add_argument("--model", default="sdxl", choices=["sdxl", "flux"])
    ap.add_argument("--candidates", type=int, default=4)
    args = ap.parse_args()
    s = Session(args.session)
    from moge.model.v2 import MoGeModel

    painter, judge = Painter(args.model), Judge()
    moge = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to("cuda").eval()
    for tag in [t for t in args.tags.split(",") if t] or tags(s):
        run_tag(s, tag, args, painter, judge, moge)


if __name__ == "__main__":
    main()
