"""S4a' panorama completion: what the photo did not see, painted by a diffusion model around the photo camera.

    python -m calibrated_scene_reconstruction.recon.panorama sessions/<name> [--model sdxl|flux] [--seed 0] [--tag twin] [--style "..."]

A single photo textures only what it saw; the rest of the room used to be rule-sized walls with a tiled PBR fill.
Here the photo is pasted into an equirectangular canvas at its own camera (centre = photo camera, world axes), and the
rest is outpainted view by view (90 deg perspective views, each conditioned on everything painted so far):
front, +-60, +-120, 180 deg yaw, then up and down. The prompt is per photo (configs/<name>/completion.json, written
by the agent: the room's type and style, optionally per view what lies in that direction and what must not repeat
there, e.g. no second kitchen behind the camera, and a view's own "seed" to re-roll one direction; "tags": {tag:
{"views": {view: {seed, prompt, negative}}, "seed"}} re-rolls or re-prompts one tag's views (the twin's is "twin", a
cousin's its id) after the agent's completion review (recon.completion_review); a default is built from the scene
graph labels).
What the photo did not see ends up on opaque wall / ceiling planes, so every painted view but the floor's adds
windows, sky, corridors and open doorways to its negative prompt (painted glass bakes into a wall of blinds or of sky,
a space seen through an opening flattens onto the wall as a smear), and a view with over a quarter still to paint is
painted several times (seeds; completion.json "candidates", default 6), keeping the candidate with the least window /
sky (SAM 3) and the least depth spread over its vertical surfaces (MoGe-2; up: the flattest ceiling):
views/<v>_cand<j>.png, candidates.json.

Every generated view then gets
  - MoGe-2 depth at the known FoV; one metric scale per view, solved jointly (joint_scales): the camera height above
    the floor, the photo's own reconstruction inside the room, agreement where neighbouring views overlap,
  - SAM 3 masks of the perception vocabulary's objects / furniture, removed with LaMa -> an object-free view plate
    (structure textures come from it; generated furniture becomes its own instances, not paint on a wall).
From the depth, the walls the photo never saw (structure entries without ids) are moved to where the completed room
puts them (scene/pano/room.json; --apply writes them into configs/<name>/scene_graph.json, a backup is kept).

Writes scene/pano[/<tag>]/{pano.png, pano_known.png, views/<v>{,_plate,_mask,_cand<j>}.png, views/candidates.json,
views.npz, room.json}.
recon.layout fills unobserved texels from the view plates (bake.PanoSource).
"""
import argparse
import json
import shutil

import cv2
import numpy as np
import torch
from PIL import Image

from calibrated_scene_reconstruction.scene.graph import dumps, structure_role
from calibrated_scene_reconstruction.scene.session import Session

Z = np.array([0.0, 0.0, 1.0])
VIEWS = [("front", 0, 0), ("left60", 60, 0), ("right60", -60, 0), ("left120", 120, 0), ("right120", -120, 0),
         ("back", 180, 0), ("up", 0, 90), ("down", 0, -90)]
FOV = {"up": 120.0, "down": 120.0}  # 90 deg elsewhere; the poles need 120 to meet the ring (which reaches +-41 deg)
N = 1024  # view size; 90 deg FoV -> f = N / 2 (~11 px / deg, matched by the 4096 x 2048 canvas)
PW, PH = 4096, 2048
STRUCTURE = {"floor", "wall", "ceiling", "window", "door", "window blind", "counter"}
NEGATIVE = ("people, person, text, letters, words, signage, caption, label, watermark, logo, blurry, distorted, fisheye, "
            "cartoon, painting, illustration, 3d render, product render, isolated object, white background, floating "
            "objects, boxes on the wall, floor markings, painted lines on the floor, coloured stripes on the floor, "
            "low quality")
# the completion is reviewed by the agent per view (qa/completion/<tag>.jpg, scene graph _completion_review) and its
# objects are reconstructed in 3D (recon.room_objects); what a review rejected most often was text (gibberish signs),
# boxes or furniture glued to a wall, painted floor stripes and isolated product renders: all told to the model
# What the photo did not see becomes opaque wall / ceiling planes: a window painted there bakes into a wall of frosted
# glass or of sky (the blinds or the view outside have no depth), and a corridor or a second room seen through an
# opening flattens onto the wall as a smear. Every painted view but the floor's is told so, and a view with much to
# paint is painted several times and judged on both.
UNSEEN = 0.25  # share of a view still to paint above which its candidates are judged
UNSEEN_NEGATIVE = "window, windows, glass, window blinds, curtains, sky, outdoor view, corridor, hallway, open doorway"
CANDIDATES = 6  # completions tried per such view (completion.json "candidates" overrides)
WINDOW_WEIGHT = 5.0  # score per share of window / sky pixels (a 10 % window outweighs a depth spread of 2.4)
TEXT_WEIGHT = 20.0  # score per share of text / sign pixels (2 % of the view in gibberish lettering outweighs a 10 % window)
BLANK_WEIGHT = 4.0  # score per share of featureless painted pixels beyond BLANK_FREE: the depth spread alone prefers a
BLANK_FREE = 0.45   # bare flat wall (lowest spread), and the room the completion stands for would be empty


def view_R(heading, yaw_deg, pitch_deg):
    """World-from-camera rotation (OpenCV axes) of a view yawed from the photo heading and pitched up."""
    a, p = np.radians(yaw_deg), np.radians(pitch_deg)
    h = np.array([np.cos(a) * heading[0] - np.sin(a) * heading[1], np.sin(a) * heading[0] + np.cos(a) * heading[1], 0])
    right = np.cross(h, Z)
    fwd = np.cos(p) * h + np.sin(p) * Z
    down = np.sin(p) * h - np.cos(p) * Z
    return np.stack([right, down, fwd], 1)


def view_K(name="", n=N):
    f = n / 2 / np.tan(np.radians(FOV.get(name, 90.0)) / 2)
    return np.array([[f, 0, n / 2], [0, f, n / 2], [0, 0, 1.0]])


def pano_dirs():
    lon = (np.arange(PW) + 0.5) / PW * 2 * np.pi - np.pi
    lat = np.pi / 2 - (np.arange(PH) + 0.5) / PH * np.pi
    lon, lat = np.meshgrid(lon, lat)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)


def project(d_world, R, K):
    """Pixel (u, v) of world directions in a camera; z <= 0 -> nan."""
    dc = d_world @ R
    z = dc[..., 2]
    ok = z > 1e-6
    u = np.where(ok, dc[..., 0] / np.where(ok, z, 1) * K[0, 0] + K[0, 2], np.nan)
    v = np.where(ok, dc[..., 1] / np.where(ok, z, 1) * K[1, 1] + K[1, 2], np.nan)
    return u, v


def paste(canvas, known, img, R, K, only_unknown=True):
    """Write a perspective image into the canvas (where it projects, and optionally only where nothing is yet)."""
    D = pano_dirs()
    u, v = project(D, R, K)
    h, w = img.shape[:2]
    inside = np.isfinite(u) & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
    if only_unknown:
        inside &= ~known
    s = cv2.remap(img.astype(np.float32), np.nan_to_num(u).astype(np.float32), np.nan_to_num(v).astype(np.float32),
                  cv2.INTER_LINEAR)
    canvas[inside] = s[inside]
    known |= inside


def render(canvas, known, R, K, n=N):
    """Perspective view of the canvas, and which of its pixels are already known."""
    uu, vv = np.meshgrid(np.arange(n) + 0.5, np.arange(n) + 0.5)
    rays = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu)], -1)
    d = rays @ R.T
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    lon = np.arctan2(d[..., 1], d[..., 0])
    lat = np.arcsin(np.clip(d[..., 2], -1, 1))
    x = ((lon + np.pi) / (2 * np.pi) * PW - 0.5).astype(np.float32)
    y = ((np.pi / 2 - lat) / np.pi * PH - 0.5).astype(np.float32)
    img = cv2.remap(canvas, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    k = cv2.remap(known.astype(np.uint8), x, y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP).astype(bool)
    return img, k


class Painter:
    """Inpainting model: SDXL inpainting (open weights) or FLUX.1 Fill [dev] (4-bit transformer, CPU offload)."""

    def __init__(self, model):
        self.model = model
        if model == "sdxl":
            from diffusers import AutoPipelineForInpainting
            self.pipe = AutoPipelineForInpainting.from_pretrained(
                "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16").to("cuda")
        else:
            from diffusers import BitsAndBytesConfig, FluxFillPipeline, FluxTransformer2DModel
            from transformers import BitsAndBytesConfig as TBnb
            from transformers import T5EncoderModel
            repo = "black-forest-labs/FLUX.1-Fill-dev"
            q = dict(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
            tr = FluxTransformer2DModel.from_pretrained(repo, subfolder="transformer", torch_dtype=torch.bfloat16,
                                                        quantization_config=BitsAndBytesConfig(**q))
            t5 = T5EncoderModel.from_pretrained(repo, subfolder="text_encoder_2", torch_dtype=torch.bfloat16,
                                                quantization_config=TBnb(**q))
            self.pipe = FluxFillPipeline.from_pretrained(repo, transformer=tr, text_encoder_2=t5,
                                                         torch_dtype=torch.bfloat16)
            self.pipe.enable_model_cpu_offload()

    def __call__(self, img, mask, prompt, negative, seed):
        g = torch.Generator("cuda").manual_seed(seed)
        im, m = Image.fromarray(img), Image.fromarray(mask.astype(np.uint8) * 255)
        if self.model == "sdxl":
            out = self.pipe(prompt=prompt, negative_prompt=negative, image=im, mask_image=m, height=N, width=N,
                            strength=0.99, guidance_scale=7.0, num_inference_steps=30, generator=g).images[0]
        else:
            out = self.pipe(prompt=prompt, image=im, mask_image=m, height=N, width=N, guidance_scale=30.0,
                            num_inference_steps=28, max_sequence_length=512, generator=g).images[0]
        return np.asarray(out.convert("RGB").resize((N, N)))


class Judge:
    """Score of a candidate completion of a mostly unseen direction, lower is better:
      - window / sky: SAM 3's share of the painted pixels (a window has nothing behind it in the room),
      - text / sign: SAM 3's share of the painted pixels (diffusion lettering is gibberish),
      - blank: the share of featureless painted pixels beyond 45 % (a bare wall has the least depth spread, and the
        room it stands for would be empty: the completion is where the room's objects come from),
      - depth spread: MoGe-2's horizontal distance to the painted vertical surfaces, 85th over 15th percentile, beyond
        what one wall seen across a 90 deg view spans (1.5); a corridor or a room beyond an opening reaches 3+,
      - the up view instead: the share of its painted surface that is not one flat ceiling facing down."""

    def __init__(self):
        from moge.model.v2 import MoGeModel
        from transformers import Sam3Model, Sam3Processor

        self.sam = Sam3Model.from_pretrained("facebook/sam3").to("cuda").eval()
        self.proc = Sam3Processor.from_pretrained("facebook/sam3")
        self.moge = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to("cuda").eval()

    def segment(self, pil, prompts, shape):
        m = np.zeros(shape, bool)
        for prompt in prompts:
            inputs = self.proc(images=pil, text=prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = self.sam(**inputs)
            res = self.proc.post_process_instance_segmentation(
                out, threshold=0.45, mask_threshold=0.5, target_sizes=inputs.get("original_sizes").tolist())[0]
            for k in res["masks"]:
                k = k.cpu().numpy().astype(bool)
                if k.mean() < 0.5:  # a "sign" answer covering the view is the wall
                    m |= k
        return m

    def __call__(self, img, painted, name):
        pil = Image.fromarray(img)
        glass = self.segment(pil, ("window", "sky"), painted.shape)
        lettering = self.segment(pil, ("text", "sign"), painted.shape)
        n = max(int(painted.sum()), 1)
        window = float((glass & painted).sum() / n)
        text = float((lettering & painted).sum() / n)
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        mu = cv2.boxFilter(g, -1, (21, 21))
        sd = np.sqrt(np.maximum(cv2.boxFilter(g * g, -1, (21, 21)) - mu * mu, 0))
        blank = float(((sd < 4.0) & painted).sum() / n)  # featureless: a bare wall or ceiling
        x = torch.tensor(img / 255.0, dtype=torch.float32, device="cuda").permute(2, 0, 1)
        with torch.no_grad():
            o = self.moge.infer(x, fov_x=FOV.get(name, 90.0))
        P = o["points"].cpu().numpy()
        ok = o["mask"].cpu().numpy().astype(bool) & np.isfinite(P).all(-1) & painted & ~glass
        nrm = o["normal"].cpu().numpy() if "normal" in o else None
        if name == "up":  # the ceiling: share of the painted surface facing straight down (a vaulted or faceted
            # ceiling bakes into polygons on the flat ceiling plane)
            flat = float((np.abs(nrm[..., 2]) > 0.95)[ok].mean()) if nrm is not None and ok.sum() > 2000 else 1.0
            score = WINDOW_WEIGHT * window + TEXT_WEIGHT * text + 2.0 * (1.0 - flat)
            return score, {"window_sky": round(window, 3), "text": round(text, 3), "flat_ceiling": round(flat, 3),
                           "score": round(score, 3)}
        spread = 1.0
        if nrm is not None:
            ok &= np.abs(nrm[..., 1]) < 0.3  # vertical surfaces (camera y is down)
        if ok.sum() > 2000:
            d = np.hypot(P[ok][:, 0], P[ok][:, 2])
            spread = float(np.percentile(d, 85) / max(np.percentile(d, 15), 1e-6))
        score = (WINDOW_WEIGHT * window + TEXT_WEIGHT * text + BLANK_WEIGHT * max(0.0, blank - BLANK_FREE)
                 + max(0.0, float(np.log(spread / 1.5))))
        return score, {"window_sky": round(window, 3), "text": round(text, 3), "blank": round(blank, 3),
                       "depth_spread": round(spread, 2), "score": round(score, 3)}


def default_prompt(s: Session):
    sg = s.scene_graph
    labels = sorted({v.get("_label", k) for sec in ("static_objects", "dynamic_objects") for k, v in sg.get(sec, {}).items()}
                    | {spec.get("category", g) for g, spec in sg.get("movable_furniture", {}).items()})
    return {"prompt": f"a photo of the rest of the same indoor room, same style, same lighting and materials, with "
                      f"{', '.join(labels[:8])}, realistic interior photography, natural light",
            "negative": NEGATIVE,
            "_note": "default built from the scene graph labels; the agent should describe the room type and style"}


def view_prompt(cfg, name):
    """A view's prompt: configs' "views": {name: {"prompt": ...}} first (the agent says what is in that direction,
    e.g. a dining area behind the camera), else the room prompt (+ ceiling / floor for up / down); a cousin's style
    ("style": its theme's look) is appended to every view."""
    v = cfg.get("views", {}).get(name, {})
    p = cfg["prompt"]
    if v.get("prompt"):
        p = v["prompt"]
    elif name == "up":
        p = cfg.get("prompt_up", "the ceiling of " + p)
    elif name == "down":
        p = cfg.get("prompt_down", "the floor of " + p)
    return f"{p}, {cfg['style']}" if cfg.get("style") else p


def complete(s, out, model, seed, cfg, dilate=9):
    """Paste the photo, outpaint the views in order; returns {name: (img, photo pixels, R)}."""
    img = np.asarray(Image.open(s / "capture" / "rgb.png").convert("RGB")).astype(np.float32)
    geo = s.geometry()
    T = np.asarray(s.world_frame()[2])
    canvas, known = np.zeros((PH, PW, 3), np.float32), np.zeros((PH, PW), bool)
    paste(canvas, known, img, T[:3, :3], geo["K"].astype(np.float64), only_unknown=False)
    photo = known.copy()  # what the photo itself shows (texture comes from the photo there, never from a view)
    heading = T[:3, 2] * np.array([1, 1, 0])
    heading /= np.linalg.norm(heading)
    painter = Painter(model)
    judge, chosen = None, {}
    n_cand = int(cfg.get("candidates", CANDIDATES))
    for f in (out / "views").glob("*_cand*.png"):
        f.unlink()
    views = {}
    for k, (name, yaw, pitch) in enumerate(VIEWS):
        R = view_R(heading, yaw, pitch)
        K = view_K(name)
        v, kv = render(canvas, known, R, K)
        kp = render(canvas, photo, R, K)[1]
        miss = ~kv
        if miss.mean() < 0.01:
            views[name] = (v.astype(np.uint8), kp, R)
            continue
        m = cv2.dilate(miss.astype(np.uint8), np.ones((dilate, dilate), np.uint8)).astype(bool)
        src = v.astype(np.uint8).copy()
        src[miss] = 127
        vcfg = cfg.get("views", {}).get(name, {})
        unseen = miss.mean() > UNSEEN and name != "down"
        neg = vcfg.get("negative") or (
            cfg.get("negative", NEGATIVE) + (", furniture, chair, stool, table" if name in ("up", "down") else "")
            + (", " + UNSEEN_NEGATIVE if name != "down" else ""))
        a = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 3)[..., None]  # feather into the known pixels
        a[miss] = 1.0
        # the agent pins one direction's seed in completion.json; else a mostly unseen one keeps the best candidate
        seeds = [vcfg["seed"]] if "seed" in vcfg else [seed + k + 7919 * j for j in range(n_cand if unseen else 1)]
        best = None
        for j, vseed in enumerate(seeds):
            gen = painter(src, m, view_prompt(cfg, name), neg, vseed).astype(np.float32)
            blend = (a * gen + (1 - a) * v).clip(0, 255)
            if len(seeds) == 1:
                best = (0.0, blend, {"seed": vseed})
                break
            judge = judge or Judge()
            score, info = judge(blend.astype(np.uint8), miss, name)
            info["seed"] = vseed
            Image.fromarray(blend.astype(np.uint8)).save(out / "views" / f"{name}_cand{j}.png")
            print(f"[pano] {name} candidate {j} (seed {vseed}): {info}", flush=True)
            chosen.setdefault(name, {"candidates": []})["candidates"].append(info)
            if best is None or score < best[0]:
                best = (score, blend, info)
        blend = best[1]
        if name in chosen:
            chosen[name]["kept"] = best[2]["seed"]
            print(f"[pano] {name}: kept seed {best[2]['seed']}", flush=True)
        views[name] = (blend.astype(np.uint8), kp, R)
        paste(canvas, known, blend, R, K)
        Image.fromarray(blend.astype(np.uint8)).save(out / "views" / f"{name}.png")
        print(f"[pano] {name}: {miss.mean() * 100:.0f}% painted", flush=True)
    (out / "views" / "candidates.json").write_text(json.dumps(chosen, indent=1))
    del painter, judge
    torch.cuda.empty_cache()
    Image.fromarray(canvas.clip(0, 255).astype(np.uint8)).save(out / "pano.png")
    Image.fromarray(known.astype(np.uint8) * 255).save(out / "pano_known.png")
    return views


def load_views(s, out):
    """The views of an earlier run (--reuse): generated images as saved, photo pixels recomputed."""
    img = np.asarray(Image.open(s / "capture" / "rgb.png").convert("RGB")).astype(np.float32)
    T = np.asarray(s.world_frame()[2])
    canvas, photo = np.zeros((PH, PW, 3), np.float32), np.zeros((PH, PW), bool)
    paste(canvas, photo, img, T[:3, :3], s.geometry()["K"].astype(np.float64), only_unknown=False)
    heading = T[:3, 2] * np.array([1, 1, 0])
    heading /= np.linalg.norm(heading)
    views = {}
    for name, yaw, pitch in VIEWS:
        R = view_R(heading, yaw, pitch)
        v, kp = render(canvas, photo, R, view_K(name))
        f = out / "views" / f"{name}.png"
        views[name] = (np.asarray(Image.open(f).convert("RGB")) if f.exists() else v.astype(np.uint8), kp, R)
    return views


def depths(s, views, masks):
    """MoGe-2 per view at the known FoV, then one metric scale per view (joint_scales). The floor below the camera,
    measured per view: up-facing points outside every object mask, their lowest layer (table tops and seats face up
    too); it only weakly informs the joint solve."""
    from moge.model.v2 import MoGeModel

    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to("cuda").eval()
    h_cam = float(np.asarray(s.world_frame()[2])[2, 3])
    res = {}
    for name, (img, kv, R) in views.items():
        x = torch.tensor(img / 255.0, dtype=torch.float32, device="cuda").permute(2, 0, 1)
        with torch.no_grad():
            o = model.infer(x, fov_x=FOV.get(name, 90.0))
        P = o["points"].cpu().numpy()
        nrm = o["normal"].cpu().numpy() if "normal" in o else None
        ok = o["mask"].cpu().numpy().astype(bool) & np.isfinite(P).all(-1)
        Pw = P @ R.T  # camera-relative, world axes
        Nw = nrm @ R.T if nrm is not None else None
        floor = ok & (Pw[..., 2] < 0) & ~masks[name][0]
        if Nw is not None:
            floor &= Nw[..., 2] > 0.9
        below = -Pw[floor][:, 2]
        floor_scale = None
        if len(below) > 20000:  # else too few floor pixels (a wall-filled view) to measure the camera height on
            low = below[below >= np.percentile(below, 60)]  # the lowest layer: the floor itself
            floor_scale = h_cam / float(np.median(low))
        res[name] = {"P": Pw, "N": Nw, "ok": ok, "R": R, "known": kv, "floor_scale": floor_scale}
    del model
    torch.cuda.empty_cache()
    scales = joint_scales(s, res, masks)
    cam = np.asarray(s.world_frame()[2])[:3, 3]
    for name, r in res.items():
        r["scale"] = scales[name]
        r["Pw"] = r["P"] * r["scale"] + cam  # world points
    return res


def _grid_dirs(name, R, step=8):
    """World directions of a pixel grid of a view, and the grid's rows / cols."""
    K = view_K(name)
    v, u = np.mgrid[step // 2:N:step, step // 2:N:step]
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones(u.shape)], -1)
    return (d / np.linalg.norm(d, axis=-1, keepdims=True)) @ R.T, v, u


def _lookup(name, R, d_world):
    """Row / col of world directions in a view, and whether they fall inside it."""
    K = view_K(name)
    dc = d_world @ R
    z = dc[..., 2]
    u = dc[..., 0] / np.maximum(z, 1e-6) * K[0, 0] + K[0, 2]
    v = dc[..., 1] / np.maximum(z, 1e-6) * K[1, 1] + K[1, 2]
    inside = (z > 0.2) & (u >= 0) & (u < N - 1) & (v >= 0) & (v < N - 1)
    return np.clip(v.astype(int), 0, N - 1), np.clip(u.astype(int), 0, N - 1), inside


def joint_scales(s, res, masks, irls=5):
    """One metric scale per view, solved jointly on log scales (weighted least squares, Huber-reweighted):
      - the camera height over the view's floor (up-facing points, their lowest layer), where it has enough floor,
      - where a view holds photo pixels, its depth there = the photo's metric reconstruction, inside the room only
        (through a window the photo depth is the street, the generated view's is the glass),
      - where two views overlap, both give the same depth, weighted by how well the depth maps agree in shape:
        this carries the scale into the views with no floor (a wall straight ahead, the ceiling).
    Each constraint's sigma comes from its own spread, so glass or a hallucinated jog cannot outvote the floor."""
    names = list(res)
    idx = {n: i for i, n in enumerate(names)}
    dist = {n: np.linalg.norm(r["P"], axis=-1) for n, r in res.items()}
    usable = {n: r["ok"] & ~masks[n][0] & (dist[n] > 1e-3) for n, r in res.items()}
    rows, rhs, sig, what = [], [], [], []

    def add(coef, b, sigma, label):
        c = np.zeros(len(names))
        for n, w in coef.items():
            c[idx[n]] = w
        rows.append(c), rhs.append(b), sig.append(sigma), what.append(label)

    geo = s.geometry()
    R_wc, o, T = s.world_frame()
    cam = np.asarray(T)[:3, 3]
    W = (geo["points"][geo["mask"]] - o) @ np.asarray(R_wc).T
    W = W[np.random.default_rng(0).choice(len(W), min(len(W), 60000), replace=False)]
    sg = s.scene_graph
    floor = np.asarray(next(v for k, v in sg["structure"].items() if structure_role(k, v) == "floor")["range"], float)
    ceil = float(next((v["at"] for k, v in sg["structure"].items() if structure_role(k, v) == "ceiling"), 3.0))
    inside_room = np.all((W[:, :2] > floor[:, 0] - 0.25) & (W[:, :2] < floor[:, 1] + 0.25), 1) & \
        (W[:, 2] > -0.1) & (W[:, 2] < ceil + 0.1)
    W = W[inside_room]
    rel = W - cam
    d_true = np.linalg.norm(rel, axis=-1)
    dirs = rel / np.maximum(d_true, 1e-6)[:, None]
    for n in names:
        vi, ui, inside = _lookup(n, res[n]["R"], dirs)
        m = inside & res[n]["known"][vi, ui] & res[n]["ok"][vi, ui]
        if m.sum() >= 500:
            r = np.log(d_true[m] / np.maximum(dist[n][vi[m], ui[m]], 1e-6))
            iqr = float(np.subtract(*np.percentile(r, [75, 25])))
            add({n: 1}, float(np.median(r)), max(iqr / 1.35, 0.05), f"photo {n} ({m.sum()} px)")
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            na, nb = names[a], names[b]
            d, vb, ub = _grid_dirs(nb, res[nb]["R"])
            va, ua, inside = _lookup(na, res[na]["R"], d)
            m = inside & usable[nb][vb, ub] & usable[na][va, ua]
            if m.sum() < 200:
                continue
            r = np.log(dist[na][va[m], ua[m]] / dist[nb][vb[m], ub[m]])  # log s_b - log s_a
            iqr = float(np.subtract(*np.percentile(r, [75, 25])))
            add({nb: 1, na: -1}, float(np.median(r)), max(iqr / 1.35, 0.05), f"overlap {na}/{nb} ({m.sum()} px)")
    for n in names:
        if res[n]["floor_scale"]:
            add({n: 1}, float(np.log(res[n]["floor_scale"])), 0.1, f"floor {n}")
    if not any(w.startswith(("photo", "floor")) for w in what):
        return {n: 1.0 for n in names}
    A, b, sig = np.array(rows), np.array(rhs), np.array(sig)
    w = 1 / sig
    for _ in range(irls):
        x = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)[0]
        z = np.abs(A @ x - b) / sig
        w = np.where(z <= 2, 1, 2 / np.maximum(z, 1e-9)) / sig  # Huber: an inconsistent constraint stops pulling
    res_z = (A @ x - b) / sig
    for label, zz in zip(what, res_z):
        if abs(zz) > 2:
            print(f"[pano] scale constraint off by {zz:+.1f} sigma: {label}")
    for n in names:
        fs = res[n]["floor_scale"]
        print(f"[pano] {n}: depth scale {np.exp(x[idx[n]]):.3f} (floor alone {fs and round(fs, 3)})")
    return {n: float(np.exp(x[idx[n]])) for n in names}


def object_masks(views, prompts):
    from calibrated_scene_reconstruction.scene.capture.perceive import run_sam3

    out = {}
    for name, (img, kv, _) in views.items():
        inst = [i for i in run_sam3(Image.fromarray(img), prompts, "cuda") if i["score"] > 0.45]
        m = np.zeros(img.shape[:2], bool)
        for i in inst:
            if i["mask"].mean() < 0.6:  # a "wall" answer to an object prompt covers the view
                m |= i["mask"]
        out[name] = (m & ~kv, [{k: v for k, v in i.items() if k != "mask"} for i in inst])
    return out


def plates(views, masks, out, dilate=21):
    from simple_lama_inpainting import SimpleLama

    lama = SimpleLama()
    res = {}
    for name, (img, kv, _) in views.items():
        m = cv2.dilate(masks[name][0].astype(np.uint8), np.ones((dilate, dilate), np.uint8)).astype(bool) & ~kv
        p = img.copy()
        if m.any():
            p[m] = np.asarray(lama(Image.fromarray(img), Image.fromarray(m.astype(np.uint8) * 255)).resize((N, N)))[m]
        Image.fromarray(p).save(out / "views" / f"{name}_plate.png")
        Image.fromarray(m.astype(np.uint8) * 255).save(out / "views" / f"{name}_mask.png")
        res[name] = p
    return res


def room_from_depth(s, deps, masks, margin=0.1):
    """Unseen room-bounding walls (no ids, on the floor's boundary) at the completed room's extent: per wall, the 3rd percentile of the generated structure
    points along its axis (heights 0.3 m .. ceiling - 0.3 m), never inside what the photo or the scene graph holds."""
    sg = s.scene_graph
    st = sg["structure"]
    ceil = float(next((v["at"] for k, v in st.items() if structure_role(k, v) == "ceiling"), 2.5))
    pts = []
    for name, r in deps.items():
        sel = r["ok"] & ~masks[name][0]
        P = r["Pw"][sel]
        pts.append(P[(P[:, 2] > 0.3) & (P[:, 2] < ceil - 0.3)])
    P = np.concatenate(pts)
    occupied = [np.asarray(f["box"], float) for f in sg["furniture"].values()]
    geo = s.geometry()
    R_wc, o, _ = s.world_frame()
    obs = (geo["points"][geo["mask"]] - o) @ R_wc.T
    floor = np.asarray(next(v for k, v in st.items() if structure_role(k, v) == "floor")["range"], float)
    walls = {}
    for name, w in st.items():
        if structure_role(name, w) != "wall" or w.get("ids") or w.get("closure") or w["plane"] not in "xy":
            continue
        ax = "xy".index(w["plane"])
        if np.abs(floor[ax] - w["at"]).min() > 1e-3:  # an inner jog (a return between two back walls), not a bound
            continue
        neg = w.get("normal", "+") == "+"  # inward normal + : the wall bounds the room from below on its axis
        c = P[:, ax]
        new = float(np.percentile(c, 3) if neg else np.percentile(c, 97))
        inner = min([b[0][ax] for b in occupied] + [np.percentile(obs[:, ax], 1)]) - margin if neg else \
            max([b[1][ax] for b in occupied] + [np.percentile(obs[:, ax], 99)]) + margin
        new = min(new, inner) if neg else max(new, inner)
        new = float(np.clip(new, w["at"] - 6, w["at"] + 6))
        walls[name] = {"old": w["at"], "at": round(new, 3)}
    return {"walls": walls}


def apply_room(s: Session, room):
    """Move the unseen walls; every range endpoint equal to a moved wall's old coordinate follows it."""
    path = s.config_dir / "scene_graph.json"
    sg = json.loads(path.read_text())
    if not (s.config_dir / "scene_graph.pre_pano.json").exists():
        shutil.copy(path, s.config_dir / "scene_graph.pre_pano.json")
    for name, w in room["walls"].items():
        ax = "xy".index(sg["structure"][name]["plane"])
        old, new = w["old"], w["at"]
        sg["structure"][name]["at"] = new
        for k, v in sg["structure"].items():
            rng = v.get("range")
            if not rng:
                continue
            if v["plane"] == "z":
                rng[ax] = [new if abs(x - old) < 1e-3 else x for x in rng[ax]]
            elif v["plane"] != sg["structure"][name]["plane"]:
                rng[0] = [new if abs(x - old) < 1e-3 else x for x in rng[0]]
    sg["_pano_note"] = "unseen walls moved by recon.panorama (completed room depth); before: scene_graph.pre_pano.json"
    path.write_text(dumps(sg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--model", default="sdxl", choices=["sdxl", "flux"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="subfolder (a cousin, a model comparison)")
    ap.add_argument("--apply", action="store_true", help="write the completed room's walls into scene_graph.json")
    ap.add_argument("--reuse", action="store_true", help="keep the generated views, redo masks / plates / depth")
    ap.add_argument("--style", default="", help="appended to every view's prompt (a cousin's theme)")
    args = ap.parse_args()
    s = Session(args.session)
    out = s / "scene" / "pano" / args.tag if args.tag else s / "scene" / "pano"
    (out / "views").mkdir(parents=True, exist_ok=True)
    cp = s.config_dir / "completion.json"
    cfg = json.loads(cp.read_text()) if cp.exists() else default_prompt(s)
    if args.style:
        cfg["style"] = args.style
    tag_cfg = cfg.get("tags", {}).get(args.tag or "twin", {})  # the agent's review of this tag's views: re-rolls,
    for name, v in tag_cfg.get("views", {}).items():          # prompts of its own (completion review)
        cfg.setdefault("views", {})[name] = {**cfg.get("views", {}).get(name, {}), **v}
    if "seed" in tag_cfg:
        args.seed = int(tag_cfg["seed"])
    views = load_views(s, out) if args.reuse else complete(s, out, args.model, args.seed, cfg)
    from calibrated_scene_reconstruction.scene.capture.perceive import PROMPTS
    prompts = [p for p in s.config("perception.json", {"prompts": PROMPTS})["prompts"] if p not in STRUCTURE]
    masks = object_masks(views, prompts)
    plates(views, masks, out)
    for f in (out / "views").glob("*_mask_pano.png"):  # recon.pano_furniture starts again from these masks
        f.unlink()
    deps = depths(s, views, masks)
    np.savez_compressed(out / "views.npz", names=np.array(list(views)), R=np.stack([v[2] for v in views.values()]),
                        K=np.stack([view_K(n) for n in views]), known=np.stack([v[1] for v in views.values()]),
                        depth=np.stack([np.linalg.norm(deps[n]["P"], axis=-1) * deps[n]["scale"] for n in views]),
                        ok=np.stack([deps[n]["ok"] for n in views]), cam=np.asarray(s.world_frame()[2])[:3, 3],
                        scale=np.array([deps[n]["scale"] for n in views]))
    room = room_from_depth(s, deps, masks)
    room.update(model=args.model, seed=args.seed, prompt=cfg["prompt"], style=cfg.get("style", ""))
    (out / "room.json").write_text(json.dumps(room, indent=1))
    print("[pano] unseen walls:", {k: (v["old"], v["at"]) for k, v in room["walls"].items()})
    for name, (m, inst) in masks.items():
        print(f"[pano] {name}: objects {sorted({i['label'] for i in inst})}")
    if args.apply:
        apply_room(s, room)
        print("[pano] scene_graph.json updated")


if __name__ == "__main__":
    main()
