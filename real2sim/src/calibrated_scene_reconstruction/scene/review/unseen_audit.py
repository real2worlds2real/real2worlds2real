"""Automatic audit of the renders of what the photo did not see (apps/scene/render_views.py --unseen).

    python -m calibrated_scene_reconstruction.recon.unseen_audit sessions/<name> [--tags pano,cousin_00] [--pano sdxl]

The agent used to judge renders/views/<tag>_unseen_{left,back,right,far}.jpg by eye. Each render is now checked
against two references it should agree with:
  - the completion itself (scene/pano[/<tag>]/pano.png, a cousin's own completion for a cousin) seen from the render
    camera's direction: DINOv2 similarity of render and reference (a missing wall, a black hole, a stretched or
    smeared bake no longer looks like what was painted there; parallax from the render's eye is small for left /
    back / right, the far view has no reference),
  - the same tag's render from the photo camera (<tag>_photo.jpg), whose surfaces are the photo's own texture: the
    unseen render's fine detail relative to it (a smeared fill) and its share of flat tiles,
and against image statistics: black / missing surfaces (pixels at the renderer's black), sky where a wall should be
(blue-sky pixels the reference does not have: the dome seen through a hole), stretched texture (structure-tensor
coherent tiles of one orientation over an area, not a single edge: a texture projected at a grazing angle streaks)
beyond the reference's own. A bad verdict is any of: black, sky, streaks, DINOv2 similarity < 0.35, fine detail
< 0.5 x the completion's in that direction (without a reference: < 0.25 x the photo render's), implausible scale; warn:
similarity < 0.55, detail < 0.7 (0.4), flat tiles a quarter beyond the completion's own (a plain wall is flat in both;
without a reference: more than half flat).
The completion's depth scales (views.npz) and moved walls (room.json) give the scale check: a direction whose scale
is far from the ring's median while nothing anchored it (no floor, no photo pixels) carries implausible walls.

Writes qa/unseen_audit.json: per tag and view a verdict (ok / warn / bad), the reasons, the metrics, and the
completion views to re-do (recon.panorama, completion.json views.<v>.seed / views.<v>.prompt).
"""
import argparse
import json

import cv2
import numpy as np
from PIL import Image

from calibrated_scene_reconstruction.scene.session import Session
from calibrated_scene_reconstruction.scene.background.panorama import PH, PW, VIEWS

W, H = 960, 640  # apps/scene/render_views.py
FOCAL = 0.62  # sim.usd.pinhole
RENDER_YAW = {"unseen_left": 90, "unseen_right": -90, "unseen_back": 180, "unseen_far": 180}
NEAR_VIEWS = 61  # pano views within this many degrees of a render's heading are the ones that textured it
# thresholds
BLACK = 0.02  # share of pixels at the renderer's black
SKY, SKY_NOREF = 0.03, 0.02  # share of blue-sky pixels beyond the reference's (bad); without a reference (warn)
STREAK = 0.12  # share of streaked tiles beyond the reference's
SMEAR_BAD, SMEAR_WARN = 0.25, 0.4  # fine detail relative to the photo-camera render
FLAT, FLAT_REF = 0.5, 0.25  # share of featureless tiles; beyond the completion's own (a plain wall is flat in both)
REF_SMEAR_BAD, REF_SMEAR_WARN = 0.5, 0.7  # fine detail relative to the completion seen from the render's heading
SIM_BAD, SIM_WARN = 0.35, 0.55  # DINOv2 class-token similarity to the completion
SCALE = 1.6  # a direction's depth scale off the ring median by this factor


def render_cameras(s: Session):
    """World-from-camera (OpenCV) rotations of the unseen renders, as apps/scene/render_views.py places them."""
    cam = s.manifest["photo_camera"]
    Tw = np.asarray(cam["T_world_cam_cv"], float)
    f = Tw[:2, 2] / max(np.linalg.norm(Tw[:2, 2]), 1e-6)
    out = {}
    for k in ("unseen_left", "unseen_right", "unseen_back"):
        a = np.radians(RENDER_YAW[k])
        d = np.r_[np.cos(a) * f[0] - np.sin(a) * f[1], np.sin(a) * f[0] + np.cos(a) * f[1], -0.15]
        out[k] = look(d)
    return out


def look(d):
    fwd = d / np.linalg.norm(d)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(fwd, right), fwd], 1)


def pano_view(canvas, R, w=W, h=H):
    """The equirectangular completion seen through a render camera."""
    f = FOCAL * w
    uu, vv = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    d = np.stack([(uu - w / 2) / f, (vv - h / 2) / f, np.ones_like(uu)], -1) @ R.T
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    lon, lat = np.arctan2(d[..., 1], d[..., 0]), np.arcsin(np.clip(d[..., 2], -1, 1))
    x = ((lon + np.pi) / (2 * np.pi) * PW - 0.5).astype(np.float32)
    y = ((np.pi / 2 - lat) / np.pi * PH - 0.5).astype(np.float32)
    return cv2.remap(canvas, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def tiles(a, t=16):
    h, w = a.shape[:2]
    return a[: h // t * t, : w // t * t].reshape(h // t, t, w // t, t, *a.shape[2:]).swapaxes(1, 2)


def stats(img):
    """Image statistics on a 480 x 320 copy: black, sky, streaked / flat tiles, fine-detail energy."""
    im = cv2.resize(img, (480, 320), interpolation=cv2.INTER_AREA).astype(np.float32)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    gray = im.mean(-1)
    black = float((im.max(-1) < 10).mean())
    sky = float(((b > r + 25) & (b > g + 8) & (b > 110)).mean())
    gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    jxx, jyy, jxy = (tiles(a).mean((2, 3)) for a in (gx * gx, gy * gy, gx * gy))
    energy = jxx + jyy
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / np.maximum(energy, 1e-6)
    ori = np.round((0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2) / np.pi * 8).astype(int) % 8
    mid = cv2.GaussianBlur(gray, (0, 0), 1.5) - cv2.GaussianBlur(gray, (0, 0), 6)
    fine = gray - cv2.GaussianBlur(gray, (0, 0), 1.0)
    mid_t, fine_t = tiles(mid).std((2, 3)), tiles(fine).std((2, 3))
    flat = mid_t < 1.0
    lines = (coh > 0.85) & (energy > 100) & ~flat
    # stretched texture: parallel streaks over an area (3 x 3 tiles of one orientation survive an erosion; a single
    # edge, a door frame, is one tile wide and does not)
    k3 = np.ones((3, 3), np.uint8)
    streak = np.zeros_like(lines)
    for o in range(8):
        m = lines & ((ori == o) | (ori == (o + 1) % 8))
        streak |= cv2.dilate(cv2.erode(m.astype(np.uint8), k3), k3).astype(bool) & m
    return {"black": round(black, 4), "sky": round(sky, 4), "streak": round(float(streak.mean()), 4),
            "flat": round(float(flat.mean()), 4),
            "fine": round(float(np.median(fine_t[~flat])) if (~flat).any() else 0.0, 3)}


class Dino:
    """DINOv2 class token and patch tokens (16 x 16 at 224 px), on the CPU: a few renders do not need the GPU."""

    def __init__(self):
        import torch
        from transformers import AutoModel

        self.torch = torch
        self.model = AutoModel.from_pretrained("facebook/dinov2-base").eval()

    def __call__(self, img):
        x = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        with self.torch.no_grad():
            h = self.model(pixel_values=self.torch.from_numpy(x.transpose(2, 0, 1)[None]).float()).last_hidden_state
        n = self.torch.nn.functional.normalize
        return n(h[0, 0], dim=-1).numpy(), n(h[0, 1:], dim=-1).numpy()


def scale_check(pdir):
    """Directions whose depth scale is off the ring median with nothing anchoring it, and the walls moved."""
    z = np.load(pdir / "views.npz")
    names, scale, known = [str(n) for n in z["names"]], z["scale"], z["known"]
    ring = [i for i, n in enumerate(names) if n not in ("up", "down")]
    med = float(np.median(scale[ring]))
    off = {}
    for i in ring:
        anchored = known[i].mean() > 0.02
        if abs(np.log(scale[i] / med)) > np.log(SCALE) and not anchored:
            off[names[i]] = round(float(scale[i] / med), 2)
    room = json.loads((pdir / "room.json").read_text()) if (pdir / "room.json").exists() else {}
    return off, room.get("walls", {})


def near_views(yaw):
    return [n for n, y, p in VIEWS if p == 0 and abs((y - yaw + 180) % 360 - 180) < NEAR_VIEWS]


def audit(s: Session, tag, pdir, dino, cams):
    out = {}
    photo = s / "renders" / "views" / f"{tag}_photo.jpg"
    base = stats(np.asarray(Image.open(photo).convert("RGB"))) if photo.exists() else None
    canvas = np.asarray(Image.open(pdir / "pano.png").convert("RGB"), np.float32) if pdir else None
    off, walls = scale_check(pdir) if pdir else ({}, {})
    for view, yaw in RENDER_YAW.items():
        f = s / "renders" / "views" / f"{tag}_{view}.jpg"
        if not f.exists():
            continue
        img = np.asarray(Image.open(f).convert("RGB"))
        m = stats(img)
        ref = None
        if canvas is not None and view in cams:
            ref = pano_view(canvas, cams[view]).clip(0, 255).astype(np.uint8)
            mr = stats(ref)
            m["ref_sky"], m["ref_streak"], m["ref_flat"] = mr["sky"], mr["streak"], mr["flat"]
            m["fine_vs_ref"] = round(m["fine"] / max(mr["fine"], 1e-3), 3)
            (ci, pi), (cr, pr) = dino(img), dino(ref)
            m["dino_global"] = round(float(ci @ cr), 3)
            m["dino_median"] = round(float(np.median((pi * pr).sum(-1))), 3)
        if base:
            m["fine_vs_photo"] = round(m["fine"] / max(base["fine"], 1e-3), 3)
        bad, warn = [], []
        if m["black"] > BLACK:
            bad.append(f"black / missing surface ({m['black'] * 100:.1f} % of the view)")
        sky = m["sky"] - m.get("ref_sky", 0.0)
        if sky > (SKY if ref is not None else SKY_NOREF):
            (bad if ref is not None else warn).append(f"sky where a wall should be ({sky * 100:.1f} % blue-sky pixels"
                                                      f"{' the completion does not have' if ref is not None else ''})")
        streak = m["streak"] - m.get("ref_streak", 0.0)
        if streak > STREAK:
            bad.append(f"stretched / streaked texture ({streak * 100:.0f} % of tiles beyond the completion's own)")
        if "fine_vs_ref" in m:  # against what was painted there: a plain wall is not a smear of a busy one
            if m["fine_vs_ref"] < REF_SMEAR_WARN:
                (bad if m["fine_vs_ref"] < REF_SMEAR_BAD else warn).append(
                    f"smeared: fine detail {m['fine_vs_ref']:.2f} x the completion's in that direction")
        elif base and m["fine_vs_photo"] < SMEAR_WARN:
            (bad if m["fine_vs_photo"] < SMEAR_BAD else warn).append(
                f"smeared: fine detail {m['fine_vs_photo']:.2f} x the photo-camera render's")
        flat = m["flat"] - m.get("ref_flat", 0.0)
        if flat > (FLAT_REF if "ref_flat" in m else FLAT):
            warn.append(f"large featureless region ({flat * 100:.0f} % flat tiles beyond the completion's own: a "
                        f"surface no view covered?)")
        if "dino_global" in m and m["dino_global"] < SIM_WARN:
            (bad if m["dino_global"] < SIM_BAD else warn).append(
                f"departs from the completion (DINOv2 similarity {m['dino_global']:.2f})")
        near = near_views(yaw)
        sc = {n: off[n] for n in near if n in off}
        if sc:
            moved = {k: v for k, v in walls.items() if abs(v["at"] - v["old"]) > 0.05}
            bad.append(f"implausible scale: depth scale of {sc} x the ring median with no floor or photo anchor; walls "
                       f"it moved are suspect ({ {k: (v['old'], v['at']) for k, v in moved.items()} })")
        verdict = "bad" if bad else "warn" if warn else "ok"
        redo = sorted(sc) if sc and len(bad) == 1 else near if bad else []
        action = ""
        if redo:
            action = (f"recon.panorama again with completion.json views.<v>.seed changed (or a sharper "
                      f"views.<v>.prompt) for {redo}")
            if sc:
                action += "; the scale needs an anchor: a prompt that shows the floor in that direction"
        out[view] = {"verdict": verdict, "reasons": bad + warn, "redo_views": redo, "action": action, "metrics": m}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--tags", default="", help="render tags (default: every tag with unseen renders)")
    ap.add_argument("--pano", default="", help="the twin's completion: scene/pano/<pano> (cousins: their own)")
    args = ap.parse_args()
    s = Session(args.session)
    tags = [t for t in args.tags.split(",") if t] or sorted(
        {p.name.split("_unseen_")[0] for p in (s / "renders" / "views").glob("*_unseen_*.jpg")})
    cams, dino = render_cameras(s), Dino()
    report = {}
    for tag in tags:
        own = s / "scene" / "pano" / tag
        pdir = own if (own / "views.npz").exists() else s / "scene" / "pano" / args.pano
        pdir = pdir if (pdir / "pano.png").exists() and (pdir / "views.npz").exists() else None
        report[tag] = audit(s, tag, pdir, dino, cams)
        for view, r in report[tag].items():
            print(f"[audit] {tag} {view}: {r['verdict']}" + (f" -- {'; '.join(r['reasons'])}" if r["reasons"] else "")
                  + (f" -> re-do {r['redo_views']}" if r["redo_views"] else ""))
    s.write_json("qa/unseen_audit.json", report)


if __name__ == "__main__":
    main()
