"""Agent review of the room completions (README: after `cousin_rooms`): every completed view of the twin and of every
cousin, as the agent judges it -- is it a plausible, realistic room of this kind, consistent with the photos?

    python -m calibrated_scene_reconstruction.recon.completion_review sessions/<name> [--tags twin,cousin_00]

A completion is the source of the room's unseen objects (recon.room_objects reconstructs them in 3D), so what it
paints must be something a real room of this type holds, standing on the floor at a believable size: no gibberish
text or signs, no boxes or furniture glued to a wall, no painted floor stripes, no isolated product renders, no
second kitchen behind the camera. The candidate scores of recon.panorama (window / sky, text, depth spread) are
printed on each view. The agent writes its verdict per tag in the scene graph:
    "_completion_review": {"<tag>": {"ok": true | false, "views": {"<view>": "ok" | "<what is wrong>"}, "note": ...}}
and re-rolls or re-prompts a rejected view in completion.json "tags": {"<tag>": {"views": {"<view>": {"seed": n,
"prompt": ...}}}} (recon.panorama, recon.cousin_rooms), then reruns from `panorama` / `cousin_rooms`.

Writes qa/completion/<tag>.jpg (the six ring views large, up / down small, the photo pixels outlined).
"""
import argparse
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw

from calibrated_scene_reconstruction.scene.session import Session

RING = ["front", "left60", "left120", "back", "right120", "right60"]
T = 512  # tile size


def tags(s: Session):
    return ["twin"] + sorted(p.stem for p in (s / "cousins").glob("cousin_*.json"))


def view_dir(s: Session, tag):
    return s / "scene" / "pano" / ("" if tag == "twin" else tag) / "views"


def sheet(s: Session, tag):
    d = view_dir(s, tag)
    if not (d / "front.png").exists():
        return None
    cands = json.loads((d / "candidates.json").read_text()) if (d / "candidates.json").exists() else {}
    z = np.load(d.parent / "views.npz") if (d.parent / "views.npz").exists() else None
    known = dict(zip([str(n) for n in z["names"]], z["known"])) if z is not None else {}
    S = Image.new("RGB", (3 * T, 2 * T + T // 2 + 24), (18, 18, 18))
    draw = ImageDraw.Draw(S)
    for k, name in enumerate(RING + ["up", "down"]):
        f = d / f"{name}.png"
        if not f.exists():
            continue
        im = np.asarray(Image.open(f).convert("RGB")).copy()
        if name in known:  # the photo's own pixels: outlined (not the completion's)
            edge = cv2.morphologyEx(known[name].astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)) > 0
            im[edge] = (255, 220, 0)
        big = k < 6
        size = T if big else T // 2
        x, y = (k % 3) * T, (k // 3) * T if big else 2 * T + 24
        if not big:
            x = (k - 6) * (T // 2)
        S.paste(Image.fromarray(im).resize((size, size)), (x, y))
        c = cands.get(name, {})
        kept = next((i for i in c.get("candidates", []) if i.get("seed") == c.get("kept")), {})
        label = f"{tag} {name}" + (f"  seed {kept.get('seed')} text {kept.get('text', '-')} window "
                                   f"{kept.get('window_sky')} spread {kept.get('depth_spread', '-')}" if kept else "")
        draw.rectangle([x, y, x + 6 * len(label) + 6, y + 14], fill=(0, 0, 0))
        draw.text((x + 3, y + 1), label, fill=(255, 255, 0))
    draw.text((T + 8, 2 * T + 6), "up, down (small): ceiling / floor colour only; structure gets clean materials",
              fill=(200, 200, 200))
    out = s / "qa" / "completion"
    out.mkdir(parents=True, exist_ok=True)
    S.save(out / f"{tag}.jpg", quality=88)
    return out / f"{tag}.jpg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--tags", default="", help="comma-separated (default: twin and every cousin)")
    args = ap.parse_args()
    s = Session(args.session)
    review = s.scene_graph.get("_completion_review", {})
    for tag in [t for t in args.tags.split(",") if t] or tags(s):
        f = sheet(s, tag)
        r = review.get(tag)
        print(f"[completion] {tag}: {f or 'no completion'}"
              + (f" -- reviewed: {'ok' if r.get('ok') else 'rejected'}" if r else " -- not reviewed"), flush=True)


if __name__ == "__main__":
    main()
