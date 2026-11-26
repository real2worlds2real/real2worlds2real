"""S4a background plate: remove every object instance (dynamic, static, movable furniture), the real robot in the
photo (scene graph `photo_robot`) and the capture rig (`clutter`) with LaMa.

    python -m calibrated_scene_reconstruction.recon.inpaint sessions/<name>

The plate textures structure/furniture surfaces, so nothing that becomes its own 3D instance may stay painted on a
wall or counter. Writes scene/{plate.png, plate_mask.png}.
"""
import argparse

import cv2
import numpy as np
from PIL import Image

from calibrated_scene_reconstruction.scene.session import Session


def object_ids(sg):
    ids = []
    for group in ("dynamic_objects", "static_objects", "movable_furniture"):
        for spec in sg.get(group, {}).values():
            ids += spec["ids"]
    ids += sg.get("photo_robot", {}).get("ids", [])  # the real robot: the simulated one stands in its place
    ids += sg.get("clutter", {}).get("ids", [])  # the capture rig (perception.json `clutter`): cameras, tripods, cables
    return sorted(set(ids))


def removed_mask(s: Session, dilate=25):
    masks = s.masks()
    m = np.zeros(masks.shape[1:], np.uint8)
    for i in object_ids(s.scene_graph):
        m |= masks[i].astype(np.uint8)
    # grow the mask so contact shadows and anti-aliased rims go too
    return cv2.dilate(m, np.ones((dilate, dilate), np.uint8)).astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    args = ap.parse_args()
    s = Session(args.session)
    out = s / "scene"
    out.mkdir(parents=True, exist_ok=True)
    img = Image.open(s / "capture" / "rgb.png").convert("RGB")
    m = removed_mask(s)
    mask_img = Image.fromarray(m.astype(np.uint8) * 255)
    mask_img.save(out / "plate_mask.png")

    from simple_lama_inpainting import SimpleLama

    plate = np.asarray(SimpleLama()(img, mask_img).resize(img.size))
    a = np.asarray(img).copy()
    a[m] = plate[m]  # keep original pixels outside the mask (LaMa works at 1/8 resolution)
    Image.fromarray(a).save(out / "plate.png")
    print(f"plate saved ({m.mean() * 100:.1f}% of pixels inpainted)")


if __name__ == "__main__":
    main()
