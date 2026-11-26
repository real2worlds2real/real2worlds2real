"""S1 perception: MoGe-2 metric point map + intrinsics, SAM 3 text-prompted instance masks.

    python -m calibrated_scene_reconstruction.recon.perceive sessions/<name>

Writes parsing/{geometry.npz, masks.npz, instances.json, overlay.png}. Instance ids are the row index in masks.npz
and are referenced by configs/<name>/scene_graph.json, so keep PROMPTS stable once a scene graph exists.
"""
from calibrated_scene_reconstruction.core.perception.moge import run_moge as run_moge  # noqa: F401
import argparse
import json

import cv2
import numpy as np
import torch
from PIL import Image

from calibrated_scene_reconstruction.core.gpu import device, share
from calibrated_scene_reconstruction.scene.session import Session

PROMPTS = [
    "cup", "bowl", "bar stool", "refrigerator", "countertop", "cooktop", "floor", "wall",
    "window", "kitchen cabinet", "bench", "ceramic jar", "picture frame", "faucet", "ceiling",
    "sink", "plant", "wicker basket", "apple", "bread", "window blind",
]




def sam_queries(pil, prompts, threshold, thresholds, regions):
    """Full-image queries plus explicitly labelled image regions for small task objects.

    Region boxes use the current capture image's pixels. They select evidence,
    never impose a mask or a metric size. Each detection still comes from SAM.
    """
    queries = [(pil, p, p, (0, 0), thresholds.get(p, threshold)) for p in prompts]
    for region in regions:
        box = region['box']
        if (len(box) != 4 or any(type(v) is not int for v in box)
                or not 0 <= box[0] < box[2] <= pil.width
                or not 0 <= box[1] < box[3] <= pil.height):
            raise ValueError('perception region box must be integer x0,y0,x1,y1 inside the capture image')
        prompt = region['prompt']
        label = region.get('label', prompt)
        queries.append((pil.crop(box), prompt, label, tuple(box[:2]),
                        region.get('threshold', thresholds.get(label, threshold))))
    return queries


def run_sam3(pil, prompts, device, threshold=0.4, thresholds=None, regions=None, *, reuse_image_features=True):
    """`thresholds` {prompt: score}: per prompt (perception.json): a small thing far off scores low, a low threshold for every prompt would let all of them in."""
    from calibrated_scene_reconstruction.core.perception.sam import load_sam3

    queries = sam_queries(pil, prompts, threshold, thresholds or {}, regions or [])
    share()
    model, proc = load_sam3(device)
    instances = []
    cached_image, vision, image_inputs = None, None, None
    for query_image, prompt, label, (x0, y0), score_threshold in queries:
        with torch.no_grad():
            if reuse_image_features and hasattr(model, 'get_vision_features'):
                # The official SAM3 interface separates image encoding from
                # text queries. Cache one actual image only; a crop or another
                # view gets fresh features and retains its own pixel sizes.
                if cached_image is not query_image:
                    image_inputs = proc(images=query_image, return_tensors='pt').to(device)
                    vision = model.get_vision_features(pixel_values=image_inputs['pixel_values'])
                    cached_image = query_image
                inputs = proc(text=prompt, return_tensors='pt').to(device)
                outputs = model(vision_embeds=vision, **inputs)
                original_sizes = image_inputs['original_sizes']
            else:
                inputs = proc(images=query_image, text=prompt, return_tensors="pt").to(device)
                outputs = model(**inputs)
                original_sizes = inputs['original_sizes']
        res = proc.post_process_instance_segmentation(
            outputs, threshold=score_threshold, mask_threshold=0.5,
            target_sizes=original_sizes.tolist())[0]
        for m, s, b in zip(res["masks"], res["scores"], res["boxes"]):
            mask = np.zeros((pil.height, pil.width), dtype=bool)
            mask[y0:y0 + query_image.height, x0:x0 + query_image.width] = m.cpu().numpy().astype(bool)
            offset = (x0, y0, x0, y0)
            instances.append({"label": label, "score": float(s),
                              "box": [float(v) + off for v, off in zip(b, offset)], "mask": mask})
        print(f"[sam3] {label}: {len(res['scores'])}" + (f" (region at {x0},{y0})" if query_image is not pil else ""))
    del model, vision, image_inputs
    torch.cuda.empty_cache()
    return instances


def overlay(img, instances):
    vis = img.astype(np.float32).copy()
    rng = np.random.default_rng(0)
    for inst in instances:
        vis[inst["mask"]] = 0.5 * vis[inst["mask"]] + 0.5 * rng.integers(40, 255, 3)
    vis = vis.clip(0, 255).astype(np.uint8)
    for i, inst in enumerate(instances):
        x0, y0 = int(inst["box"][0]), int(inst["box"][1])
        cv2.putText(vis, f"{i}:{inst['label']} {inst['score']:.2f}", (x0 + 3, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
    return Image.fromarray(vis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    args = ap.parse_args()
    s = Session(args.session)
    from .rectification import ensure as ensure_rectified_capture
    ensure_rectified_capture(s)
    out = s / "parsing"
    out.mkdir(parents=True, exist_ok=True)
    pil = Image.open(s / "capture" / "rgb.png").convert("RGB")
    img = np.asarray(pil)
    pcfg = s.config("perception.json", {"prompts": PROMPTS})
    prompts = pcfg["prompts"]

    cam = s.config("camera.json", {})  # the calibrated phone, for images of its orientation (the fov is size-free)
    fov_x = cam["fov_x_deg"] if cam and (cam["size"][0] > cam["size"][1]) == (img.shape[1] > img.shape[0]) else None
    if fov_x:
        print(f"[moge] field of view {fov_x:.1f} deg from configs/{s.name}/camera.json")
    geo = run_moge(img, device(), fov_x)
    H, W = img.shape[:2]
    K = geo["intrinsics"].copy()
    K[0] *= W
    K[1] *= H
    np.savez_compressed(out / "geometry.npz", points=geo["points"], depth=geo["depth"], mask=geo["mask"], K=K)
    for stale in ("geometry.moge.npz", "world_frame.json"):  # recon.multiview's copy and autograph's frame of the old map
        (out / stale).unlink(missing_ok=True)
    print("[moge] fx=%.1f depth %.2f-%.2f m" % (K[0, 0], np.nanmin(geo["depth"][geo["mask"]]),
                                                 np.nanmax(geo["depth"][geo["mask"]])))
    instances = run_sam3(pil, prompts, device(), thresholds=pcfg.get("thresholds"), regions=pcfg.get("regions"))
    np.savez_compressed(out / "masks.npz", masks=np.stack([i["mask"] for i in instances]))
    meta = [{k: v for k, v in inst.items() if k != "mask"} | {"id": i, "area": int(inst["mask"].sum())}
            for i, inst in enumerate(instances)]
    (out / "instances.json").write_text(json.dumps(meta, indent=1))
    overlay(img, instances).save(out / "overlay.png")
    print(f"done: {len(instances)} instances")


if __name__ == "__main__":
    main()
