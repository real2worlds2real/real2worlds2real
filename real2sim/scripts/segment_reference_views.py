"""Optional native SAM3 command; caller owns its model environment and local weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    args = parser.parse_args()
    import torch
    import transformers
    from transformers import Sam3Model, Sam3Processor

    from real2sim_harness.artifacts import atomic_json, sha256_file
    from real2sim_harness.segmentation import associate, mask_records, prompt_box_inputs
    request = json.loads(args.request.read_text())
    views, prompts = request["views"], request["prompts"]
    threshold = float(request.get("threshold", .3))
    if (not 1 <= len(views) <= 1000 or not 1 <= len(prompts) <= 32
            or not all(isinstance(p, str) and p.strip() for p in prompts)
            or len(set(prompts)) != len(prompts) or not 0 <= threshold <= 1):
        raise ValueError("Use 1..1000 views,1..32 prompts and a threshold in [0,1]")
    if not args.weights.is_dir():
        raise ValueError("Supply operator-owned local SAM3 weights; this command does not log in or download")
    for view in views:
        if not isinstance(view.get('prompt_boxes', {}), dict) or set(view.get('prompt_boxes', {}))-set(prompts):
            raise ValueError('Spatial hint keys must match declared prompt names')
    model = Sam3Model.from_pretrained(str(args.weights.resolve()), local_files_only=True).to(args.device).eval()
    processor = Sam3Processor.from_pretrained(str(args.weights.resolve()), local_files_only=True)
    args.output.mkdir(parents=True, exist_ok=False)
    records, previous, identifiers = [], {}, {prompt: 0 for prompt in prompts}
    weight_files = sorted({*args.weights.glob("*.safetensors"), *args.weights.glob("pytorch_model*.bin"),
                           *args.weights.glob("config.json")})
    hashes = {str(p.name): sha256_file(p) for p in weight_files}
    for i, view in enumerate(views):
        path = (args.request.parent / view["image"]).resolve()
        image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        if min(image.size) < 8 or max(image.size) > 4096:
            raise ValueError("Use displayed images with axes between8 and4096pixels")
        entry = {"view": i, "source_sha256": sha256_file(path), "image_size": list(image.size),
                 "time_seconds": view.get("time_seconds"), "instances": [],
                 "spatial_prompts": view.get('prompt_boxes', {})}
        inputs = processor(images=image, return_tensors="pt").to(args.device)
        with torch.inference_mode():
            vision = model.get_vision_features(pixel_values=inputs["pixel_values"])
        for prompt_index, prompt in enumerate(prompts):
            spatial = prompt_box_inputs(view, prompt, image.size)
            text = processor(**(spatial or {'text': prompt}), return_tensors="pt").to(args.device)
            text.pop('original_sizes', None)  # Processor metadata is not a model argument.
            with torch.inference_mode():
                prediction = model(vision_embeds=vision, **text)
            result = processor.post_process_instance_segmentation(prediction, threshold=threshold,
                mask_threshold=.5, target_sizes=inputs["original_sizes"].tolist())[0]
            current = mask_records(result["masks"].detach().cpu().numpy(), result["scores"].detach().cpu().numpy())
            identifiers[prompt] = associate(previous.get(prompt, []), current, identifiers[prompt],
                                             min_iou=float(request.get("association_min_iou", .1)))
            previous[prompt] = current
            union = np.zeros((image.height, image.width), bool)
            for j, row in enumerate(current):
                name = f"view-{i:05d}-prompt-{prompt_index:02d}-instance-{j:03d}.png"
                Image.fromarray(row["mask"].astype(np.uint8) * 255).save(args.output / name)
                union |= row["mask"]
                entry["instances"].append({k: v for k, v in row.items() if k != "mask"} | {"prompt": prompt, "mask": name})
            Image.fromarray(union.astype(np.uint8) * 255).save(args.output / f"view-{i:05d}-prompt-{prompt_index:02d}-union.png")
        records.append(entry)
    atomic_json(args.output / "segmentation_report.json", {"ok": True, "views": records,
        "backend": {"name": "SAM3 via Transformers", "version": transformers.__version__, "weight_hashes": hashes},
        "limits": ["Masks are model predictions. Inspect boundaries before using them as geometric evidence.",
                   "Spatial boxes are caller annotations, not measured geometry; review masks even with positive/negative hints.",
                   "Track IDs are conservative adjacent-mask IoU proposals, not native learned video tracking.",
                   "Identity after occlusion or camera jumps requires review; unions may merge distinct objects."]})


if __name__ == "__main__":
    main()
