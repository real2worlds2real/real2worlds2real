"""Optional real-reference-conditioned Difix backend for Gaussian refinement."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    from real2sim_harness.artifacts import atomic_json, sha256_file
    sys.path.insert(0, str(args.checkout.resolve() / "src"))
    from pipeline_difix import DifixPipeline
    request = json.loads(args.request.read_text())
    images, references = request["images"], request.get("references", [])
    if not 1 <= len(images) <= 32 or len(references) != len(images) or not args.weights.is_dir():
        raise ValueError("Provide local Difix weights and one actual capture reference per proposed view")
    pipeline = DifixPipeline.from_pretrained(str(args.weights.resolve()), local_files_only=True).to("cuda")
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for i, (image_path, reference_path) in enumerate(zip(images, references, strict=True)):
        image = Image.open(image_path).convert("RGB")
        reference = Image.open(reference_path).convert("RGB")
        if not 32 <= min(image.size) <= max(image.size) <= 2048:
            raise ValueError("Use proposal images from 32 to 2048 pixels per axis")
        # Pad rather than stretch the proposal; output retains its exact camera domain.
        width, height = image.size
        padded = Image.fromarray(np.pad(np.asarray(image), ((0, (-height) % 8), (0, (-width) % 8), (0, 0)), mode="edge"))
        ref = reference.resize(padded.size, Image.Resampling.LANCZOS)
        with torch.no_grad():
            result = pipeline("remove degradation", image=padded, ref_image=ref,
                              num_inference_steps=1, timesteps=[199], guidance_scale=0.).images[0]
        result = result.crop((0, 0, width, height)).convert("RGB")
        if result.size != image.size or not np.isfinite(np.asarray(result)).all():
            raise ValueError("Difix changed the proposal pixel domain or returned invalid values")
        target = args.output / f"anchor-{i:03d}.png"
        result.save(target)
        records.append({"image": target.name, "input_sha256": sha256_file(Path(image_path)),
                        "reference_sha256": sha256_file(Path(reference_path)),
                        "output_sha256": sha256_file(target)})
    revision = subprocess.check_output(["git", "-C", str(args.checkout), "rev-parse", "HEAD"], text=True).strip()
    weight_files = sorted({*args.weights.rglob("*.safetensors"), *args.weights.rglob("pytorch_model*.bin")})
    weight_hashes = {str(path.relative_to(args.weights)): sha256_file(path) for path in weight_files}
    atomic_json(args.output / "restoration_report.json", {"ok": True, "images": records,
        "backend": {"name": "Difix real-reference-conditioned restoration", "version": revision,
                    "weight_hashes": weight_hashes},
        "reference_conditioned": True,
        "scope": "Generated proposals conditioned on actual captures; not measured hidden appearance"})


if __name__ == "__main__":
    main()
