"""Run the pinned official VGGT-Omega implementation with auditable outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantiles(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {}
    return {
        str(percentile): float(np.percentile(finite, percentile))
        for percentile in (1, 5, 50, 95, 99)
    }


def scalar_frame(values: np.ndarray) -> np.ndarray:
    frame = np.asarray(values)
    if frame.ndim == 3 and frame.shape[-1] == 1:
        frame = frame[..., 0]
    if frame.ndim != 2:
        raise ValueError(f"Expected one scalar image, got shape {frame.shape}")
    return frame


def scalar_preview(values: np.ndarray, output: Path) -> Image.Image:
    finite = values[np.isfinite(values)]
    if not finite.size:
        image = np.zeros(values.shape, dtype=np.uint8)
    else:
        low, high = np.percentile(finite, [5, 95])
        scale = max(float(high - low), 1e-8)
        image = np.clip((values - low) / scale, 0.0, 1.0)
        image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
        image = np.asarray(image * 255, dtype=np.uint8)
    preview = Image.fromarray(image).convert("L")
    preview.save(output)
    return preview


def preprocessing_records(paths: list[Path], resolution: int, patch_size: int = 16):
    records = []
    for path in paths:
        with Image.open(path) as image:
            original_width, original_height = image.size
        crop_left = crop_top = 0
        crop_width, crop_height = original_width, original_height
        aspect = original_height / max(original_width, 1)
        if aspect < 0.5:
            crop_width = min(original_width, max(1, int(round(original_height / 0.5))))
            crop_left = max((original_width - crop_width) // 2, 0)
        elif aspect > 2.0:
            crop_height = min(original_height, max(1, int(round(original_width * 2.0))))
            crop_top = max((original_height - crop_height) // 2, 0)
        cropped_aspect = crop_height / max(crop_width, 1)
        token_number = (resolution // patch_size) ** 2
        width_patches = max(1, int(np.round(np.sqrt(token_number / cropped_aspect))))
        height_patches = max(1, int(np.round(token_number / width_patches)))
        resized_width = width_patches * patch_size
        resized_height = height_patches * patch_size
        records.append(
            {
                "original_size": [original_width, original_height],
                "crop_box": [
                    crop_left,
                    crop_top,
                    crop_left + crop_width,
                    crop_top + crop_height,
                ],
                "cropped_size": [crop_width, crop_height],
                "resized_size": [resized_width, resized_height],
            }
        )
    processed_width = max(record["resized_size"][0] for record in records)
    processed_height = max(record["resized_size"][1] for record in records)
    for record in records:
        resized_width, resized_height = record["resized_size"]
        pad_left = (processed_width - resized_width) // 2
        pad_top = (processed_height - resized_height) // 2
        pad_right = processed_width - resized_width - pad_left
        pad_bottom = processed_height - resized_height - pad_top
        crop_left, crop_top, _, _ = record["crop_box"]
        crop_width, crop_height = record["cropped_size"]
        scale_x = resized_width / crop_width
        scale_y = resized_height / crop_height
        record.update(
            {
                "padding": [pad_left, pad_top, pad_right, pad_bottom],
                "processed_size": [processed_width, processed_height],
                "original_to_processed": [
                    [scale_x, 0.0, pad_left - scale_x * crop_left],
                    [0.0, scale_y, pad_top - scale_y * crop_top],
                    [0.0, 0.0, 1.0],
                ],
            }
        )
    return records


def evidence_panel(
    processed: Image.Image,
    depth: Image.Image,
    confidence: Image.Image,
    output: Path,
) -> None:
    images = [processed.convert("RGB"), depth.convert("RGB"), confidence.convert("RGB")]
    width, height = images[0].size
    label_height = 24
    panel = Image.new("RGB", (width * 3, height + label_height), (20, 20, 20))
    draw = ImageDraw.Draw(panel)
    for index, (label, image) in enumerate(
        zip(("MODEL INPUT", "RELATIVE DEPTH", "CONFIDENCE"), images, strict=True)
    ):
        panel.paste(image, (index * width, label_height))
        draw.text((index * width + 8, 6), label, fill=(255, 255, 255))
    panel.save(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--structural-smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("VGGT-Omega requires CUDA")
    images = [path.resolve() for path in args.image]
    if any(not path.is_file() for path in images):
        raise FileNotFoundError("One or more input images are missing")
    if args.structural_smoke and args.checkpoint is not None:
        raise ValueError("Structural smoke must not load a checkpoint")
    if not args.structural_smoke and (args.checkpoint is None or not args.checkpoint.is_file()):
        raise FileNotFoundError("The approved VGGT-Omega checkpoint is required")
    if args.resolution not in {256, 416, 512}:
        raise ValueError("Resolution must be 256, 416 or 512")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    preprocess = preprocessing_records(images, args.resolution)
    torch.manual_seed(0)
    model = VGGTOmega().eval()
    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True)
    model = model.to("cuda")
    inputs = load_and_preprocess_images(
        [str(path) for path in images], image_resolution=args.resolution
    ).to("cuda")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.monotonic()
    with torch.inference_mode():
        predictions = model(inputs)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    extrinsics, intrinsics = encoding_to_camera(
        predictions["pose_enc"], predictions["images"].shape[-2:]
    )
    depth = predictions["depth"][0].detach().float().cpu().numpy()
    confidence = predictions["depth_conf"][0].detach().float().cpu().numpy()
    extrinsics_np = extrinsics[0].detach().float().cpu().numpy()
    intrinsics_np = intrinsics[0].detach().float().cpu().numpy()
    np.savez_compressed(
        output / "predictions.npz",
        depth=depth,
        confidence=confidence,
        extrinsics=extrinsics_np,
        intrinsics=intrinsics_np,
    )
    for index, frame_depth in enumerate(depth):
        input_image = np.asarray(
            inputs[index].detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
            * 255,
            dtype=np.uint8,
        )
        processed = Image.fromarray(input_image)
        processed.save(output / f"processed_{index:03d}.png")
        depth_image = scalar_preview(
            scalar_frame(frame_depth), output / f"depth_{index:03d}.png"
        )
        confidence_image = scalar_preview(
            scalar_frame(confidence[index]), output / f"confidence_{index:03d}.png"
        )
        evidence_panel(
            processed,
            depth_image,
            confidence_image,
            output / f"evidence_{index:03d}.png",
        )

    repo = Path(os.environ.get("REAL2SIM_VGGT_REPO",
                             Path(__file__).resolve().parents[1] / ".tools/vggt-omega")).resolve()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    report = {
        "schema_version": 1,
        "ok": not args.structural_smoke,
        "structural_smoke": args.structural_smoke,
        "warning": "Random weights: outputs are not geometry evidence"
        if args.structural_smoke
        else None,
        "backend": "vggt-omega",
        "backend_commit": commit,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "gpu_capability": list(torch.cuda.get_device_capability()),
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "checkpoint_sha256": sha256(args.checkpoint) if args.checkpoint else None,
        "resolution": args.resolution,
        "input_tensor_shape": list(inputs.shape),
        "elapsed_seconds": round(elapsed, 4),
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 4),
        "images": [
            {
                "index": index,
                "name": path.name,
                "sha256": sha256(path),
                "preprocess": preprocess[index],
            }
            for index, path in enumerate(images)
        ],
        "depth_quantiles": [quantiles(scalar_frame(frame)) for frame in depth],
        "confidence_quantiles": [quantiles(scalar_frame(frame)) for frame in confidence],
        "cameras": [
            {
                "index": index,
                "world_to_camera_opencv": extrinsics_np[index].tolist(),
                "intrinsics": intrinsics_np[index].tolist(),
            }
            for index in range(len(images))
        ],
        "artifacts": {
            "predictions": "predictions.npz",
            "depth_previews": [f"depth_{index:03d}.png" for index in range(len(images))],
            "confidence_previews": [
                f"confidence_{index:03d}.png" for index in range(len(images))
            ],
            "processed_inputs": [
                f"processed_{index:03d}.png" for index in range(len(images))
            ],
            "evidence_panels": [
                f"evidence_{index:03d}.png" for index in range(len(images))
            ],
            "camera_priors": [
                f"camera_priors/reference_{index:03d}.json" for index in range(len(images))
            ],
        },
    }
    prior_dir = output / "camera_priors"
    prior_dir.mkdir()
    for index, camera in enumerate(report["cameras"]):
        prior = {
            "schema_version": 1,
            "source": "vggt-omega",
            "reference_index": index,
            "reference_name": images[index].name,
            "coordinate_frame": "shared_vggt_world_with_opencv_cameras",
            "metric_scale": False,
            "preprocess": preprocess[index],
            "world_to_camera_opencv": camera["world_to_camera_opencv"],
            "intrinsics_processed_pixels": camera["intrinsics"],
        }
        (prior_dir / f"reference_{index:03d}.json").write_text(
            json.dumps(prior, indent=2), encoding="utf-8"
        )
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
