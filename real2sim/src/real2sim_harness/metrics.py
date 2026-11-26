from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image


def _binary_mask(path: Path, size: tuple[int, int] | None = None) -> tuple[list[bool], tuple[int, int]]:
    image = Image.open(path).convert("RGBA")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    alpha = image.getchannel("A")
    if alpha.getextrema() == (255, 255):
        gray = image.convert("L")
        pixels = [value >= 128 for value in gray.get_flattened_data()]
    else:
        pixels = [value >= 128 for value in alpha.get_flattened_data()]
    return pixels, image.size


def _shape(mask: list[bool], size: tuple[int, int]) -> dict[str, Any]:
    width, height = size
    indices = [index for index, value in enumerate(mask) if value]
    if not indices:
        return {"area_fraction": 0.0, "bbox": None, "centroid": None}
    xs = [index % width for index in indices]
    ys = [index // width for index in indices]
    return {
        "area_fraction": round(len(indices) / (width * height), 6),
        "bbox": [min(xs), min(ys), max(xs) + 1, max(ys) + 1],
        "centroid": [round(sum(xs) / len(xs), 3), round(sum(ys) / len(ys), 3)],
    }


def silhouette_metrics(reference: Path, candidate: Path) -> dict[str, Any]:
    reference_mask, size = _binary_mask(reference)
    candidate_mask, _ = _binary_mask(candidate, size)
    intersection = sum(a and b for a, b in zip(reference_mask, candidate_mask, strict=True))
    union = sum(a or b for a, b in zip(reference_mask, candidate_mask, strict=True))
    return {
        "iou": round(intersection / union, 6) if union else 1.0,
        "reference": _shape(reference_mask, size),
        "candidate": _shape(candidate_mask, size),
        "size": list(size),
    }


def compare_mask_sets(reference_dir: Path, candidate_dir: Path) -> dict[str, Any]:
    names = sorted(path.name for path in reference_dir.glob("mask_*.png"))
    missing = [name for name in names if not (candidate_dir / name).is_file()]
    views = {
        name.removeprefix("mask_").removesuffix(".png"): silhouette_metrics(
            reference_dir / name, candidate_dir / name
        )
        for name in names
        if (candidate_dir / name).is_file()
    }
    scores = [value["iou"] for value in views.values()]
    return {
        "ok": bool(views) and not missing,
        "mean_iou": round(sum(scores) / len(scores), 6) if scores else None,
        "missing": missing,
        "views": views,
    }
