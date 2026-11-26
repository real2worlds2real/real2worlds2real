from __future__ import annotations

from fractions import Fraction
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from PIL import ExifTags, Image

from .artifacts import sha256_file
from .multiview import colmap_capabilities
from .vggt import vggt_status


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, tuple) and len(value) == 2:
            return float(Fraction(*value))
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _image_record(path: Path, index: int) -> dict[str, Any]:
    with Image.open(path) as image:
        exif = image.getexif()
        width, height = image.size
        orientation = int(exif.get(ExifTags.Base.Orientation, 1))
        display_size = [height, width] if orientation in {5, 6, 7, 8} else [width, height]
        focal_mm = _number(exif.get(ExifTags.Base.FocalLength))
        focal_35mm = _number(exif.get(ExifTags.Base.FocalLengthIn35mmFilm))
        fx_prior = display_size[0] * focal_35mm / 36.0 if focal_35mm else None
        return {
            "index": index,
            "name": path.name,
            "sha256": sha256_file(path),
            "format": image.format,
            "mode": image.mode,
            "stored_size": [width, height],
            "display_size": display_size,
            "exif_orientation": orientation,
            "camera": {
                "make": str(exif.get(ExifTags.Base.Make) or "") or None,
                "model": str(exif.get(ExifTags.Base.Model) or "") or None,
                "lens_model": str(exif.get(ExifTags.Base.LensModel) or "") or None,
                "focal_length_mm": focal_mm,
                "focal_length_35mm": focal_35mm,
                "fx_prior_px": round(fx_prior, 3) if fx_prior else None,
            },
        }


def geometry_backend_status() -> dict[str, dict[str, Any]]:
    from .segmentation import backend_status as segmentation_status

    return {
        "opencv_pnp": {
            "available": find_spec("cv2") is not None,
            "role": "metric pose from known 3D-2D correspondences",
        },
        "colmap": {
            **colmap_capabilities(),
            "role": "feature geometry, bundle adjustment and multi-view reconstruction",
        },
        "vggt_omega": vggt_status(),
        "sam3": segmentation_status(),
    }


def analyze_reference_images(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("At least one reference image is required")
    records = []
    for index, source in enumerate(paths):
        path = source.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(_image_record(path, index))
    hashes = [record["sha256"] for record in records]
    sizes = {tuple(record["display_size"]) for record in records}
    cameras = {
        (record["camera"]["make"] or "", record["camera"]["model"] or "")
        for record in records
        if record["camera"]["make"] or record["camera"]["model"]
    }
    count = len(records)
    if count == 1:
        route = {
            "mode": "single_view",
            "preferred": "intrinsics prior followed by PnP after a metric draft exists",
            "limits": [
                "one image does not determine hidden geometry",
                "absolute scale requires a known dimension or other metric evidence",
            ],
        }
    else:
        route = {
            "mode": "multi_view",
            "preferred": "VGGT-Omega camera/depth prior refined by COLMAP bundle adjustment",
            "limits": [
                "the views must depict one consistent object state",
                "low overlap or texture can make feature-only reconstruction fail",
            ],
        }
    return {
        "schema_version": 1,
        "count": count,
        "unique_images": len(set(hashes)),
        "duplicate_indices": [
            index for index, digest in enumerate(hashes) if digest in hashes[:index]
        ],
        "consistent_display_size": len(sizes) == 1,
        "camera_signatures": [list(value) for value in sorted(cameras)],
        "views_with_focal_prior": sum(
            record["camera"]["fx_prior_px"] is not None for record in records
        ),
        "route": route,
        "backends": geometry_backend_status(),
        "views": records,
    }
