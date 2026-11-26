from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifacts import atomic_json
from .camera import solve_pnp


def _map_to_processed(points, preprocess):
    import numpy as np

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("image_points must contain pixel [x, y] pairs")
    original_width, original_height = preprocess["original_size"]
    if (
        np.any(values[:, 0] < 0)
        or np.any(values[:, 0] >= original_width)
        or np.any(values[:, 1] < 0)
        or np.any(values[:, 1] >= original_height)
    ):
        raise ValueError("An image point lies outside the original image")
    left, top, right, bottom = preprocess["crop_box"]
    if (
        np.any(values[:, 0] < left)
        or np.any(values[:, 0] >= right)
        or np.any(values[:, 1] < top)
        or np.any(values[:, 1] >= bottom)
    ):
        raise ValueError("An image point lies in an area cropped before VGGT-Omega inference")
    homogeneous = np.column_stack((values, np.ones(len(values))))
    transform = np.asarray(preprocess["original_to_processed"], dtype=np.float64)
    return (transform @ homogeneous.T).T[:, :2]


def _sample_bilinear(field, points):
    import numpy as np

    values = np.asarray(field, dtype=np.float64)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    height, width = values.shape
    x = points[:, 0]
    y = points[:, 1]
    if np.any(x < 0) or np.any(x > width - 1) or np.any(y < 0) or np.any(y > height - 1):
        raise ValueError("A transformed point lies outside the VGGT-Omega prediction")
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = x - x0
    wy = y - y0
    return (
        values[y0, x0] * (1 - wx) * (1 - wy)
        + values[y0, x1] * wx * (1 - wy)
        + values[y1, x0] * (1 - wx) * wy
        + values[y1, x1] * wx * wy
    )


def _original_intrinsics(processed_intrinsics, preprocess):
    import numpy as np

    matrix = np.asarray(processed_intrinsics, dtype=np.float64)
    transform = np.asarray(preprocess["original_to_processed"], dtype=np.float64)
    return np.linalg.inv(transform) @ matrix


def align_vggt_prior(
    prior_output: Path,
    camera_solution_dir: Path,
    *,
    reference_index: int,
    object_points: list[list[float]],
    image_points: list[list[float]],
    max_reprojection_error_px: float,
) -> dict[str, Any]:
    import numpy as np

    if len(object_points) != len(image_points) or len(object_points) < 4:
        raise ValueError("Provide at least four paired object and image points")
    if max_reprojection_error_px <= 0:
        raise ValueError("Reprojection tolerance must be positive")
    report_path = prior_output / "report.json"
    predictions_path = prior_output / "predictions.npz"
    if not report_path.is_file() or not predictions_path.is_file():
        raise FileNotFoundError("Run the VGGT-Omega reference prior first")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("ok") or report.get("structural_smoke"):
        raise ValueError("Only a real-checkpoint VGGT-Omega result can be aligned")
    if not 0 <= reference_index < len(report["images"]):
        raise ValueError("reference_index is out of range")

    with np.load(predictions_path) as predictions:
        depth = predictions["depth"]
        confidence = predictions["confidence"]
        intrinsics = predictions["intrinsics"]
    preprocess = report["images"][reference_index]["preprocess"]
    processed_points = _map_to_processed(image_points, preprocess)
    sampled_depth = _sample_bilinear(depth[reference_index], processed_points)
    sampled_confidence = _sample_bilinear(confidence[reference_index], processed_points)
    if np.any(~np.isfinite(sampled_depth)) or np.any(sampled_depth <= 0):
        raise ValueError("One or more correspondence depths are invalid")

    original_intrinsics = _original_intrinsics(intrinsics[reference_index], preprocess)
    width, height = preprocess["original_size"]
    solution = solve_pnp(
        object_points,
        image_points,
        width=width,
        height=height,
        fx=float(original_intrinsics[0, 0]),
        fy=float(original_intrinsics[1, 1]),
        cx=float(original_intrinsics[0, 2]),
        cy=float(original_intrinsics[1, 2]),
        distortion=[],
    )
    solution.update(
        {
            "source": "vggt_omega_intrinsics_pnp",
            "reference_index": reference_index,
            "reference_name": report["images"][reference_index]["name"],
            "reference_sha256": report["images"][reference_index]["sha256"],
            "vggt_prior": {
                "processed_image_points": processed_points.tolist(),
                "sampled_depth": sampled_depth.tolist(),
                "sampled_confidence": sampled_confidence.tolist(),
                "limits": [
                    "VGGT-Omega supplied intrinsics and learned depth evidence",
                    "metric pose comes from explicit model-to-pixel PnP correspondences",
                    "other reference cameras remain priors until independently aligned",
                ],
            },
        }
    )
    observed_error = float(solution["reprojection"]["max_px"])
    if observed_error > max_reprojection_error_px:
        result = {
            "schema_version": 1,
            "ok": False,
            "status": "rejected_reprojection_error",
            "backend": "vggt-omega-intrinsics-pnp",
            "gate": {
                "observed_max_reprojection_error_px": observed_error,
                "allowed_max_reprojection_error_px": max_reprojection_error_px,
            },
            "camera": solution,
            "written_camera_indices": [],
        }
        atomic_json(prior_output / "alignment.json", result)
        return result

    camera_solution_dir.mkdir(parents=True, exist_ok=True)
    destination = camera_solution_dir / f"reference_{reference_index:03d}.json"
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing.get("source") != "vggt_omega_intrinsics_pnp":
            result = {
                "schema_version": 1,
                "ok": True,
                "status": "precise_existing_camera_preserved",
                "backend": "vggt-omega-intrinsics-pnp",
                "camera": solution,
                "written_camera_indices": [],
                "preserved_existing_camera_indices": [reference_index],
            }
            atomic_json(prior_output / "alignment.json", result)
            return result
    atomic_json(destination, solution)
    result = {
        "schema_version": 1,
        "ok": True,
        "status": "reference_camera_ready",
        "backend": "vggt-omega-intrinsics-pnp",
        "camera": solution,
        "written_camera_indices": [reference_index],
        "preserved_existing_camera_indices": [],
    }
    atomic_json(prior_output / "alignment.json", result)
    return result
