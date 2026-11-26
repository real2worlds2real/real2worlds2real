from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from .artifacts import atomic_json, sha256_file


def colmap_capabilities() -> dict[str, Any]:
    try:
        import pycolmap
    except ImportError:
        return {"available": False, "learned_features": False}
    return {
        "available": True,
        "version": pycolmap.__version__,
        "build": pycolmap.COLMAP_build,
        "cuda": bool(pycolmap.has_cuda),
        "learned_features": os.environ.get("REAL2SIM_HARNESS_COLMAP_ONNX", os.environ.get("UCB_REMO_COLMAP_ONNX")) == "1",
    }


def _camera_record(camera) -> dict[str, Any]:
    return {
        "camera_id": int(camera.camera_id),
        "model": camera.model_name,
        "width": int(camera.width),
        "height": int(camera.height),
        "params": [float(value) for value in camera.params],
        "params_info": camera.params_info,
        "prior_focal_length": bool(camera.has_prior_focal_length),
    }


def _image_record(image) -> dict[str, Any]:
    pose = image.cam_from_world()
    return {
        "image_id": int(image.image_id),
        "name": image.name,
        "camera_id": int(image.camera_id),
        "world_to_camera": pose.matrix().tolist(),
        "camera_center": image.projection_center().tolist(),
        "observed_points3d": int(image.num_points3D),
    }


def run_colmap(
    images: list[Path],
    output: Path,
    *,
    camera_mode: str = "auto",
    device: str = "auto",
    features: str = "sift",
    num_threads: int = 8,
) -> dict[str, Any]:
    if len(images) < 2:
        raise ValueError("COLMAP needs at least two reference images")
    if camera_mode not in {"auto", "single", "per_folder", "per_image"}:
        raise ValueError("Unsupported COLMAP camera mode")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("Unsupported COLMAP device")
    if features not in {"sift", "aliked"}:
        raise ValueError("Unsupported COLMAP feature backend")
    if not 1 <= num_threads <= 32:
        raise ValueError("COLMAP num_threads must be between 1 and 32")
    resolved = [path.resolve() for path in images]
    if any(not path.is_file() for path in resolved):
        raise FileNotFoundError("One or more COLMAP input images do not exist")
    if len({sha256_file(path) for path in resolved}) != len(resolved):
        raise ValueError("Duplicate images do not provide multi-view geometry")

    import pycolmap

    if features == "aliked" and not colmap_capabilities()["learned_features"]:
        raise RuntimeError(
            "This PyCOLMAP build lacks declared ONNX support; use SIFT, VGGT-Omega, "
            "or an ONNX-enabled COLMAP build with REAL2SIM_HARNESS_COLMAP_ONNX=1"
        )

    output = output.resolve()
    image_dir = output / "images"
    sparse_dir = output / "sparse"
    database = output / "database.db"
    if output.exists():
        raise FileExistsError(f"COLMAP output already exists: {output}")
    image_dir.mkdir(parents=True)
    sparse_dir.mkdir()
    for index, source in enumerate(resolved):
        shutil.copy2(source, image_dir / f"{index:03d}{source.suffix.lower()}")

    modes = {
        "auto": pycolmap.CameraMode.AUTO,
        "single": pycolmap.CameraMode.SINGLE,
        "per_folder": pycolmap.CameraMode.PER_FOLDER,
        "per_image": pycolmap.CameraMode.PER_IMAGE,
    }
    devices = {
        "auto": pycolmap.Device.auto,
        "cpu": pycolmap.Device.cpu,
        "cuda": pycolmap.Device.cuda,
    }
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.num_threads = num_threads
    matching = pycolmap.FeatureMatchingOptions()
    matching.num_threads = num_threads
    if features == "aliked":
        extraction.type = pycolmap.FeatureExtractorType.ALIKED_N16ROT
        matching.type = pycolmap.FeatureMatcherType.ALIKED_LIGHTGLUE
    pycolmap.extract_features(
        database,
        image_dir,
        camera_mode=modes[camera_mode],
        extraction_options=extraction,
        device=devices[device],
    )
    pycolmap.match_exhaustive(
        database,
        matching_options=matching,
        device=devices[device],
    )
    mapping = pycolmap.IncrementalPipelineOptions()
    mapping.num_threads = num_threads
    mapping.min_model_size = 2
    reconstructions = pycolmap.incremental_mapping(
        database, image_dir, sparse_dir, options=mapping
    )
    if not reconstructions:
        raise RuntimeError("COLMAP could not register a reconstruction")
    reconstruction_id, reconstruction = max(
        reconstructions.items(),
        key=lambda item: (item[1].num_reg_images(), item[1].num_points3D()),
    )
    selected = sparse_dir / "selected"
    selected.mkdir()
    reconstruction.write(selected)
    reconstruction.export_PLY(output / "sparse.ply")
    report = {
        "schema_version": 1,
        "ok": True,
        "backend": "pycolmap",
        "backend_version": pycolmap.__version__,
        "camera_mode": camera_mode,
        "device": device,
        "features": features,
        "num_threads": num_threads,
        "input_images": len(images),
        "reconstruction_count": len(reconstructions),
        "selected_reconstruction": int(reconstruction_id),
        "registered_images": int(reconstruction.num_reg_images()),
        "points3d": int(reconstruction.num_points3D()),
        "mean_reprojection_error_px": float(reconstruction.compute_mean_reprojection_error()),
        "mean_track_length": float(reconstruction.compute_mean_track_length()),
        "cameras": [
            _camera_record(camera)
            for _, camera in sorted(reconstruction.cameras.items())
        ],
        "images": [
            _image_record(image) for _, image in sorted(reconstruction.images.items())
        ],
        "artifacts": {
            "database": "database.db",
            "sparse_model": "sparse/selected",
            "sparse_point_cloud": "sparse.ply",
        },
    }
    atomic_json(output / "report.json", report)
    return report
