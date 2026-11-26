"""Conditioned planar motion measurements using existing calibrated cameras."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from .artifacts import atomic_json, sha256_file
from .physics_inference import _digest, _track
from .posed_observations import camera_spec


def project_to_plane(observations, request):
    plane = request["plane"]
    origin, u, v = (np.asarray(plane[key], float) for key in ("origin_m", "u", "v"))
    if (any(a.shape != (3,) or not np.isfinite(a).all() for a in (origin, u, v))
            or not np.isclose(u @ u, 1) or not np.isclose(v @ v, 1) or abs(u @ v) > 1e-6):
        raise ValueError("Use a finite plane origin and orthonormal u/v axes")
    for key in ("source", "point_on_plane_assumption", "frame"):
        if not isinstance(plane.get(key), str) or not plane[key].strip():
            raise ValueError(f"Plane requires {key}")
    metric = request["metric_evidence"]
    if not metric.get("source") or type(metric.get("metric_verified")) is not bool:
        raise ValueError("Declare metric scale source and metric_verified status")
    residual_scale = float(request["residual_scale_m"])
    if not np.isfinite(residual_scale) or residual_scale <= 0:
        raise ValueError("residual_scale_m must be positive")
    if ("camera" in request) == ("cameras_by_frame" in request):
        raise ValueError("Provide one fixed camera or explicit per-frame cameras")
    fixed = camera_spec(request["camera"]) if "camera" in request else None
    if fixed and observations.get("camera_motion", {}).get("kind") not in {"fixed", "registered"}:
        raise ValueError("One camera cannot explain unresolved moving-camera observations")
    cameras = {str(i): camera_spec(row) for i, row in request.get("cameras_by_frame", {}).items()}
    selected = request.get("track_ids", [t["id"] for t in observations["tracks"]])
    if not selected or len(set(selected)) != len(selected) or set(selected) - {t["id"] for t in observations["tracks"]}:
        raise ValueError("Select existing unique track IDs")
    normal = np.cross(u, v)
    result = copy.deepcopy(observations)
    result["tracks"] = []
    diagnostics = []
    for row in observations["tracks"]:
        if row["id"] not in selected:
            continue
        times, pixels, measured = _track(row)
        if row["units"] != "px" or pixels.shape[1] != 2:
            raise ValueError("Plane projection takes 2D pixel measurements")
        frames = row.get("frame_indices")
        if not fixed and (frames is None or len(frames) != len(times)):
            raise ValueError("Per-frame cameras need original frame_indices")
        values, states, reasons = [], [], []
        for i, pixel in enumerate(pixels):
            state = row.get("sample_state", ["observed"] * len(times))[i]
            view = fixed if fixed is not None else cameras.get(str(frames[i]))
            reason, point = None, [None, None]
            if not measured[i]:
                reason = "not_observed"
                state = "missing"
            elif view is None:
                reason = "missing_registered_camera"
                state = "missing"
            else:
                if row.get("image_size") != [view["width"], view["height"]]:
                    raise ValueError("Track image_size must match the calibrated camera pixel domain")
                if not (0 <= pixel[0] < view["width"] and 0 <= pixel[1] < view["height"]):
                    raise ValueError("Measured pixel is outside the supplied camera image")
                ray = view["pose"][:3, :3].T @ np.linalg.solve(view["K"], [*pixel, 1.])
                denominator = float(ray @ normal)
                cosine = abs(denominator) / np.linalg.norm(ray)
                distance = float((origin - view["center"]) @ normal / denominator) if cosine > 1e-3 else -1.
                if cosine <= 1e-3 or distance <= 0:
                    reason = "grazing_or_behind_camera_intersection"
                    state = "missing"
                else:
                    delta = view["center"] + distance * ray - origin
                    point = [float(delta @ u), float(delta @ v)]
            values.append(point)
            states.append(state)
            reasons.append(reason)
        projected = {**row, "values": values, "sample_state": states, "units": "m",
                     "frame": plane["frame"], "residual_scale": residual_scale,
                     "source": row["source"] + "; ray/plane projection conditional on supplied geometry",
                     "observable": "plane_projection:" + row.get("observable", "image_point")}
        result["tracks"].append(projected)
        diagnostics.append({"id": row["id"], "reasons": reasons, "observed_samples": states.count("observed")})
    result["metric_evidence"] = copy.deepcopy(metric)
    result["projection_plane"] = copy.deepcopy(plane)
    result["projection_is_conditional"] = True
    result["projection_evidence"] = {"observations_sha256": _digest(observations),
                                     "request": copy.deepcopy(request)}
    return result, diagnostics


def prepare(request, base, output):
    base, output = Path(base), Path(output)
    path = base / request["observations"]
    observations, quality = project_to_plane(json.loads(path.read_text()), request)
    atomic_json(output / "observations.json", observations)
    atomic_json(output / "quality.json", quality)
    report = {"ok": True, "acceptance": False, "observations": "observations.json",
              "quality": "quality.json", "input_sha256": sha256_file(path), "preview_images": [],
              "metric_evidence": request["metric_evidence"],
              "warning": "Coordinates are conditional on camera, scale and the specified point-on-plane assumption; not recovered 3D ground truth."}
    atomic_json(output / "report.json", report)
    return report
