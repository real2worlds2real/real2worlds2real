"""Observed landmark/board similarity registration of a complete reconstruction."""
from __future__ import annotations

import numpy as np

from .artifacts import sha256_file
from .posed_observations import file_path, load_views, write_report


def similarity_fit(source, target):
    source, target = np.asarray(source, float), np.asarray(target, float)
    if (source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3
            or len(source) < 3 or not np.isfinite(source).all() or not np.isfinite(target).all()):
        raise ValueError("Use at least three finite paired 3D landmarks")
    a, b = source - source.mean(0), target - target.mean(0)
    if min(np.linalg.matrix_rank(a), np.linalg.matrix_rank(b)) < 2:
        raise ValueError("Registration landmarks must not be collinear")
    left, singular, right = np.linalg.svd(b.T @ a / len(a))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(left @ right)
    rotation = left @ correction @ right
    scale = float((singular * np.diag(correction)).sum() / (a * a).sum(axis=1).mean())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Landmarks do not admit a positive similarity")
    matrix = np.eye(4)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = target.mean(0) - scale * rotation @ source.mean(0)
    return matrix, scale


def robust_similarity(source, target, *, tolerance_m=.005, trials=256):
    source, target = np.asarray(source, float), np.asarray(target, float)
    similarity_fit(source, target)  # Validate before sampling.
    if not np.isfinite(tolerance_m) or tolerance_m <= 0 or type(trials) is not int or not 1 <= trials <= 4096:
        raise ValueError("Use positive metric tolerance and 1..4096 trials")
    rng, best, best_key = np.random.default_rng(0), None, None
    for _ in range(trials):
        indices = rng.choice(len(source), 3, replace=False)
        try:
            matrix, _ = similarity_fit(source[indices], target[indices])
        except ValueError:
            continue
        residual = np.linalg.norm(source @ matrix[:3, :3].T + matrix[:3, 3] - target, axis=1)
        keep = residual <= tolerance_m
        key = (int(keep.sum()), -float(np.median(residual[keep])) if keep.any() else -np.inf)
        if best_key is None or key > best_key:
            best_key, best = key, keep
    if best is None or best.sum() < max(3, int(np.ceil(len(source) * .5))):
        raise ValueError("No consistent majority of metric landmarks")
    matrix, scale = similarity_fit(source[best], target[best])
    residual = np.linalg.norm(source @ matrix[:3, :3].T + matrix[:3, 3] - target, axis=1)
    keep = residual <= tolerance_m
    if keep.sum() < max(3, int(np.ceil(len(source) * .5))):
        raise ValueError("Refitted metric landmarks lost majority consistency")
    matrix, scale = similarity_fit(source[keep], target[keep])
    residual = np.linalg.norm(source @ matrix[:3, :3].T + matrix[:3, 3] - target, axis=1)
    keep = residual <= tolerance_m
    if keep.sum() < max(3, int(np.ceil(len(source) * .5))):
        raise ValueError("Final metric registration has insufficient consistent landmarks")
    return matrix, scale, residual, keep


def board_landmarks(views, board, *, max_reprojection_px=2., min_ray_angle_deg=.5):
    from .calibration import detect_board
    observations, rejected = {}, []
    for i, view in enumerate(views):
        k = view["K"]
        camera = {"image_size": [view["width"], view["height"]], "intrinsics":
                  {"fx": k[0, 0], "fy": k[1, 1], "cx": k[0, 2], "cy": k[1, 2]}}
        try:
            hit = detect_board(view["image_path"], board, camera)
        except ValueError as error:
            rejected.append({"view": i, "reason": str(error)})
            continue
        for identifier, pixel, point in zip(hit["corner_ids"], hit["pixels"], hit["points_m"], strict=True):
            row = observations.setdefault(identifier, {"metric": point, "sightings": []})
            row["sightings"].append((i, np.asarray(pixel, float)))
    source, target, details = [], [], []
    for identifier, row in observations.items():
        if len(row["sightings"]) < 2:
            continue
        equations = []
        for i, (u, v) in row["sightings"]:
            p = views[i]["K"] @ views[i]["pose"][:3]
            equations.extend([u * p[2] - p[0], v * p[2] - p[1]])
        homogeneous = np.linalg.svd(equations)[2][-1]
        if abs(homogeneous[3]) < 1e-12:
            continue
        point = homogeneous[:3] / homogeneous[3]
        errors, rays, positive = [], [], True
        for i, pixel in row["sightings"]:
            view = views[i]
            q = view["pose"][:3, :3] @ point + view["pose"][:3, 3]
            positive &= q[2] > 0
            predicted = view["K"] @ q
            errors.append(float(np.linalg.norm(predicted[:2] / max(predicted[2], 1e-12) - pixel)))
            ray = point - view["center"]
            rays.append(ray / max(np.linalg.norm(ray), 1e-12))
        angle = float(np.degrees(np.arccos(np.clip(np.asarray(rays) @ np.asarray(rays).T, -1, 1))).max())
        accepted = bool(positive and max(errors) <= max_reprojection_px and angle >= min_ray_angle_deg)
        details.append({"corner_id": identifier, "views": [i for i, _ in row["sightings"]],
                        "residual_px": errors, "ray_angle_deg": angle, "used": accepted})
        if accepted:
            source.append(point)
            target.append(row["metric"])
    if len(source) < 6:
        raise ValueError("Need six board corners triangulated with useful baseline and consistent projections")
    return np.asarray(source), np.asarray(target), {"corners": details, "rejected_views": rejected}


def register(request, base, output):
    views, hashes = load_views(request, base, required=("image",) if "board" in request else (),
                               min_views=1 if "board" in request else 0)
    diagnostics = {}
    if ("board" in request) == ("correspondences" in request):
        raise ValueError("Supply either an observed board or explicit paired 3D landmarks")
    if "board" in request:
        source, target, diagnostics = board_landmarks(views, request["board"])
    else:
        evidence = request["correspondences"]
        if not isinstance(evidence.get("source"), str) or not evidence["source"].strip():
            raise ValueError("Describe independent metric landmark provenance")
        source, target = np.asarray(evidence["reconstruction_points"], float), np.asarray(evidence["metric_points_m"], float)
    matrix, scale, residual, keep = robust_similarity(source, target, **request.get("registration", {}))
    rotation, translation = matrix[:3, :3] / scale, matrix[:3, 3]
    registered = []
    for view in views:
        pose = np.eye(4)
        pose[:3, :3] = view["pose"][:3, :3] @ rotation.T
        pose[:3, 3] = scale * view["pose"][:3, 3] - pose[:3, :3] @ translation
        registered.append({"world_to_camera": pose.tolist(), "intrinsics": view["K"].tolist(),
                           "image_size": [view["width"], view["height"]],
                           "source": "Metric registration of " + view["source"]})
    from pathlib import Path
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cloud = None
    if request.get("surface_points"):
        from .measurement import load_metric_points
        path = file_path(base, request["surface_points"])
        hashes[str(path)] = sha256_file(path)
        points = load_metric_points(path)
        np.save(output / "registered_points_m.npy", points @ matrix[:3, :3].T + translation)
        cloud = "registered_points_m.npy"
    return write_report(output, {"schema_version": 1, "ok": True, "operation": "metric_registration",
        "frame": request["frame"], "input_hashes": hashes, "source_to_metric": matrix.tolist(),
        "metres_per_source_unit": scale, "registered_cameras": registered, "registered_points": cloud,
        "landmark_residual_m": residual.tolist(), "inliers": keep.tolist(),
        "inlier_rms_m": float(np.sqrt(np.mean(residual[keep] ** 2))), "board_diagnostics": diagnostics,
        "preview_images": [], "limits": ["Board registration is local to observed evidence; reconstruction drift elsewhere remains possible.",
            "Original geometry and camera files are unchanged. Apply the transform once to all participants.",
            "Low residual does not certify unseen shape or calibrated material parameters."]})
