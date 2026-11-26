"""Source-matched rendered visibility and depth-supported Gaussian cleanup."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .artifacts import sha256_file
from .gaussian_background import _matrix_columns, read_gaussians, similarity
from .posed_observations import file_path, load_views, project, write_report


def depth_evidence(points, views, *, near_ratio=.85, far_ratio=1.1, absolute_m=.03):
    if not np.isfinite([near_ratio, far_ratio, absolute_m]).all() or not 0 < near_ratio < 1 <= far_ratio <= 2 or absolute_m < 0:
        raise ValueError("Invalid free-space depth tolerances")
    seen, front = np.zeros(len(points), np.int32), np.zeros(len(points), np.int32)
    for view in views:
        if "depth" not in view:
            continue
        for start in range(0, len(points), 65536):
            uv, z, valid = project(points[start:start + 65536], view)
            pixel = np.rint(np.clip(uv, [0, 0], [view["width"] - 1, view["height"] - 1])).astype(int)
            observed = view["depth"][pixel[:, 1], pixel[:, 0]]
            valid &= observed > 0
            if "ignore" in view:
                valid &= view["ignore"][pixel[:, 1], pixel[:, 0]] == 0
            # Behind the model surface is occlusion, not free-space evidence.
            visible = valid & (z <= observed * far_ratio + absolute_m)
            empty = visible & (z < observed * near_ratio - absolute_m)
            seen[start:start + len(z)] += visible
            front[start:start + len(z)] += empty
    return seen, front


def filter_visibility(request, base, output):
    from plyfile import PlyData, PlyElement
    views, hashes = load_views(request, base)
    source = file_path(base, request["gaussians"])
    digest = sha256_file(source)
    hashes[str(source)] = digest
    vertices = read_gaussians(source)
    matrix, _ = similarity(request.get("source_to_frame", np.eye(4).tolist()))
    points = _matrix_columns(vertices, ["x", "y", "z"])
    points = points @ matrix[:3, :3].T + matrix[:3, 3]
    seen, front = depth_evidence(points, views, **request.get("depth_tolerances", {}))
    counts, render_views = np.zeros(len(points), np.int32), 0
    for row in request["views"]:
        if "particle_visibility" not in row:
            continue
        if row.get("gaussian_source_sha256") != digest:
            raise ValueError("Renderer visibility must identify this exact Gaussian source/order")
        path = file_path(base, row["particle_visibility"])
        hashes[str(path)] = sha256_file(path)
        visibility = np.load(path, allow_pickle=False)
        if visibility.shape != (len(points),) or not np.isfinite(visibility).all() or (visibility < 0).any():
            raise ValueError("Renderer visibility must be one finite nonnegative value per source particle")
        counts += visibility > 0
        render_views += 1
    options = request.get("filter", {})
    min_drawn = options.get("min_drawn_views", 0)
    minimum = options.get("min_depth_views", 5)
    fraction = options.get("max_free_space_fraction")
    if type(min_drawn) is not int or not 0 <= min_drawn <= render_views or type(minimum) is not int or minimum < 1:
        raise ValueError("Invalid visibility/depth view thresholds; supply actual renderer evidence")
    if fraction is not None and (not np.isfinite(fraction) or not 0 <= fraction <= 1 or not any("depth" in v for v in views)):
        raise ValueError("Free-space filtering requires depth evidence and a fraction from 0 to 1")
    if min_drawn == 0 and fraction is None:
        raise ValueError("Explicitly choose a visibility or free-space filter")
    ratio = front / np.maximum(seen, 1)
    undrawn = counts < min_drawn
    floaters = (seen >= minimum) & (ratio > fraction) if fraction is not None else np.zeros(len(points), bool)
    keep = ~(undrawn | floaters)
    if not keep.any():
        raise ValueError("Cleanup would remove every Gaussian; inspect evidence and thresholds")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertices[keep].copy(), "vertex")], text=False).write(str(output / "filtered.ply"))
    np.savez_compressed(output / "visibility.npz", draw_count=counts, depth_view_count=seen,
                        free_space_count=front, free_space_fraction=ratio, kept=keep)
    if sha256_file(source) != digest:
        raise ValueError("Gaussian source changed during cleanup")
    return write_report(output, {"schema_version": 1, "ok": True, "operation": "gaussian_visibility",
        "frame": request["frame"], "input_hashes": hashes, "source_sha256": digest,
        "source_ply": "filtered.ply", "particle_evidence": "visibility.npz", "preview_images": [],
        "counts": {"source": len(points), "kept": int(keep.sum()), "undrawn": int(undrawn.sum()),
                   "free_space": int(floaters.sum()), "render_views": render_views},
        "limits": ["Depths rendered by the current Gaussian model are estimates, not real surface measurements.",
                   "Frustum inclusion is not renderer visibility; particle order must match the hashed source.",
                   "Filtered geometry can expose holes; inspect and prepare/export it before scene acceptance."]})
