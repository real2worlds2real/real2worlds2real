"""Occlusion-aware support-plane votes from registered observed masks."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .artifacts import sha256_file
from .measurement import load_metric_points
from .posed_observations import file_path, load_views, project, write_report
from .support_geometry import complete_candidate, point_cluster_grid, refine_height


def recover(request, base, output):
    views, hashes = load_views(request, base, required=("mask",))
    depth_options = request.get('depth_consistency')
    if depth_options is not None:
        if not isinstance(depth_options, dict) or set(depth_options) - {'absolute_m', 'relative'}:
            raise ValueError('Depth consistency accepts absolute_m and relative tolerances')
        absolute = depth_options.get('absolute_m', .02)
        relative = depth_options.get('relative', .01)
        if (any(isinstance(x, bool) or not isinstance(x, (int, float))
                or not np.isfinite(x) or x < 0 for x in (absolute, relative))
                or absolute + relative <= 0):
            raise ValueError('Depth tolerances must be finite, nonnegative and not both zero')
        if len(views) < 2 or any('depth' not in view for view in views):
            raise ValueError('Depth consistency needs at least two registered views, each with depth')
    plane = request["plane"]
    origin, u, v = (np.asarray(plane[name], dtype=float) for name in ("origin_m", "u", "v"))
    if (
        any(a.shape != (3,) or not np.isfinite(a).all() for a in (origin, u, v))
        or not np.isclose(np.linalg.norm(u), 1)
        or not np.isclose(np.linalg.norm(v), 1)
        or abs(u @ v) > 1e-6
    ):
        raise ValueError("Support plane requires a finite origin and orthonormal u/v axes")
    bounds = np.asarray(plane["bounds_m"], dtype=float)
    step = float(plane.get("cell_m", 0.01))
    if (
        bounds.shape != (2, 2)
        or not np.isfinite(bounds).all()
        or not (bounds[1] > bounds[0]).all()
        or not np.isfinite(step)
        or step <= 0
    ):
        raise ValueError("Supply increasing plane-coordinate bounds and a positive cell size")
    cells = np.ceil((bounds[1] - bounds[0]) / step)
    if not np.isfinite(cells).all() or (cells > 1_000_000).any() or np.prod(cells) > 1_000_000:
        raise ValueError("Support grid exceeds one million cells; increase cell_m")
    shape = cells.astype(int)
    surface = None
    height_report = {"status": "supplied_plane", "applied": False}
    cluster_report = {"status": "not_requested"}
    if request.get("surface_points"):
        surface_path = file_path(base, request["surface_points"])
        hashes[str(surface_path)] = sha256_file(surface_path)
        surface = load_metric_points(surface_path)
        if request.get("plane_refinement"):
            origin, height_report = refine_height(
                surface, origin, u, v, bounds, request["plane_refinement"]
            )
    elif request.get("plane_refinement") or request.get("point_evidence"):
        raise ValueError(
            "Height refinement and point evidence require actual surface_points in the common frame"
        )
    # Keep every cell inside the requested bounds; the final cell may be narrower.
    x_edges = np.minimum(bounds[0, 0] + np.arange(shape[0] + 1) * step, bounds[1, 0])
    y_edges = np.minimum(bounds[0, 1] + np.arange(shape[1] + 1) * step, bounds[1, 1])
    x, y = np.meshgrid((x_edges[:-1] + x_edges[1:]) / 2, (y_edges[:-1] + y_edges[1:]) / 2)
    points = origin + x.reshape(-1, 1) * u + y.reshape(-1, 1) * v
    support_value = request.get("support_value", 255)
    occluder_value = request.get("occluder_value", 127)
    background_value = request.get("background_value", 0)
    labels = (support_value, occluder_value, background_value)
    if len(set(labels)) != 3 or any(type(n) is not int or not 0 <= n <= 255 for n in labels):
        raise ValueError("Use three distinct byte labels for support, occluder and background")
    support = np.zeros(len(points), dtype=np.uint16)
    background = np.zeros_like(support)
    occluded = np.zeros_like(support)
    depth_agreement = np.zeros_like(support)
    depth_foreground = np.zeros_like(support)
    depth_unknown = np.zeros_like(support)
    depth_free_space = np.zeros_like(support)
    for view in views:
        if not np.isin(view["mask"], labels).all():
            raise ValueError("Support masks must contain only the three declared labels")
        uv, projected_depth, valid = project(points, view)
        indices = np.flatnonzero(valid)
        pix = np.rint(uv[valid]).astype(int)
        values = view["mask"][pix[:, 1], pix[:, 0]]
        if "ignore" in view:
            values = values.copy()
            values[view["ignore"][pix[:, 1], pix[:, 0]] > 0] = occluder_value
        if depth_options is not None:
            measured = view['depth'][pix[:, 1], pix[:, 0]]
            tolerance = np.maximum(absolute, relative * measured)
            unknown = measured <= 0
            difference = projected_depth[valid] - measured
            foreground = ~unknown & (difference > tolerance)
            free_space = ~unknown & (difference < -tolerance)
            agreement = ~unknown & ~foreground & ~free_space
            depth_agreement[indices] += agreement
            depth_foreground[indices] += foreground
            depth_unknown[indices] += unknown
            depth_free_space[indices] += free_space
            # Ignore/occluder labels remain unknown even when depth disagrees.
            # A foreground occluder cannot disprove the plane behind it.
            values = values.copy()
            usable = values != occluder_value
            values[usable & free_space] = background_value
            values[unknown | foreground] = occluder_value
        support[indices] += values == support_value
        background[indices] += values == background_value
        occluded[indices] += values == occluder_value
    minimum = request.get("min_support_views", 2 if depth_options is not None else 1)
    min_evidence = request.get("min_evidence_views", 1)
    min_background = request.get("min_background_views", 1)
    fraction = request.get("max_background_fraction", 0.25)
    if (
        any(
            type(n) is not int or not 1 <= n <= len(views)
            for n in (minimum, min_evidence, min_background)
        )
        or not 0 <= fraction < 1
        or (depth_options is not None and minimum < 2)
    ):
        raise ValueError("Invalid vote thresholds")
    observed = support.astype(int) + background.astype(int)
    bg_fraction = background / np.maximum(observed, 1)
    rejected = (
        (observed >= min_evidence) & (background >= min_background) & (bg_fraction > fraction)
    )
    retained = (support >= minimum) & ~rejected
    state = np.zeros(len(points), np.uint8)
    state[rejected] = 2
    state[retained] = 1
    from pathlib import Path

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    shape = x.shape
    candidate = retained.reshape(shape).copy()
    if request.get("completion"):
        candidate = complete_candidate(
            candidate, observed.reshape(shape), occluded.reshape(shape), step, request["completion"]
        )
    if request.get("point_evidence") is not None:
        cluster, cluster_report = point_cluster_grid(
            surface, origin, u, v, x_edges, y_edges, request["point_evidence"]
        )
        if cluster is not None:
            candidate &= cluster
    inferred = candidate & (state.reshape(shape) != 1)
    edge_report = None
    if request.get("edge_refinement"):
        from .support_edges import refine_edges
        edge_bounds = bounds
        if candidate.any():
            iy, ix = np.nonzero(candidate)
            edge_bounds = [[x_edges[ix.min()], y_edges[iy.min()]],
                           [x_edges[ix.max() + 1], y_edges[iy.max() + 1]]]
        edge_report = refine_edges(views, origin, u, v, edge_bounds, request["edge_refinement"],
                                   support_value=support_value, occluder_value=occluder_value)
    np.savez_compressed(
        output / "footprint.npz",
        state=state.reshape(shape),
        support_votes=support.reshape(shape),
        background_votes=background.reshape(shape),
        occluded_views=occluded.reshape(shape),
        x_edges_m=x_edges,
        y_edges_m=y_edges,
        origin_m=origin,
        u=u,
        v=v,
        candidate_footprint=candidate,
        inferred_completion=inferred,
        depth_agreement_views=depth_agreement.reshape(shape),
        depth_foreground_views=depth_foreground.reshape(shape),
        depth_unknown_views=depth_unknown.reshape(shape),
        depth_free_space_views=depth_free_space.reshape(shape),
    )
    colors = np.array([[128, 128, 128], [58, 189, 126], [217, 80, 70]], dtype=np.uint8)
    Image.fromarray(colors[state.reshape(shape)][::-1]).save(output / "footprint.png")
    candidate_colors = np.zeros((*shape, 3), np.uint8)
    candidate_colors[candidate] = [58, 189, 126]
    candidate_colors[inferred] = [220, 170, 50]
    Image.fromarray(candidate_colors[::-1]).save(output / "candidate.png")
    area = np.diff(y_edges)[:, None] * np.diff(x_edges)[None, :]
    return write_report(
        output,
        {
            "schema_version": 1,
            "edge_refinement": edge_report,
            "ok": True,
            "operation": "support_footprint",
            "frame": request["frame"],
            "input_hashes": hashes,
            "grid": "footprint.npz",
            "preview_images": ["footprint.png", "candidate.png"],
            "plane_height": height_report,
            "point_cluster": cluster_report,
            "depth_consistency": None if depth_options is None else {
                'absolute_m': absolute, 'relative': relative,
                'min_support_views': minimum,
                'scope': 'Supplied camera-Z depth and labels; not independent depth measurement or camera recovery'},
            "candidate_cells": int(candidate.sum()),
            "inferred_completion_cells": int(inferred.sum()),
            "supported_cells": int(retained.sum()),
            "supported_area_m2": float(area.reshape(-1)[retained].sum()),
            "unknown_cells": int((state == 0).sum()),
            "rejected_cells": int((state == 2).sum()),
            "limits": [
                "Occluders do not vote against the support.",
                "Unobserved regions stay unknown; no hidden geometry is invented.",
                "Plane height/pose and metric accuracy come from supplied evidence.",
                "This grid is evidence for modeling, not a collision mesh or simulator acceptance.",
            ],
        },
    )
