"""Optional observed-plane diagnostics and explicit footprint completion hypotheses."""

from __future__ import annotations

import cv2
import numpy as np


def refine_height(points, origin, u, v, bounds, settings):
    normal = np.cross(u, v)
    relative = points - origin
    uv = np.column_stack([relative @ u, relative @ v])
    distances = relative @ normal
    reach, bin_size = float(settings.get("search_m", 0.3)), float(settings.get("bin_m", 0.005))
    minimum = settings.get("min_points", 100)
    if (
        not np.isfinite([reach, bin_size]).all()
        or not 0 < bin_size < reach
        or reach / bin_size > 10000
        or type(minimum) is not int
        or minimum < 10
    ):
        raise ValueError("Invalid support height search/bin/min_points")
    keep = (
        (uv >= bounds[0]).all(axis=1) & (uv <= bounds[1]).all(axis=1) & (np.abs(distances) <= reach)
    )
    samples = distances[keep]
    result = {
        "method": "observed point density along the supplied plane normal",
        "selected_points": len(samples),
        "applied": False,
        "ambiguous_layers": False,
    }
    if len(samples) < minimum:
        return origin, {**result, "status": "insufficient_points"}
    edges = np.linspace(-reach, reach, int(np.ceil(2 * reach / bin_size)) + 1)
    histogram, edges = np.histogram(samples, bins=edges)
    peak = int(histogram.argmax())
    in_peak = (samples >= edges[peak]) & (samples <= edges[peak + 1])
    shift = float(np.median(samples[in_peak]))
    centers = (edges[:-1] + edges[1:]) / 2
    far = np.abs(centers - shift) > max(0.03, 3 * bin_size)
    ratio = float(histogram[far].max() / max(histogram[peak], 1)) if far.any() else 0.0
    result.update(
        status="estimated",
        proposed_shift_m=shift,
        dominant_points=int(histogram[peak]),
        secondary_peak_fraction=ratio,
        ambiguous_layers=ratio > 0.15,
    )
    if settings.get("apply", False) and not result["ambiguous_layers"]:
        result["applied"] = True
        return origin + shift * normal, result
    return origin, result


def point_cluster_grid(points, origin, u, v, x_edges, y_edges, settings):
    band, block = float(settings.get("band_m", 0.015)), float(settings.get("block_m", 0.05))
    if not np.isfinite([band, block]).all() or band <= 0 or block <= 0:
        raise ValueError("Point footprint band/block sizes must be positive")
    normal = np.cross(u, v)
    relative = points - origin
    uv = np.column_stack([relative @ u, relative @ v])
    near = np.abs(relative @ normal) <= band
    low = np.array([x_edges[0], y_edges[0]])
    high = np.array([x_edges[-1], y_edges[-1]])
    near &= (uv >= low).all(axis=1) & (uv <= high).all(axis=1)
    shape = np.ceil((high - low) / block) + 1
    if (shape > 2000).any() or np.prod(shape) > 1_000_000:
        raise ValueError("Point evidence grid too large; increase block_m")
    grid = np.zeros(tuple(shape.astype(int)[::-1]), np.uint8)
    ij = np.floor((uv[near] - low) / block).astype(int)
    if len(ij) < 3:
        return None, {"status": "insufficient_points", "near_plane_points": len(ij)}
    grid[ij[:, 1], ij[:, 0]] = 1
    connected = cv2.dilate(grid, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(connected, connectivity=8)
    chosen = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if settings.get("seed_uv_m") is not None:
        seed = np.asarray(settings["seed_uv_m"], dtype=float)
        if (
            seed.shape != (2,)
            or not np.isfinite(seed).all()
            or (seed < low).any()
            or (seed > high).any()
        ):
            raise ValueError("seed_uv_m must lie in the support bounds")
        seed_cell = np.floor((seed - low) / block).astype(int)
        candidate = labels[seed_cell[1], seed_cell[0]]
        if candidate:
            chosen = int(candidate)
    component = labels == chosen
    # Convex completion is an explicit modeling hypothesis, separate from raw mask votes.
    hull = np.zeros_like(grid)
    contour = np.argwhere(component)[:, ::-1].astype(np.int32)
    cv2.fillConvexPoly(hull, cv2.convexHull(contour), 1)
    x = (x_edges[:-1] + x_edges[1:]) / 2
    y = (y_edges[:-1] + y_edges[1:]) / 2
    gx, gy = (
        np.floor((axis - start) / block).astype(int) for axis, start in ((x, low[0]), (y, low[1]))
    )
    return hull[np.ix_(gy, gx)] > 0, {
        "status": "estimated_cluster_hull",
        "near_plane_points": len(ij),
        "components": count - 1,
        "selected_component": chosen,
        "interpretation": "inferred hull; not a measured closed boundary",
    }


def complete_candidate(observed, support, occluded, step, settings):
    candidate = observed.copy().astype(np.uint8)
    for operation, name in ((cv2.MORPH_CLOSE, "close_radius_m"), (cv2.MORPH_OPEN, "open_radius_m")):
        radius = float(settings.get(name, 0))
        if not np.isfinite(radius) or radius < 0 or radius / step > 100:
            raise ValueError("Completion radii must be nonnegative and at most 100 cells")
        if radius:
            size = 2 * int(np.ceil(radius / step)) + 1
            candidate = cv2.morphologyEx(candidate, operation, np.ones((size, size), np.uint8))
    extension = float(settings.get("extend_under_occlusion_m", 0))
    minimum = settings.get("min_occluded_views", 3)
    fraction = float(settings.get("occlusion_fraction", 0.6))
    if (
        not np.isfinite(extension)
        or not 0 <= extension / step <= 100
        or type(minimum) is not int
        or minimum < 1
        or not 0 < fraction <= 1
    ):
        raise ValueError("Invalid occluded-region completion limits")
    eligible = (occluded >= minimum) & (
        occluded / np.maximum(occluded.astype(float) + support, 1) >= fraction
    )
    for _ in range(int(np.ceil(extension / step))):
        grown = (cv2.dilate(candidate, np.ones((3, 3), np.uint8)) > 0) & (
            eligible | (candidate > 0)
        )
        if np.array_equal(grown, candidate > 0):
            break
        candidate = grown.astype(np.uint8)
    return candidate > 0
