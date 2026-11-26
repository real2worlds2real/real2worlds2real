"""Trajectory diagnostics shared by Blender simulation and ordinary Python tests."""

from __future__ import annotations

import numpy as np


def cloth_stretch_check(metrics, limit=2.0):
    """A deliberately loose cloth sanity bound, not a calibrated fabric model."""
    limit = float(limit)
    if not np.isfinite(limit) or limit <= 1:
        raise ValueError("Cloth stretch limit must be finite and greater than one")
    ratio = metrics.get("edge_length_ratio_p95")
    return {"ok": ratio is not None and ratio <= limit, "edge_length_ratio_p95": ratio,
            "limit": limit, "meaning": "95th-percentile edge extension sanity bound; no real-world material calibration claim"}


def trajectory_metrics(points, edges):
    """Separate genuine shape change from an object's rigid translation/rotation.

    These are numerical diagnostics, not a material calibration or contact proof.
    All vertices participate in the rigid fit; edges refer to the same control mesh.
    """
    points = np.asarray(points, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if points.ndim != 3 or points.shape[0] < 3 or points.shape[1] < 4 or points.shape[2] != 3:
        raise ValueError("Expected at least three frames of a fixed mesh with four vertices")
    if not np.isfinite(points).all():
        return {"ok": False, "error": "Non-finite simulated vertex coordinates"}
    reference = points[0]
    diagonal = float(np.linalg.norm(np.ptp(reference, axis=0)))
    if diagonal <= 1e-8:
        return {"ok": False, "error": "Degenerate reference mesh"}
    centered = reference - reference.mean(axis=0)
    residuals, displacements = [], []
    rest_lengths = np.linalg.norm(reference[edges[:, 0]] - reference[edges[:, 1]], axis=1)
    usable = rest_lengths > diagonal * 1e-8
    ratios = []
    usable_edges = np.flatnonzero(usable)
    worst_stretch = None
    largest_ratio = -1.0
    for sample_index, current in enumerate(points):
        target = current - current.mean(axis=0)
        left, _, right = np.linalg.svd(centered.T @ target)
        correction = np.eye(3)
        correction[-1, -1] = np.linalg.det(left @ right)
        rotation = left @ correction @ right
        residuals.append(float(np.sqrt(np.mean(np.sum((centered @ rotation - target) ** 2, axis=1)))))
        displacements.append(float(np.linalg.norm(current - reference, axis=1).max()))
        lengths = np.linalg.norm(current[edges[:, 0]] - current[edges[:, 1]], axis=1)
        current_ratios = lengths[usable] / rest_lengths[usable]
        ratios.extend(current_ratios.tolist())
        if len(current_ratios) and float(current_ratios.max()) > largest_ratio:
            largest_ratio = float(current_ratios.max())
            edge_index = usable_edges[int(current_ratios.argmax())]
            vertices = edges[edge_index]
            worst_stretch = {
                "sample_index": sample_index,
                "edge_vertex_indices": vertices.tolist(),
                "reference_length_m": float(rest_lengths[edge_index]),
                "current_length_m": float(lengths[edge_index]),
                "reference_midpoint_m": reference[vertices].mean(axis=0).tolist(),
                "current_midpoint_m": current[vertices].mean(axis=0).tolist(),
            }
    shape_change = max(residuals) / diagonal
    bounded = max(displacements) < diagonal * 50 and float(np.abs(points).max()) < 1000
    return {
        "ok": bounded and shape_change > 1e-4,
        "finite": True,
        "bounded": bounded,
        "nonrigid_motion": shape_change > 1e-4,
        "reference_diagonal_m": diagonal,
        "max_displacement_m": max(displacements),
        "nonrigid_rms_m_by_sample": residuals,
        "max_nonrigid_rms_relative_to_diagonal": shape_change,
        "edge_length_ratio_min": float(np.min(ratios)) if ratios else None,
        "edge_length_ratio_p95": float(np.percentile(ratios, 95)) if ratios else None,
        "edge_length_ratio_p99": float(np.percentile(ratios, 99)) if ratios else None,
        "edge_length_ratio_max": float(np.max(ratios)) if ratios else None,
        "worst_stretch": worst_stretch,
        "warnings": ["Edge stretch exceeds 2x the initial length; inspect the reported edge location and rest length"] if largest_ratio > 2 else [],
        "error": None if bounded and shape_change > 1e-4 else (
            "Unbounded motion" if not bounded else "Only rigid motion or negligible shape change observed"
        ),
    }
