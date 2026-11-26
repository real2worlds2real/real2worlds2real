"""Explicit finite static box colliders shared by liquid export and execution."""

from __future__ import annotations

import math


def static_colliders(configuration):
    rows = configuration.get("static_colliders", [])
    if not isinstance(rows, list) or len(rows) > 200:
        raise ValueError("static_colliders must be a bounded list")
    result, ids = [], set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "size_m", "center_m"}:
            raise ValueError("A static box needs id, size_m and center_m")
        name = row["id"]
        if not isinstance(name, str) or not name.strip() or name in ids:
            raise ValueError("Static collider IDs must be unique nonempty strings")
        ids.add(name)
        for key in ("size_m", "center_m"):
            values = row[key]
            if (
                not isinstance(values, list)
                or len(values) != 3
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
            ):
                raise ValueError("Static box vectors need three finite numbers")
        if min(row["size_m"]) <= 0:
            raise ValueError("Static box extents must be positive")
        result.append(row)
    if type(configuration.get("use_ground_plane", True)) is not bool:
        raise ValueError("use_ground_plane must be a boolean")
    return result


def audit_static_boxes(points, boxes, *, tolerance_m):
    """Sampled point/box penetration, not a contact-force certificate."""
    import numpy as np

    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if not np.isfinite(points).all() or not math.isfinite(tolerance_m) or tolerance_m < 0:
        raise ValueError("Finite points and nonnegative tolerance are required")
    checks = []
    for box in boxes:
        q = np.abs(points - np.asarray(box["center_m"])) - np.asarray(box["size_m"]) / 2
        distance = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)
        depth = float(max(0.0, -distance.min())) if len(points) else 0.0
        checks.append(
            {"id": box["id"], "maximum_sampled_penetration_m": depth, "ok": depth <= tolerance_m}
        )
    return {
        "ok": all(row["ok"] for row in checks),
        "boxes": checks,
        "tolerance_m": tolerance_m,
        "scope": "Signed-distance check of recorded particle centers; not surface-shell, subframe or contact-force certification",
    }
