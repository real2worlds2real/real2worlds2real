"""Continuous centerline binding for editable rod/cable surface meshes."""

from __future__ import annotations

import numpy as np


def quaternion_matrices(quaternions):
    q = np.asarray(quaternions, dtype=float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def material_frames(quaternions, segments, fractions):
    """Blend neighboring material frames, keeping quaternion hemispheres consistent."""
    count = len(quaternions)
    station = np.clip(np.asarray(segments) + np.asarray(fractions) - 0.5, 0, count - 1)
    first = np.floor(station).astype(int)
    second = np.minimum(first + 1, count - 1)
    alpha = station - first
    a, b = np.asarray(quaternions)[first], np.asarray(quaternions)[second].copy()
    b[np.sum(a * b, axis=1) < 0] *= -1
    return quaternion_matrices(a * (1 - alpha[:, None]) + b * alpha[:, None])


def bind_vertices(vertices, points, quaternions):
    vertices, points = np.asarray(vertices), np.asarray(points)
    edge = np.diff(points, axis=0)
    length2 = np.sum(edge * edge, axis=1)
    if np.min(length2) < 1e-12:
        raise ValueError("Strand control points must be distinct")
    # Bounded batches avoid a dense high-detail-mesh x segment allocation.
    rows = []
    for start in range(0, len(vertices), 2048):
        v = vertices[start : start + 2048]
        fractions = np.clip(
            np.einsum("vsi,si->vs", v[:, None, :] - points[:-1], edge) / length2, 0, 1
        )
        delta = v[:, None, :] - (points[:-1] + fractions[:, :, None] * edge)
        segment = np.argmin(np.sum(delta * delta, axis=2), axis=1)
        u = fractions[np.arange(len(v)), segment]
        center = points[segment] + u[:, None] * edge[segment]
        rotation = material_frames(quaternions, segment, u)
        offset = np.einsum("vji,vj->vi", rotation, v - center)
        rows.append((segment, u, offset))
    return {
        "segment": np.concatenate([r[0] for r in rows]),
        "fraction": np.concatenate([r[1] for r in rows]),
        "offset": np.concatenate([r[2] for r in rows]),
    }


def deform_vertices(binding, points, quaternions):
    segment, u = binding["segment"], binding["fraction"]
    points = np.asarray(points)
    center = points[segment] * (1 - u[:, None]) + points[segment + 1] * u[:, None]
    return center + np.einsum(
        "vij,vj->vi", material_frames(quaternions, segment, u), binding["offset"]
    )
