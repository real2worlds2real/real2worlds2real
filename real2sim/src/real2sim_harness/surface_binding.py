"""Triangle attachment for a detailed render mesh driven by a cloth control mesh."""

from __future__ import annotations

import numpy as np


def triangle_frames(triangles):
    first = triangles[:, 1] - triangles[:, 0]
    normal = np.cross(first, triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(first, axis=-1)
    areas = np.linalg.norm(normal, axis=-1)
    if np.any(lengths < 1e-10) or np.any(areas < 1e-12):
        raise ValueError("Cannot bind or render a degenerate control triangle")
    first = first / lengths[:, None]
    normal = normal / areas[:, None]
    second = np.cross(normal, first)
    return np.stack([first, second, normal], axis=1)


def closest_weights(triangles, points):
    """Closest point on each paired triangle, with barycentric coordinates."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    e0, e1, rel = b - a, c - a, points - a
    d00 = np.einsum("ij,ij->i", e0, e0)
    d01 = np.einsum("ij,ij->i", e0, e1)
    d11 = np.einsum("ij,ij->i", e1, e1)
    d20 = np.einsum("ij,ij->i", rel, e0)
    d21 = np.einsum("ij,ij->i", rel, e1)
    denom = d00 * d11 - d01**2
    v = (d11 * d20 - d01 * d21) / np.maximum(denom, 1e-30)
    w = (d00 * d21 - d01 * d20) / np.maximum(denom, 1e-30)
    projected = np.stack([1 - v - w, v, w], axis=1)
    options = [projected]
    for i, j in ((0, 1), (1, 2), (2, 0)):
        edge = triangles[:, j] - triangles[:, i]
        t = np.einsum("ij,ij->i", points - triangles[:, i], edge)
        t = np.clip(t / np.maximum(np.einsum("ij,ij->i", edge, edge), 1e-30), 0, 1)
        weights = np.zeros_like(projected)
        weights[:, i], weights[:, j] = 1 - t, t
        options.append(weights)
    options = np.stack(options, axis=1)
    locations = np.einsum("nki,nij->nkj", options, triangles)
    distance = np.sum((locations - points[:, None]) ** 2, axis=-1)
    distance[(projected < -1e-7).any(axis=1) | (denom < 1e-20), 0] = np.inf
    best = np.argmin(distance, axis=1)
    return options[np.arange(len(points)), best], distance[np.arange(len(points)), best]


def build_binding(control_points, faces, visual_points, candidates=16):
    control_points, visual_points = np.asarray(control_points), np.asarray(visual_points)
    faces = np.asarray(faces, dtype=np.int32)
    all_triangles = control_points[faces]
    areas = np.linalg.norm(np.cross(all_triangles[:, 1] - all_triangles[:, 0], all_triangles[:, 2] - all_triangles[:, 0]), axis=-1)
    valid_ids = np.flatnonzero(areas > 1e-12)
    if not len(valid_ids):
        raise ValueError("The control mesh has no valid triangles")
    triangles = all_triangles[valid_ids]
    count = min(candidates, len(triangles))
    if len(triangles) > count:
        from scipy.spatial import cKDTree
        tree = cKDTree(triangles.mean(axis=1))
    ids, all_weights, all_offsets = [], [], []
    for start in range(0, len(visual_points), 8192):
        points = visual_points[start:start + 8192]
        if len(triangles) > count:
            _, candidates_idx = tree.query(points, k=count)
            candidates_idx = np.asarray(candidates_idx).reshape(len(points), count)
        else:
            candidates_idx = np.tile(np.arange(count), (len(points), 1))
        candidate_triangles = triangles[candidates_idx].reshape(-1, 3, 3)
        weights, distances = closest_weights(candidate_triangles, np.repeat(points, count, axis=0))
        weights = weights.reshape(-1, count, 3)
        distances = distances.reshape(-1, count)
        best = distances.argmin(axis=1)
        selected = candidates_idx[np.arange(len(points)), best]
        weights = weights[np.arange(len(points)), best]
        chosen = triangles[selected]
        centers = np.einsum("ni,nij->nj", weights, chosen)
        frames = triangle_frames(chosen)
        offsets = np.einsum("nij,nj->ni", frames, points - centers)
        ids.append(valid_ids[selected])
        all_weights.append(weights)
        all_offsets.append(offsets)
    return {
        "triangle_ids": np.concatenate(ids).astype(np.int32),
        "weights": np.concatenate(all_weights).astype(np.float32),
        "offsets": np.concatenate(all_offsets).astype(np.float32),
    }


def deform_surface(control_points, faces, binding):
    triangles = np.asarray(control_points)[np.asarray(faces)[binding["triangle_ids"]]]
    centers = np.einsum("ni,nij->nj", binding["weights"], triangles)
    frames = triangle_frames(triangles)
    return centers + np.einsum("ni,nij->nj", binding["offsets"], frames)
