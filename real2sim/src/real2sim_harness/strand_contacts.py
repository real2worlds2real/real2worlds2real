"""Audit retained capsule contacts independently of solver success flags."""

from __future__ import annotations

import numpy as np

from .strand_geometry import quaternion_matrices


def segment_distances(p0, p1, q0, q1):
    """Closest distances for corresponding finite-segment pairs, including parallel pairs."""
    u, v, w = p1 - p0, q1 - q0, p0 - q0
    a = np.sum(u * u, axis=-1)
    b = np.sum(u * v, axis=-1)
    c = np.sum(v * v, axis=-1)
    d = np.sum(u * w, axis=-1)
    e = np.sum(v * w, axis=-1)
    if np.any(a <= 1e-20) or np.any(c <= 1e-20):
        raise ValueError("Capsule axes must have positive length")
    denominator = a * c - b * b
    parallel = denominator <= 1e-12 * a * c
    s = np.clip(np.divide(b * e - c * d, denominator, out=np.zeros_like(a), where=~parallel), 0, 1)
    t = (b * s + e) / c
    below, above = t < 0, t > 1
    s = np.where(below, np.clip(-d / a, 0, 1), s)
    s = np.where(above, np.clip((b - d) / a, 0, 1), s)
    t = np.clip(t, 0, 1)
    return np.linalg.norm(w + s[..., None] * u - t[..., None] * v, axis=-1)


def audit_capsules(poses, runtime):
    c = runtime["configuration"]
    ids = []
    half = []
    radii = []
    names = []
    excluded = set()
    for rod in runtime["rods"]:
        begin = len(ids)
        for i, body in enumerate(rod["body_ids"]):
            ids.append(body)
            half.append(rod["lengths"][i] / 2)
            radii.append(rod["specification"]["radius"])
            names.append(f"{rod['specification']['name']}:{i}")
        excluded.update((begin + i, begin + i + 1) for i in range(len(rod["body_ids"]) - 1))
    for b in c["bodies"]:
        index = len(ids)
        body = runtime["bodies"][b["name"]]["body"]
        ids.append(body)
        half.append(b["half_height"])
        radii.append(b["radius"])
        names.append(b["name"])
        for rod in runtime["rods"]:
            for anchor in rod["specification"]["attachments"]:
                if anchor["body"] != b["name"]:
                    continue
                neighbor = rod["body_ids"][:2] if anchor["node"] == 0 else rod["body_ids"][-2:]
                excluded.update((ids.index(owner), index) for owner in neighbor)
    first, second = np.triu_indices(len(ids), k=1)
    selected = np.array(
        [(int(a), int(b)) not in excluded for a, b in zip(first, second, strict=True)]
    )
    first, second = first[selected], second[selected]
    radius = np.asarray(radii)
    radius_sum = radius[first] + radius[second]
    tolerance = np.clip(0.2 * radius_sum, 0.00025, 0.001)
    worst = 0.0
    floor_worst = 0.0
    failures = []
    obstacle_worst = 0.0
    obstacle_failures = []
    for frame, all_poses in enumerate(poses):
        q = all_poses[ids]
        offset = quaternion_matrices(q[:, 3:])[:, :, 2] * np.asarray(half)[:, None]
        starts, ends = q[:, :3] - offset, q[:, :3] + offset
        floor = float(
            np.min(np.minimum(starts[:, 2], ends[:, 2]) - radius - c["environment"]["floor_height"])
        )
        floor_worst = min(floor_worst, floor)
        for fixture in c["environment"].get("capsule_obstacles", []):
            center = np.asarray(fixture["center_m"])
            offset = np.asarray(fixture["axis"]) * fixture["half_height_m"]
            other0 = np.broadcast_to(center - offset, starts.shape)
            other1 = np.broadcast_to(center + offset, ends.shape)
            clearance = (
                segment_distances(starts, ends, other0, other1) - radius - fixture["radius_m"]
            )
            j = int(np.argmin(clearance))
            obstacle_worst = min(obstacle_worst, float(clearance[j]))
            if clearance[j] < -0.001 and len(obstacle_failures) < 16:
                obstacle_failures.append(
                    {
                        "frame": frame,
                        "capsule": names[j],
                        "obstacle": fixture["id"],
                        "clearance_m": float(clearance[j]),
                    }
                )
        if len(first):
            distance = segment_distances(starts[first], ends[first], starts[second], ends[second])
            clearance = distance - radius_sum
            index = int(np.argmin(clearance))
            worst = min(worst, float(clearance[index]))
            bad = np.flatnonzero(clearance < -tolerance)
            if len(bad) and len(failures) < 16:
                j = int(bad[np.argmin(clearance[bad])])
                failures.append(
                    {
                        "frame": frame,
                        "capsules": [names[first[j]], names[second[j]]],
                        "clearance_m": float(clearance[j]),
                        "tolerance_m": float(tolerance[j]),
                    }
                )
    return {
        "ok": not failures and not obstacle_failures and floor_worst >= -0.001,
        "cached_frames_checked": len(poses),
        "pairs_per_frame": len(first),
        "worst_capsule_clearance_m": worst,
        "worst_floor_clearance_m": floor_worst,
        "witnesses": failures,
        "worst_obstacle_clearance_m": obstacle_worst,
        "obstacle_witnesses": obstacle_failures,
        "scope": "Geometric capsule-proxy checks at retained timestamps, excludes adjacent rod segments and declared attachment neighborhoods; not an exact visual-mesh or continuous-time knot certificate.",
    }
