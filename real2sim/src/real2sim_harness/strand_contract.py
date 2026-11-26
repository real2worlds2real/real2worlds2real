"""Backend-independent contracts for open, unbranched slender bodies."""

from __future__ import annotations

import math

import numpy as np


def validate_configuration(c):
    if c.get("schema_version") != 1 or not c.get("strands"):
        raise ValueError("Expected schema 1 and at least one strand")
    bodies = {b["name"]: b for b in c.get("bodies", [])}
    if len(bodies) != len(c.get("bodies", [])):
        raise ValueError("Body names must be unique")
    for b in bodies.values():
        matrix = np.asarray(b["matrix_world"], dtype=float)
        if (
            matrix.shape != (4, 4)
            or not np.isfinite(matrix).all()
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)
            or np.linalg.det(matrix[:3, :3]) < 0
        ):
            raise ValueError(
                "Bake scale/shear/mirrors into rigid geometry; body frames must be rigid"
            )
        for key in ("mass_kg", "radius", "half_height"):
            if not math.isfinite(b[key]) or b[key] <= 0:
                raise ValueError("Body mass and capsule dimensions must be finite and positive")
    for s in c["strands"]:
        points = np.asarray(s["points"], dtype=float)
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or not 4 <= len(points) <= 513
            or not np.isfinite(points).all()
        ):
            raise ValueError("Use 4..513 finite centerline points")
        if np.min(np.linalg.norm(np.diff(points, axis=0), axis=1)) < 1e-5:
            raise ValueError("Repeated strand centerline point")
        for key in (
            "radius",
            "density",
            "stretch_rigidity",
            "shear_rigidity",
            "bend_rigidity",
            "twist_rigidity",
        ):
            if not math.isfinite(s[key]) or s[key] <= 0:
                raise ValueError(
                    "Strand dimensions and section rigidities must be finite and positive"
                )
        for key in ("stretch_damping", "bend_damping", "twist_damping"):
            if not math.isfinite(s[key]) or s[key] < 0:
                raise ValueError("Strand damping must be finite and nonnegative")
        owners = set()
        for a in s.get("attachments", []):
            if (
                type(a["node"]) is not int
                or a["node"] not in {0, len(points) - 1}
                or a["body"] not in bodies
                or a["node"] in owners
            ):
                raise ValueError(
                    "Each open strand endpoint may attach once to a declared rigid body"
                )
            owners.add(a["node"])
        if not s.get("visual_objects"):
            raise ValueError("A strand needs editable visible surface geometry")
    seen = set()
    for m in c.get("motion", []):
        if m["body"] not in bodies or m["body"] in seen:
            raise ValueError("Each motion names one unique rigid body")
        seen.add(m["body"])
        k = np.asarray(m["keyframes"], dtype=float)
        if (
            k.ndim != 2
            or k.shape[1] != 4
            or len(k) < 2
            or not np.isfinite(k).all()
            or not np.allclose(k[0], [0, 0, 0, 0])
            or np.any(np.diff(k[:, 0]) <= 0)
        ):
            raise ValueError(
                "Translation keyframes are [seconds,dx,dy,dz], start at [0,0,0,0], strictly ordered"
            )
    env = c["environment"]
    gravity = np.asarray(env["gravity"], dtype=float)
    if gravity.shape != (3,) or not np.isfinite(gravity).all():
        raise ValueError("Use a finite gravity vector")
    for key in ("friction", "air_drag_rate_per_second"):
        if not math.isfinite(env[key]) or env[key] < 0:
            raise ValueError("Use finite nonnegative environment parameters")
    if not math.isfinite(env["floor_height"]):
        raise ValueError("Use finite floor height")
    for key, default in (("contact_stiffness", 1e5), ("contact_damping", 1.0)):
        value = env.get(key, default)
        if not math.isfinite(value) or value < 0 or (key == "contact_stiffness" and value == 0):
            raise ValueError("Use positive contact stiffness and nonnegative damping")
    names = set()
    for fixture in env.get("capsule_obstacles", []):
        center, axis = np.asarray(fixture["center_m"]), np.asarray(fixture["axis"])
        if (
            center.shape != (3,)
            or axis.shape != (3,)
            or not np.isfinite([center, axis]).all()
            or not np.isclose(np.linalg.norm(axis), 1)
        ):
            raise ValueError("Obstacle needs a finite center and unit axis")
        if not fixture.get("id") or fixture["id"] in names:
            raise ValueError("Obstacle identifiers must be unique")
        names.add(fixture["id"])
        if any(
            not math.isfinite(fixture[k]) or fixture[k] <= 0 for k in ("radius_m", "half_height_m")
        ):
            raise ValueError("Obstacle capsule dimensions must be positive")
    solver = c["solver"]
    if not 4 <= solver["substeps"] <= 96 or not 4 <= solver["iterations"] <= 64:
        raise ValueError("Substeps/iterations outside supported budget")
    if type(solver.get("contact_history", False)) is not bool:
        raise ValueError("Contact history must be boolean")
    return c


def translation_at(keyframes, time):
    k = np.asarray(keyframes, dtype=float)
    index = int(np.clip(np.searchsorted(k[:, 0], time, side="right") - 1, 0, len(k) - 2))
    u = float(np.clip((time - k[index, 0]) / (k[index + 1, 0] - k[index, 0]), 0, 1))
    return k[index, 1:] + u * u * (3 - 2 * u) * (k[index + 1, 1:] - k[index, 1:])
