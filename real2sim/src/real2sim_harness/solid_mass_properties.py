"""Integrate an explicitly oriented closed material surface, in metres.

Density is a caller-supplied prior or measurement. Topological closure does not
prove absence of self-intersections or overlapping material; verify source
geometry before using the resulting coupled mass properties.
"""

from __future__ import annotations

import numpy as np
from .physx_mass_properties import validate_mass_properties


def from_mesh(vertices, triangles, *, density_kg_m3, source):
    """Return volume and mass/COM/inertia about COM in the input coordinate axes.

    Inner cavity faces must point into the cavity. Do not pass a convex hull in
    place of hollow source material, or add overlapping solids as disjoint mass.
    """
    v, f = np.asarray(vertices, float), np.asarray(triangles)
    density = density_kg_m3
    if isinstance(density, (bool, str)) or not np.isscalar(density):
        raise ValueError("Density must be positive and finite")
    density = float(density)
    if not np.isfinite(density) or density <= 0:
        raise ValueError("Density must be positive and finite")
    if v.ndim != 2 or v.shape[1] != 3 or len(v) < 4 or not np.isfinite(v).all():
        raise ValueError("Finite vertices Nx3 required")
    if (
        f.ndim != 2
        or f.shape[1] != 3
        or len(f) < 4
        or f.dtype.kind not in "iu"
        or f.min() < 0
        or f.max() >= len(v)
    ):
        raise ValueError("Valid triangle indices required")
    # Exact seams may be welded; no tolerance-based shape change is made.
    v, inverse = np.unique(v, axis=0, return_inverse=True)
    f = inverse[f]
    edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    undirected = np.sort(edges, axis=1)
    _, ids, counts = np.unique(undirected, axis=0, return_inverse=True, return_counts=True)
    directions = np.where(edges[:, 0] < edges[:, 1], 1, -1)
    if np.any(counts != 2) or np.any(np.bincount(ids, weights=directions) != 0):
        raise ValueError("Material surface must be closed with consistent orientation")
    origin = v.mean(axis=0)
    p = v[f] - origin
    area = np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    if np.any(area <= 0):
        raise ValueError("Degenerate material triangles")
    volumes = np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])) / 6
    volume = float(volumes.sum())
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError("Material volume must be positive and outward oriented")
    sums = p.sum(axis=1)
    center = np.einsum("i,ij->j", volumes, sums) / (4 * volume)
    second = np.einsum("i,ij,ik->jk", volumes, sums, sums) / 20
    second += np.einsum("i,iaj,iak->jk", volumes, p, p) / 20
    covariance = second - volume * np.outer(center, center)
    inertia = float(density) * (np.eye(3) * np.trace(covariance) - covariance)
    record = validate_mass_properties(
        {
            "mass_kg": volume * float(density),
            "center_of_mass_m": (origin + center).tolist(),
            "inertia_kg_m2": inertia.tolist(),
            "source": source,
        }
    )
    return {
        "volume_m3": volume,
        "density_kg_m3": float(density),
        "mass_properties": record,
        "assumptions": [
            "Input is a non-self-intersecting oriented material boundary in metres.",
            "Uniform declared density; density is not inferred from images.",
        ],
    }


def combine(records, *, source, disjoint_material=False):
    """Combine mass records expressed in one shared frame by the parallel-axis rule."""
    if disjoint_material is not True:
        raise ValueError("Declare disjoint material volumes; union overlapping solids upstream")
    if not records:
        raise ValueError("At least one mass record required")
    rows = [validate_mass_properties(r) for r in records]
    mass = sum(r["mass_kg"] for r in rows)
    center = sum(r["mass_kg"] * np.array(r["center_of_mass_m"]) for r in rows) / mass
    inertia = np.zeros((3, 3))
    for r in rows:
        d = np.array(r["center_of_mass_m"]) - center
        inertia += np.array(r["inertia_kg_m2"]) + r["mass_kg"] * (
            np.eye(3) * (d @ d) - np.outer(d, d)
        )
    return validate_mass_properties(
        {
            "mass_kg": mass,
            "center_of_mass_m": center.tolist(),
            "inertia_kg_m2": inertia.tolist(),
            "source": source,
        }
    )
