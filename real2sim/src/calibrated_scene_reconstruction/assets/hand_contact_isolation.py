"""Conservative geometry witness for a particle-wide friction intervention.

Particle material friction affects every rigid partner. A hand experiment may
interpret that intervention as finger friction only while the whole cloth is
clear of all other colliders. Overlapping boxes are inconclusive, even when
selected nodes or a grasp patch appear clear. This is a sampled-state witness,
not continuous collision detection or a task certificate.
"""
import numpy as np


def material_samples(points, faces, indices):
    """Sample only the intended material patch, including triangle interiors.

    Vertices elsewhere on the sheet must not witness this patch's contact.
    Interior samples matter for triangle-contact cloth whose vertices can lie
    away from a finger even when the triangle between them touches it.
    """
    p, f, ids = np.asarray(points, float), np.asarray(faces), np.asarray(indices)
    if (p.ndim != 2 or p.shape[1:] != (3,) or not np.isfinite(p).all()
            or f.ndim != 2 or f.shape[1:] != (3,) or not len(f) or f.dtype.kind not in 'iu'
            or f.min() < 0 or f.max() >= len(p)
            or ids.ndim != 1 or ids.dtype.kind not in 'iu' or len(ids) < 3
            or ids.min() < 0 or ids.max() >= len(p) or len(np.unique(ids)) != len(ids)):
        raise ValueError('Finite native cloth and a distinct indexed material region required')
    selected = np.zeros(len(p), bool)
    selected[ids] = True
    triangles = p[f[selected[f].all(1)]]
    return np.concatenate([p[ids], triangles.mean(1),
                           (triangles[:, 0]+triangles[:, 1])/2,
                           (triangles[:, 1]+triangles[:, 2])/2,
                           (triangles[:, 2]+triangles[:, 0])/2])


def clearance(points, rigid_bounds, contact_margins, *, expected_colliders):
    """Use actual world-space geometry, including the full sheet and supports.

    A separating gap between enclosing boxes lower-bounds the distance between
    the contained surfaces. The caller must include the actual contact margins
    of both partners, rather than just the physical cloth thickness.
    """
    points = np.asarray(points, float)
    names = set(expected_colliders)
    if (points.ndim != 2 or points.shape[1:] != (3,) or not len(points)
            or not np.isfinite(points).all() or not names
            or set(rigid_bounds) != names or set(contact_margins) != names):
        raise ValueError('Complete finite cloth and all non-hand colliders required')
    lower, upper = points.min(0), points.max(0)
    checks = {}
    for name in sorted(names):
        bounds = np.asarray(rigid_bounds[name], float)
        margin = contact_margins[name]
        if (bounds.shape != (2, 3) or not np.isfinite(bounds).all()
                or (bounds[1] < bounds[0]).any()
                or not np.isfinite(margin) or margin <= 0):
            raise ValueError('Finite world bounds and positive pair contact margin required')
        separation = np.maximum(0., np.maximum(bounds[0]-upper, lower-bounds[1]))
        gap = float(np.linalg.norm(separation))
        checks[name] = {'pass': bool(gap > margin),
                        'surface_distance_lower_bound_m': gap,
                        'required_pair_contact_margin_m': float(margin)}
    return {'pass': all(row['pass'] for row in checks.values()), 'colliders': checks,
            'cloth_bounds': [lower.tolist(), upper.tolist()],
            'rigid_bounds': {name: np.asarray(rigid_bounds[name],float).tolist() for name in sorted(names)},
            'scope': 'Whole-sheet separation at this native sample only; overlapping bounds '
                     'are inconclusive. All causal-interval samples and a matched native '
                     'friction-response control are required separately.'}
