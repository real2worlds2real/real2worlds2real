"""Locate reported crossing examples in the unchanged material chart.

This is a diagnostic for the agent's next construction proposal. It neither
changes a physical verdict nor identifies the cause of an intersection.
"""
import numpy as np


def locate(points, faces, rest, crossing):
    p, f, r = np.asarray(points, float), np.asarray(faces), np.asarray(rest, float)
    pairs = np.asarray(crossing['example_face_pairs'])
    if (p.ndim != 2 or p.shape[1:] != (3,) or r.shape != p.shape
            or not np.isfinite(p).all() or not np.isfinite(r).all()
            or f.ndim != 2 or f.shape[1:] != (3,) or not len(f)
            or f.dtype.kind not in 'iu' or f.min() < 0 or f.max() >= len(p)):
        raise ValueError('Finite corresponding physical/material nodes and indexed triangles required')
    if not pairs.size:
        pairs = np.empty((0, 2), dtype=int)
    if (pairs.ndim != 2 or pairs.shape[1:] != (2,) or pairs.dtype.kind not in 'iu'
            or (pairs.size and (pairs.min() < 0 or pairs.max() >= len(f)))):
        raise ValueError('Reported crossing examples must name current source triangles')
    examples = []
    low, span = r.min(0), np.ptp(r, axis=0)
    resolved = span > 1e-12
    for pair in pairs:
        sides = []
        for index in pair:
            ids = f[index]
            fraction = np.zeros((3, 3))
            fraction[:, resolved] = (r[ids][:, resolved]-low[resolved])/span[resolved]
            sides.append({'face': int(index), 'node_indices': ids.tolist(),
                          'world_centre_m': p[ids].mean(0).tolist(),
                          'material_centre_m': r[ids].mean(0).tolist(),
                          'material_fraction': fraction.mean(0).tolist()})
        examples.append(sides)
    return {'examples': examples, 'resolved_material_axes': resolved.tolist(),
            'reported_intersecting_pairs': int(crossing['intersecting_pairs']),
            'all_pairs_located': len(examples) == crossing['intersecting_pairs'],
            'task_approved': False,
            'scope': 'Reported examples only; material fractions use the original rest-chart bounds. '
                     'Locations do not establish the failure cause or change any physical verdict.'}
