"""Canonical rigid triangle indices at the validator's numerical weld precision."""
import numpy as np


def canonical_triangles(vertices, faces):
    vertices = np.asarray(vertices, float); faces = np.asarray(faces)
    # Match the metric validator's 8-decimal vertex grouping. Keeping original
    # representative coordinates avoids rounding the entire authored surface.
    _, first, inverse = np.unique(np.round(vertices, 8), axis=0,
                                  return_index=True, return_inverse=True)
    representative = first[inverse]
    displacement = float(np.linalg.norm(vertices - vertices[representative], axis=1).max())
    if displacement > 2e-8:
        raise ValueError('Numerical canonicalization exceeded its metric bound')
    canonical = representative[faces]
    keep = np.linalg.norm(np.cross(vertices[canonical[:, 1]] - vertices[canonical[:, 0]],
                                   vertices[canonical[:, 2]] - vertices[canonical[:, 0]]), axis=1) > 1e-18
    return canonical, keep, {'maximum_vertex_reassignment_m': displacement,
                             'removed_numerical_zero_area_triangles': int((~keep).sum()),
                             'scope': 'Numerical duplicate indexing only; no hole repair or shape inflation'}
