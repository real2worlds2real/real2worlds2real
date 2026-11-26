"""Geometric conditioning of actual bonded nodes, never a mechanical certificate.

A whole assembly can have well-spread anchors while every disconnected sheet
is attached along an almost straight line. Audit each sheet independently.
The rigid-motion calculation diagnoses a possible unintended hinge; it does
not replace native flexible response or decide whether a hinge is appropriate.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def audit(points, faces, roots):
    points, faces, roots = np.asarray(points, float), np.asarray(faces), np.asarray(roots)
    if (points.ndim != 2 or points.shape[1:] != (3,) or not len(points)
            or not np.isfinite(points).all() or faces.ndim != 2 or faces.shape[1:] != (3,)
            or not np.issubdtype(faces.dtype, np.integer) or not len(faces)
            or faces.min() < 0 or faces.max() >= len(points)
            or roots.ndim != 1 or not np.issubdtype(roots.dtype, np.integer)
            or len(np.unique(roots)) != len(roots)
            or len(roots) and (roots.min() < 0 or roots.max() >= len(points))):
        raise ValueError('Finite actual surface triangles and unique actual bound node indices required')
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    count, owner = connected_components(coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                                                   shape=(len(points), len(points))), directed=False)
    rows = []
    for k in range(count):
        ids = np.flatnonzero(owner == k)
        bound = roots[owner[roots] == k]
        surface = points[ids]
        # RMS distance from the actual surface centroid is rotationally invariant.
        extent = float(np.sqrt(np.square(surface - surface.mean(0)).sum(1).mean()))
        if extent <= 0:
            raise ValueError('Degenerate connected surface extent')
        if len(bound):
            centered = points[bound] - points[bound].mean(0)
            covariance = centered.T @ centered / len(bound)
            gram = np.trace(covariance) * np.eye(3) - covariance
            eigenvalues, eigenvectors = np.linalg.eigh(gram)
            eigenvalues = np.maximum(eigenvalues, 0.)
            tolerance = max(eigenvalues.max(), extent**2) * 128 * np.finfo(float).eps
            rank = int(np.count_nonzero(eigenvalues > tolerance))
        else:
            gram = np.zeros((3, 3)); eigenvalues = np.zeros(3); eigenvectors = np.eye(3); rank = 0
        sensitivity = np.sqrt(eigenvalues)
        rows.append({'component': k, 'vertices': len(ids), 'bound_vertices': len(bound),
                     'surface_rms_extent_m': extent, 'rotational_constraint_rank': rank,
                     'unconstrained_rigid_rotation_directions': 3 - rank,
                     'rotation_gram_m2': gram.tolist(),
                     'rms_anchor_displacement_per_radian_m': sensitivity.tolist(),
                     'normalized_rotational_sensitivity': (sensitivity / extent).tolist(),
                     'weakest_rotation_axis': eigenvectors[:, 0].tolist()})
    return {'components': rows, 'native_tested': False, 'approved': False,
            'scope': 'Actual anchor geometry for each connected sheet, including nearly collinear bases. '
                     'Sensitivity predicts only small rigid rotation about the bound-node centroid. '
                     'Source-reviewed connection semantics, finite material geometry and native response remain required.'}
