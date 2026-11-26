"""Ordered forward intersections with actual triangles, deduplicated at shared edges."""
import numpy as np


def hits(triangles, origin, direction):
    e1 = triangles[:, 1] - triangles[:, 0]
    e2 = triangles[:, 2] - triangles[:, 0]
    h = np.cross(direction, e2)
    determinant = np.einsum('ij,ij->i', e1, h)
    valid = abs(determinant) > 1e-14
    inverse = np.divide(1., determinant, out=np.zeros_like(determinant), where=valid)
    delta = origin - triangles[:, 0]
    u = np.einsum('ij,ij->i', delta, h) * inverse
    q = np.cross(delta, e1)
    v = (q @ direction) * inverse
    distance = np.einsum('ij,ij->i', e2, q) * inverse
    ids = np.flatnonzero(valid & (u >= -1e-7) & (v >= -1e-7)
                        & (u + v <= 1 + 1e-7) & (distance >= 0))
    ids = ids[np.argsort(distance[ids], kind='stable')]
    if len(ids):
        ids = ids[np.r_[True, np.diff(distance[ids]) > 1e-7]]
    return distance[ids], ids
