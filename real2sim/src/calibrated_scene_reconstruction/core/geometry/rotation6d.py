"""Continuous 6-D rotations used by dataset export and policy clients."""
import numpy as np


def rot6d(R):
    """(..., 3, 3) -> (..., 6): the first two columns of the rotation (a continuous representation)."""
    R = np.asarray(R)
    return np.concatenate([R[..., :, 0], R[..., :, 1]], -1)

def rot_from6d(d):
    """(..., 6) -> (..., 3, 3) by Gram-Schmidt."""
    d = np.asarray(d, float)
    a, b = d[..., :3], d[..., 3:6]
    x = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b - (x * b).sum(-1, keepdims=True) * x
    y = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.stack([x, y, np.cross(x, y)], -1)
