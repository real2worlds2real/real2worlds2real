"""Numerical kernels shared by scene, physics, tasks and data generation.

Callers retain their empty-mask conventions and camera/quaternion representation.
"""
import numpy as np


def min_jerk(n):
    """The original n-sample minimum-jerk schedule, excluding the initial zero."""
    t = np.arange(1, n + 1) / n
    return 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5


def mask_iou(a, b, *, empty=0.0, coerce=False):
    """Mask overlap; keep the caller's original empty-union and conversion policy."""
    if coerce:
        a, b = np.asarray(a, bool), np.asarray(b, bool)
    union = (a | b).sum()
    if not union and empty != 0:
        return float(empty)
    return (a & b).sum() / max(union, 1)


def look_at_matrix(eye, target, up=(0.0, 0.0, 1.0)):
    """Camera-to-world rotation: columns are right, up, back; camera looks down -Z."""
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z], 1)
