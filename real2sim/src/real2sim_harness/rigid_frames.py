"""Extract rigid orientation without confusing uniform geometry scale with rotation."""

import numpy as np


def rotation_only(matrix):
    linear = np.asarray(matrix, dtype=float)[:3, :3]
    if linear.shape != (3, 3) or not np.isfinite(linear).all():
        raise ValueError("A finite 3x3 or 4x4 joint transform is required")
    left, scale, right = np.linalg.svd(linear)
    if scale.min() <= 1e-10:
        raise ValueError("Collapsed joint transforms cannot define a rigid frame")
    if scale.max() / scale.min() > 1.0001:
        raise ValueError("Nonuniform scale or shear on a joint frame must be baked into geometry before rigid integration")
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        raise ValueError("Mirrored joint frames must be baked into geometry before rigid integration")
    return rotation
