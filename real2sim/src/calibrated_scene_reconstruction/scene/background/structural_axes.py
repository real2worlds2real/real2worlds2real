"""Explicit section axes for source-authored noncircular structural members.

A centreline fixes two rotation degrees of freedom. A rectangular section still
needs a surface normal: an arbitrary camera/up-vector roll can swap width and
depth while leaving the centreline unchanged.
"""
import numpy as np


def section_frame(start, end, depth_normal):
    """Return a right-handed local-to-world frame: X width, Y depth, Z length.

    The agent supplies the source-reviewed normal of the broad face. Its small
    component along the centreline is projected out, never replaced by a preset
    world-up direction. A parallel/undefined normal is rejected.
    """
    a, b, normal = (np.asarray(x, float) for x in (start, end, depth_normal))
    if any(x.shape != (3,) or not np.isfinite(x).all() for x in (a, b, normal)):
        raise ValueError('Finite three-dimensional member endpoints and face normal required')
    axis = b-a
    if np.linalg.norm(axis) < 1e-8 or np.linalg.norm(normal) < 1e-8:
        raise ValueError('Nonzero member and explicit face normal required')
    axis /= np.linalg.norm(axis)
    depth = normal - (normal @ axis)*axis
    if np.linalg.norm(depth) < .01*np.linalg.norm(normal):
        raise ValueError('Face normal is parallel to the structural member')
    depth /= np.linalg.norm(depth)
    return np.column_stack([np.cross(depth, axis), depth, axis])
