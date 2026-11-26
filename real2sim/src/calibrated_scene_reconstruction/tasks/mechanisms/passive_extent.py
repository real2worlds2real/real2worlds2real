"""Actual initial object extent for explicitly normalized passive measurements."""
import numpy as np


def initial_extent(parts, poses, flexible_states, samples):
    """Use all actual initial surface points, in the fixed simulation world.

The denominator is the largest world-axis bounding-box span at the first
recorded native sample. It is fixed for the entire trace; later deformation or
drift cannot enlarge the normalization. Missing parts never shrink the object
silently. This geometric scale is not a measured real-world dimension.
    """
    from .passive_response import rigid_drift
    points = []
    for name, part in parts.items():
        if part['kind'] == 'rigid':
            if name not in poses or len(poses[name]) != samples:
                raise ValueError('Normalized drift requires every rigid object part at each native sample')
            rigid_drift(part['V'], poses[name])
            pose = np.asarray(poses[name], float)[0]
            value = np.asarray(part['V'], float) @ pose[3:].reshape(3, 3).T + pose[:3]
        else:
            value = np.asarray((flexible_states or {}).get(name), float)
            if (value.ndim != 3 or value.shape[0] != samples or value.shape[2] != 3
                    or value.shape[1] != len(part['V']) or not np.isfinite(value).all()):
                raise ValueError('Normalized drift requires actual states of every flexible object part')
            value = value[0]
        if not len(value) or not np.isfinite(value).all():
            raise ValueError('Finite nonempty native initial object surface required')
        points.append(value)
    if not points:
        raise ValueError('Complete native object geometry required')
    span = np.ptp(np.concatenate(points), axis=0)
    extent = float(span.max())
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError('Positive finite initial object extent required')
    return {'value_m': extent, 'world_axis_spans_m': span.tolist(),
            'reference': 'All actual initial object surface points in the fixed simulation world',
            'definition': 'Maximum initial world-axis bounding-box span; fixed throughout this trace'}
