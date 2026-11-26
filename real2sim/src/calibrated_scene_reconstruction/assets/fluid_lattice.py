"""Finite close-packed fluid seeds; callers clip to observed cavity and level.

Nearest-neighbour spacing is explicit. FCC bulk volume per particle is
spacing**3 / sqrt(2), unlike a cubic seed that undergoes large initial
compression in a dense particle solver. Native settling is still required.
"""
import numpy as np


def fcc(bounds, spacing, *, maximum_points=2_000_000):
    bounds = np.asarray(bounds, float)
    if (bounds.shape != (2, 3) or not np.isfinite(bounds).all()
            or np.any(bounds[0] >= bounds[1]) or not np.isfinite(spacing) or spacing <= 0
            or type(maximum_points) is not int or maximum_points <= 0):
        raise ValueError('Finite ordered metric bounds, spacing and positive point budget required')
    cell = np.sqrt(2)*spacing
    lo = np.floor(bounds[0]/cell).astype(int)
    hi = np.ceil(bounds[1]/cell).astype(int)
    count = int(np.prod(hi-lo+1))*4
    if count > maximum_points:
        raise ValueError('Fluid seed exceeds explicit point budget')
    grid = np.stack(np.meshgrid(*(np.arange(a, b+1) for a, b in zip(lo, hi)), indexing='ij'), axis=-1).reshape(-1, 3)
    offsets = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    points = ((grid[:, None]+offsets)*cell).reshape(-1, 3)
    points = points[np.all((points >= bounds[0]) & (points <= bounds[1]), axis=1)]
    return points, {'layout': 'fcc', 'spacing_m': float(spacing),
                    'bulk_cell_volume_m3': float(spacing**3/np.sqrt(2)),
                    'particles': len(points), 'native_qualified': False}
