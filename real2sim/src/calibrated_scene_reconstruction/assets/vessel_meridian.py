"""Simple physical meridians for source-modeled open vessels.

Watertight triangles and positive signed volume do not rule out an inner wall
crossing the outer wall. Validate the actual 2D material boundary before
revolution. Construction preserves the supplied exterior; thickness is an
explicit hypothesis, never an inferred measurement or a contact certificate.
"""
import numpy as np


def validate(profile):
    points = np.asarray(profile, float)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
            or not np.isfinite(points).all() or (points[:, 0] < 0).any()):
        raise ValueError('Finite nonnegative-radius meridian vertices required')
    scale = float(np.ptp(points, axis=0).max())
    tolerance = max(scale * 1e-10, 1e-14)
    if scale <= tolerance:
        raise ValueError('Nondegenerate material section required')
    following = np.roll(points, -1, axis=0)
    if (np.linalg.norm(following-points, axis=1) <= tolerance).any():
        raise ValueError('Material meridian contains a zero-length edge')

    def cross(a, b, c):
        u, v = b-a, c-a
        return float(u[0]*v[1]-u[1]*v[0])

    area_tolerance = tolerance * scale
    for i, (a, b) in enumerate(zip(points, following)):
        for j in range(i+1, len(points)):
            if j == i+1 or i == 0 and j == len(points)-1:
                continue
            c, d = points[j], following[j]
            if (np.maximum(np.minimum(a, b), np.minimum(c, d))
                    > np.minimum(np.maximum(a, b), np.maximum(c, d)) + tolerance).any():
                continue
            values = np.array([cross(a, b, c), cross(a, b, d),
                               cross(c, d, a), cross(c, d, b)])
            signs = np.where(np.abs(values) <= area_tolerance, 0., np.sign(values))
            if signs[0]*signs[1] <= 0 and signs[2]*signs[3] <= 0:
                raise ValueError(f'Material meridian has crossing or touching nonadjacent edges {i} and {j}')
    area = float(np.sum(points[:, 0]*following[:, 1]-points[:, 1]*following[:, 0])/2)
    if abs(area) <= area_tolerance:
        raise ValueError('Positive material section area required')
    return {'simple_material_section': True, 'area': abs(area),
            'vertices': len(points), 'scope': 'Planar material section only; native contact still required'}


def hollow(outer, radial_thickness, *, bottom_thickness=None):
    """Return a closed material section inside an unchanged rising exterior.

The exterior starts at the base axis, then its footprint, and rises to the
open rim. Radii may increase or decrease with height. The interior follows
the same sampled exterior with positive radial thickness; a horizontal floor
closes it above the base. Rolled or undercut profiles need a different explicit
construction and are rejected if their heights are not monotone.
"""
    points = np.asarray(outer, float)
    wall = float(radial_thickness)
    floor = wall if bottom_thickness is None else float(bottom_thickness)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
            or not np.isfinite(points).all() or not np.isfinite([wall, floor]).all()
            or wall <= 0 or floor <= 0 or not np.allclose(points[0], [0., 0.], atol=1e-12)
            or points[1, 1] != 0 or points[1, 0] <= wall
            or (np.diff(points[1:, 1]) <= 0).any() or points[-1, 1] <= floor
            or (points[1:, 0] <= wall).any()):
        raise ValueError('Vessel exterior needs a base footprint, increasing heights and positive wall/floor thickness')
    radius_at_floor = float(np.interp(floor, points[1:, 1], points[1:, 0]))
    inner = [[r-wall, z] for r, z in points[:0:-1] if z > floor]
    inner.extend([[radius_at_floor-wall, floor], [0., floor]])
    result = np.r_[points, inner]
    validate(result)
    return result.tolist()
