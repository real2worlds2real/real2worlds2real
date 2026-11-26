"""Convex material cells of an open revolved vessel, preserving its cavity.

Profiles are authored source geometry, not fitted sizes or contents boundaries.
This auxiliary particle-compatible boundary omits fine external ridges; those
remain in the precision rigid collider. Native cooking and contact checks are
mandatory. A convex hull of the entire vessel is never constructed.
"""
import numpy as np
from calibrated_scene_reconstruction.assets.convex_source_solids import _hull


def build(outer_profile, inner_profile, floor, *, segments=64, maximum_cells=4096):
    outer = np.asarray(outer_profile, float)
    inner = np.asarray(inner_profile, float)
    for p in (outer, inner):
        if (p.ndim != 2 or p.shape[1] != 2 or len(p) < 2 or not np.isfinite(p).all()
                or np.any(np.diff(p[:, 0]) <= 0) or np.any(p[:, 1] <= 0) or p[0, 0] != 0):
            raise ValueError('Increasing positive source radius profiles starting at z=0 required')
    h = float(outer[-1, 0])
    if not np.isclose(inner[-1, 0], h) or not np.isfinite(floor) or not 0 < floor < h:
        raise ValueError('Common source rim and physical material floor required')
    if type(segments) is not int or not 32 <= segments <= 256:
        raise ValueError('Explicit finite radial tessellation in 32..256 required')
    levels = np.unique(np.r_[outer[:, 0], inner[:, 0], floor])
    if len(levels)*segments > maximum_cells:
        raise ValueError('Source vessel exceeds convex-cell budget')
    ro = np.interp(levels, *outer.T)
    ri = np.interp(levels, *inner.T)
    if np.any(ri >= ro):
        raise ValueError('Source inner wall must remain inside the outer wall')
    angles = np.arange(segments+1)*2*np.pi/segments
    cells = []
    for i in range(len(levels)-1):
        a, b = levels[i:i+2]
        solid_floor = b <= floor
        for j in range(segments):
            v = []
            for k, z in ((i, a), (i+1, b)):
                radii = [0., ro[k]] if solid_floor else [ri[k], ro[k]]
                for radius in radii:
                    for theta in angles[j:j+2]:
                        v.append([radius*np.cos(theta), radius*np.sin(theta), z])
            mesh = _hull(np.unique(np.asarray(v), axis=0))
            cells.append({'V': np.asarray(mesh.vertices), 'F': np.asarray(mesh.faces),
                          'source_interval': [float(a), float(b)], 'source_sector': j})
    report = {'cells': len(cells), 'segments': segments, 'floor_m': float(floor),
              'rim_m': h, 'source_boundary_error_bound_m': float(ro.max()*(1-np.cos(np.pi/segments))),
              'solid_volume_m3': float(sum(_volume(c) for c in cells)),
              'cavity_filled': False, 'outer_thread_ridges_included': False,
              'native_approved': False, 'scope': __doc__.strip()}
    return cells, report


def _volume(cell):
    tri = cell['V'][cell['F']]
    return np.einsum('ij,ij->i', tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum()/6
