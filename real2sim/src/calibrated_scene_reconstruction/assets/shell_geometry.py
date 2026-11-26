"""Closed axisymmetric shells for reconstructed assets."""
import numpy as np


def shell(*, outer_radius, inner_radius, height, floor, segments=192, surface_side='outer', z_step=None, outer_profile=None, inner_profile=None):
    """An open-top closed-bottom shell with optional radial profiles."""
    if not np.isfinite([outer_radius, inner_radius, height, floor]).all() or not 0 < inner_radius < outer_radius or (not 0 < floor < height) or (type(segments) is not int) or (segments < 32) or (surface_side not in {'outer', 'inner'}):
        raise ValueError('Invalid shell dimensions or surface side')
    step = z_step or height / 24
    theta = np.arange(segments) * 2 * np.pi / segments
    profiles = {'outer': outer_profile, 'inner': inner_profile}
    for profile in profiles.values():
        if profile is not None:
            values = np.asarray(profile, float)
            if values.ndim != 2 or values.shape[1] != 2 or len(values) < 2 or (not np.isfinite(values).all()) or (np.diff(values[:, 0]) <= 0).any() or (values[:, 1] <= 0).any() or (values[0, 0] > 0) or (values[-1, 0] < height):
                raise ValueError('Radius profiles require increasing z/radius pairs covering the shell')
    rings = []
    for radius, lower, side in ((outer_radius, 0.0, 'outer'), (inner_radius, floor, 'inner')):
        levels = np.linspace(lower, height, max(2, int(np.ceil((height - lower) / step)) + 1))
        if side == 'inner':
            levels = levels[::-1]
        for z in levels:
            profile = profiles[side]
            radius_at_z = radius if profile is None else np.interp(z, *np.asarray(profile).T)
            r = np.full(segments, radius_at_z, dtype=float)
            rings.append(np.c_[r * np.cos(theta), r * np.sin(theta), np.full(segments, z)])
    if len(rings) * segments > 1000000:
        raise ValueError('Shell tessellation budget exceeded')
    vertices = np.concatenate(rings)
    triangles = []
    for i in range(len(rings) - 1):
        a = i * segments + np.arange(segments)
        b = i * segments + (np.arange(segments) + 1) % segments
        triangles.extend(np.c_[a, b, b + segments].tolist())
        triangles.extend(np.c_[a, b + segments, a + segments].tolist())
    bottom, inside = (len(vertices), len(vertices) + 1)
    vertices = np.vstack([vertices, [0, 0, 0], [0, 0, floor]])
    for j in range(segments):
        k = (j + 1) % segments
        triangles.append([bottom, k, j])
        a = (len(rings) - 1) * segments
        triangles.append([inside, a + j, a + k])
    return (vertices, np.asarray(triangles, np.int32))
