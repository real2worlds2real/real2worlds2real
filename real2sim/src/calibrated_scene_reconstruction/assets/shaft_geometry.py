"""Measure a straight exposed shaft from its actual labeled mesh vertices.

The fitted cylinder describes a conservative finite insertion region; it never
replaces source geometry or authorizes collision-free motion. Bulged attachment
vertices shorten that region instead of being silently counted as a thin shaft.
"""
import numpy as np
from scipy.optimize import least_squares


def measure(vertices, *, attachment_hint, tolerance_m):
    v = np.asarray(vertices, float); attachment = np.asarray(attachment_hint, float)
    if (v.ndim != 2 or v.shape[1] != 3 or len(v) < 12 or attachment.shape != (3,)
            or not np.isfinite(v).all() or not np.isfinite(attachment).all()
            or not np.isfinite(tolerance_m) or tolerance_m <= 0):
        raise ValueError('Finite labeled shaft vertices, attachment hint and positive tolerance required')
    centre = v.mean(0)
    _, singular, axes = np.linalg.svd(v - centre, full_matrices=False)
    if singular[0] <= 3 * singular[1] or singular[-1] <= 0:
        raise ValueError('A distinctly elongated three-dimensional shaft is required')
    axis = axes[0] * (1 if axes[0] @ (centre - attachment) >= 0 else -1)
    e1, e2 = axes[1:]
    projection = (v - centre) @ axis
    low, high = projection.min(), projection.max()
    middle = v[(projection > low + .2 * (high-low)) & (projection < low + .8 * (high-low))]
    if len(middle) < 12:
        raise ValueError('Insufficient interior side surface for a shaft fit')
    def frame(p):
        direction = axis + p[2] * e1 + p[3] * e2
        return centre + p[0] * e1 + p[1] * e2, direction / np.linalg.norm(direction)
    def radii(points, origin, direction):
        d = points - origin
        return np.linalg.norm(d - (d @ direction)[:, None] * direction, axis=1)
    radius = float(np.median(radii(middle, centre, axis)))
    if radius <= tolerance_m:
        raise ValueError('Shaft radius is below the requested geometric resolution')
    def residual(p):
        origin, direction = frame(p)
        return radii(middle, origin, direction) - p[4]
    fit = least_squares(residual, [0., 0., 0., 0., radius],
                        bounds=([-radius, -radius, -.2, -.2, radius/2],
                                [radius, radius, .2, .2, radius*2]),
                        loss='soft_l1', f_scale=tolerance_m/4,
                        xtol=1e-12, ftol=1e-12, gtol=1e-12)
    error = float(np.abs(residual(fit.x)).max())
    if not fit.success or error > tolerance_m:
        raise ValueError('Labeled side surface is not straight/cylindrical within tolerance')
    origin, direction = frame(fit.x)
    projection = (v - origin) @ direction; radial = radii(v, origin, direction)
    # Preserve an exact enclosure of the measured interior side surface.
    radius = float(radii(middle, origin, direction).max())
    outside = radial > radius + tolerance_m
    start = float(projection[outside].max() + tolerance_m) if outside.any() else float(projection.min())
    end = float(projection.max()); usable = projection >= start
    if end - start <= 3 * radius or not usable.any():
        raise ValueError('No sufficiently long conservative shaft interval reaches the free tip')
    radius = float(radial[usable].max())
    return {'tip': (origin + direction * end).tolist(),
            'root': (origin + direction * start).tolist(),
            'axis': direction.tolist(), 'radius_m': radius,
            'usable_length_m': end - start, 'maximum_fit_residual_m': error,
            'excluded_attachment_vertices': int((~usable).sum()),
            'scope': 'Measured finite shaft enclosure; full source surface and native contacts remain required.'}
