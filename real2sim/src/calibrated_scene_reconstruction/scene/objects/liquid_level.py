"""Estimate a resting liquid level from visible boundaries and known cavities.

The section callback supplies the actual cavity/plane intersection in world
coordinates. Observations must identify a liquid boundary, not a label or a
reflection. This initializer cannot identify material optics or approve an asset.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree


def fit_level(views, section, bounds, *, pixel_sigma=2.):
    """Fit one plane height without changing vessel geometry or dropping views.

Each view has K, R (camera to world), C and visible boundary pixels. Section
accepts height in metres and returns densely sampled finite boundary points.
Use a refractive image model separately if wall refraction is significant.
"""
    bounds = np.asarray(bounds, float)
    if (bounds.shape != (2,) or not np.isfinite(bounds).all()
            or bounds[0] >= bounds[1] or not np.isfinite(pixel_sigma) or pixel_sigma <= 0):
        raise ValueError('Finite ordered height bounds and positive pixel uncertainty required')
    if len(views) < 3:
        raise ValueError('At least three independent calibrated views required')
    rows = []
    for view in views:
        k, r, c, p = [np.asarray(view[key], float) for key in ('K', 'R', 'C', 'pixels')]
        if (k.shape != (3, 3) or r.shape != (3, 3) or c.shape != (3,)
                or p.ndim != 2 or p.shape[1:] != (2,) or len(p) < 3
                or not all(np.isfinite(v).all() for v in (k, r, c, p))
                or not np.allclose(r.T@r, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(r), 1.) or abs(np.linalg.det(k)) < 1e-10):
            raise ValueError('Finite calibrated cameras and visible boundary samples required')
        rows.append((k, r, c, p))
    if max(np.linalg.norm(row[2]-rows[0][2]) for row in rows) < 1e-6:
        raise ValueError('Distinct camera centres required')

    def errors(height):
        ring = np.asarray(section(float(height)), float)
        if ring.ndim != 2 or ring.shape[1:] != (3,) or len(ring) < 8 or not np.isfinite(ring).all():
            raise ValueError('Cavity section must provide a finite resolved boundary')
        result = []
        for k, r, c, p in rows:
            q = (ring-c)@r
            if (q[:, 2] <= 0).any():
                raise ValueError('Liquid boundary behind a source camera')
            uv = q@k.T
            uv = uv[:, :2]/uv[:, 2, None]
            result.append(cKDTree(uv).query(p)[0])
        return result

    def objective(height):
        # Equal view weighting; denser manual labels must not dominate.
        return np.mean([np.mean(2*pixel_sigma**2*(np.sqrt(1+(e/pixel_sigma)**2)-1))
                        for e in errors(height)])

    grid = np.linspace(*bounds, 129)
    scores = np.asarray([objective(h) for h in grid])
    best = int(scores.argmin())
    lo, hi = grid[max(0, best-1)], grid[min(len(grid)-1, best+1)]
    fit = minimize_scalar(objective, bounds=(lo, hi), method='bounded', options={'xatol': 1e-9})
    e = errors(fit.x)
    return {'height_m': float(fit.x), 'optimizer_converged': bool(fit.success),
            'at_search_boundary': bool(min(fit.x-bounds[0], bounds[1]-fit.x) < (bounds[1]-bounds[0])/128),
            'per_view_errors_px': [v.tolist() for v in e],
            'per_view_p95_px': [float(np.quantile(v, .95)) for v in e],
            'median_error_px': float(np.median(np.concatenate(e))),
            'source_views_removed': False, 'accepted': False,
            'limitations': ['Known cavity and calibrated cameras required',
                            'Boundary identity, refraction and camera uncertainty need independent review',
                            'Height estimate is not a native-fluid, material or task qualification']}
