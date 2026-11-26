"""Fit a visible circular boundary across calibrated views, without contacts.

Samples must label one physical edge, not a shaft, reflection or rear edge.
This produces an initializer and per-view evidence, never an asset approval.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def visible_boundary(mask, *, domain=None, maximum_points=96, clip_margin=1):
    """Sample observed outer arcs, excluding image/known-domain cut edges.

    A cropped object's cut line is not its physical boundary. The domain must
    come from source visibility, not a candidate model. This only prepares an
    initializer; original masks and views remain unchanged for source checks.
    """
    import cv2
    mask = np.asarray(mask)
    if (mask.ndim != 2 or not mask.size or mask.dtype.kind not in 'biu'
            or not np.isin(mask, [0, 1]).all()
            or type(maximum_points) is not int or maximum_points < 5
            or type(clip_margin) is not int or clip_margin < 1):
        raise ValueError('Binary source mask and positive bounded sampling settings required')
    mask = mask.astype(bool)
    valid = np.ones_like(mask) if domain is None else np.asarray(domain)
    if valid.shape != mask.shape or valid.dtype.kind not in 'biu' or not np.isin(valid, [0, 1]).all():
        raise ValueError('Matching binary source visibility domain required')
    valid = valid.astype(bool)
    contours, _ = cv2.findContours((mask & valid).astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    points = np.concatenate([p[:, 0] for p in contours]) if contours else np.empty((0, 2), int)
    eroded = cv2.erode(valid.astype('uint8'), np.ones((2*clip_margin+1, 2*clip_margin+1), 'uint8'),
                      borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    kept = points[eroded[points[:, 1], points[:, 0]]]
    if len(kept) > maximum_points:
        selected = kept[np.linspace(0, len(kept)-1, maximum_points).astype(int)]
    else:
        selected = kept
    return selected.astype(float), {'boundary_pixels': len(points), 'visible_arc_pixels': len(kept),
                                    'clipped_edge_pixels': len(points)-len(kept),
                                    'sampled_pixels': len(selected), 'source_mask_changed': False}


def fit_circle(views, centre, normal, radius, *, pixel_sigma=2.):
    """Fit source arcs; camera R maps camera coordinates into world coordinates.

    Initial values are hypotheses. No contact force, clearance target or real
    object measurement enters the objective. Camera calibration uncertainty is
    not estimated by this fit. Keep occluded arcs absent instead of inventing
    their edge samples.
    """
    centre, normal = np.asarray(centre, float), np.asarray(normal, float)
    if (centre.shape != (3,) or normal.shape != (3,)
            or not np.isfinite(np.r_[centre, normal, radius, pixel_sigma]).all()
            or np.linalg.norm(normal) < 1e-10 or radius <= 0 or pixel_sigma <= 0):
        raise ValueError('Finite centre, normal and positive radius/pixel uncertainty required')
    if len(views) < 3:
        raise ValueError('At least three source views required')
    rows = []
    for view in views:
        k, r, c, p = (np.asarray(view[key], float) for key in ('K', 'R', 'C', 'pixels'))
        if (k.shape != (3, 3) or r.shape != (3, 3) or c.shape != (3,)
                or p.ndim != 2 or p.shape[1:] != (2,) or len(p) < 5
                or not all(np.isfinite(x).all() for x in (k, r, c, p))
                or abs(np.linalg.det(k)) < 1e-12
                or not np.allclose(r.T@r, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(r), 1.)):
            raise ValueError('Calibrated cameras and at least five finite visible edge samples required')
        rows.append((k, r, c, np.c_[p, np.ones(len(p))]))
    centres = np.array([row[2] for row in rows])
    if np.max(np.linalg.norm(centres-centres[0], axis=1)) < 1e-8:
        raise ValueError('Distinct camera centres required to resolve scale')
    normal = normal/np.linalg.norm(normal)
    basis = np.linalg.svd(normal[None], full_matrices=True)[2][1:].T

    def geometry(q):
        frame = Rotation.from_rotvec(basis@q[3:5]).as_matrix()@basis
        return centre+radius*q[:3], frame, radius*np.exp(q[5])

    def errors(q):
        point, frame, size = geometry(q)
        values = []
        for k, r, c, pixels in rows:
            h = k@r.T@np.c_[size*frame, point-c]
            if abs(np.linalg.det(h)) < 1e-12:
                values.append(np.full(len(pixels), 1e6))
                continue
            inv = np.linalg.inv(h)
            conic = inv.T@np.diag([1., 1., -1.])@inv
            gradient = 2*(pixels@conic)[:, :2]
            values.append(np.sum(pixels*(pixels@conic), axis=1)
                          / np.maximum(np.linalg.norm(gradient, axis=1), 1e-12))
        return values

    fitted = least_squares(lambda q: np.concatenate(errors(q)), np.zeros(6),
                           bounds=([-4., -4., -4., -1., -1., -1.5],
                                   [4., 4., 4., 1., 1., 1.5]),
                           loss='soft_l1', f_scale=pixel_sigma, max_nfev=500)
    point, frame, size = geometry(fitted.x)
    angles = np.linspace(0, 2*np.pi, 256, endpoint=False)
    ring = point+size*np.c_[np.cos(angles), np.sin(angles)]@frame.T
    positive_depth = all(np.min(((ring-c)@r)[:, 2]) > 0 for _, r, c, _ in rows)
    per_view = [abs(e) for e in errors(fitted.x)]
    all_errors = np.concatenate(per_view)
    return {'centre': point.tolist(), 'normal': np.cross(*frame.T).tolist(),
            'radius': float(size), 'optimizer_converged': bool(fitted.success),
            'all_views_positive_depth': bool(positive_depth),
            'per_view_errors_px': [e.tolist() for e in per_view],
            'per_view_p95_px': [float(np.percentile(e, 95)) for e in per_view],
            'median_error_px': float(np.median(all_errors)),
            'source_views_removed': False, 'proposal_only': True,
            'limitations': 'Circular-edge hypothesis; labels and camera calibration require independent review. Low residual does not prove physical support or approve geometry.'}
