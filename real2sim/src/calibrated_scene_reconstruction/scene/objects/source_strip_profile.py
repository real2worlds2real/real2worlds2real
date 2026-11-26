"""Independent RGB boundary diagnostics for a resolved straight source strip.

This is a source proposal, never an acceptance test or a model-error tolerance.
The blurred-strip assumption is explicit and may fail for texture, occlusion,
transparency, nonlinear camera processing or unresolved width/blur ambiguity.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.special import ndtr


def coverage(x, centre, width, sigma):
    x = np.asarray(x, float)
    if not np.isfinite([centre, width, sigma]).all() or min(width, sigma) <= 0:
        raise ValueError('Finite positive strip width and blur required')
    return ndtr((x-centre+width/2)/sigma)-ndtr((x-centre-width/2)/sigma)


def fit(x, rgb, *, width_hint):
    """Estimate latent edges from RGB only; retain source residual/ambiguity.

    Each channel has its own linear background and strip contrast. Several
    width/blur seeds reduce local initialization bias. No geometry, camera pose,
    target dimension or candidate mask is accepted by this function.
    """
    x = np.asarray(x, float);rgb = np.asarray(rgb, float)
    if (x.ndim != 1 or len(x) < 15 or rgb.shape != (len(x), 3)
            or not np.isfinite(x).all() or not np.isfinite(rgb).all()
            or np.any(np.diff(x) <= 0) or not 0 < width_hint < np.ptp(x)
            or rgb.min() < 0 or rgb.max() > 1):
        raise ValueError('Ordered resolved RGB profile and positive source width hint required')
    span = np.ptp(x)
    def evaluate(q):
        weight = coverage(x, q[0], q[1], q[2])
        design = np.c_[np.ones(len(x)), x/span, weight]
        colours = np.linalg.lstsq(design, rgb, rcond=None)[0]
        return design@colours, colours
    def residual(q):
        predicted, colours = evaluate(q)
        foreground = colours[0]+colours[2]
        # Prevent an arbitrarily thin, infinitely bright latent strip from
        # explaining a broad blurred observation. These are RGB bounds, not
        # an invented object dimension or confidence interval.
        physical = np.r_[np.minimum(foreground, 0), np.maximum(foreground-1, 0)]
        return np.r_[(predicted-rgb).ravel(), physical*np.sqrt(len(x))]
    bounds = ([-span*.3, .35, .15], [span*.3, span*.7, span*.3])
    candidates = []
    for width_scale, sigma in [(1., .6), (1., 2.), (.7, 1.5), (1.3, .6)]:
        seed = np.clip([0., width_hint*width_scale, sigma],
                       np.array(bounds[0])+1e-8, np.array(bounds[1])-1e-8)
        result = least_squares(residual, seed, bounds=bounds, max_nfev=100)
        candidates.append(result)
    result = min(candidates, key=lambda r: np.mean(residual(r.x)**2))
    predicted, colours = evaluate(result.x)
    _, singular, _ = np.linalg.svd(result.jac, full_matrices=False)
    return {'centre_px': float(result.x[0]), 'width_px': float(result.x[1]),
            'blur_sigma_px': float(result.x[2]),
            'rgb_rmse': float(np.sqrt(np.mean((predicted-rgb)**2))),
            'latent_colours': colours.tolist(), 'predicted_rgb': predicted.tolist(),
            'jacobian_singular_values': singular.tolist(),
            'condition_number': float(singular[0]/max(singular[-1], 1e-12)),
            'solutions': [{'width_px': float(r.x[1]), 'sigma_px': float(r.x[2]),
                           'loss': float(np.mean(residual(r.x)**2))} for r in candidates],
            'proposal_only': True, 'accepted': False,
            'scope': 'Independent source RGB blurred-strip diagnostic; width/blur identifiability and source-only visual review remain required.'}
