"""Articulation-independent metric seed from visible fixed-root CAD surfaces.

Observed tracks query the CAD surface. Unobserved CAD backsides and wheel cavities
are not treated as missing photo evidence. Multi-start disagreement abstains.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits
from .visual_robot_fit import rotation


@threadpool_limits.wrap(limits=1, user_api='blas')
def fit(model, observed):
    observed = np.asarray(observed, float)
    if observed.ndim != 2 or observed.shape[1] != 3 or len(observed) < 100 or not np.isfinite(observed).all():
        raise ValueError('Fixed-base fitting needs100 independent finite observed tracks')
    fixed = model.fixed_root()
    source, _, _ = fixed.points()
    tree = cKDTree(source)
    floor = fixed.floor_height()
    size = np.ptp(source, axis=0)
    height = source[:, 2].max()-floor
    initial_scale = max(.01, float(np.percentile(observed[:, 2], 98)/height))
    rng = np.random.default_rng(0)
    observed = observed[rng.permutation(len(observed))[:4000]]
    validation, train = observed[::4], observed[np.arange(len(observed)) % 4 != 0]
    centre = np.median(train[:, :2], axis=0)
    radius = max(initial_scale, float(np.linalg.norm(np.ptp(train[:, :2], axis=0))))
    lower = np.r_[centre-radius, -2*np.pi, np.log(initial_scale*.5)]
    upper = np.r_[centre+radius, 2*np.pi, np.log(initial_scale*2)]
    hypotheses = []
    for yaw in np.radians(np.arange(0, 360, 30)):
        for scale in (initial_scale*.8, initial_scale, initial_scale*1.25):
            values = np.r_[centre, yaw, np.log(scale)]
            for _ in range(35):
                k = np.exp(values[3])
                local = (train-[*values[:2], -k*floor])@rotation(values[2])/k
                distance, nearest = tree.query(local)
                chosen = np.argsort(distance)[:max(40, int(.75*len(train)))]
                target = source[nearest[chosen]]
                def residual(candidate):
                    k = np.exp(candidate[3])
                    q = (train[chosen]-[*candidate[:2], -k*floor])@rotation(candidate[2])/k
                    return (q-target).ravel()
                result = least_squares(residual, values, bounds=(lower, upper), loss='soft_l1',
                                       f_scale=.01, max_nfev=15, ftol=1e-5, xtol=1e-5, gtol=1e-5)
                delta = np.linalg.norm(result.x-values)
                values = result.x
                if delta < 1e-5:
                    break
            k = float(np.exp(values[3]))
            local = (validation-[*values[:2], -k*floor])@rotation(values[2])/k
            distance = tree.query(local)[0]
            score = float(np.mean(np.sort(distance)[:max(20, int(.75*len(distance)))]))
            hypotheses.append({'values': values, 'score': score, 'distance': distance, 'local': local})
    hypotheses.sort(key=lambda h: h['score'])
    best = hypotheses[0]
    plausible = [h for h in hypotheses if h['score'] <= best['score']*1.15+.001]
    scales = np.exp([h['values'][3] for h in plausible])
    spread = float(np.ptp(scales)/np.median(scales))
    coverage = float((best['distance'] < .02).mean())
    residual = float(np.median(best['distance']))
    inliers = best['local'][best['distance'] < .02]
    span = np.ptp(inliers, axis=0)/size if len(inliers) else np.zeros(3)
    reasons = []
    if residual > .02 or coverage < .7:
        reasons.append('Fixed-root CAD is not supported by enough independent observed surface tracks')
    if spread > .05:
        reasons.append('Similarly supported rigid-base hypotheses disagree in scale')
    if max(span[:2]) < .5 or span[2] < .3:
        reasons.append('Fixed-base observations have insufficient spatial extent to determine scale')
    k = float(np.exp(best['values'][3]))
    return {'ok': not reasons, 'reasons': reasons, 'map_units_per_metre': k,
            'base_xy_map': best['values'][:2].tolist(), 'yaw_deg': float(np.degrees(best['values'][2])),
            'base_z_map': float(-k*floor), 'residual_median_m': residual, 'observed_coverage': coverage,
            'plausible_scale_spread': spread, 'plausible_scales': scales.tolist(),
            'observed_span_over_cad': span.tolist(), 'validation_points': len(validation),
            'hypotheses': len(hypotheses), 'criterion': 'visible tracks to fixed CAD surface; 20mm,70% support,5% multistart scale spread',
            'limitations': ['Visibility is partial; no missing-observation penalty on CAD backsides',
                           'Root yaw can remain ambiguous; articulation fit still searches the full circle']}
