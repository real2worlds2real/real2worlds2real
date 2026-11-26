"""Estimate robot scale, root pose and articulation from observed surface points.

No hardware joint file, task size or previous fit is an input. CAD home is only
one initialization hypothesis. Independent scene points select hypotheses and
ambiguous scale solutions abstain instead of silently becoming metric truth.
"""
from dataclasses import dataclass
import re

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits


@dataclass
class Settings:
    keep: float = .65
    outer_iterations: int = 30
    inner_evaluations: int = 15
    yaw_step_deg: int = 30
    coarse_iterations: int = 5
    finalists: int = 6
    points_per_link: int = 160
    max_scene_points: int = 12000
    min_scene_points: int = 400
    min_coverage: float = .30
    max_residual_m: float = .02
    max_scale_spread: float = .05
    visible_link_fraction: float = .75
    min_link_extent_m: float = .06
    exclude_links: tuple = ('hand|index|middle|ring|pinky|thumb',)
    exclude_joints: tuple = ('index|middle|ring|pinky|thumb',)


def rotation(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def sample_surfaces(model, cfg):
    local = {}
    areas = {name: float(np.prod(np.sort(np.ptp(points, axis=0))[-2:]))
             for name, points in model.surfaces.items()}
    area_max = max(areas.values())
    for name, points in model.surfaces.items():
        if any(re.search(pattern, name, re.I) for pattern in cfg.exclude_links):
            continue
        if np.linalg.norm(np.ptp(points, axis=0)) < cfg.min_link_extent_m:
            continue
        count = max(8, round(cfg.points_per_link*np.sqrt(areas[name]/area_max)))
        local[name] = points[np.linspace(0, len(points)-1, min(len(points), count)).astype(int)]
    return local


def evidence_metrics(model, observed, candidate, settings=None):
    """Apply the same held-out 3D evidence criterion to a refined pose."""
    cfg = settings or Settings()
    local = sample_surfaces(model, cfg)
    transforms = model.transforms(candidate['estimated_joints_rad'])
    source = np.concatenate([points@transforms[name][:3, :3].T+transforms[name][:3, 3]
                             for name, points in local.items()])
    labels = np.concatenate([np.full(len(points), i) for i, points in enumerate(local.values())])
    k = candidate['map_units_per_metre']
    predicted = k*source@rotation(np.radians(candidate['yaw_deg'])).T+[
        *candidate['base_xy_map'], candidate['base_z_map']]
    observed = np.asarray(observed, float)
    selected = observed[np.random.default_rng(0).permutation(len(observed))[:cfg.max_scene_points]]
    validation = selected[::4]
    distance = cKDTree(validation).query(predicted)[0]/k
    residual = float(np.median(np.sort(distance)[:int(cfg.keep*len(distance))]))
    coverage = float((distance < cfg.max_residual_m).mean())
    supported = [name for i, name in enumerate(local)
                 if (distance[labels == i] < cfg.max_residual_m).mean() >= .2]
    reasons = []
    if residual > cfg.max_residual_m:
        reasons.append('CAD/observation residual exceeds the shared visual criterion')
    if coverage < cfg.min_coverage:
        reasons.append('Too little CAD surface is supported by independent observations')
    if len(supported) < 2:
        reasons.append('Fewer than two CAD links have observed surface support')
    return {'ok': not reasons, 'reasons': reasons, 'residual_median_m': residual,
            'surface_coverage': coverage, 'supported_links': supported, 'validation_points': len(validation)}


@threadpool_limits.wrap(limits=1, user_api='blas')
def fit(model, scene, initial=None, settings=None, base_scene=None):
    """Fit CAD into an upright, provisionally scaled map with floor at z=0.

    ``initial`` may contain CAD ``home`` and coarse root ``xy/yaw_deg``. It must
    not contain measured capture joints or a previous-session fitted state.
    Returned ``map_units_per_metre`` rescales the map by its reciprocal.
    """
    cfg = settings or Settings()
    if not 0 < cfg.keep <= 1 or cfg.points_per_link < 8:
        raise ValueError('Valid trimmed coverage and per-link sampling required')
    if (not 0 < cfg.visible_link_fraction <= 1 or cfg.yaw_step_deg not in (15, 30, 45, 60, 90)
            or min(cfg.outer_iterations, cfg.inner_evaluations, cfg.coarse_iterations, cfg.finalists) < 1
            or cfg.min_scene_points < 8 or cfg.max_scene_points < cfg.min_scene_points
            or not 0 < cfg.min_coverage <= 1 or not cfg.max_residual_m > 0):
        raise ValueError('Invalid shared visual-fit search or evidence settings')
    observed = np.asarray(scene, float)
    if observed.ndim != 2 or observed.shape[1] != 3 or not np.isfinite(observed).all():
        raise ValueError('Finite (N,3) observed robot points required')
    if len(observed) < cfg.min_scene_points:
        raise ValueError('Too few independent robot surface observations')
    initial = initial or {}
    rng = np.random.default_rng(0)
    selected = rng.permutation(len(observed))[:cfg.max_scene_points]
    observed = observed[selected]
    validation = observed[::4]
    training = observed[np.arange(len(observed)) % 4 != 0]
    tree = cKDTree(training)
    base_tree = None
    if base_scene is not None:
        base_scene = np.asarray(base_scene, float)
        if base_scene.ndim != 2 or base_scene.shape[1] != 3 or not np.isfinite(base_scene).all() or len(base_scene) < 100:
            raise ValueError('Fixed-base evidence needs at least100 finite independent tracks')
        base_tree = cKDTree(base_scene)
    joints = [j for j in model.movable if j.upper > j.lower and not any(
        re.search(pattern, j.name, re.I) for pattern in cfg.exclude_joints)]
    homes = initial.get('home') or {}
    zero = {j.name: float(np.clip(homes.get(j.name, 0.), j.lower, j.upper)) for j in model.movable}
    floor = model.floor_height()
    # Cache each selected local surface so changing an articulation never
    # resamples its evidence or weights a highly tessellated link more heavily.
    # Approximate surface area controls density; tiny cameras cannot outweigh
    # the body supplying metric scale.
    local = sample_surfaces(model, cfg)
    if len(local) < 2:
        raise ValueError('At least two CAD link surfaces required for visual calibration')
    names = list(local)
    labels = np.concatenate([np.full(len(local[n]), i, int) for i, n in enumerate(names)])
    base_labels = np.isin(labels, [i for i, name in enumerate(names) if name in model.fixed_root().surfaces])
    qlo = np.array([j.lower for j in joints])
    qhi = np.array([j.upper for j in joints])
    home = np.array([zero[j.name] for j in joints])

    def points(values):
        state = {**zero, **dict(zip((j.name for j in joints), values[4:]))}
        transforms = model.transforms(state)
        source = np.concatenate([local[n] @ transforms[n][:3, :3].T + transforms[n][:3, 3] for n in names])
        scale = np.exp(values[3])
        placed = scale * source @ rotation(values[2]).T + [values[0], values[1], -scale*floor]
        return placed, state

    # Bootstrap root location is only a search centre; the whole robot's
    # orientation is unknown and gets a complete yaw search.
    centre = np.asarray(initial.get('xy', np.median(training[:, :2], axis=0)), float)
    if 'map_units_per_metre' in initial:
        scale0 = float(initial['map_units_per_metre'])
    else:
        home_points, _, _ = model.points(zero, exclude_links=cfg.exclude_links)
        home_height = float(np.percentile(home_points[:, 2]-floor, 98))
        observed_height = float(np.percentile(training[:, 2], 98))
        scale0 = max(.01, observed_height/home_height)
    yaw0 = np.radians(float(initial.get('yaw_deg', 0.)))
    if centre.shape != (2,) or not np.isfinite([*centre, scale0, yaw0]).all() or scale0 <= 0:
        raise ValueError('Finite XY/yaw and positive initial map scale required')
    radius = max(.5*scale0, float(np.linalg.norm(np.ptp(training[:, :2], axis=0))))
    lower = np.r_[centre-radius, yaw0-2*np.pi, np.log(scale0*.5), qlo]
    upper = np.r_[centre+radius, yaw0+2*np.pi, np.log(scale0*2.), qhi]
    hypotheses = []
    articulation_seeds = []
    for seed in (home, .5*(home+np.clip(np.zeros_like(home), qlo, qhi)),
                 .25*qlo+.75*qhi, .75*qlo+.25*qhi):
        if not any(np.allclose(seed, previous) for previous in articulation_seeds):
            articulation_seeds.append(seed)
    for yaw in yaw0+np.radians(np.arange(0, 360, cfg.yaw_step_deg)):
        for scale in (scale0*.8, scale0, scale0*1.25):
            for articulation in articulation_seeds:
                hypotheses.append(np.r_[centre, yaw, np.log(scale), articulation])

    def optimize(start, iterations):
        values = np.clip(start, lower+1e-10, upper-1e-10)
        for _ in range(iterations):
            source, _ = points(values)
            distance, nearest = tree.query(source)
            # Trim within each rigid link before trimming invisible links.
            # Global point trimming alone lets a well-fitted torso discard an
            # entire misposed arm, so articulation never receives its evidence.
            by_link = []
            for index in range(len(names)):
                index_points = np.flatnonzero(labels == index)
                keep_points = index_points[np.argsort(distance[index_points])[:max(4, int(cfg.keep*len(index_points)))]]
                by_link.append((float(np.median(distance[keep_points])), keep_points))
            by_link.sort(key=lambda pair: pair[0])
            visible = max(2, int(np.ceil(cfg.visible_link_fraction*len(by_link))))
            chosen = np.concatenate([pair[1] for pair in by_link[:visible]])
            target = training[nearest[chosen]]
            backward_distance, backward_nearest = cKDTree(source).query(training)
            back = np.argsort(backward_distance)[:max(8, int(cfg.keep*len(training)))]
            back_source = backward_nearest[back]
            back_weight = np.sqrt(len(chosen)/len(back))
            base_chosen, base_target, base_back = [], None, []
            if base_tree is not None:
                base_indices = np.flatnonzero(base_labels)
                base_distance, base_nearest = base_tree.query(source[base_indices])
                selected_base = np.argsort(base_distance)[:max(8, int(cfg.keep*len(base_indices)))]
                base_chosen = base_indices[selected_base]
                base_target = base_scene[base_nearest[selected_base]]
                base_back_distance, base_back_nearest = cKDTree(source[base_indices]).query(base_scene)
                base_back = np.argsort(base_back_distance)[:max(8, int(cfg.keep*len(base_scene)))]
                base_back_source = base_indices[base_back_nearest[base_back]]

            def residual(candidate):
                pred, _ = points(candidate)
                # Fixed correspondences inside each ICP step; scale-normalized
                # residuals keep shrinking CAD from artificially winning.
                scale = np.exp(candidate[3])
                values = [((pred[chosen]-target)/scale).ravel(),
                          (back_weight*(pred[back_source]-training[back])/scale).ravel()]
                if len(base_chosen):
                    values.append((np.sqrt(len(chosen)/len(base_chosen))*
                                   (pred[base_chosen]-base_target)/scale).ravel())
                    values.append((np.sqrt(len(chosen)/len(base_back))*
                                   (pred[base_back_source]-base_scene[base_back])/scale).ravel())
                return np.concatenate(values)

            result = least_squares(residual, values, bounds=(lower, upper),
                                   max_nfev=cfg.inner_evaluations, loss='soft_l1', f_scale=.01,
                                   diff_step=1e-4, ftol=1e-4, xtol=1e-4, gtol=1e-4)
            delta = np.linalg.norm(result.x-values)
            values = result.x
            if delta < 1e-5:
                break
        pred, state = points(values)
        scale = float(np.exp(values[3]))
        forward = cKDTree(validation).query(pred)[0]/scale
        backward = cKDTree(pred).query(validation)[0]/scale
        score = float(np.mean(np.sort(forward)[:max(8, int(cfg.keep*len(forward)))]) +
                      np.mean(np.sort(backward)[:max(8, int(cfg.keep*len(backward)))]))
        if base_tree is not None:
            base_forward = base_tree.query(pred[base_labels])[0]/scale
            base_backward = cKDTree(pred[base_labels]).query(base_scene)[0]/scale
            score += float(np.mean(np.sort(base_forward)[:max(8, int(cfg.keep*len(base_forward)))])+
                           np.mean(np.sort(base_backward)[:max(8, int(cfg.keep*len(base_backward)))]))
        return dict(values=values, state=state, score=score, forward=forward, backward=backward,
                    map_units_per_metre=scale)

    # First compare root/yaw/scale hypotheses briefly; retain several distinct
    # minima rather than treating one home-pose fit as an observed articulation.
    coarse = []
    for i, start in enumerate(hypotheses):
        coarse.append(optimize(start, min(cfg.coarse_iterations, cfg.outer_iterations)))
        if (i+1) % 12 == 0:
            print(f'[robot_visual] root/articulation search {i+1}/{len(hypotheses)}; '
                  f'best held-out residual {min(r["score"] for r in coarse):.4f}m', flush=True)
    coarse.sort(key=lambda result: result['score'])
    finalists = [optimize(result['values'], cfg.outer_iterations) for result in coarse[:cfg.finalists]]
    finalists.sort(key=lambda result: result['score'])
    best = finalists[0]
    plausible = [r for r in finalists if r['score'] <= best['score']*1.15+.001]
    scales = np.array([r['map_units_per_metre'] for r in plausible])
    spread = float(np.ptp(scales)/np.median(scales))
    residual = float(np.median(np.sort(best['forward'])[:int(cfg.keep*len(best['forward']))]))
    coverage = float((best['forward'] < cfg.max_residual_m).mean())
    supported_links = [name for i, name in enumerate(names)
                       if (best['forward'][labels == i] < cfg.max_residual_m).mean() >= .2]
    reasons = []
    if residual > cfg.max_residual_m:
        reasons.append('CAD/observation residual exceeds the shared visual criterion')
    if coverage < cfg.min_coverage:
        reasons.append('Too little CAD surface is supported by independent observations')
    if len(supported_links) < 2:
        reasons.append('Fewer than two CAD links have observed surface support')
    if spread > cfg.max_scale_spread:
        reasons.append('Several similarly supported articulation hypotheses disagree in scale')
    estimated = [j.name for j in joints]
    return {'ok': not reasons, 'reasons': reasons, 'source': 'visual CAD articulation, no supplied joint state',
            'map_units_per_metre': best['map_units_per_metre'],
            'metres_per_map_unit': 1/best['map_units_per_metre'],
            'base_xy_map': best['values'][:2].tolist(),
            'base_z_map': float(-best['map_units_per_metre']*floor),
            'yaw_deg': float(np.degrees(best['values'][2])),
            'estimated_joints_rad': {name: best['state'][name] for name in estimated},
            'unestimated_joints': sorted(set(zero)-set(estimated)),
            'residual_median_m': residual, 'surface_coverage': coverage,
            'plausible_scale_spread': spread, 'plausible_scales': scales.tolist(),
            'validation_points': len(validation), 'training_points': len(training),
            'model_points': len(labels), 'model_links': names,
            'supported_links': supported_links,
            'hypotheses': len(hypotheses), 'finalist_scores': [r['score'] for r in finalists],
            'initial_scale_hypothesis': scale0,
            'floor_constraint': 'fixed CAD root/wheel surfaces rest on the observed floor',
            'limitations': ['Unestimated finger states remain unobserved, not measured home values',
                           'Multi-start spread is an ambiguity diagnostic, not a ground-truth accuracy bound']}
