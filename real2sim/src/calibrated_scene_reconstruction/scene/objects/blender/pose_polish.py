"""Bounded exact-raster pose refinement, preserving geometry and source gates."""
import numpy as np
from scipy.optimize import minimize

from . import evidence
from .contract import write_json


def run(vertices, faces, pose, observations, *, model_parts=None,
        parameters=('xy', 'yaw_deg'), budget=40, initial_metrics=None, checkpoint=None):
    if (not parameters or len(set(parameters)) != len(parameters)
            or set(parameters) - {'xy', 'yaw_deg'} or type(budget) is not int
            or not 1 <= budget <= 120):
        raise ValueError('Only bounded xy/yaw refinement with a finite evaluation budget is supported')
    best_pose = dict(pose)
    best_metrics = initial_metrics if initial_metrics is not None else evidence.score(
        vertices, faces, pose, observations, model_parts=model_parts)[0]
    history = []
    width = (2 if 'xy' in parameters else 0) + (1 if 'yaw_deg' in parameters else 0)
    report = {'geometry_changed': False, 'height_changed': False, 'acceptance_changed': False,
              'parameters': list(parameters), 'bounded_xy_m': .001, 'bounded_yaw_deg': .2,
              'evaluations': history, 'task_approved': False}

    def save():
        report['height_changed']=best_pose.get('support_z')!=pose.get('support_z')
        report['height_policy']=('actual support triangles' if pose.get('support_triangles') else
                                 'actual reviewed support plane' if pose.get('support_plane') else 'fixed source height')
        report.update(pose=best_pose, source_pass=bool(best_metrics['ok']),
                      source_metrics={k: best_metrics[k] for k in ('iou_median', 'local_difference', 'ok')})
        if checkpoint is not None:
            write_json(checkpoint, report)

    class Accepted(Exception):
        pass

    def objective(offset):
        nonlocal best_pose, best_metrics
        candidate = dict(pose); offset = np.asarray(offset, float)
        if offset.shape != (width,) or not np.isfinite(offset).all() or (np.abs(offset) > 1 + 1e-10).any():
            raise ValueError('CPU pose optimizer exceeded its geometric bounds')
        cursor = 0
        if 'xy' in parameters:
            candidate['xy'] = (np.asarray(pose['xy']) + .001 * offset[:2]).tolist(); cursor = 2
        if 'yaw_deg' in parameters:
            candidate['yaw_deg'] = float(pose['yaw_deg'] + .2 * offset[cursor])
        if candidate.get('support_plane') or candidate.get('support_triangles'):
            from .fit import ground_pose
            candidate=ground_pose(vertices,model_parts,candidate,faces=faces)
        current, _ = evidence.score(vertices, faces, candidate, observations, model_parts=model_parts)
        score = evidence.search_objective(current)
        history.append({'offset': offset.tolist(), 'source_score': score, 'ok': bool(current['ok'])})
        if (bool(current['ok']), -score) > (bool(best_metrics['ok']), -evidence.search_objective(best_metrics)):
            best_pose, best_metrics = candidate, current
        save()
        # Exact source gates decide acceptance. Further optimization after a
        # passing pose wastes time and risks hiding a valid intermediate state.
        if current['ok']:
            raise Accepted()
        return score

    save()
    if not best_metrics['ok']:
        try:
            minimize(objective, np.zeros(width), method='Nelder-Mead', bounds=[(-1, 1)] * width,
                     options={'maxfev': budget, 'xatol': .001, 'fatol': 1e-7,
                              'initial_simplex': np.r_[np.zeros((1, width)), np.eye(width) * .15]})
        except Accepted:
            pass
    save()
    return best_pose, best_metrics, report
