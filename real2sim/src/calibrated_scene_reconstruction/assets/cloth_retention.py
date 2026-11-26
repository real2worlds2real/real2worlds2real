"""Observe retained material in actual body-local native snapshots.

This measures placement, not material identification or task success. Opening
column occupancy is kept separate from material actually below the rim.
"""
import numpy as np

from .cloth_container_state import check


def measure(points, rest, faces, body_poses, times, opening, request, *,
            released_at_s, physical_pass, complete):
    p, poses, t = map(lambda x: np.asarray(x, float), [points, body_poses, times])
    start = request['observation_start_s']
    duration = request['minimum_duration_s']
    loss = request['maximum_inside_loss_fraction']
    if (p.ndim != 3 or p.shape[2:] != (3,) or poses.shape != (len(p), 12)
            or t.shape != (len(p),) or len(t) < 1
            or not all(np.isfinite(x).all() for x in [p, poses, t])
            or (np.diff(t) <= 0).any()
            or not np.isfinite([start, duration, loss, released_at_s]).all()
            or start < released_at_s or duration < .5 or not 0 <= loss <= 1):
        raise ValueError('Actual ordered native snapshots and at least0.5s fully released retention interval required')
    rows = []
    for points_i, pose, time in zip(p, poses, t):
        rotation = pose[3:].reshape(3, 3)
        if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0)
                or not np.isclose(np.linalg.det(rotation), 1., atol=1e-5, rtol=0)):
            raise ValueError('Actual proper body frames required for retained material measurements')
        row = check((points_i - pose[:3]) @ rotation, rest, faces, opening,
                    request['requirements'])
        rows.append({'time_s': float(time), **row})
    # A snapshot at the release boundary can precede that step's clearing of
    # a persistent native drive. Observe strictly after its release time.
    window = [i for i, time in enumerate(t)
              if time >= start - 1e-9 and time > released_at_s + 1e-9]
    initial = rows[0]['measurements']['inside_fraction']
    lowest = min((rows[i]['measurements']['inside_fraction'] for i in window), default=0.)
    checks = {
        'complete_native_run': complete is True,
        'native_physical_validity': physical_pass is True,
        'unforced_observation_duration': bool(len(window) > 1 and t[window[-1]] - t[window[0]] >= duration - 1e-8),
        'resolved_observation_sampling': bool(len(window) > 1 and np.diff(t[window]).max() <= .05 + 1e-8),
        'initial_distribution': rows[0]['pass'],
        'retained_distribution': bool(window and all(rows[i]['pass'] for i in window)),
        'bounded_inside_material_loss': bool(window and initial - lowest <= loss + 1e-9),
    }
    return {'pass': all(checks.values()), 'checks': checks, 'rows': rows,
            'request': request, 'all_force_release_time_s': float(released_at_s),
            'initial_inside_fraction': initial, 'lowest_observed_inside_fraction': lowest,
            'maximum_inside_loss_fraction': max(0., initial - lowest),
            'task_approved': False,
            'scope': 'Actual body-local material area after fixture release; not contact force, source fit, grasp accessibility or extraction approval.'}
