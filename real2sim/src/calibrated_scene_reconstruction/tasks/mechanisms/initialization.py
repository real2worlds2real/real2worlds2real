"""Native soft-body initialization candidates, separate from task validation.

The agent may archive a relaxed candidate as declared model data, then rebuild
and rerun image, material, contact and task gates. A relaxation trace never
approves a task or silently changes a delivered asset's initial state.
"""
import numpy as np


def relaxation_trace(ctx, name, metric, *, seconds=2., sample_every=64):
    dt = float(ctx.world.get_physics_dt())
    if (not np.isfinite([dt, seconds]).all() or not 0 < dt <= 1/960
            or not dt <= seconds <= 10 or type(sample_every) is not int or sample_every < 1):
        raise ValueError('Bounded native initialization at960Hz or finer required')
    initial = ctx.nodes(name).copy()
    points, times, measurements = [initial], [0.], [metric.check(initial)]
    for step in range(round(seconds/dt)):
        ctx.world.step(render=False)
        current = ctx.nodes(name).copy()
        if (step+1) % sample_every == 0:
            points.append(current); times.append((step+1)*dt); measurements.append(metric.check(current))
    if times[-1] < (step+1)*dt:
        points.append(current); times.append((step+1)*dt); measurements.append(metric.check(current))
    return {'initialization_only': True, 'delivery_gate_eligible': False,
            'positions_prescribed': False, 'external_cloth_forces_applied': False,
            'physics_dt': dt, 'points': np.asarray(points), 'times': np.asarray(times),
            'material': measurements,
            'scope': 'Actual native solver states; evaluate the rebuilt candidate independently before approval'}
