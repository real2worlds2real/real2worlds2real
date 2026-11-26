"""Bound the real native reset step without replacing simulated state."""
import math


def reset(world, *, bootstrap_hz=None):
    dt=float(world.get_physics_dt())
    if not math.isfinite(dt) or dt<=0:
        raise ValueError('Finite positive task physics timestep required')
    if bootstrap_hz is None:
        world.reset()
    else:
        if (type(bootstrap_hz) is not int or not 1/dt-1e-6<=bootstrap_hz<=1000000):
            raise ValueError('Reset clock must be between the task clock and1000000Hz')
        try:
            world.set_simulation_dt(physics_dt=1/bootstrap_hz)
            world.reset()
        finally:
            world.set_simulation_dt(physics_dt=dt)
    return {'task_physics_dt_s':dt,'reset_physics_dt_s':dt if bootstrap_hz is None else 1/bootstrap_hz,
            'state_overwritten':False,'scope':'Real native reset with its actual positions and velocities retained; no task or contact approval.'}
