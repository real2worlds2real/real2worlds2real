"""Closing-axis force control for two finite-force rigid contact tools.

This commands tool loads, never material nodes. A force cap on a position
servo alone does not establish preload after reaching the position target.
"""


def apply(force, velocity, axis, preload, damping, limits):
    """Replace only the closing-axis component; retain tangential positioning.

    Inputs are native torch tensors. Thumb then index approach along +axis
    and -axis. Preserve the position servo's common translation mode; replace
    only its differential closing mode. Damping must be discretized for the
    controller's sample/hold interval, not continuous high-gain damping.
    The caller must also
    cap the total wrench and check actual contact, penetration and retention.
    """
    import math
    import torch
    if not math.isfinite(preload) or preload <= 0:
        raise ValueError('Positive explicit pinch preload required')
    if (force.shape != (2, 3) or velocity.shape != (2, 3) or axis.shape != (3,)
            or damping.shape != (2, 1) or limits.shape != (2, 1)
            or not all(torch.isfinite(v).all() for v in [force, velocity, axis, damping, limits])
            or not torch.isclose(torch.linalg.vector_norm(axis), axis.new_tensor(1.))
            or (damping < 0).any() or (limits <= 0).any()):
        raise ValueError('Finite ordered tools, unit normal and positive source force limits required')
    sign = force.new_tensor([1., -1.])
    bound = limits[:, 0]
    normal_velocity=velocity@axis
    normal = (force@axis).mean() + sign * preload - damping[:, 0] * (normal_velocity-normal_velocity.mean())
    normal = torch.maximum(-bound, torch.minimum(bound, normal))
    return force + (normal - force @ axis)[:, None] * axis
