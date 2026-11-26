"""Pass finite external wrenches through the active native tensor backend.

Isaac RigidPrim views may use NumPy or Torch independently of whether PhysX
runs on the GPU. Never infer the tensor type from gpu_dynamics alone.
"""
import numpy as np


def apply(view, forces, torques=None, *, positions=None, is_global=True):
    if type(is_global) is not bool:
        raise ValueError('Explicit wrench coordinate frame required')
    values = {}
    for name, raw in [('forces', forces), ('torques', torques), ('positions', positions)]:
        if raw is None:
            continue
        value = np.asarray(raw, dtype=np.float32)
        if (value.ndim != 2 or value.shape[1:] != (3,) or not len(value)
                or not np.isfinite(value).all()):
            raise ValueError('Finite N x 3 force/torque/position arrays required')
        values[name] = value
    if 'forces' not in values:
        raise ValueError('An explicit external force array is required')
    if any(value.shape != values['forces'].shape for value in values.values()):
        raise ValueError('Wrench components must address the same bodies')
    converted = {name: view._backend_utils.convert(value, device=view._device)
                 for name, value in values.items()}
    view.apply_forces_and_torques_at_pos(**converted, is_global=is_global)
