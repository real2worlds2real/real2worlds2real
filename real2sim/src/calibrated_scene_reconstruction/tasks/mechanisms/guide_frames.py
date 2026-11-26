"""Coherent finite-force guide targets; never material or native state writes."""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def paths(centres, reference, translation, times, rotations, *, angle_step_rad):
    """Move a group through a proper frame, bounding linear-path chord loss.

    The existing force controller interpolates target positions linearly.
    Subdivide rotations explicitly so that this interpolation cannot silently
    reverse or substantially compress the group's relative target offsets.
    This is a construction proposal, not a rigid material patch or fit proof.
    """
    c, ref, path, t, r = map(lambda v: np.asarray(v, float),
                           [centres, reference, translation, times, rotations])
    if (c.ndim != 2 or c.shape[1:] != (3,) or not len(c) or ref.shape != (3,)
            or path.ndim != 2 or path.shape[1:] != (4,) or len(path) < 2
            or t.ndim != 1 or len(t) < 2 or r.shape != (len(t), 3, 3)
            or not all(np.isfinite(v).all() for v in [c, ref, path, t, r])
            or (np.diff(t) <= 0).any() or (np.diff(path[:, 0]) <= 0).any()
            or t[0] < 0 or path[0, 0] != t[0] or path[-1, 0] != t[-1]
            or not np.isfinite(angle_step_rad) or not 0 < angle_step_rad <= np.pi/12):
        raise ValueError('Finite guide centres, shared bounded clock and small rotation step required')
    if (not np.allclose(np.swapaxes(r, 1, 2) @ r, np.eye(3), atol=1e-8, rtol=0)
            or not np.allclose(np.linalg.det(r), 1., atol=1e-8, rtol=0)):
        raise ValueError('Guide frames must be proper unscaled rotations, never reflections')
    if not np.allclose(r[0], np.eye(3), atol=1e-8, rtol=0) or not np.allclose(path[0, 1:], ref, atol=1e-10, rtol=0):
        raise ValueError('The shared frame must start at the actual reference centre without rotation')
    key_rotation = Rotation.from_matrix(r)
    angles = (key_rotation[:-1].inv() * key_rotation[1:]).magnitude()
    samples = list(path[:, 0])
    for a, b, angle in zip(t[:-1], t[1:], angles):
        count = max(1, int(np.ceil(angle / angle_step_rad)))
        if count > 1000:
            raise ValueError('Guide-frame rotation exceeds the sampling budget')
        samples.extend(np.linspace(a, b, count + 1))
    clock = np.unique(samples)
    if len(clock) > 5000:
        raise ValueError('Guide-frame path exceeds the sampling budget')
    orientation = Slerp(t, key_rotation)(clock)
    position = np.array([np.interp(clock, path[:, 0], path[:, i]) for i in [1, 2, 3]]).T
    targets = np.einsum('tij,nj->nti', orientation.as_matrix(), c-ref) + position
    result = np.concatenate([np.broadcast_to(clock[None, :, None], (*targets.shape[:2], 1)), targets], axis=2)
    actual_step = (orientation[:-1].inv() * orientation[1:]).magnitude().max()
    return {'paths': result.tolist(), 'maximum_rotation_step_rad': float(actual_step),
            'maximum_relative_target_distance_loss': float(1-np.cos(actual_step/2)),
            'native_positions_prescribed': False, 'task_approved': False,
            'scope': 'Shared frame targets for temporary finite-force guides. '
                     'Linear interpolation has the reported chord bound; actual material and release remain unchecked.'}
