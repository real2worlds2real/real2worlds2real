"""Measure actual relative thread motion; never prescribe a helix.

Pose agreement alone is not proof of contact. The caller must additionally
record native contacts, force bounds, retention, release and negative controls.
"""
import numpy as np
from scipy.spatial.transform import Rotation






def motion(base_p, base_q, moving_p, moving_q, *, lead_m_per_turn, axis=(0, 0, 1),
           tolerance_m, minimum_turns=.5, base_origin=(0., 0., 0.), moving_origin=(0., 0., 0.)):
    arrays = [np.asarray(v, float) for v in (base_p, base_q, moving_p, moving_q)]
    bp, bq, mp, mq = arrays
    n = len(bp)
    if (n < 3 or bp.shape != (n, 3) or mp.shape != (n, 3) or bq.shape != (n, 4)
            or mq.shape != (n, 4) or any(not np.isfinite(v).all() for v in arrays)):
        raise ValueError('At least three finite paired poses required; quaternions are wxyz')
    if not np.isfinite([lead_m_per_turn, tolerance_m, minimum_turns]).all() or lead_m_per_turn == 0 or tolerance_m <= 0 or minimum_turns <= 0:
        raise ValueError('Finite signed lead, positive tolerance and minimum turns required')
    axis = np.asarray(axis, float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or abs(np.linalg.norm(axis)-1) > 1e-6:
        raise ValueError('Thread axis must be normalized in the base body frame')
    if any(np.max(abs(np.linalg.norm(q, axis=1)-1)) > 1e-4 for q in (bq, mq)):
        raise ValueError('Unit quaternions required')
    base = Rotation.from_quat(bq[:, [1, 2, 3, 0]])
    moving = Rotation.from_quat(mq[:, [1, 2, 3, 0]])
    origins = [np.asarray(v, float) for v in (base_origin, moving_origin)]
    if any(v.shape != (3,) or not np.isfinite(v).all() for v in origins):
        raise ValueError('Finite link-local functional origins required')
    bp = bp+base.apply(origins[0])
    mp = mp+moving.apply(origins[1])
    relative = base.inv()*moving
    q = relative.as_quat()
    twist = q[:, :3] @ axis
    if np.min(q[:, 3]**2+twist**2) < 1e-6:
        raise ValueError('Undefined twist at 180 degree swing')
    angle = np.unwrap(2*np.arctan2(twist, q[:, 3]))
    # Quaternion sign changes may introduce whole turns; unwrap above removes them.
    if np.max(np.abs(np.diff(angle))) >= .9*np.pi:
        raise ValueError('Undersampled thread rotation')
    turns = (angle-angle[0])/(2*np.pi)
    local = base.inv().apply(mp-bp)
    displacement = local-local[0]
    axial = displacement@axis
    drift = displacement-axial[:, None]*axis
    residual = axial-lead_m_per_turn*turns
    enough = abs(float(turns[-1])) >= minimum_turns
    lateral = float(np.linalg.norm(drift, axis=1).max())
    return {'pass': bool(enough and np.max(abs(residual)) <= tolerance_m and lateral <= tolerance_m),
            'turns': float(turns[-1]), 'axial_m': float(axial[-1]),
            'lead_m_per_turn': lead_m_per_turn,
            'max_lead_error_m': float(np.max(abs(residual))),
            'max_lateral_drift_m': lateral,
            'tolerance_m': tolerance_m, 'minimum_turns': minimum_turns}


def precision(*, extent_m, resolution, pitch_m, depth_m, clearance_m, contact_offset_m, dt, max_radians_s):
    """Conservative authoring budget, followed by actual cooking/contact tests."""
    values = [extent_m, resolution, pitch_m, depth_m, clearance_m, contact_offset_m, dt, max_radians_s]
    if not np.isfinite(values).all() or min(values) <= 0:
        raise ValueError('Positive finite contact precision inputs required')
    voxel = extent_m/resolution
    conditions = {'feature_resolution': voxel <= min(pitch_m/8, depth_m/3, clearance_m/2),
                  'contact_offset': contact_offset_m <= clearance_m/3,
                  'rotation_step': dt*max_radians_s <= np.pi/90}
    return {'pass': all(conditions.values()), 'checks': conditions, 'voxel_m': voxel,
            'note': 'Budget only; actual native SDF cooking and functional tests still required'}


def engagement_window(base_band, moving_band, pitch_m):
    """Relative-Z interval with a full pitch of untapered tooth overlap.

    Bands come from the actual generated geometry in each body frame. Half a
    pitch at each end is its finite lead-in/runout, not a fitted-error filter.
    Entry and disengagement still remain in the full saved motion trace.
    """
    a, b = np.asarray(base_band, float)
    c, d = np.asarray(moving_band, float)
    if not np.isfinite([a, b, c, d, pitch_m]).all() or pitch_m <= 0 or a >= b or c >= d:
        raise ValueError('Finite ordered thread bands and positive pitch required')
    lo = a+2*pitch_m-d
    hi = b-2*pitch_m-c
    if lo >= hi:
        raise ValueError('Thread bands have no fully engaged interval')
    return float(lo), float(hi)


def seating_stall(values, *, seated_z, tolerance_m, angle_tolerance_rad=.01, window_s=.5):
    """Bound actual relative angle spread over a full window, not noisy velocity.

    Values use the contact runner's recorded base/moving poses and axial column.
    A rotating or reciprocating cap cannot pass merely because its net angle is0.
    """
    v=np.asarray(values,float)
    if v.ndim!=2 or v.shape[1]!=24 or len(v)<3 or not np.isfinite(v).all():return False
    v=v[v[:,0]>=v[-1,0]-window_s-1e-7]
    if len(v)<3 or v[-1,0]-v[0,0]<window_s-1e-5:return False
    if np.max(abs(v[:,-1]-seated_z))>tolerance_m:return False
    b=Rotation.from_quat(v[:,4:8][:,[1,2,3,0]]);c=Rotation.from_quat(v[:,11:15][:,[1,2,3,0]])
    q=(b.inv()*c).as_quat();angle=np.unwrap(2*np.arctan2(q[:,2],q[:,3]))
    return bool(np.ptp(angle)<=angle_tolerance_rad)


def torque_limited_stall(values, *, torque_limit_Nm, axis, window_s=3.):
    """Detect an immobile, torque-saturated tool; never approve a task from it.

    This only permits an early failed experiment instead of a long timeout.
    Actual motion, release and every acceptance criterion still apply.
    """
    v=np.asarray(values,float);axis=np.asarray(axis,float)
    if (v.ndim!=2 or v.shape[1]!=24 or len(v)<3 or not np.isfinite(v).all()
            or torque_limit_Nm<=0 or window_s<.5 or axis.shape!=(3,)
            or not np.isclose(np.linalg.norm(axis),1.)):return False
    v=v[v[:,0]>=v[-1,0]-window_s-1e-7]
    if len(v)<3 or v[-1,0]-v[0,0]<window_s-1e-5:return False
    if np.mean(abs(v[:,20:23]@axis)>=.99*torque_limit_Nm)<.9:return False
    b=Rotation.from_quat(v[:,4:8][:,[1,2,3,0]]);c=Rotation.from_quat(v[:,11:15][:,[1,2,3,0]])
    q=(b.inv()*c).as_quat();angle=np.unwrap(2*np.arctan2(q[:,2],q[:,3]))
    return bool(np.ptp(angle)<.02 and np.ptp(v[:,-1])<.00005)
