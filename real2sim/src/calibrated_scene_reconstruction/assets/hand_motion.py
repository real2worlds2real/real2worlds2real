"""Bind a contact control to a saved plan and compare executed command traces."""
from copy import deepcopy
import numpy as np


def translation_seconds(distance_m,maximum_speed_m_s):
    """Duration whose cubic smoothstep peak respects the declared speed."""
    if (not np.isfinite([distance_m,maximum_speed_m_s]).all()
            or min(distance_m,maximum_speed_m_s)<=0):
        raise ValueError('Positive finite travel and speed required')
    return float(1.5*distance_m/maximum_speed_m_s)


def force_limited_seconds(distance_m,requested_s,masses_kg,force_limits_N,*,force_fraction=.25):
    """Lower-bound a cubic translation by finite free-tool acceleration.

    Reserve most of the actuator budget for tracking and contact. This is a
    trajectory proposal, not a reach/contact certificate; native arrival and
    material capture must still be measured. It grants no gravity allowance.
    """
    mass,force=np.asarray(masses_kg,float),np.asarray(force_limits_N,float)
    if (mass.ndim!=1 or mass.shape!=force.shape or not len(mass)
            or not np.isfinite([mass,force]).all()or (mass<=0).any()or(force<=0).any()
            or not np.isfinite([distance_m,requested_s,force_fraction]).all()
            or distance_m<0 or requested_s<=0 or not 0<force_fraction<=.5):
        raise ValueError('Finite nonnegative travel, positive tool budgets and bounded force reserve required')
    acceleration=float((force/mass).min())*force_fraction
    return max(float(requested_s),float(np.sqrt(6*distance_m/acceleration)))


def orientation_path(start,goal):
    """Continuous shortest rotation path; never blend rotation matrices."""
    from scipy.spatial.transform import Rotation
    start,goal=np.asarray(start,float),np.asarray(goal,float)
    for r in [start,goal]:
        if (r.shape!=(3,3) or not np.isfinite(r).all() or not np.allclose(r.T@r,np.eye(3),atol=1e-7)
                or not np.isclose(np.linalg.det(r),1.,atol=1e-7)):
            raise ValueError('Proper finite unscaled rotation endpoints required')
    start=start.copy();delta=Rotation.from_matrix(goal@start.T).as_rotvec()
    def at(fraction):
        if not np.isfinite(fraction) or not 0<=fraction<=1:raise ValueError('Bounded finite rotation path fraction required')
        return Rotation.from_rotvec(delta*fraction).as_matrix()@start
    return at


def decode(value, *, subject, candidate, hand, node_count, closure_mode=None,material_axis_frame=None):
    if (value.get('version')!=1 or value.get('subject_sha256')!=subject
            or value.get('candidate_sha256')!=candidate or value.get('hand_sha256')!=hand):
        raise ValueError('Replay motion belongs to different hand, candidate, physics or task inputs')
    saved_mode=value.get('closure_mode','coupled')
    if saved_mode not in {'open_descent','coupled'} or closure_mode is not None and saved_mode!=closure_mode:
        raise ValueError('Replay closure mode differs; legacy motions require explicit coupled mode')
    saved_frame=value.get('pinch_material_frame','spatial')
    if saved_frame not in {'spatial','opening_plane'} or material_axis_frame is not None and saved_frame!=material_axis_frame:
        raise ValueError('Replay material direction frame differs from the positive motion')
    selection=deepcopy(value['selection']);patch=np.asarray(selection['material_patch'])
    index=selection['material_node']
    if (type(index) is not int or not 0<=index<node_count or patch.ndim!=1 or len(patch)<3
            or patch.dtype.kind not in 'iu' or patch.min()<0 or patch.max()>=node_count
            or len(np.unique(patch))!=len(patch) or index not in patch or selection.get('on_actual_material') is not True):
        raise ValueError('Replay needs a resolved source-indexed interior material patch')
    selection['material_patch']=patch
    for key in ['point','rotation']:selection[key]=np.asarray(selection[key],float)
    wrist=np.asarray(value['wrist_at_world'],float);rotation=np.asarray(value['rotation_world'],float)
    for point in [selection['point'],wrist]:
        if point.shape!=(3,) or not np.isfinite(point).all():raise ValueError('Finite replay positions required')
    for r in [selection['rotation'],rotation]:
        if (r.shape!=(3,3) or not np.isfinite(r).all() or not np.allclose(r.T@r,np.eye(3),atol=1e-7)
                or not np.isclose(np.linalg.det(r),1.,atol=1e-7)):
            raise ValueError('Replay orientations must be proper unscaled rotations')
    return selection,wrist,rotation


def commands_match(positive, negative):
    """A shortened slip control must replay the same command prefix exactly."""
    report={'pass':False,'samples_compared':len(negative)}
    if not negative or len(negative)>len(positive):return report
    try:
        errors={'position_m':0.,'rotation_matrix':0.,'joint_rad':0.,'hinge_rad':0.}
        for a,b in zip(positive,negative):
            if (a['phase']!=b['phase'] or not np.isfinite([a['time_s'],b['time_s']]).all()
                    or abs(a['time_s']-b['time_s'])>1e-8):return report
            if set(a['finger_joint_targets_rad'])!=set(b['finger_joint_targets_rad']):return report
            pairs=[('position_m',a['wrist_target'],b['wrist_target'],(3,)),
                   ('rotation_matrix',a['wrist_rotation_target'],b['wrist_rotation_target'],(3,3)),
                   ('joint_rad',list(a['finger_joint_targets_rad'].values()),
                    [b['finger_joint_targets_rad'][k] for k in a['finger_joint_targets_rad']],(len(a['finger_joint_targets_rad']),)),
                   ('hinge_rad',[a['lid_target_rad']],[b['lid_target_rad']],(1,))]
            for key,x,y,shape in pairs:
                x,y=np.asarray(x,float),np.asarray(y,float)
                if x.shape!=shape or y.shape!=shape or not np.isfinite([x,y]).all():return report
                errors[key]=max(errors[key],float(abs(x-y).max()))
        report.update(pass_=all(x<=1e-10 for x in errors.values()),maximum_command_differences=errors)
        report['pass']=report.pop('pass_')
    except (KeyError,TypeError,ValueError):return report
    return report
