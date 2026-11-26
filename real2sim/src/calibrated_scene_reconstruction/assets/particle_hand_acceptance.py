"""Evaluate an isolated PBD hand-friction contrast, not merely a lifted pose."""
import numpy as np

from calibrated_scene_reconstruction.assets.hand_contact_isolation import clearance


def isolation_interval(history, *, start, end, physics_hz):
    if not history or not np.isfinite([start,end,physics_hz]).all() or end<=start or physics_hz<=0:
        raise ValueError('Resolved native isolated-hold interval required')
    times=np.array([r['time_s'] for r in history],float)
    if (not np.isfinite(times).all() or abs(times[0]-start)>1e-8
            or abs(times[-1]-end)>1e-8
            or not np.allclose(np.diff(times),1/physics_hz,rtol=0,atol=1e-8)):
        raise ValueError('Isolation must cover every physics step of the causal interval')
    inventory=set(history[0]['rigid_bounds'])
    for row in history:
        margins={name:c['required_pair_contact_margin_m'] for name,c in row['colliders'].items()}
        actual=clearance(row['cloth_bounds'],row['rigid_bounds'],margins,expected_colliders=inventory)
        if not actual['pass'] or row.get('pass') is not True or actual['colliders']!=row['colliders']:
            raise ValueError('Whole-sheet isolation from all non-hand colliders failed or changed')
    return {'pass':True,'steps':len(times)-1,'start_s':float(start),'end_s':float(end)}


def compare(positive,negative,positive_states,negative_states,positive_isolation,negative_isolation,*,physics_hz):
    """Both runs must lift with friction before its isolated negative ablation.

    Callers additionally validate current producer/control hashes, identical
    physical inputs and exact executed motion, all contact/material checks,
    the selected-patch witnesses and full task requirements.
    """
    receipts=negative.get('native_zero_friction') or {}
    if (receipts.get('method')!='particle_base_and_self_scale'
            or receipts.get('native_readback_after_step') is not True
            or receipts.get('affects_all_particle_rigid_contacts') is not True
            or not 0<receipts.get('base_friction',0)<=1e-6
            or negative.get('friction_zero_readback') is not True):
        raise ValueError('Stepped native particle-wide friction intervention required')
    before=[np.asarray(s['precontrol_points'],float) for s in [positive_states,negative_states]]
    if (before[0].ndim!=2 or before[0].shape[1:]!=(3,) or not len(before[0])
            or before[0].shape!=before[1].shape or not np.isfinite(before).all()):
        raise ValueError('Two complete native pre-intervention cloth states required')
    mismatch=float(np.linalg.norm(before[0]-before[1],axis=1).max())
    slips=[];isolations=[];wrists=[]
    for result,trace,initial,history in zip([positive,negative],[positive_states,negative_states],before,
                                          [positive_isolation,negative_isolation]):
        rows=result['rows'];ids=[i for i,row in enumerate(rows) if row['phase']=='hold']
        if not ids:raise ValueError('Actual native isolated hold samples required')
        control=result['particle_control'];wrist=np.asarray(control['wrist_at_intervention'],float)
        if wrist.shape!=(3,) or not np.isfinite(wrist).all():raise ValueError('Actual wrist at intervention required')
        wrists.append(wrist)
        start=result['phases']['hold'][0];end=rows[ids[-1]]['time_s']
        if not np.isclose(start,receipts['time_s'],rtol=0,atol=1e-8):
            raise ValueError('Particle friction intervention must begin at the same isolated hold')
        isolations.append(isolation_interval(history,start=start,end=end,physics_hz=physics_hz))
        patch=np.asarray(trace['material_patch']);points=np.asarray(trace['points'])
        if (patch.ndim!=1 or patch.dtype.kind not in 'iu' or len(patch)<3
                or patch.min()<0 or patch.max()>=len(initial)
                or points.shape!=(len(rows),*initial.shape) or not np.isfinite(points).all()):
            raise ValueError('Resolved actual material patch and corresponding native states required')
        slips.append([float(np.asarray(rows[i]['wrist_actual'])[2]-wrist[2]
            -np.median(points[i,patch,2]-initial[patch,2])) for i in ids])
    positive_peak=float(np.abs(slips[0]).max());negative_slip=slips[1][-1]
    wrist_mismatch=float(np.linalg.norm(wrists[0]-wrists[1]))
    contrast=negative_slip>.015 and negative_slip>3*positive_peak
    termination=negative.get('control_termination') or {}
    matching=positive.get('motion_sha256')==negative.get('motion_sha256') and mismatch<=.001 and wrist_mismatch<=.001
    return {'pass':bool(positive.get('lift',{}).get('pass') is True and contrast and matching
                       and termination.get('reason')=='measured_slip'),
            'maximum_preintervention_state_difference_m':mismatch,
            'preintervention_wrist_difference_m':wrist_mismatch,
            'positive_hold_maximum_slip_m':positive_peak,'negative_hold_downward_slip_m':negative_slip,
            'isolated_native_intervals':isolations,'native_material_readback':True,
            'scope':'Isolated native particle-friction contrast only; no independent task or material approval.'}
