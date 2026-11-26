"""Independent source-tool contact qualification; no object/task approval."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from calibrated_scene_reconstruction.assets.acceptance_subject import sha


def contact_hand_signature(plan):
    """Bind tool construction conservatively, separately from the task motion.

    These fields do not enter FingertipFixture construction. All other fields,
    including unknown future ones, remain bound. Original complete control
    inputs are still hash-checked; this is no qualification of task trajectories.
    """
    task_only={'source','hold_s','lift_height_m','lift_speed_m_s','initial_material_distribution'}
    payload=json.dumps({k:v for k,v in plan.items() if k not in task_only},
                       sort_keys=True,separators=(',',':'),allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def compare(positive, negative, positive_trace, negative_trace, positive_protocol, negative_protocol,*,maximum_prelift_difference_m=.001):
    a,b=positive,negative;ta,tb=positive_trace,negative_trace
    def parameters(p):
        return {k:v for k,v in p['parameters'].items() if k not in {'output','zero_friction_lift'}}
    same_inputs=(a['input_sha256']==b['input_sha256'] and a['source_tool']==b['source_tool']
                 and positive_protocol['inputs']==negative_protocol['inputs']
                 and parameters(positive_protocol)==parameters(negative_protocol))
    def equal(x,y,tolerance=0.):
        x,y=np.asarray(x),np.asarray(y)
        return bool(x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all()
                    and np.allclose(x,y,rtol=0.,atol=tolerance))
    commands=equal(ta['commands'],tb['commands'])
    x,y=np.asarray(ta['prelift_points']),np.asarray(tb['prelift_points'])
    mismatch=(float(np.linalg.norm(x-y,axis=1).max()) if x.shape==y.shape and x.ndim==2 and x.shape[1:]==(3,) and len(x)
              and np.isfinite(x).all() and np.isfinite(y).all() else float('inf'))
    states=(equal(ta['rest'],tb['rest']) and equal(ta['faces'],tb['faces'])
            and equal(ta['points'][0],tb['points'][0],1e-6)
            and 0<maximum_prelift_difference_m<=.001 and mismatch<=maximum_prelift_difference_m)
    def physical(r):
        return bool(r['complete'] and r['physical_pass'] and r['failure'] is None
                    and r['native_physics_errors']['pass']
                    and all(row['physical_pass'] and row['material']['pass'] for row in r['rows']))
    def measured_lift(trace):
        before,after=np.asarray(trace['prelift_points']),np.asarray(trace['points'][-1])
        if before.shape!=after.shape or before.ndim!=2 or not len(before):raise ValueError('Matching nonempty native coupon states required')
        return float(np.median(after[:,2]-before[:,2]))
    lift_a,lift_b=measured_lift(ta),measured_lift(tb)
    def loaded_hold(r,trace):
        rows=[row for row in r['rows'] if row['phase']=='hold']
        end=r['rows'][-1]['time_s']
        plateau=[state for state,row in zip(trace['points'],r['rows'])if row['phase']=='hold' and row['time_s']>=end-.3]
        drift=max((abs(float(np.median(state[:,2]-trace['points'][-1,:,2])))for state in plateau),default=float('inf'))
        return bool(r['hold_seconds']>=1 and len(rows)>=2 and rows[-1]['time_s']-rows[0]['time_s']>=.9
                    and r['lift_retained'] and drift<.01 and r['stationary_hold_drift_m']<.01
                    and np.isclose(drift,r['stationary_hold_drift_m'],rtol=0,atol=1e-6)
                    and lift_a>.05 and abs(lift_a-.08)<.02
                    and np.isclose(lift_a,r['material_lift_m'],rtol=0,atol=1e-6)
                    and all(all(count>0 for count in row['contact']['near_surface_samples'].values())
                            and len(row['contact']['near_surface_samples'])==2 for row in rows))
    update=b.get('zero_material_update') or {}
    zero=bool(b['zero_friction_lift'] and not a['zero_friction_lift']
              and update.get('method')=='existing_usd_material' and update.get('combine_mode')=='multiply'
              and update.get('native_readback_after_step') is True
              and not b['lift_retained'] and lift_b<lift_a-.03
              and np.isclose(lift_b,b['material_lift_m'],rtol=0,atol=1e-6))
    no_material_actuation=all(p.get('cloth_forces') is False and p.get('cloth_attachments') is False
                             for p in [positive_protocol,negative_protocol])
    checks={'same_current_sources_and_profile':same_inputs,'same_executed_commands':commands,
            'same_initial_and_prelift_material':states,'positive_physical_validity':physical(a),
            'negative_physical_validity':physical(b),'sustained_loaded_hold':loaded_hold(a,ta),
            'matched_native_zero_friction_slip':zero,'no_material_actuation':no_material_actuation}
    return {'pass':all(checks.values()),'checks':checks,'task_approved':False,
            'maximum_prelift_state_difference_m':mismatch,'allowed_prelift_state_difference_m':maximum_prelift_difference_m,
            'scope':'Independent source-tip/coupon contact qualification only. Full-sheet geometry, source state, extraction, passive stability and delivery remain separate.'}


def verify(positive,negative,output):
    directories=[Path(positive).resolve(),Path(negative).resolve()]
    if directories[0]==directories[1]:raise ValueError('Distinct positive and negative native controls required')
    reports=[];protocols=[];traces=[];evidence={}
    for directory in directories:
        report=json.loads((directory/'result.json').read_text());protocol=json.loads((directory/'protocol.json').read_text())
        if report.get('synthetic_control') is not True or report['input_sha256']!=protocol['inputs']:
            raise ValueError('Independent current native coupon controls required')
        for path,checksum in protocol['inputs'].items():
            if sha(path)!=checksum:raise ValueError('Stale control dependency: '+path)
        for path,checksum in report['source_tool'].get('sha256',{}).items():
            if sha(path)!=checksum:raise ValueError('Stale imported hand geometry: '+path)
        for name in ['result.json','protocol.json','states.npz','initial.usda','source.py','hand_input.json','profile.json']:
            evidence[str(directory/name)]=sha(directory/name)
        with np.load(directory/'states.npz') as data:trace={key:data[key].copy()for key in data.files}
        if len(trace['points'])!=len(report['rows']) or not np.allclose(trace['times'],[row['time_s']for row in report['rows']],rtol=0,atol=1e-8):
            raise ValueError('Native trace and measured rows differ')
        reports.append(report);protocols.append(protocol);traces.append(trace)
    result=compare(*reports,*traces,*protocols)
    result['evidence_sha256']=evidence;result['input_sha256']=reports[0]['input_sha256']
    result['input_sha256'].update(reports[0]['source_tool'].get('sha256',{}))
    from .contact_control_profile import edge_range
    result['runtime_profile']={'profile':json.loads((directories[0]/'profile.json').read_text()),'solver':'PGS',
        'hand_sha256':sha(directories[0]/'hand_input.json'),
        'hand_contact_signature':contact_hand_signature(json.loads((directories[0]/'hand_input.json').read_text())),
        'source_tool':reports[0]['source_tool'],
        'normal_preload':protocols[0]['parameters']['normal_preload'],
        'coupon_normal_force_N':protocols[0]['parameters']['normal_force_N'],
        'material_edges_m':edge_range(traces[0]['rest'],traces[0]['faces'])}
    result['input_sha256'][str(Path(__file__).resolve())]=sha(__file__)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def bind(proof,hand_plan,settings,physics_hz,rest,faces,*,solver,collision_approximation,normal_preload,normal_force_N,sdf_resolution=None):
    """Reject reuse of a passing tool coupon at a different task profile."""
    from .contact_control_profile import edge_range,resolution_matches,validate
    path=Path(proof).resolve();report=json.loads(path.read_text())
    if report.get('pass') is not True:raise ValueError('Passing independent contact-tool qualification required')
    if report.get('input_sha256',{}).get(str(Path(__file__).resolve()))!=sha(__file__):
        raise ValueError('Current contact-qualification verifier required')
    for mapping in ['input_sha256','evidence_sha256']:
        for source,checksum in report[mapping].items():
            if sha(source)!=checksum:raise ValueError('Stale contact qualification: '+source)
    runtime=report['runtime_profile'];profile=runtime['profile'];source=runtime['source_tool']
    validate(profile)
    hand_matches=(contact_hand_signature(json.loads(Path(hand_plan).read_text()))==runtime['hand_contact_signature']
                  if 'hand_contact_signature' in runtime else sha(hand_plan)==runtime['hand_sha256'])
    checks={'hand':hand_matches,'physics_hz':physics_hz==profile['physics_hz'],
        'solver':solver==runtime['solver'],
        'material_and_contact':all(settings.get(k)==v for k,v in profile['settings'].items()),
        'material_resolution':resolution_matches(runtime['material_edges_m'],edge_range(rest,faces)),
        'closing_control_mode':normal_preload is runtime['normal_preload'],
        'qualified_force_envelope':bool(np.isfinite(normal_force_N) and 0<normal_force_N<=runtime['coupon_normal_force_N']+1e-7),
        'collision_approximation':collision_approximation==source['collision_approximation'],
        'sdf_resolution':sdf_resolution==source['sdf_resolution']}
    if not all(checks.values()):raise ValueError('Task contact profile differs from qualified tool controls: '+str(checks))
    return {'pass':True,'proof':str(path),'sha256':sha(path),'checks':checks,'task_approved':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('positive');p.add_argument('negative');p.add_argument('--output',required=True)
    a=p.parse_args();result=verify(a.positive,a.negative,a.output);print(json.dumps(result['checks']))
    raise SystemExit(0 if result['pass'] else 2)
