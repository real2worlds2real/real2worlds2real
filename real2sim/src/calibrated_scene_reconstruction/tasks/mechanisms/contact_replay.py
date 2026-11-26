"""Exact finite-tool command replay for a matched cloth friction intervention."""
import hashlib
import json
from pathlib import Path
import numpy as np


BOUND_FIELDS=('candidate_sha256','candidate_mesh_sha256','assembly','cloth_link','body_link',
    'input_sha256','fixture','fixture_timing_seconds','record_failed_lift_attempt',
    'grasp_force_proposal','closing_axis_preload','material_contact_mode','contact_tool_qualification',
    'closure_mode','open_descent_seconds','expected_closure_contacts','pinch_material_axis',
    'pinch_material_frame','inside_finger','requested_inside_finger','material_accuracy_mode',
    'diagnose_thickness_residual')


def decode(commands,phases,*,physics_hz):
    c=np.asarray(commands,float)
    if (c.ndim!=2 or c.shape[1:]!=(16,)or not len(c)or not np.isfinite(c).all()
            or type(physics_hz)is not int or physics_hz<=0):
        raise ValueError('Finite source-tip commands at an explicit native physics clock required')
    rotations=c[:,3:12].reshape(-1,3,3)
    if (not np.allclose(rotations.transpose(0,2,1)@rotations,np.eye(3),atol=1e-7)
            or not np.allclose(np.linalg.det(rotations),1.,atol=1e-7)
            or (c[:,12]<0).any()or(c[:,12]>.05).any()
            or not np.isin(c[:,14],[0.,1.]).all()or(c[:,15]<=0).any()):
        raise ValueError('Proper rotations, bounded jaw opening and finite positive load commands required')
    labels=np.full(len(c),'',dtype=object)
    for name,interval in phases.items():
        t=np.asarray(interval,float)
        if t.shape!=(2,)or not np.isfinite(t).all()or t[0]<0 or t[1]<=t[0]:
            raise ValueError('Ordered finite phase intervals required')
        bounds=np.rint(t*physics_hz).astype(int);start,end=bounds
        if (not np.allclose(bounds/physics_hz,t,atol=1e-8,rtol=0.)
                or not 0<=start<end<=len(c)or(labels[start:end]!='').any()):
            raise ValueError('Recorded command phases overlap or do not match the native clock')
        labels[start:end]=name
    if (labels=='').any():raise ValueError('Every native command must belong to its recorded phase')
    return c.copy(),labels.tolist()


def load(directory,expected,*,physics_hz):
    directory=Path(directory).resolve()
    result=json.loads((directory/'result.json').read_text());protocol=json.loads((directory/'protocol.json').read_text())
    if (result.get('complete')is not True or result.get('physical_pass')is not True
            or result.get('failure')is not None or(result.get('lift')or{}).get('pass')is not True
            or result.get('native_physics_errors',{}).get('pass')is not True
            or protocol.get('zero_friction_lift')is not False):
        raise ValueError('Complete physically valid positive lift required for a matched replay')
    if any(protocol.get(k)!=expected.get(k)for k in BOUND_FIELDS):
        raise ValueError('Replay source, geometry, material, tool or control inputs differ')
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if any(sha(path)!=value for path,value in protocol['input_sha256'].items()):
        raise ValueError('Positive replay inputs changed')
    with np.load(directory/'states.npz')as z:trace={k:z[k].copy()for k in z.files}
    if (len(trace['points'])!=len(result['rows'])or not np.allclose(trace['times'],[r['time_s']for r in result['rows']],atol=1e-8,rtol=0.)):
        raise ValueError('Positive native rows and states do not match')
    commands,labels=decode(trace['commands'],result['phases'],physics_hz=physics_hz)
    record_steps=set(np.rint(trace['times']*physics_hz).astype(int).tolist())
    if min(record_steps)!=0 or max(record_steps)!=len(commands):
        raise ValueError('Complete positive recording must span the executed command trace')
    evidence={str(directory/name):sha(directory/name)for name in ['result.json','protocol.json','states.npz','source.py','initial.usda','candidate_input.json','hand_input.json']}
    return {'result':result,'protocol':protocol,'trace':trace,'commands':commands,
            'labels':labels,'record_steps':record_steps,
            'evidence':{'positive':str(directory),'sha256':evidence,'task_approved':False,
                'scope':'Exact source-tip pose, jaw, lid and load commands; no material targets or replanning. Native validity and causal acceptance remain separate.'}}
