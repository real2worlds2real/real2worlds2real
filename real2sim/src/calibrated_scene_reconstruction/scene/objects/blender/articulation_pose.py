"""Source-only articulation pose sweep; never changes assets or acceptance.

Keeps every rigid shape, base pose, source mask and camera fixed. This checks
whether a numeric initial-angle proxy is supported by calibrated observations.
A passing silhouette interval is not a confidence interval or a semantic closed
state: internal contact, visibility and mechanism checks remain independent.
"""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np


def posed(vertices,parts,joints,index,position):
    from scipy.spatial.transform import Rotation
    joint=joints[index];kind=joint['type']
    if kind not in {'revolute','prismatic'}:raise ValueError('Moving articulation required')
    bounds=np.asarray(joint['limits'],float)
    if not np.isfinite(position) or not bounds[0]<=position<=bounds[1]:raise ValueError('Pose must lie inside joint limits')
    selected={joint['child']};pending=[joint['child']]
    while pending:
        parent=pending.pop()
        for edge in joints:
            if edge['parent']!=parent:continue
            if edge['child'] in selected:raise ValueError('Articulation subtree is cyclic or ambiguous')
            selected.add(edge['child']);pending.append(edge['child'])
    if joint['parent'] in selected:raise ValueError('Parent cannot move with its own child')
    result=np.array(vertices,float,copy=True);axis=np.asarray(joint['axis'],float)
    axis/=np.linalg.norm(axis);delta=position-joint.get('initial_position',0.)
    rotation=Rotation.from_rotvec(axis*delta).as_matrix() if kind=='revolute' else np.eye(3)
    anchor=np.asarray(joint['anchor0'],float)
    for part in parts:
        if part['name'] not in selected:continue
        if part['kind']!='rigid':raise ValueError('Flexible motion cannot be inferred from a rigid subtree sweep')
        a,n=part['vertex_start'],part['vertex_count']
        result[a:a+n]=(result[a:a+n]-anchor)@rotation.T+anchor if kind=='revolute' else result[a:a+n]+axis*delta
    return result


def run(session,obj,candidate,output,*,joint_index,positions):
    from .evidence import load,score
    from .source_acceptance import policy,scoped_comparison
    from .material_reference import mesh_parts
    from .store import digest
    output=Path(output).resolve();candidate=Path(candidate).resolve()
    if not output.is_relative_to(session.dir.resolve()/'qa') or not candidate.is_relative_to(session.dir.resolve()/'qa'):
        raise ValueError('Current-session QA candidate and output required')
    positions=np.asarray(positions,float)
    if positions.ndim!=1 or not 3<=len(positions)<=101 or not np.isfinite(positions).all() or not (np.diff(positions)>0).all():
        raise ValueError('Three to101 finite increasing source-pose samples required')
    info=json.loads(candidate.read_text());mesh=Path(info['path'])/'mesh.npz'
    with np.load(mesh) as raw:data={k:raw[k].copy() for k in raw.files}
    parts=mesh_parts(data);joints=json.loads(str(data['joints_json']));joint=joints[joint_index]
    observations=load(session,obj,parts=True);requested=policy(session,obj)
    scoped=scoped_comparison(data['F'],observations,parts,requested) if requested else None
    if scoped is not None:
        selected,observations,_=scoped;faces=data['F'][selected]
        child=next(p for p in parts if p['name']==joint['child'])
        if not selected[child['face_start']:child['face_start']+child['triangle_count']].any():
            raise ValueError('Moving child is excluded from source observations')
    else:
        faces=data['F'];observations=({k:v for k,v in observations[0].items() if k not in {'_source_parts','_source_material'}},*observations[1:])
    output.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,output/'source.py')
    inputs=[candidate,mesh,session.dir/'parsing/posed_frames/cameras.npz',session.dir/'splat/object_fit'/f'{obj}.npz']
    rows=[]
    for position in positions:
        v=posed(data['V'],parts,joints,joint_index,float(position))
        metrics,_=score(v,faces,info['pose'],observations)
        row={'position':float(position),'metrics':metrics};rows.append(row)
        (output/'progress.json').write_text(json.dumps(row,indent=2)+'\n')
        print('SOURCE_JOINT_POSE',position,metrics['iou_median'],metrics['local_difference'],metrics['ok'],flush=True)
    accepted=[r['position'] for r in rows if r['metrics']['ok']]
    result={'source_only':True,'task_approved':False,'initial_state_approved':False,'object':obj,'joint':joint,
        'rows':rows,'sampled_passing_interval':[min(accepted),max(accepted)] if accepted else None,
        'sampled_best_position':min(rows,key=lambda r:r['metrics']['loss'])['position'],
        'range_boundary_passes':[rows[0]['metrics']['ok'],rows[-1]['metrics']['ok']],
        'input_sha256':{str(p):digest(p) for p in inputs},
        'scope':__doc__}
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    from calibrated_scene_reconstruction.scene.session import Session
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('session');p.add_argument('object');p.add_argument('candidate');p.add_argument('--output',required=True);p.add_argument('--joint-index',type=int,required=True);p.add_argument('--range',type=float,nargs=2,required=True);p.add_argument('--samples',type=int,default=17);a=p.parse_args()
    run(Session(a.session),a.object,a.candidate,a.output,joint_index=a.joint_index,positions=np.linspace(*a.range,a.samples))


if __name__=='__main__':main()
