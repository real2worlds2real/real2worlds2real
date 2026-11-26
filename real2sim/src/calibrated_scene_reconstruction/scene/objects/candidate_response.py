"""Compare an unapproved native candidate trace with independent source images.

A same-capture object round supplies only a body pose proposal. It supplies no
shape or response approval. The candidate must match actual source silhouettes
at initialization and throughout unforced simulation. This diagnostic cannot
export geometry or approve the task.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from calibrated_scene_reconstruction.scene.objects.blender.store import digest


def probe_contact_length(mesh, report):
    """A numerical length scale from authored geometry, never measured size truth.

    Preserve the legacy explicit radius when present. New agent recipes need
    not use that parameter name; their first rigid link supplies the scale.
    Loose sheets cannot enlarge this contact-resolution initializer.
    """
    radius = report.get('params', {}).get('body_radius')
    basis = 'legacy authored body_radius parameter'
    if radius is None:
        parts = json.loads(str(mesh['links_json']))
        rigid = next((p for p in parts if p['kind'] == 'rigid'), None)
        if rigid is None:
            raise ValueError('A rigid body is required for this passive assembly probe')
        a, n = rigid['vertex_start'], rigid['vertex_count']
        vertices = np.asarray(mesh['V'][a:a+n], float)
        if not len(vertices) or not np.isfinite(vertices).all():
            raise ValueError('Finite primary rigid geometry required')
        radius = .5*float(np.ptp(vertices[:, :2], axis=0).min())
        basis = 'half the smaller horizontal extent of authored rigid link '+rigid['name']
    if not isinstance(radius, (int, float)) or not np.isfinite(radius) or radius <= 0:
        raise ValueError('Finite positive authored contact length required')
    return float(radius), {'length_m': float(radius), 'basis': basis,
                           'scope': 'Numerical probe initialization only; no semantic opening or size approval.'}


def verified_protocol(probe,candidate,mesh,report=None):
    """Bind an actual trace to its solver, material and optional force code."""
    protocol=probe/'probe_input.json';snapshot=probe/'candidate_input.json'
    if not protocol.is_file() or not snapshot.is_file():return False,[]
    profile=json.loads(protocol.read_text())
    files=[protocol,snapshot]
    valid=(digest(snapshot)==digest(candidate) and profile.get('candidate_sha256')==digest(candidate)
           and Path(profile['input']).resolve()==candidate.resolve()
           and profile.get('mesh_sha256')==digest(mesh))
    if report is not None:
        valid &= report.get('probe_input_sha256')==digest(protocol)
    for key,name in [('source_sha256','source.py'),('physics_priors_sha256','physics_priors.json'),
                     ('mechanisms_sha256','mechanisms.json')]:
        path=probe/name;files.append(path)
        valid &= path.is_file() and profile.get(key)==digest(path)
    experimental=bool(profile.get('arguments',{}).get('experimental_bending'))
    extra=profile.get('experimental_files_sha256',{})
    if experimental:
        valid &= {'experimental_bending_input.json','experimental_bending_natural.npz',
                  'experimental_experimental_bending.py','experimental_discrete_bending.py',
                  'experimental_cloth_projection.py','experimental_cloth_curvature.py'}<=extra.keys()
    if profile.get('arguments',{}).get('defer_experimental_startup'):
        valid &= 'experimental_experimental_startup.py' in extra
        if report is not None:
            path=probe/'experimental_startup.json';files.append(path)
            startup=report.get('experimental_startup') or {}
            valid &= (path.is_file() and report.get('experimental_startup_sha256')==digest(path)
                      and startup.get('all_settings_restored') is True
                      and startup.get('positions_written') is False
                      and startup.get('velocities_written') is False)
    for name,sha in extra.items():
        path=probe/name
        if Path(name).name!=name or not path.is_file():valid=False;continue
        files.append(path);valid &= digest(path)==sha
    preparation=profile.get('preparation_files_sha256',{})
    if profile.get('arguments',{}).get('preparation_plan'):
        valid &= {'preparation_preparation.py','preparation_force_fixture.py','preparation_input.json'}<=preparation.keys()
    for name,sha in preparation.items():
        path=probe/name
        if Path(name).name!=name or not path.is_file():valid=False;continue
        files.append(path);valid &= digest(path)==sha
    return bool(valid),[p for p in files if p.is_file()]


def placement(vertices,parts,source_vertices,source_parts,pose):
    from .blender.fit import rigid_anchor
    if not parts or not source_parts or parts[0]['name']!=source_parts[0]['name']:
        raise ValueError('Source pose proposal needs the same primary rigid link identity')
    a=np.radians(pose['yaw_deg']);c,s=np.cos(a),np.sin(a)
    rotation=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    delta=rigid_anchor(source_vertices,source_parts)-rigid_anchor(vertices,parts)
    translation=np.r_[np.asarray(pose['xy'])+delta@rotation[:2,:2].T,pose['support_z']]
    return rotation,translation


def candidate_placement(info,vertices,parts,source_vertices,source_parts,fallback):
    """Use the hash-bound candidate pose when it has already been source fit.

    An older object's fitted pose is only a fallback for legacy proposals.
    Reusing it after a candidate pose refinement introduces an artificial
    source error even at the beginning of a valid trace.
    """
    if 'pose' not in info:
        return placement(vertices,parts,source_vertices,source_parts,fallback),'legacy_source_round'
    pose=info['pose'];values=np.array([*pose['xy'],pose['support_z'],pose['yaw_deg']],float)
    if values.shape!=(4,) or not np.isfinite(values).all():raise ValueError('Finite candidate pose required')
    angle=np.radians(pose['yaw_deg']);c,s=np.cos(angle),np.sin(angle)
    rotation=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    return (rotation,np.array([*pose['xy'],pose['support_z']])),'candidate_input'


def actual_vertices(vertices,parts,trace,index,rotation,translation):
    """Use each actual native rigid pose and actual nodal positions."""
    body=parts[0]['name'];initial=np.asarray(trace[body+'_poses'][0],float)
    origin,frame=initial[:3],initial[3:].reshape(3,3)
    result=[];flexible=[p for p in parts if p['kind']!='rigid']
    if len(flexible)!=1:raise ValueError('This trace adapter requires one explicitly recorded flexible link')
    for part in parts:
        a,n=part['vertex_start'],part['vertex_count']
        if part['kind']=='rigid':
            transform=np.asarray(trace[part['name']+'_poses'][index],float)
            points=vertices[a:a+n]@transform[3:].reshape(3,3).T+transform[:3]
        else:points=np.asarray(trace['points'][index],float)
        if points.shape!=(n,3) or not np.isfinite(points).all():raise ValueError('Native trace topology changed')
        result.append(((points-origin)@frame)@rotation.T+translation)
    return np.concatenate(result)


def comparison(s,observed,initial,worst,faces,target,output,device,transmitting=None):
    from .physical_observations import compare
    from .object_refine import _frame
    before,first=compare(initial,faces,target,observed,device=device,renders=True,transmitting_faces=transmitting)
    after,last=compare(worst,faces,target,observed,device=device,renders=True,transmitting_faces=transmitting)
    z,frames,masks,windows=observed
    selected=list(dict.fromkeys([*np.argsort(after['local_per_frame'])[-4:][::-1],
                                *range(0,len(frames),max(1,len(frames)//4))]))[:8]
    tiles=[]
    for k in selected:
        source=_frame(s,z,int(frames[k]),windows[k],masks[k].shape)
        row=[]
        for label,mask,colour in [('Source',None,None),('Observed part',masks[k],(240,40,40)),
                                  ('Native initial',first[k],(40,220,40)),('Native maximum motion',last[k],(40,200,240))]:
            tile=source.copy()
            if mask is not None:
                contours,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(tile,contours,-1,colour,2)
            tile=cv2.resize(tile,(256,256));cv2.putText(tile,label,(4,18),0,.45,(255,255,0),1)
            cv2.putText(tile,str(z['names'][frames[k]]),(4,36),0,.45,(255,255,0),1)
            row.append(tile)
        tiles.append(np.hstack(row))
    path=output/'source_native_comparison.jpg';cv2.imwrite(str(path),np.vstack(tiles)[:,:,::-1])
    return {'image':str(path.relative_to(s.dir.resolve())),
            'frames':[str(z['names'][frames[k]]) for k in selected],
            'initial_part_iou':before['iou_median'],'maximum_motion_part_iou':after['iou_median'],
            'scope':'Actual static source images, independent part labels and recorded native states'}


def failed_probe(candidate,probe,output,session_root):
    """Keep an incomplete native experiment as failure, never a material fit."""
    error=json.loads((probe/'error.json').read_text())
    snapshot=probe/'candidate_input.json';protocol=probe/'probe_input.json'
    candidate_verified=snapshot.exists() and digest(snapshot)==digest(candidate)
    verified=candidate_verified
    profile=json.loads(protocol.read_text()) if protocol.exists() else None
    if profile is not None:
        verified &= (profile.get('candidate_sha256')==digest(candidate)
                     and Path(profile['input']).resolve()==candidate.resolve())
        for key,name in [('source_sha256','source.py'),('physics_priors_sha256','physics_priors.json'),
                         ('mechanisms_sha256','mechanisms.json')]:
            verified &= (probe/name).exists() and profile.get(key)==digest(probe/name)
        mesh=Path(json.loads(candidate.read_text())['path'])/'mesh.npz'
        verified &= (mesh.resolve().is_relative_to(session_root.resolve()) and mesh.exists()
                     and profile.get('mesh_sha256')==digest(mesh))
    else:verified=False
    progress=json.loads((probe/'progress.json').read_text()) if (probe/'progress.json').exists() else None
    files=[candidate,*[p for p in probe.iterdir() if p.is_file() and
        (p.name in {'error.json','progress.json','probe_input.json','candidate_input.json','source.py',
                    'physics_priors.json','mechanisms.json','initial.usda','native_physics_errors.json',
                    'authored_sampling.json','native_contact_settings.json','attempt_status.json',
                    'native_constitutive_authoring.json','geometry_preparation.json','prepared_admissibility.json','prepared_source.npz'}
         or p.name.startswith('state_') and p.suffix=='.npz')]]
    result={'diagnostic_only':True,'task_approved':False,'pass':False,
            'native_completed':False,'native_input_hashes_verified':bool(verified),
            'native_candidate_hash_verified':bool(candidate_verified),
            'probe_protocol_available':profile is not None,
            'diagnosis':{'category':'native_experiment_failed','physical_parameters_identified':False,
                'next_step':'Resolve the recorded native error and material/contact/numerical validity. An incomplete experiment cannot identify visual softness or certify source response.'},
            'native_error':error,'last_recorded_state':progress,
            'inputs_sha256':{str(p.resolve().relative_to(session_root.resolve())):digest(p) for p in files},
            'scope':'Failed native experiment retained without source/task approval; older probes without a protocol cannot certify actuation or solver settings'}
    output.mkdir(parents=True,exist_ok=False)
    (output/'response.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def run(s,obj,source_round,candidate,probe,output,*,device='cpu'):
    from .blender import evidence
    from .mechanisms import declaration
    from .physical_observations import observations as part_observations
    from calibrated_scene_reconstruction.scene.review.observation_match import ObservedState,response_diagnosis,material_motion_profile
    candidate,probe,output=map(lambda p:Path(p).resolve(),[candidate,probe,output])
    for path in [candidate,probe,output]:
        if not path.is_relative_to(s.dir.resolve()):raise ValueError('Candidate response must stay inside its source capture')
    if not (probe/'result.json').exists():
        return failed_probe(candidate,probe,output,s.dir)
    report=json.loads((probe/'result.json').read_text())
    if Path(report['input']).resolve()!=candidate:raise ValueError('Native trace names a different candidate')
    if (report.get('external_actuation') is not False or any(report.get(key) is not False for key in
            ['diagnostic_contact_disabled','diagnostic_gravity_disabled','diagnostic_rigid_contact_disabled'])):
        raise ValueError('Source passive response needs actual gravity/contact with no external actuation')
    info=json.loads(candidate.read_text());mesh=Path(info['path'])/'mesh.npz'
    if not mesh.resolve().is_relative_to(s.dir.resolve()):raise ValueError('Candidate mesh belongs to another capture')
    source=s/'objects'/obj/f'round_{source_round}'
    inputs=json.loads((source/'input.json').read_text())['inputs']
    for name in ['parsing/posed_frames/cameras.npz','splat/align.json',f'splat/object_fit/{obj}.npz']:
        if inputs.get(name)!=digest(s/name):raise ValueError('Source pose proposal uses stale observations: '+name)
    fit=json.loads((source/'fit.json').read_text())
    with np.load(source/'mesh.npz') as data:
        source_vertices=data['V'];source_parts=json.loads(str(data['links_json']))
    with np.load(mesh) as data:
        from .blender.material_reference import mesh_parts
        vertices,faces=data['V'],data['F'];parts=mesh_parts(data)
    (rotation,translation),pose_source=candidate_placement(info,vertices,parts,source_vertices,source_parts,fit['pose'])
    reference=vertices@rotation.T+translation
    plan=declaration(s)
    observations=[item for item in (plan or {}).get('observations',[]) if item['link'].split('/')[0]==obj]
    if not observations:raise ValueError('Declare source-backed physical observations before candidate response assessment')
    output.mkdir(parents=True,exist_ok=False);results={}
    with np.load(probe/'states.npz') as trace:
        states=report['states']
        for item in observations:
            link=item['link'].split('/')[1];part=next(p for p in parts if p['name']==link)
            if part['kind']=='rigid':raise ValueError('This trace adapter evaluates the recorded flexible link')
            target=np.zeros(len(faces),bool)
            target[part['face_start']:part['face_start']+part['triangle_count']]=True
            observed=(part_observations(s,obj,item['part_prompt'],exterior_to=item.get('exterior_to'))
                      if item.get('part_prompt') else None)
            from .optical_visibility import transmitting_faces
            optical=transmitting_faces(obj,item,parts,len(faces))
            observer=ObservedState(reference,faces,target,evidence.load(s,obj),
                max_motion_px=item['max_projection_motion_px'],device=device,part_observations=observed,transmitting_faces=optical)
            samples=[]
            for index,state in enumerate(states):
                actual=actual_vertices(vertices,parts,trace,index,rotation,translation)
                samples.append({'time_s':state['time_s'],'result':observer.check(actual)})
            times=np.asarray([row['time_s'] for row in samples],float)
            if len(times)<2 or times[0]!=0 or np.any(np.diff(times)<=0):raise ValueError('Ordered initial and intermediate native states required')
            covered=bool(times[-1]+1e-8>=item['duration_s'] and np.diff(times).max()<=.05+1e-8)
            physical=bool(report.get('physical_state_pass',False) and all(
                row['material']['pass'] and row['self_intersection']['pass']
                and row.get('solid_contact_pass',False) for row in states))
            series={'pass':bool(covered and physical and all(row['result']['pass'] for row in samples)),
                    'samples':samples,'duration_s':float(times[-1]),'maximum_sample_interval_s':float(np.diff(times).max()),
                    'required_duration_covered':covered,'material_and_self_contact_pass':physical,
                    'solid_contact_and_sampling_checked':'physical_state_pass' in report,'external_actuation':False}
            visual=None
            if observed is not None:
                worst=max(range(len(samples)),key=lambda k:samples[k]['result']['maximum_view_p95_visible_material_motion_px'])
                folder=output/item['id'];folder.mkdir()
                visual=comparison(s,observed,actual_vertices(vertices,parts,trace,0,rotation,translation),
                                  actual_vertices(vertices,parts,trace,worst,rotation,translation),
                                  faces,target,folder,device,optical)
                visual['maximum_motion_time_s']=samples[worst]['time_s']
            profile=material_motion_profile(trace['rest'],trace['points'],body_poses=trace[parts[0]['name']+'_poses'])
            results[item['id']]={**series,'diagnosis':response_diagnosis(series),'comparison':visual,
                                 'material_motion_profile':profile}
    protocol_valid,protocol_files=verified_protocol(probe,candidate,mesh,report)
    files=[candidate,mesh,probe/'result.json',probe/'states.npz',source/'fit.json',source/'mesh.npz',*protocol_files]
    result={'diagnostic_only':True,'task_approved':False,'source_pose_round':source_round,
            'native_input_hashes_verified':bool(protocol_valid and report.get('candidate_sha256')==digest(candidate)
                                               and report.get('mesh_sha256')==digest(mesh)),
            'native_candidate_hash_verified':report.get('candidate_sha256')==digest(candidate),
            'pose_proposal_is_approval':False,'observations':results,
            'source_pose_selection':pose_source,
            'experimental_constitutive_model':report.get('experimental_bending'),
            'inputs_sha256':{str(path.relative_to(s.dir.resolve())):digest(path) for path in map(Path.resolve,files)},
            'scope':'Current-capture source pixels and unforced native states; no user stiffness target or true dimensions'}
    (output/'response.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('session');p.add_argument('object');p.add_argument('source_round',type=int)
    p.add_argument('candidate');p.add_argument('probe');p.add_argument('output');p.add_argument('--device',default='cpu')
    a=p.parse_args()
    from calibrated_scene_reconstruction.scene.session import Session
    result=run(Session(a.session),a.object,a.source_round,a.candidate,a.probe,a.output,device=a.device)
    print(json.dumps({k:{'pass':v['pass'],'diagnosis':v['diagnosis']} for k,v in result['observations'].items()}
                     if 'observations' in result else result['diagnosis'],indent=2))
