"""Turn a valid native preparation trace into geometry for agent rebuilding.

No fixtures, native state overrides or task approval enter the proposed asset.
Bake observed rigid-link transforms and joint frames in the primary rigid
frame; retain the complete cloth material chart and face/material ownership.
"""
import copy
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.transform import Rotation

from calibrated_scene_reconstruction.assets.mechanisms import joint_frames,quaternion_matrix,revolute_position
from .store import digest


def pose(value):
    value=np.asarray(value,float)
    if value.shape!=(12,) or not np.isfinite(value).all():raise ValueError('Finite native rigid pose required')
    R=value[3:].reshape(3,3)
    if not np.allclose(R.T@R,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(R),1.,atol=1e-5):
        raise ValueError('Native rigid pose must have a proper unscaled rotation')
    return value[:3],R


def bake(mesh,cloth_link,anchor,points,rigid_poses,*,volume_states=None):
    result={k:v.copy() for k,v in mesh.items()};parts=json.loads(str(mesh['links_json']))
    flexible=[p for p in parts if p['kind']!='rigid' and p['name']!=cloth_link]
    captures={} if volume_states is None else volume_states
    if (set(captures)!={p['name'] for p in flexible}
            or any(p['kind'] not in {'deformable','rope'} or p.get('backend')!='volume_fem' for p in flexible)):
        raise ValueError('Every moving flexible part needs its own recorded native state')
    if not any(p['name']==anchor and p['kind']=='rigid' for p in parts):raise ValueError('Primary frame must be a rigid link')
    transforms={p['name']:pose(rigid_poses[p['name']]) for p in parts if p['kind']=='rigid'}
    p0,R0=transforms[anchor];vertices=mesh['V'].astype(float).copy()
    volumes=copy.deepcopy(json.loads(str(mesh.get('volume_initial_json',np.array('{}')))))
    posed_volumes={}
    for part in flexible:
        from calibrated_scene_reconstruction.assets.volume_initial_state import proposal,validate,skin_mapping
        name=part['name'];capture=captures[name]
        if name not in volumes:raise ValueError('Original volume material reference is missing')
        original,_=validate(volumes[name]);reference=capture['reference']
        # The capture may move/deform the material, never replace its reference
        # or topology with a recooked or already-compressed rest state.
        for kind in ['simulation','collision']:
            actual=np.asarray(reference[kind+'_rest_points_local'],float)
            if (actual.shape!=original[kind+'_rest_points'].shape
                    or not np.allclose(actual,original[kind+'_rest_points'],atol=1e-6,rtol=0.)
                    or not np.array_equal(reference[kind+'_tetrahedra'],original[kind+'_indices'])):
                raise ValueError('Native capture changed the original volume reference or topology')
        posed,state,_=proposal(original['skin_rest_points'],reference,
            capture['simulation_points'],capture['collision_points'],p0,R0,
            basis='Actual complete native volume capture; original material reference retained. Fresh unforced reload required.')
        a,b,n,m=[part[k] for k in ['vertex_start','face_start','vertex_count','triangle_count']]
        skin_mapping(state,posed,mesh['F'][b:b+m]-a)
        if posed.shape!=(n,3):raise ValueError('Complete source-indexed volume skin required')
        posed_volumes[name]=posed;volumes[name]={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in state.items()}
    normals=None
    if 'normals' in mesh:
        normals=np.asarray(mesh['normals'],float).copy()
        if (normals.shape!=(len(mesh['F']),3,3) or not np.isfinite(normals).all()
                or not np.allclose(np.linalg.norm(normals,axis=2),1.,atol=1e-4)):
            raise ValueError('Finite unit face-corner normals required before native pose baking')
    for part in parts:
        a,n=part['vertex_start'],part['vertex_count']
        if part['name']==cloth_link:
            if part['kind']!='cloth' or np.asarray(points).shape!=(n,3) or not np.isfinite(points).all():
                raise ValueError('Exact source-indexed native cloth state required')
            world=np.asarray(points,float)
        elif part['kind']=='rigid':
            p,R=transforms[part['name']];world=mesh['V'][a:a+n]@R.T+p
        else:
            vertices[a:a+n]=posed_volumes[part['name']]
            continue
        vertices[a:a+n]=(world-p0)@R0
        if normals is not None and part['kind']=='rigid':
            b,k=part['face_start'],part['triangle_count']
            normals[b:b+k]=normals[b:b+k]@R.T@R0
    if normals is not None:
        for part in [p for p in parts if p['kind']!='rigid']:
            b,k=part['face_start'],part['triangle_count']
            triangles=vertices[mesh['F'][b:b+k]]
            area=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
            length=np.linalg.norm(area,axis=1)
            if (length<=1e-20).any():raise ValueError('Native flexible surface contains a degenerate shading triangle')
            normals[b:b+k]=(area/length[:,None])[:,None,:]
        result['normals']=normals
    joints=copy.deepcopy(json.loads(str(mesh.get('joints_json',np.array('[]')))))
    for joint in joints:
        a,b=joint['parent'],joint['child'];pA,RA=transforms[a];pB,RB=transforms[b]
        f0,f1=joint_frames(joint)
        if joint['type']=='revolute':joint['initial_position']=revolute_position(RA,RB,f0,f1)
        axis=RA@np.asarray(joint.get('axis',[1.,0.,0.]),float)
        if joint['type']=='prismatic':
            joint['initial_position']=float(((pB+RB@joint['anchor1'])-(pA+RA@joint['anchor0']))@axis)
        if 'axis' in joint:joint['axis']=(R0.T@axis).tolist()
        for i,(p,R,f) in enumerate([(pA,RA,f0),(pB,RB,f1)]):
            joint['anchor'+str(i)]=((np.asarray(joint['anchor'+str(i)])@R.T+p-p0)@R0).tolist()
            q=Rotation.from_matrix(R0.T@R@quaternion_matrix(f)).as_quat()
            joint['frame'+str(i)]=np.r_[q[3],q[:3]].tolist()
    result.update(V=vertices,joints_json=np.array(json.dumps(joints)))
    if flexible:result['volume_initial_json']=np.array(json.dumps(volumes))
    return result


def selected_state(native,mode,time_s):
    if mode=='geometry_seed':
        if time_s is None:return len(native['states'])-1
        matches=[i for i,row in enumerate(native['states']) if np.isfinite(time_s) and abs(row['time_s']-time_s)<1e-9]
        if len(matches)!=1:raise ValueError('Geometry seed must select one actual recorded state')
        return matches[0]
    if mode=='released_candidate':
        if time_s is not None:raise ValueError('Default proposals use the final released state only')
        index=len(native['states'])-1
        if native.get('material_preparation') and native['states'][index]['phase']!='preparation_released':
            raise ValueError('Construction forces must be released before proposing the final state')
        preparation=native.get('material_preparation') or {}
        if preparation.get('scheduled_checkpoints',0) and preparation.get('checkpoints_pass') is not True:
            raise ValueError('Every declared preparation checkpoint must pass before proposing a released candidate')
        if preparation.get('hinge_active_s') and (not preparation.get('hinge_effort_cleared')
                or (preparation.get('hinge_release_receipt') or {}).get('native_command_readback')!=0.):
            raise ValueError('Released hinge requires an explicit zero native actuation readback')
        if preparation.get('hinge_control')=='native_drive' and (
                preparation.get('hinge_release_receipt') or {}).get('native_drive_gains_zero') is not True:
            raise ValueError('Released native drive requires zero stiffness and damping readback')
        construction=(native.get('material_preparation') or {}).get('container_state')
        if construction is not None and construction.get('pass') is not True:
            raise ValueError('Prepared material distribution does not satisfy its declared container-state requirements')
        final=preparation.get('final_release_conditions')
        if final is not None and (not final.get('pass') or preparation.get('released_conditions_retained') is not True):
            raise ValueError('Preparation conditions were lost after force release')
        return index
    if mode!='preparation_seed' or time_s is None or not np.isfinite(time_s):
        raise ValueError('Explicit preparation_seed mode and recorded time required for an actuated construction seed')
    matches=[i for i,row in enumerate(native['states']) if abs(row['time_s']-time_s)<1e-9]
    if len(matches)!=1 or not native.get('material_preparation'):
        raise ValueError('Select an actual recorded material-preparation state')
    index=matches[0]
    if native['states'][index]['phase']!='material_pose_preparation':
        raise ValueError('Preparation seeds must disclose that construction fixtures were still active')
    return index


def hypothesis_inputs(probe,protocol):
    """Use the trace's bound material/representation pair, never one stale half."""
    paths=[Path(probe)/name for name in ['mechanisms.json','physics_priors.json']]
    keys=['mechanisms_sha256','physics_priors_sha256']
    if not any(path.exists() for path in paths) and not any(key in protocol for key in keys):
        return {},[]  # Legacy traces use the unchanged session declaration.
    if any(not path.is_file() or digest(path)!=protocol.get(key) for path,key in zip(paths,keys)):
        raise ValueError('Native representation and physical hypotheses must be a complete hash-bound pair')
    return {'mechanism_hypothesis':json.loads(paths[0].read_text()),
            'physics_hypothesis':json.loads(paths[1].read_text())},paths


def volume_captures(probe,mesh,native,index,data):
    parts=[p for p in json.loads(str(mesh['links_json']))
           if p['kind'] in {'deformable','rope'}]
    if not parts:return {}
    reference_file=Path(probe)/'native_volume_references.npz'
    receipt=native.get('native_volume_reference') or {}
    if (receipt.get('file')!=reference_file.name or not reference_file.is_file()
            or receipt.get('sha256')!=digest(reference_file)):
        raise ValueError('Every volume requires its hash-bound original native reference')
    captures={};count=len(native['states'])
    with np.load(reference_file) as references:
        for part in parts:
            name=part['name'];reference={}
            for key in ['simulation_rest_points_local','collision_rest_points_local',
                        'simulation_tetrahedra','collision_tetrahedra','local_to_world_row_matrix']:
                full=name+'_'+key
                if full not in references:raise ValueError('Incomplete original native volume reference: '+name)
                reference[key]=references[full].copy()
            values={}
            for key,suffix in [('simulation_points','simulation_points'),('collision_points','surface_points')]:
                full=name+'_'+suffix
                if full not in data or len(data[full])!=count:
                    raise ValueError('Incomplete actual native volume trace: '+name)
                values[key]=data[full][index].copy()
            captures[name]={**values,'reference':reference}
    return captures


def run(s,obj,cloth_link,anchor,candidate,probe,out,*,mode='released_candidate',time_s=None):
    candidate=Path(candidate).resolve();probe=Path(probe).resolve();out=Path(out).resolve();root=s.dir.resolve()
    if not all(p.is_relative_to(root/'qa') for p in [candidate,probe,out]):
        raise ValueError('Current-session QA candidate, completed native trace and new output required')
    info=json.loads(candidate.read_text());source=(Path(info['path'])/'mesh.npz').resolve()
    if info.get('requires_material_restore') and mode!='geometry_seed':
        raise ValueError('Temporary construction material can only supply a geometry seed; restore material and validate a fresh release')
    if info.get('requires_numerical_restore') and mode!='geometry_seed':
        raise ValueError('Temporary construction numerics can only supply a geometry seed; restore the qualified profile and validate a fresh release')
    if not source.is_relative_to(root):raise ValueError('Current-session source mesh required')
    native=json.loads((probe/'result.json').read_text());protocol=json.loads((probe/'probe_input.json').read_text())
    geometry_only=mode=='geometry_seed'
    trace_available=(native.get('complete') is True or
        geometry_only and native.get('complete') is False and native.get('stopped_after_physical_failure') is True
        and native.get('physical_state_pass') is False)
    if (not trace_available
            or (not geometry_only and native.get('physical_state_pass') is not True)
            or native.get('native_physics_errors',{}).get('pass') is not True
            or protocol['mesh_sha256']!=digest(source) or protocol['candidate_sha256']!=digest(candidate)
            or native.get('probe_input_sha256')!=digest(probe/'probe_input.json')):
        raise ValueError('Complete passing native state trace must belong to this exact candidate and mesh')
    arguments=protocol['arguments']
    hypotheses,hypothesis_files=hypothesis_inputs(probe,protocol)
    forbidden=['no_gravity','no_contact','no_rigid_contact']
    if not geometry_only:forbidden+=['experimental_bending','experimental_particle_shell','experimental_native_viscosity_proof','continue_invalid_diagnostic']
    if (arguments.get('assembly')!=obj or any(arguments.get(k) for k in forbidden)):
        raise ValueError('Native construction requires real gravity/contact and the ordinary material backend')
    index=selected_state(native,mode,time_s);last=native['states'][index]
    from calibrated_scene_reconstruction.assets.response_policy import material_verdict
    material=material_verdict(last['material'],native.get('material_accuracy_mode','quantitative'))
    if (not material['pass'] or not last['self_intersection']['pass'] or not last['solid_contact_pass']
            or (last.get('finite_sheet_clearance') is not None and not last['finite_sheet_clearance']['pass'])):
        raise ValueError('Last native cloth/material/contact state failed')
    with np.load(source) as data:mesh={k:data[k].copy() for k in data.files}
    with np.load(probe/'states.npz') as data:
        if len(data['points'])!=len(native['states']):raise ValueError('Native state trace length changed')
        rigid={p['name']:data[p['name']+'_poses'][index].copy() for p in json.loads(str(mesh['links_json'])) if p['kind']=='rigid'}
        captures=volume_captures(probe,mesh,native,index,data)
        result=bake(mesh,cloth_link,anchor,data['points'][index],rigid,volume_states=captures)
    out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,out/'source.py')
    np.savez_compressed(out/'mesh.npz',**result)
    initial_states={}
    if mode=='released_candidate' and hasattr(s,'config'):
        from calibrated_scene_reconstruction.assets.joint_state import measure as initial_joint_state
        for mechanism in s.config('agent/mechanisms.json',{}).get('mechanisms',[]):
            if not mechanism.get('initial_state'):continue
            matching=[j for j in json.loads(str(result['joints_json']))
                      if set(mechanism.get('links',[]))=={obj+'/'+j['parent'],obj+'/'+j['child']}]
            if len(matching)!=1:raise ValueError('Released initial articulation is missing or ambiguous')
            initial_states[mechanism['id']]=initial_joint_state(mechanism,matching[0]['initial_position'],default=matching[0]['limits'])
        if initial_states:
            (out/'initial_joint_state.json').write_text(json.dumps(initial_states,indent=2)+'\n')
            if not all(row['pass'] for row in initial_states.values()):
                raise ValueError('Released proposal does not satisfy the declared source initial articulation state')
    interface_plan=(s.config('agent/interface_checks.json',{}).get('objects',{}).get(obj,[])
                    if mode=='released_candidate' and hasattr(s,'config') else [])
    interface_plan=[r for r in interface_plan if r.get('phase','initial_closed')=='initial_closed']
    if interface_plan and mode=='released_candidate':
        from calibrated_scene_reconstruction.assets.metric_mesh import asset
        from calibrated_scene_reconstruction.assets.interface_geometry import from_poses
        interfaces=from_poses(asset({'mesh':str(source)})['parts'],interface_plan,{name:pose(value) for name,value in rigid.items()})
        (out/'interface_review.json').write_text(json.dumps(interfaces,indent=2)+'\n')
        if not interfaces['pass']:raise ValueError('Released candidate does not retain its declared mating interfaces')
    from .initial_state import policy
    from calibrated_scene_reconstruction.assets.cloth_state import check
    # policy preserves source support, mechanisms, material scope and required
    # contact effects when checking a new physical/representation hypothesis.
    # This does not adopt those hypotheses or approve the reconstructed object.
    admissibility=check(result,policy(s,obj,**hypotheses)) if hypotheses else check(result,policy(s,obj))
    proposed={**info,'path':str(out),'proposal_only':True,'requires_agent_rebuild':True,
              'preparation_only':True,'source_pose_approved':False,'native_approved':False,'task_approved':False}
    if info.get('link_physics'):
        proposed['link_physics']=copy.deepcopy(info['link_physics'])
        _,anchor_rotation=pose(rigid[anchor])
        from .mechanism_pose import rotate_physics
        for name,settings in proposed['link_physics'].items():
            if 'revolution_axis' not in settings:continue
            _,part_rotation=pose(rigid[name])
            proposed['link_physics'][name]=rotate_physics(settings,anchor_rotation.T@part_rotation)
    proposed.update(native_geometry_mode=mode,requires_additional_preparation=mode!='released_candidate',
                    native_trace_physical_pass=native.get('physical_state_pass') is True,
                    native_trace_complete=native.get('complete') is True,
                    native_trace_is_acceptance_evidence=not geometry_only,
                    selected_native_time_s=last['time_s'],selected_native_phase=last['phase'],
                    previous_construction_goals=(native.get('material_preparation') or {}).get('container_state'))
    if hypothesis_files:
        for path in hypothesis_files:shutil.copy2(path,out/path.name)
        proposed['native_preparation_hypotheses']={path.stem:str(out/path.name) for path in hypothesis_files}
    (out/'candidate.json').write_text(json.dumps(proposed,indent=2)+'\n')
    report={'ok':bool(admissibility['pass']),'proposal_created':True,'proposal_only':True,
        'requires_agent_rebuild':True,'authored_ok':admissibility['pass'],
        'source_pose_approved':False,'native_approved':False,'task_approved':False,
        'candidate':str(out/'candidate.json'),'admissibility':admissibility,'native_time_s':last['time_s'],
        'mode':mode,'requires_additional_preparation':mode!='released_candidate',
        'selected_material_response':material,
        'native_trace_physical_pass':native.get('physical_state_pass') is True,
        'native_trace_complete':native.get('complete') is True,
        'native_trace_stopped_after_physical_failure':native.get('stopped_after_physical_failure') is True,
        'native_trace_is_acceptance_evidence':not geometry_only,
        'construction_forces_were_active':last['phase']=='material_pose_preparation',
        'previous_construction_goals':(native.get('material_preparation') or {}).get('container_state'),
        'input_sha256':{str(p):digest(p) for p in [candidate,source,probe/'result.json',probe/'probe_input.json',probe/'states.npz',Path(__file__),*hypothesis_files]},
        'evidence_images':[],
        'scope':('Geometry only from a recorded individually valid state: earlier trace failures remain failures, and this grants no released-physics evidence. ' if geometry_only else
                 'Actuated geometry seed for further preparation only; rebuilding starts a new hypothesis with no restored velocities, not a simulator checkpoint or a released asset. ' if mode=='preparation_seed' else 'Released geometry proposal only. ')+
            'Temporary forces, holders and attachments are omitted. Agent Blender rebuilding, original complete-source constraints, fresh unforced native checks and all task/prior/delivery gates remain required. Native SDF contact can pass while stricter authored contact fails; never hide that failure.'}
    if captures:
        from calibrated_scene_reconstruction.assets import volume_initial_state,tetra_boundary
        report['original_volume_references']={'parts':sorted(captures),'velocities_restored':False,
            'requires_fresh_unforced_reload':True,'task_approved':False}
        for file in [probe/'native_volume_references.npz',Path(volume_initial_state.__file__),Path(tetra_boundary.__file__)]:
            report['input_sha256'][str(file)]=digest(file)
    if interface_plan:
        from calibrated_scene_reconstruction.assets import interface_geometry,ray_triangles
        report['released_interfaces']=interfaces
        for file in [s.config_dir/'agent/interface_checks.json',Path(interface_geometry.__file__),Path(ray_triangles.__file__)]:
            report['input_sha256'][str(file)]=digest(file)
        report.setdefault('evidence_files',[]).append(str((out/'interface_review.json').relative_to(s.dir)))
    if initial_states:
        from calibrated_scene_reconstruction.assets import joint_state
        report['initial_joint_states']=initial_states
        for file in [s.config_dir/'agent/mechanisms.json',Path(joint_state.__file__)]:
            report['input_sha256'][str(file)]=digest(file)
        report.setdefault('evidence_files',[]).append(str((out/'initial_joint_state.json').relative_to(s.dir)))
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
