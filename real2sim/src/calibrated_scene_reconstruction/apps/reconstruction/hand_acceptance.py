"""Resolve a recorded native hand/cloth trial pair into input-criterion evidence."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial import cKDTree

from calibrated_scene_reconstruction.assets.acceptance_subject import sha,subject
from calibrated_scene_reconstruction.core.capabilities import entries, invoke
from calibrated_scene_reconstruction.tasks.mechanisms.hand_cloth import friction_control,lift_measurement,opening_straddle,pinch_axis_alignment,material_tangent
from calibrated_scene_reconstruction.tasks.mechanisms.hand_motion import commands_match

CHECKS=frozenset({'initial_rim_drape','actual_sharpa_thumb_index',
    'downward_approach','source_pinch_axis','inside_outside_fingers','interior_material_grasp','friction_lift',
    'zero_friction_control','no_cloth_attachments_or_nodal_forces','material_and_collision_validity','open_descent_before_closure'})

CHECKS = CHECKS | frozenset(entries('hand_checks'))


def run(s,positive,negative,out,*,criterion='sharpa_middle_friction_grasp',register=False):
    from calibrated_scene_reconstruction.assets.acceptance import declaration
    from calibrated_scene_reconstruction.assets.response_policy import mode,material_verdict
    accuracy=mode(s.config('task.json',{}).get('flexible_accuracy'))
    request=next((r for r in declaration(s.config('task.json',{}))['criteria'] if r['id']==criterion),None)
    if request is None or request.get('validation')!='native_physics' or not set(request.get('checks',[]))<=CHECKS:
        raise ValueError('This native hand/cloth checker cannot validate the requested criterion')
    root=s.dir.resolve();positive,negative,out=[Path(p).resolve() for p in [positive,negative,out]]
    if positive==negative or not all(p.is_relative_to(root/'qa') for p in [positive,negative,out]):
        raise ValueError('Distinct current-session native trials and a new QA output required')
    loaded=[];evidence={};inputs={}
    for path in [positive,negative]:
        result=json.loads((path/'result.json').read_text());protocol=json.loads((path/'protocol.json').read_text())
        from calibrated_scene_reconstruction.core.paths import ROOT
        if (result.get('diagnostic_hypothesis') or protocol.get('diagnostic_hypothesis')
                or sha(path/'source.py') not in {sha(ROOT/name) for name in entries('hand_sources').values()}):
            raise ValueError('Diagnostic hypotheses or non-current assay implementations cannot grant input acceptance')
        if (result.get('material_accuracy_mode')!=accuracy or protocol.get('material_accuracy_mode')!=accuracy
                or not result.get('rows')
                or any(row.get('material_response')!=material_verdict(row['material'],accuracy)
                       for row in result['rows'])):
            raise ValueError('Native hand trial must use the current shared flexible-response policy')
        for name,checksum in protocol['input_sha256'].items():
            if sha(name)!=checksum:raise ValueError('Trial implementation/geometry/hand inputs changed: '+name)
            inputs[name]=checksum
        if sha(path/'motion.json')!=protocol.get('executed_motion_sha256'):
            raise ValueError('Saved executed motion changed or was not recorded')
        artifacts=['result.json','protocol.json','states.npz','initial.usda','hand_input.json','candidate_input.json','source.py','motion.json']
        if protocol.get('closure_mode')=='open_descent':artifacts.append('preclosure_readiness.json')
        if protocol.get('friction_intervention_phase')=='hold_after_lift':
            from calibrated_scene_reconstruction.scene.objects.particle_hand_proof import bind
            proof=protocol['particle_contact_qualification']
            native=proof['profile_checks']['rigid_nominal']['runtime_profile']
            current=bind(proof['qualification'],native,physics_hz=protocol['simulation']['physics_hz'],
                         solver=protocol['simulation']['solver'])
            if current!=proof:raise ValueError('Particle hand proof changed or does not bind the recorded runtime')
            if sha(path/'isolation.json')!=result['particle_control']['isolation_sha256']:
                raise ValueError('Particle hand isolation recording changed')
            artifacts.append('isolation.json')
        for name in artifacts:
            file=path/name;evidence[str(file.relative_to(root))]=sha(file)
        with np.load(path/'states.npz') as data:trace={key:data[key].copy() for key in data.files}
        if len(trace['points'])!=len(result['rows']) or not np.allclose(trace['times'],[r['time_s'] for r in result['rows']],atol=1e-8):
            raise ValueError('Native state samples do not match their measured rows')
        loaded.append((result,protocol,trace,json.loads((path/'hand_input.json').read_text()),
                       json.loads((path/'candidate_input.json').read_text())))
    a,pa,ta,hand,candidate=loaded[0];b,pb,tb,other_hand,other_candidate=loaded[1]
    required_contact=request.get('interpretation',{}).get('material_contact_mode')
    if required_contact is not None and pa.get('material_contact_mode','single_patch')!=required_contact:
        raise ValueError('Recorded contact-region scope differs from the input criterion')
    obj=pa['assembly'];mesh=Path(candidate['path'])/'mesh.npz';case=pa['case'];expected=subject(s,obj,mesh,case)
    geometry_plan=s.config('agent/geometry_checks.json',{}).get('objects',{}).get(obj,[])
    if geometry_plan:
        from calibrated_scene_reconstruction.scene.objects.blender.geometry_checks import measure
        from calibrated_scene_reconstruction.core.paths import ROOT
        if not measure(mesh.parent,geometry_plan,ROOT)['pass']:
            raise ValueError('Current source solid/void geometry checks failed')
    if pa['subject_sha256']!=expected or pb['subject_sha256']!=expected or hand!=other_hand or candidate!=other_candidate:
        raise ValueError('Trial pair belongs to different or stale reconstruction inputs')
    protocols_equal={k:v for k,v in pa.items() if k!='zero_friction_lift'}=={k:v for k,v in pb.items() if k!='zero_friction_lift'}
    if pa['zero_friction_lift'] or not pb['zero_friction_lift']:raise ValueError('Supply positive then zero-finger-friction trials')
    preparatory = [invoke('hand_preparation', name, pa) for name in entries('hand_preparation')]
    open_end = preparatory[-1][-1] if preparatory else 'approach'
    phases = [phase for group in preparatory for phase in group] + ['approach','pinch','lift','hold','release']
    if pa.get('closure_mode')=='open_descent':phases.insert(phases.index('pinch'),'open_descent')
    complete=all(r.get('complete') is True and r.get('physical_pass') is True and
                 r.get('native_physics_errors',{}).get('pass') is True and
                 all(p in r.get('phases',{}) for p in
                     (phases if r is a or not r.get('control_termination') else phases[:phases.index('lift')+1])) for r in [a,b])
    def last(phase):return max(i for i,row in enumerate(a['rows']) if row['phase']==phase)
    checks={name:{'pass':False} for name in CHECKS if name!='source_pinch_axis' or name in request.get('checks',[])}
    if pa.get('closure_mode')!='open_descent' and 'open_descent_before_closure' not in request.get('checks',[]):
        checks.pop('open_descent_before_closure',None)
    if complete:
        ready_phase='open_descent' if pa.get('closure_mode')=='open_descent' else 'approach'
        start=a['rows'][0];opened=a['rows'][last(open_end)];approach=a['rows'][last(ready_phase)]
        if ready_phase=='open_descent':
            from pxr import Usd
            from calibrated_scene_reconstruction.sim.hand_collision import colliders,OpenPinchReadiness
            readiness=[]
            for path,(result,protocol,trace,plan,_) in zip([positive,negative],loaded):
                index=max(i for i,row in enumerate(result['rows']) if row['phase']=='open_descent')
                row=result['rows'][index]
                model=OpenPinchReadiness(colliders(Usd.Stage.Open(str(path/'initial.usda')),protocol['hand_root_path']),
                    [plan[f]['link'] for f in ['thumb','index']])
                selection=result.get('preclosure_selection') or result['postopening_selection']
                measured=model.check(trace['points'][index,selection['material_node']],
                    row['finger_pads'],trace['hand_links'][index],result['hand_body_names'])
                names=protocol['native_hand_joint_names'];actual=np.asarray(row['hand_joint_positions_rad'])
                error=max(abs(actual[names.index(name)]-value) for name,value in row['finger_joint_targets_rad'].items())
                open_targets=all(np.isclose(value,plan['open_state'].get(name,0.)) for name,value in row['finger_joint_targets_rad'].items())
                order=all(r['phase']!='pinch' for r in result['rows'][:index+1])
                readiness.append({'pass':bool(measured['pass'] and error<.05 and open_targets and order),
                    'actual_open_pad_capture':measured,'maximum_open_joint_error_rad':float(error),
                    'open_targets_retained':open_targets,'closure_only_after_descent':order})
            checks['open_descent_before_closure']={'pass':all(r['pass'] for r in readiness),'trials':readiness}
        pinched=a['rows'][last('pinch')];held=a['rows'][last('hold')];opening=candidate['report']['opening']
        inside=start['material_distribution']['measurements']['inside_fraction']
        from calibrated_scene_reconstruction.assets.cloth_container_state import check as initial_distribution
        pose=np.asarray(start['body_pose'])
        distribution=initial_distribution((ta['points'][0]-pose[:3])@pose[3:].reshape(3,3),
            ta['rest'],ta['faces'],opening,hand['initial_material_distribution'])
        checks['initial_rim_drape']={'pass':distribution['pass'],
            'inside_material_fraction':inside,'initial_distribution':distribution}
        for name in entries('hand_initial_state'):
            invoke('hand_initial_state', name, s, obj, pa, start, checks['initial_rim_drape'])
        if pa.get('initial_support_route') is not None:
            from calibrated_scene_reconstruction.assets.metric_mesh import asset
            from calibrated_scene_reconstruction.assets.material_support import check as support_state
            from calibrated_scene_reconstruction.assets.physical_priors import sample
            support=pa['mechanisms']['objects'][obj]['links'][pa['cloth_link']]['initial_support']
            parts={part['name']:part for part in asset({'mesh':str(mesh)})['parts']};base=parts[support['link']]
            posed=np.asarray(start['rigid_poses'][support['link']]);rotation=posed[3:].reshape(3,3)
            nominal=sample(pa['priors'],'nominal')['bodies'][obj+'/'+pa['cloth_link']]['parameters']
            measured_support=support_state((ta['points'][0]-posed[:3])@rotation,ta['rest'],ta['faces'],
                base['V'],base['F'],support,thickness_m=nominal['thickness'],
                tolerance_m=start['hand_contact']['tolerance_m'])
            checks['initial_rim_drape']['pass'] &= measured_support['pass']
            checks['initial_rim_drape']['initial_support_state']=measured_support
        for name in entries('hand_opened_state'):
            checks.update(invoke('hand_opened_state', name, a, last(open_end), opened))
        checks['actual_sharpa_thumb_index']={'pass':('sharpa' in hand['asset'].lower() and
            len(a['hand']['joint_limits_rad'])==22 and hand['thumb']['link'] in a['hand_body_names'] and
            hand['index']['link'] in a['hand_body_names']), 'source_asset':hand['asset']}
        descent=float(np.asarray(opened['wrist_actual'])[2]-np.asarray(approach['wrist_actual'])[2])
        checks['downward_approach']={'pass':descent>=.03,'actual_wrist_descent_m':descent}
        bp=np.asarray(approach['body_pose']);pads=(np.asarray(approach['finger_pads'])-bp[:3])@bp[3:].reshape(3,3)
        with np.load(mesh) as data:
            part=next(p for p in json.loads(str(data['links_json'])) if p['name']==pa['body_link'])
            vertices=data['V'][part['vertex_start']:part['vertex_start']+part['vertex_count']]
            outer=float(np.ptp(vertices[:,:2],axis=0).mean()/2)
        checks['inside_outside_fingers']=opening_straddle(pads,opening,outer)
        alignment_minimum=max(pa.get('minimum_pinch_axis_alignment',.94),request.get('interpretation',{}).get('minimum_pinch_axis_alignment',.94))
        if 'source_pinch_axis' in checks and pa.get('source_pinch_axis') is not None:
            checks['source_pinch_axis']=pinch_axis_alignment(pads,pa['source_pinch_axis'],minimum=alignment_minimum)
        required_axis=request.get('interpretation',{}).get('pinch_material_axis')
        required_frame=request.get('interpretation',{}).get('pinch_material_frame')
        if required_frame is not None and pa.get('pinch_material_frame','spatial')!=required_frame:
            raise ValueError('Native grasp direction frame differs from the input')
        if required_axis is not None and pa.get('source_material_axis')!=required_axis:
            raise ValueError('Native grasp did not use the input-specified material direction')
        patch=ta['material_patch'];rest=ta['rest'];faces=ta['faces']
        direction_patch=ta.get('approach_material_patch',patch)
        if 'source_pinch_axis' in checks and pa.get('source_material_axis') is not None:
            index=last(ready_phase)
            tangent=material_tangent((ta['points'][index]-bp[:3])@bp[3:].reshape(3,3),rest,direction_patch,pa['source_material_axis'])
            checks['source_pinch_axis']=pinch_axis_alignment(pads,tangent,up=None if pa.get('pinch_material_frame','spatial')=='spatial' else [0,0,1],minimum=alignment_minimum)
        if 'source_pinch_axis' in checks and (pa.get('source_pinch_axis') is not None or pa.get('source_material_axis') is not None):
            pose_pinched=np.asarray(pinched['body_pose']);rotation_pinched=pose_pinched[3:].reshape(3,3)
            pads_pinched=(np.asarray(pinched['finger_pads'])-pose_pinched[:3])@rotation_pinched
            axis_pinched=(pa['source_pinch_axis'] if pa.get('source_pinch_axis') is not None else
                material_tangent((ta['points'][last('pinch')]-pose_pinched[:3])@rotation_pinched,rest,direction_patch,pa['source_material_axis']))
            closed_alignment=pinch_axis_alignment(pads_pinched,axis_pinched,up=None if pa.get('source_material_axis') is not None and pa.get('pinch_material_frame','spatial')=='spatial' else [0,0,1],minimum=alignment_minimum)
            approach_alignment=checks['source_pinch_axis']
            checks['source_pinch_axis']={'pass':approach_alignment['pass'] and closed_alignment['pass'],
                'approach':approach_alignment,'pinch':closed_alignment}
        edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
        unique,counts=np.unique(edges,axis=0,return_counts=True);boundary=np.unique(unique[counts==1])
        clearance=cKDTree(rest[boundary]).query(rest[patch])[0]
        contact_rows=[r for r in a['rows'] if r['phase']=='hold']
        actual_contact=all(all(sum(n for link,n in r['selected_patch_contact']['near_surface_samples'].items()
            if link.startswith('right_'+finger) and ('_DP' in link or '_elastomer' in link))>0 for finger in ['thumb','index']) for r in contact_rows)
        contact_association=None
        if pa.get('material_contact_mode')=='same_component':
            from pxr import Usd
            from calibrated_scene_reconstruction.sim.hand_collision import colliders,Penetration
            from calibrated_scene_reconstruction.tasks.mechanisms.material_grasp import contact_regions,retained_regions
            model=Penetration(colliders(Usd.Stage.Open(str(positive/'initial.usda')),pa['hand_root_path']))
            groups={finger:[hand[finger]['link']]for finger in ['thumb','index']}
            recorded=a.get('grasp_regions') or {};radius=float(np.linalg.norm(rest[direction_patch]-
                rest[(a.get('preclosure_selection') or a['postopening_selection'])['material_node']],axis=1).max())
            contact_range=pinched['hand_contact']['near_surface_range_m']
            contact_association=contact_regions(ta['points'][last('pinch')],rest,faces,pinched['finger_pads'],
                ta['hand_links'][last('pinch')],a['hand_body_names'],model.rows,groups,
                contact_range_m=contact_range,patch_radius=radius/2,boundary_margin=radius)
            # Recompute from actual native poses. Saved metadata alone cannot
            # change a stale single disk into a passing two-region witness.
            identity=bool(contact_association.get('pass') and recorded.get('pass') and
                contact_association['material_patch']==recorded['material_patch'] and
                np.array_equal(patch,recorded['material_patch']) and b.get('grasp_regions')==recorded)
            retained=[retained_regions(ta['points'][i],faces,recorded,model,ta['hand_links'][i],a['hand_body_names'],
                a['rows'][i]['hand_contact']['tolerance_m'],a['rows'][i]['hand_contact']['near_surface_range_m'],groups)
                for i,r in enumerate(a['rows']) if r['phase']=='hold'] if identity else []
            actual_contact=bool(identity and retained and all(r['pass']for r in retained))
            contact_association['recorded_source_indices_match']=identity
            contact_association['sustained_per_finger_retention']=actual_contact
        no_other_support=all(sum(n for link,n in r['hand_contact']['near_surface_samples'].items()
            if not ('thumb' in link or 'index' in link))==0 for r in contact_rows)
        distances=[float(np.linalg.norm(ta['points'][last('pinch')][patch]-pad,axis=1).min()) for pad in pinched['finger_pads']]
        checks['interior_material_grasp']={'pass':len(patch)>=3 and clearance.min()>0 and actual_contact and no_other_support and max(distances)<.012,
            'minimum_material_boundary_distance_m':float(clearance.min()),'both_distal_fingers_contact_cloth':actual_contact,
            'other_hand_support_absent':no_other_support,'selected_patch_to_pad_distances_m':distances}
        if contact_association is not None:checks['interior_material_grasp']['actual_contact_regions']=contact_association
        lift=lift_measurement(ta['prelift_points'],ta['points'][last('hold')],patch,pinched['wrist_actual'],held['wrist_actual'])
        checks['friction_lift']=lift
        if pa.get('friction_intervention_phase')=='hold_after_lift':
            from calibrated_scene_reconstruction.tasks.mechanisms.particle_hand_acceptance import compare as particle_compare
            control=particle_compare({**a,'lift':lift},b,ta,tb,
                json.loads((positive/'isolation.json').read_text()),json.loads((negative/'isolation.json').read_text()),
                physics_hz=pa['simulation']['physics_hz'])
        else:
            control=friction_control({**a,'prelift_points':ta['prelift_points'],'lift':lift},
                                     {**b,'prelift_points':tb['prelift_points']})
            if b.get('control_termination'):
                end=b['rows'][-1];pinch=next(r for r in b['rows'][::-1] if r['phase']=='pinch')
                measured=lift_measurement(tb['prelift_points'],tb['points'][-1],tb['material_patch'],pinch['wrist_actual'],end['wrist_actual'])
                control['pass'] &= b['control_termination']['reason']=='measured_slip' and measured['vertical_slip_m']>.015
            update=b.get('native_zero_friction',{}) or {}
            control['native_material_readback']=bool(update.get('method')=='existing_usd_material'
                and update.get('combine_mode')=='multiply'
                and update.get('native_readback_after_step') is True
                and all(row.get('native_finger_friction_zero') is True for row in b['rows'] if row['phase']=='lift'))
        control['pass'] &= protocols_equal
        control['executed_commands']=commands_match(a['rows'],b['rows'])
        control['pass'] &= control['executed_commands']['pass']
        control['pass'] &= control['native_material_readback']
        checks['zero_friction_control']=control
        checks['no_cloth_attachments_or_nodal_forces']={'pass':all(p.get('cloth_forces') is False and p.get('cloth_attachments') is False for p in [pa,pb])
            and all(r['hand'].get('cloth_actuation') is False and r['hand'].get('positions_written_after_reset') is False for r in [a,b])}
        raw_material=all(row['material']['pass'] for r in [a,b] for row in r['rows'])
        checks['material_and_collision_validity']={'pass':all(all(row['physical_pass'] for row in r['rows']) for r in [a,b]),
            'material_accuracy_mode':accuracy,'quantitative_material_pass':raw_material,
            'physical_plausibility_review_required':not raw_material,
            'scope':'Specified friction interaction only; approximate material response still requires semantic and physical-plausibility review before object or scene approval.'}
    for row in checks.values():row['pass']=bool(row['pass'])
    out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,out/'source.py');inputs[str(Path(__file__).resolve())]=sha(__file__)
    from calibrated_scene_reconstruction.assets import joint_state
    inputs[str(Path(joint_state.__file__).resolve())]=sha(joint_state.__file__)
    result={'pass':bool(complete and all(row['pass'] for row in checks.values())),'criterion':criterion,
        'native_states_read':True,'backend':a['backend'],'subject_sha256':expected,'case':case,'checks':checks,
        'evidence_sha256':evidence,'input_sha256':inputs,'evidence_files':list(evidence),
        'task_approved':False,'scope':'Specified native hand/contact criterion only. Default observations, rigid precision, material qualification and full task/delivery checks remain separate.'}
    file=out/'result.json';file.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if register and result['pass']:
        path=s.config_dir/'agent/acceptance_evidence.json';plan=s.config('agent/acceptance_evidence.json',{'version':1,'criteria':{}})
        plan.setdefault('criteria',{}).setdefault(criterion,{})[case]={'path':str(file.relative_to(root)),'sha256':sha(file)}
        path.write_text(json.dumps(plan,indent=2)+'\n')
    return result


if __name__=='__main__':
    from calibrated_scene_reconstruction.scene.session import Session
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('session');p.add_argument('positive');p.add_argument('negative')
    p.add_argument('--output',required=True);p.add_argument('--register',action='store_true');args=p.parse_args()
    result=run(Session(Path(args.session).resolve()),args.positive,args.negative,args.output,register=args.register)
    print(json.dumps({'pass':result['pass'],'checks':{k:v['pass'] for k,v in result['checks'].items()}}))
    raise SystemExit(0 if result['pass'] else 2)
