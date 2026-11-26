"""Compose source comparison with native task-state inspection."""
import json
from pathlib import Path
import numpy as np
from calibrated_scene_reconstruction.scene.review.observation_match import material_motion_profile, passive_series, response_diagnosis, ObservedState
from calibrated_scene_reconstruction.scene.objects.blender import evidence
from calibrated_scene_reconstruction.core.geometry.visibility import VisibleParts

def run(ctx, contract, destination):
    """Automatic unforced preflight; failing it prevents operating the objects."""
    observations = (contract or {}).get('observations', [])
    if not observations:
        return {'pass': True, 'mechanisms': {}, 'required': False}
    from pxr import UsdGeom,PhysxSchema
    from calibrated_scene_reconstruction.assets.metric_mesh import asset
    out = Path(destination)
    if not out.resolve().is_relative_to(ctx.s.dir.resolve()):
        raise ValueError('Physical observation evidence must stay inside this session')
    out.mkdir(parents=True, exist_ok=False)
    results = {}
    passed = True
    failed_gates=[]
    for observation in observations:
        obj, target_link = observation['link'].split('/')
        target_links={ref.split('/')[1] for ref in observation.get('link_group',[observation['link']])}
        kit = asset(ctx.sg['dynamic_objects'][obj]['fix'])
        transform = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(ctx.ts.objects[obj]))
        reference, faces, target,metadata = [], [], [],[]
        offset = 0;face_offset=0
        for part in kit['parts']:
            reference.append(np.asarray(part['V'])@transform[:3, :3]+transform[3, :3])
            faces.append(np.asarray(part['F'])+offset)
            target.extend([part['name'] in target_links]*len(part['F']))
            metadata.append({'name':part['name'],'kind':part['kind'],'vertex_start':offset,'vertex_count':len(part['V']),
                             'face_start':face_offset,'triangle_count':len(part['F'])})
            offset += len(part['V'])
            face_offset+=len(part['F'])
        reference, faces = np.concatenate(reference), np.concatenate(faces)
        part=None
        if observation.get('part_prompt'):
            from calibrated_scene_reconstruction.scene.objects.physical_observations import observations as part_observations
            part=part_observations(ctx.s,obj,observation['part_prompt'],
                                   exterior_to=observation.get('exterior_to'))
        optical=None
        if observation.get('transmitting_links'):
            from calibrated_scene_reconstruction.core.paths import resolve
            from calibrated_scene_reconstruction.assets.optical_surfaces import annotate
            from calibrated_scene_reconstruction.assets.optical_print import opaque_faces
            from calibrated_scene_reconstruction.scene.objects.optical_visibility import transmitting_faces
            fix=ctx.sg['dynamic_objects'][obj]['fix']
            printing=opaque_faces(fix.get('look'),ctx.s.dir,len(faces))
            with np.load(resolve(fix['mesh'])) as data:annotate(data,metadata,opaque_faces=printing)
            optical=transmitting_faces(obj,observation,metadata,len(faces))
        source_observations=evidence.load(ctx.s,obj)
        observed_faces=faces;observed_target=np.asarray(target,bool);pose_scope=None
        from calibrated_scene_reconstruction.scene.objects.blender.source_acceptance import policy,scoped_comparison
        requests=policy(ctx.s,obj)
        other_requests=[row for row in requests if target_link not in (row.get('links') or [row.get('link')])]
        if other_requests:
            retained,source_observations,ignored=scoped_comparison(faces,evidence.load(ctx.s,obj,parts=True),metadata,other_requests)
            observed_faces=faces[retained];observed_target=observed_target[retained]
            # This measures the unrelaxed part, not acceptance of the omitted
            # poses. Required user criteria are evaluated from this same native
            # trial by task_check after all observation records are collected.
            # Requiring an earlier successful trial here would be circular.
            pose_scope={'policy':other_requests,'ignored_source_pixels':ignored,
                        'functional_acceptance_pending':True}
            if optical is not None:optical=np.asarray(optical)[retained]
        try:
            observer = ObservedState(reference, observed_faces, observed_target, source_observations,
                                 max_motion_px=observation['max_projection_motion_px'],
                                 device=ctx.world.device if ctx.world.backend == 'torch' else 'cpu',
                                 part_observations=part,transmitting_faces=optical)
        except ValueError as error:
            from calibrated_scene_reconstruction.assets.acceptance import declaration
            waived={row['gate'] for row in declaration(ctx.task)['relaxations']}
            if ('at least8 views' not in str(error) or 'observation:'+observation['id'] not in waived):raise
            observer=None
        def read():
            return np.concatenate([ctx.points(obj+'/'+p['name']) for p in kit['parts']])
        physical_contract=None
        rigid_screen=None
        if all(p['kind']=='rigid' for p in kit['parts']):
            from calibrated_scene_reconstruction.assets.observation_state import RigidState
            # Separate the rigid material/contact screen from source pose
            # agreement. A user may waive exact pose, never invalid contacts.
            rigid_screen=RigidState(kit['parts'],spacing_m=.0005,tolerance_m=.0001,
                device='cuda:0' if ctx.s.config('simulation.json',{}).get('gpu_dynamics') else 'cpu')
        if (contract or {}).get('material_state_contract'):
            from calibrated_scene_reconstruction.assets.physical_priors import expand_families,sample
            from calibrated_scene_reconstruction.assets.cloth_contact_tolerance import solid_tolerance
            plan=expand_families(ctx.s.config('agent/physics_priors.json'))
            priors=plan['bodies'];limits={};tolerances={}
            sampled=sample(plan,getattr(ctx,'physics_case','nominal'))['bodies']
            root=str(ctx.ts.objects[obj].GetPath());sdf_error=0.
            for rigid in kit['parts']:
                if rigid['kind']!='rigid':continue
                prim=ctx.stage.GetPrimAtPath(root+'/'+rigid['name']+'/surface')
                resolution=PhysxSchema.PhysxSDFMeshCollisionAPI(prim).GetSdfResolutionAttr().Get()
                if resolution:
                    native_errors=getattr(ctx,'native_errors',None)
                    if native_errors is not None:
                        resolution=native_errors.sdf_resolution(prim.GetPath(),int(resolution))['effective_resolution_bound']
                    sdf_error=max(sdf_error,2*float(np.ptp(rigid['V'],axis=0).max())/int(resolution))
            for sheet in kit['parts']:
                if sheet['kind']!='cloth':continue
                name=sheet['name'];limits[name]=priors[obj+'/'+name]['validity_hypothesis']
                prim=ctx.stage.GetPrimAtPath(root+'/'+name+'/surface')
                if sheet.get('backend','particle')=='surface_fem':
                    rest_offset=PhysxSchema.PhysxCollisionAPI(prim).GetRestOffsetAttr().Get()
                else:
                    system=PhysxSchema.PhysxParticleAPI(prim).GetParticleSystemRel().GetTargets()[0]
                    rest_offset=PhysxSchema.PhysxParticleSystem(ctx.stage.GetPrimAtPath(system)).GetRestOffsetAttr().Get()
                if rest_offset is None or not np.isfinite(rest_offset) or rest_offset<=0:
                    raise ValueError('Read actual native cloth contact thickness before source response validation')
                tolerances[name]=solid_tolerance(sampled[obj+'/'+name]['parameters'].get('thickness'),
                    sdf_error,legacy_rest_offset=float(rest_offset))
            if limits:
                from calibrated_scene_reconstruction.assets.response_policy import mode,secondary_contact_budgets
                from calibrated_scene_reconstruction.assets.object_understanding import declaration as understand
                physical_contract={'material_state_contract':contract['material_state_contract'],'links':limits,
                    'material_accuracy_mode':mode(ctx.task.get('flexible_accuracy')),
                    'secondary_contact_budgets_m':secondary_contact_budgets(kit['parts'],understand(ctx.s,obj),ctx.task.get('flexible_accuracy')),
                    'root_connection_tolerance_m':.0001,
                    'maximum_authored_solid_penetration_m':1e-6,'solid_penetration_tolerances_m':tolerances}
        material_data=json.dumps({p['name']:p['cloth_rest'] for p in kit['parts'] if p['kind']=='cloth'})
        link_data=json.dumps(metadata)
        def check(actual):
            if observer is None:
                raw_metrics,_=evidence.score(actual,faces,{'xy':[0.,0.],'yaw_deg':0.,'support_z':0.},source_observations)
                result={'pass':False,'source_fit_pass':False,'actual_image_metrics':raw_metrics,
                        'reason':'Requested source pose has insufficient visible correspondence; raw source check remains failed.',
                        'source_reference_changed_by_settling':False}
            else:result=observer.check(actual)
            if pose_scope is not None:
                raw_metrics,_=evidence.score(actual,faces,{'xy':[0.,0.],'yaw_deg':0.,'support_z':0.},evidence.load(ctx.s,obj))
                result.update(source_pose_scope=pose_scope,unscoped_actual_image_metrics=raw_metrics)
            if rigid_screen is not None:
                result['actual_material_state']=rigid_screen.check(
                    [ctx.pose(obj+'/'+p['name']) for p in kit['parts']])
                result['pass'] &= result['actual_material_state']['pass']
            if physical_contract:
                from calibrated_scene_reconstruction.assets.cloth_state import check as check_physical
                binding_targets={}
                for bond in kit.get('surface_bindings',[]):
                    surface=next(p for p in kit['parts'] if p['name']==bond['surface'])
                    position,rotation=ctx.pose(obj+'/'+bond['rigid'])
                    binding_targets[bond['surface']]=np.asarray(surface['V'])[bond['vertices']]@rotation.T+position
                result['actual_material_state']=check_physical(
                    {'V':actual,'F':faces,'links_json':link_data,'cloth_rest_json':material_data,
                     'surface_bindings_json':json.dumps(kit.get('surface_bindings',[]))},physical_contract,
                    device=ctx.world.device if ctx.world.backend=='torch' else 'cpu',binding_targets=binding_targets)
                result['pass'] &= result['actual_material_state']['pass']
            return result
        samples_out=out/(observation['id']+'_samples');samples_out.mkdir()
        np.savez_compressed(samples_out/'reference.npz',points=reference,faces=faces,target_faces=target)
        anchor=next((p['name'] for p in kit['parts'] if p['kind']=='rigid'),None);anchor_poses=[]
        def preserve_sample(row,actual):
            number=round(row['time_s']/ctx.world.get_physics_dt());stem=f'state_{number:08d}'
            extra={}
            if anchor is not None:
                position,rotation=ctx.pose(obj+'/'+anchor);anchor_poses.append(np.r_[position,rotation.ravel()])
                extra['reference_rigid_pose']=anchor_poses[-1]
            np.savez_compressed(samples_out/(stem+'.npz'),points=actual,**extra)
            (samples_out/(stem+'.json')).write_text(json.dumps(row,indent=2)+'\n')
        # No controls, no holding forces, no initial-pose overwrite. Gravity,
        # friction, joints and all real contacts remain active.
        series,states=passive_series(ctx.world,observation['duration_s'],read,check,on_sample=preserve_sample)
        before,after=series['samples'][0]['result'],series['samples'][-1]['result']
        path = out/(observation['id']+'.npz')
        np.savez_compressed(path, reference=reference, initial=states[0], final=states[-1], faces=faces,
                            sampled_states=states,times_s=[row['time_s'] for row in series['samples']])
        record = {'pass': series['pass'], 'observation_id': observation['id'],
                  'native_states_read': True, 'external_actuation': False,
                  'source_evidence_sha256': observation['evidence_files'],
                  'passive_duration_s': series['duration_s'], 'initial': before, 'passive': after,
                  'sampled_response':series,'diagnosis':response_diagnosis(series)}
        target_part=next(p for p in metadata if p['name']==target_link)
        if target_part['kind']=='cloth':
            a,n=target_part['vertex_start'],target_part['vertex_count']
            rest=next(p['cloth_rest']['points'] for p in kit['parts'] if p['name']==target_link)
            record['material_motion_profile']=material_motion_profile(rest,states[:,a:a+n],
                body_poses=anchor_poses if anchor is not None else None)
        report = path.with_suffix('.json')
        report.write_text(json.dumps(record, indent=2)+'\n')
        mechanism = results.setdefault(observation['mechanism'], {'checks': {}, 'evidence_files': []})
        # More than one physical link can share a named behavior check.
        # Preserve each source observation; the legacy behavior slot alone
        # would overwrite the first link with the last link's evidence.
        mechanism['checks']['observation:'+observation['id']] = record
        mechanism['checks'][observation['check']] = record
        mechanism['evidence_files'].extend(str(p.relative_to(ctx.s.dir)) for p in [path, report])
        mechanism['evidence_files'].extend(str(p.relative_to(ctx.s.dir)) for p in sorted(samples_out.iterdir()))
        passed &= bool(record['pass'])
        if not record['pass']:
            failed_gates.append('observation:'+observation['id'])
            from calibrated_scene_reconstruction.assets.acceptance import may_attempt
            physical_valid=all(row['result'].get('actual_material_state',{}).get('pass') is True
                               for row in series['samples'])
            if not may_attempt(ctx.task,failed_gates,physical_valid=physical_valid):
                break
    return {'pass': passed, 'mechanisms': results, 'required': True,
            'scope': 'Observed source-state consistency before task actuation; no true dimensions or property identification'}
