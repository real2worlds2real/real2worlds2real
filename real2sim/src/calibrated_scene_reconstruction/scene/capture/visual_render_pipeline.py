"""Prepare and review current-capture CAD calibration, with optional uncertain commands."""
import argparse
from dataclasses import replace
import json
import shutil
from pathlib import Path

import numpy as np

from calibrated_scene_reconstruction.core.paths import rel, resolve
from calibrated_scene_reconstruction.robots.spec import load_spec
from calibrated_scene_reconstruction.scene.session import Session
from .articulated_model import from_usd
from .robot_visual import apply_result, associate_robot_tracks, digest, observed_tracks, verify_anchor
from .visual_model_io import load, save
from .visual_mask_fit import load_views, score
from .visual_robot_fit import Settings, fit


def observation_settings(session):
    config=session.config('robot_visual.json',{})
    if config.get('state_input','none')!='none':
        raise ValueError('Visual calibration accepts no supplied joint configuration')
    return config,config.get('observations',{})


def initializer_snapshot(directory, original, content):
    """Bind immutable initializer bytes, never a canonical result overwritten later."""
    target=Path(directory)/'initializer_source.json'
    with target.open('xb') as stream:stream.write(content)
    return target


def prepare(session, initializer=None, validation=False, occluder_suffix=None, robot_suffix=None,
            background_suffix=None, joint_prior=False, rejected_frames=None):
    import pycolmap
    if (session/'qa/independent_freeze.json').exists():
        raise ValueError('Frozen session: prepare a new independent reconstruction')
    if joint_prior:
        from .joint_prior import build, SEEDS
        config=session.config('robot_calibration.json',{})
        observations=config.get('observations',{})
        if initializer is not None or validation:
            raise ValueError('Joint-prior calibration starts from this capture command, not a visual-only anchor')
    else:
        config,observations=observation_settings(session)
    alignment=json.loads((session/'splat/align.json').read_text())
    if (alignment.get('robot_visual_anchor') or alignment.get('robot_joint_anchor')) and not validation:
        raise ValueError('Visual anchor already applied; preserve this calibration and start a new session')
    approved=verify_anchor(session,session.scene_graph) if validation else None
    sim=alignment['sim3_before_visual'] if validation else alignment['sim3']
    reconstruction=pycolmap.Reconstruction(str(session/'splat/data/sparse/full'))
    robot_suffix=robot_suffix or observations.get('robot_suffix','_visual_robot_v2')
    occluder_suffix=occluder_suffix or observations.get('occluder_suffix','_visual_occluders_v3')
    background_suffix=background_suffix or observations.get('background_suffix')
    hands_suffix=observations.get('hands_suffix', '_visual_hands')
    from .robot_visual import observation_rejections
    configured_rejections=observation_rejections(reconstruction,observations.get('rejected_frames'))
    explicit_rejections=observation_rejections(reconstruction,rejected_frames)
    rejected_frames={**configured_rejections,**explicit_rejections}
    points,evidence=observed_tracks(session,reconstruction,robot_suffix,occluder_suffix=occluder_suffix,
                                    rejected_frames=rejected_frames, excluded_suffixes=(hands_suffix,),
                                    require_exclusions=True)
    points,evidence=associate_robot_tracks(points,evidence)
    world=sim['scale']*points@np.asarray(sim['R']).T+sim['t']
    floor_points, floor_prior = None, None
    if not joint_prior:
        from .placement_planes import from_photo
        from .floor_prior import estimate
        raw_floor, _, _ = from_photo(session, reconstruction)
        floor_points = sim['scale']*raw_floor@np.asarray(sim['R']).T+sim['t']
        floor_prior = estimate(floor_points)
    if len(world)<400:raise ValueError('At least400 reliable actual robot tracks required')
    spec=load_spec(config.get('robot','vega_1p_sharpa_arm'))
    model=from_usd(spec['usd'])
    prior=None;initializer_content=None
    if joint_prior:
        state_path=session.config_dir/'robot_state.json'
        prior=build(model,json.loads(state_path.read_text()),config.get('joint_uncertainty',{}))
        state=prior['nominal_rad']
        source,_,_=model.points(state,exclude_links=Settings().exclude_links)
        k=max(.01,float(np.percentile(world[:,2],98)/(np.percentile(source[:,2],98)-model.floor_height())))
        initial={'robot':spec['name'],'supplied_joint_config_read':True,'board_input_read':False,
                 'measured_task_or_support_size_read':False,'estimated_joints_rad':state,
                 'unestimated_joints':prior['unestimated_joints'], 'map_units_per_metre':k,
                 'base_xy_map':np.median(world[:,:2],axis=0).tolist(),
                 'base_z_map':-k*model.floor_height(),'yaw_deg':0.,'ok':False,
                 'source':'Current capture commanded joints as uncertain prior; not measured execution'}
    elif validation:
        if initializer is not None:raise ValueError('Approved validation resumes only its own approved hypothesis')
        initial=json.loads(json.dumps(approved))
    elif initializer is None:
        settings=replace(Settings(),outer_iterations=10,inner_evaluations=10,
                         yaw_step_deg=60,coarse_iterations=3,finalists=4)
        initial=fit(model,world,{'home':spec.get('home',{})},settings)
        initial.update(robot=spec['name'],supplied_joint_config_read=False,board_input_read=False,
                       measured_task_or_support_size_read=False)
    else:
        initializer=resolve(initializer).resolve()
        if not initializer.is_relative_to(session.dir.resolve()):
            raise ValueError('An initializer may only resume this same capture session, never another session')
        initializer_content=initializer.read_bytes()
        initial=json.loads(initializer_content)
        if (initial.get('robot')!=spec['name'] or initial.get('supplied_joint_config_read') is not False
                or initial.get('measured_task_or_support_size_read') is not False):
            raise ValueError('Current-session initializer lacks visual-only provenance')
        for path,expected in initial.get('input_sha256',{}).items():
            if digest(resolve(path))!=expected:
                raise ValueError('Initializer source changed: '+path)
    initial['cad_home_hypothesis']=spec.get('home',{})
    initial['observation_evidence']=evidence
    initial['applied']=False
    views,selection=load_views(session,reconstruction,sim,max_views=128,
                               robot_suffix=robot_suffix,occluder_suffix=occluder_suffix,
                               identity_points=world,background_suffix=background_suffix,
                               require_boundary=False,rejected_frames=rejected_frames, hands_suffix=hands_suffix)
    actual_frames=[set(names) for names in evidence['positive_observation_frames']]
    # Textureless views can constrain silhouettes without containing depth
    # features. Record support, but never mistake missing texture for bad RGB.
    selection['actual_track_counts']={view['name']:sum(view['name'] in frames for frames in actual_frames)
                                     for view in views}
    if len(views)>32:views=[views[i] for i in np.linspace(0,len(views)-1,32).astype(int)]
    selection['selected']=[v['name'] for v in views]
    root=(session/'splat/visual_render').resolve()
    numbers=[int(p.name[6:]) for p in root.glob('round_*') if p.name[6:].isdigit()]
    number=max(numbers,default=0)+1
    directory=root/f'round_{number}'/'input'
    directory.mkdir(parents=True,exist_ok=False)
    save(model,directory)
    # Different camera orientations share a padded canvas. Padding is unknown
    # evidence, with K shifted exactly; input video selection is untouched.
    height=max(v['mask'].shape[0] for v in views);width=max(v['mask'].shape[1] for v in views)
    masks,soft_masks,ignored,Ks,pads=[],[],[],[],[]
    for view in views:
        h,w=view['mask'].shape;top=(height-h)//2;left=(width-w)//2
        pad=((top,height-h-top),(left,width-w-left))
        masks.append(np.pad(view['mask'],pad,constant_values=False))
        soft_masks.append(np.pad(view['soft_mask'],pad,constant_values=0))
        ignored.append(np.pad(view['ignored'],pad,constant_values=True))
        K=view['K'].copy();K[0,2]+=left;K[1,2]+=top;Ks.append(K)
        pads.append({'frame':view['name'],'top':top,'left':left,'source_shape':[h,w]})
    np.savez_compressed(directory/'observations.npz',points=world,names=[v['name'] for v in views],
                        K=Ks,R=[v['R'] for v in views],C=[v['C'] for v in views],
                        masks=masks,soft_masks=soft_masks,ignored=ignored,
                        point_sightings=[[view['name'] in frames for view in views] for frames in actual_frames],
                        **({'floor_points': floor_points} if floor_prior is not None else {}))
    (directory/'initial.json').write_text(json.dumps(initial,indent=2)+'\n')
    if prior is not None:
        (directory/'joint_prior.json').write_text(json.dumps(prior,indent=2)+'\n')
    sources=[session/'splat/data/sparse/full'/name for name in ('points3D.bin','cameras.bin','images.bin')]
    sources.extend(Path(path) for path in model.dependencies)
    if joint_prior:
        sources.append(state_path)
        calibration_path=session.config_dir/'robot_calibration.json'
        if calibration_path.exists():sources.append(calibration_path)
    else:
        sources.append(session.config_dir/'robot_visual.json')
        sources.extend([session/'parsing/masks.npz', session/'parsing/instances.json'])
    sources.extend(path for view in views for path in view['inputs'].values())
    engine_sources=[Path(__file__).with_name(name) for name in (
        'visual_render_pipeline.py','articulated_model.py','visual_model_io.py','visual_mask_fit.py','calibration_scope.py',
        'visual_robot_fit.py','robot_visual.py','visual_render_fit.py','differentiable_robot.py',
        'visual_render_seed.py','joint_prior.py','valid_pixels.py','floor_prior.py','placement_planes.py')]
    code=directory/'engine';code.mkdir()
    for source in engine_sources:shutil.copy2(source,code/source.name)
    sources.extend(engine_sources)
    if initializer is not None:
        sources.append(initializer_snapshot(directory,initializer,initializer_content))
    if validation:sources.append(session/'splat/robot_visual_fit.json')
    metadata={'validation_only':validation,'calibration_mode':'joint_prior' if joint_prior else 'visual_only',
              'floor_prior': floor_prior,
              'source':initial['source'] if joint_prior else 'Current capture and legal CAD only; no supplied joints, board scale or task/support truth',
              'robot':spec['name'],'sim3':sim,'require_distinct_initializations':True,
              'preparation_sha256':{str(path):digest(path) for path in
                  (session/'splat/align.json',session.config_dir/'scene_graph.json')},
              'mask_selection':selection,'camera_padding':pads,'raw_image_rejections':rejected_frames,
              'images':[{key:str(path) for key,path in view['inputs'].items()} for view in views],
              'source_inputs':{str(path):digest(path) for path in sources},
              'engine_snapshots':{str(path):str(code/path.name) for path in engine_sources}}
    if joint_prior:metadata['seeds']=list(SEEDS)
    else:
        from .visual_render_seed import VISUAL_SEEDS
        metadata['seeds']=list(VISUAL_SEEDS)
    if initializer is not None:
        metadata['initializer_original']=str(initializer)
        metadata['initializer_snapshot']=str(directory/'initializer_source.json')
    (directory/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    request={'round':number,'input':rel(directory),'output':rel(directory.parent/'fit'),
             'robot':spec['name'],'robot_points':len(world),'views':len(views),
             'next':'Run scene.capture.visual_render_fit in the recon CUDA environment, then assess'}
    session.write_json('splat/robot_visual_run.json',request)
    print(json.dumps(request,indent=2),flush=True)
    return request



def current_sources(fingerprints):
    for path, expected in fingerprints.items():
        if digest(resolve(path)) != expected:
            raise ValueError('Visual calibration evidence changed: '+path)



def review_sources(fingerprints, manifest):
    """Completed workers bind archived engine bytes; capture data stays live-bound."""
    snapshots=manifest.get('engine_snapshots',{})
    bound={snapshots.get(path,path):expected for path,expected in fingerprints.items()}
    current_sources(bound)


def bind_validation(session, validation_path):
    """Certify an unchanged applied pose, never apply a nearby optimized pose."""
    if (session/'qa/independent_freeze.json').exists():
        raise ValueError('Frozen reconstruction: preserve the approved calibration')
    observation_settings(session)
    anchor=verify_anchor(session,session.scene_graph)
    path=resolve(validation_path).resolve()
    if not path.is_relative_to(session.dir.resolve()):
        raise ValueError('Validation must belong to this capture')
    proof=json.loads(path.read_text())
    if not proof.get('ok') or proof.get('applied') is not False:
        raise ValueError('Only passing, unapplied validation can certify an anchor')
    current_sources(proof['input_sha256'])
    root=session/f'splat/visual_render/round_{int(proof["render_round"])}'
    manifest=json.loads((root/'input/manifest.json').read_text())
    anchor_path=session/'splat/robot_visual_fit.json'
    prepared_sources={str(resolve(name).resolve()):value for name,value in manifest['source_inputs'].items()}
    if (manifest.get('validation_only') is not True or
            prepared_sources.get(str(anchor_path.resolve())) != digest(anchor_path)):
        raise ValueError('Validation was not prepared against this exact applied anchor')
    pose_keys=('robot','map_units_per_metre','base_xy_map','base_z_map','yaw_deg','estimated_joints_rad')
    exact=[]
    for hypothesis in proof['hypotheses']:
        if not hypothesis['ok']:continue
        candidate_path=root/'fit'/(hypothesis['seed']+'.json')
        if proof['input_sha256'].get(str(candidate_path.resolve())) != digest(candidate_path):
            raise ValueError('Passing candidate is not bound to the validation proof')
        candidate=json.loads(candidate_path.read_text())
        if all(candidate[key]==anchor[key] for key in pose_keys):exact.append(hypothesis['seed'])
    if not exact:
        raise ValueError('No passing validation hypothesis exactly matches the applied pose')
    binding={'anchor_sha256':digest(anchor_path),'validation':str(path),
             'exact_pose_hypotheses':exact,
             'input_sha256':{**proof['input_sha256'],str(path):digest(path)}}
    session.write_json('splat/robot_visual_validation.json',binding)
    return binding

def assess(session, apply=False):
    """Accept only independently supported, converged image/CAD hypotheses."""
    if (session/'qa/independent_freeze.json').exists():
        raise ValueError('Frozen reconstruction: preserve the approved calibration')
    request=json.loads((session/'splat/robot_visual_run.json').read_text())
    directory=resolve(request['input']);output=resolve(request['output'])
    for path in (directory,output):
        if not path.resolve().is_relative_to(session.dir.resolve()):
            raise ValueError('Render evidence must belong to this same capture session')
    manifest=json.loads((directory/'manifest.json').read_text())
    joint_prior=manifest.get('calibration_mode')=='joint_prior'
    if not joint_prior:observation_settings(session)
    if apply and manifest.get('validation_only'):
        raise ValueError('Validation cannot replace an approved calibration')
    review_sources(manifest['source_inputs'],manifest)
    current_sources(manifest['preparation_sha256'])
    model=load(directory)
    data=dict(np.load(directory/'observations.npz'))
    views=[]
    for i,name in enumerate(data['names']):
        views.append({'name':str(name),'K':data['K'][i],'R':data['R'][i],'C':data['C'][i],
                      'mask':data['masks'][i],'ignored':data['ignored'][i],
                      'inputs':{key:Path(value) for key,value in manifest['images'][i].items()}})
    candidates=[]
    # Every prepared starting articulation is required, including failures. A
    # successful seed must never hide an equally plausible conflicting scale.
    for seed in manifest.get('seeds',('inferred','cad_home_hypothesis','midpoint_hypothesis')):
        path=output/(seed+'.json')
        candidate=json.loads(path.read_text())
        floor = manifest.get('floor_prior')
        if floor is not None:
            if candidate.get('floor_uncertainty') != floor or abs(candidate.get('floor_offset_m', 0.)*candidate['map_units_per_metre']) > floor['max_offset_map']+1e-7:
                raise ValueError('Candidate exceeds independent observed-floor uncertainty')
        fit_info=candidate['render_fit']
        if candidate.get('seed') != seed or candidate.get('robot') != manifest['robot']:
            raise ValueError('Candidate identity differs from the prepared legal CAD')
        review_sources(fit_info['input_sha256'],manifest)
        bound_files={str(resolve(name).resolve()):value for name,value in fit_info['input_sha256'].items()}
        for file in directory.iterdir():
            if file.is_file() and bound_files.get(str(file.resolve())) != digest(file):
                raise ValueError('Candidate is not bound to the prepared input: '+str(file))
        mask_path=output/(seed+'_masks.npz')
        if digest(mask_path) != fit_info['mask_sha256']:
            raise ValueError('Rendered prediction changed after optimization')
        evidence_path=output/(seed+'_evidence.jpg')
        if digest(evidence_path) != fit_info['evidence_sha256']:
            raise ValueError('Rendered evidence changed after optimization')
        masks=dict(np.load(mask_path))
        if not np.array_equal(masks['names'],data['names']):
            raise ValueError('Prediction cameras differ from the prepared observations')
        metrics=score(model,data['points'],candidate,views,predicted_masks=masks['predicted'],
                      point_sightings=data.get('point_sightings'))
        if joint_prior:
            from .joint_prior import audit
            prior=json.loads((directory/'joint_prior.json').read_text())
            if candidate.get('supplied_joint_config_read') is not True:
                raise ValueError('Joint-prior worker must disclose supplied command input')
            checked=audit(prior,candidate['estimated_joints_rad'],candidate.get('floor_offset_m',0.))
            candidate['joint_prior_audit']=checked
            metrics['reasons'].extend(checked['reasons'])
            metrics['ok']=metrics['ok'] and checked['ok']
        if fit_info['heldout_iou'] < .85:
            metrics['ok']=False
            metrics['reasons'].append('Held-out image IoU fails the shared silhouette criterion')
        candidate.update(metrics)
        candidate['input_sha256']={**fit_info['input_sha256'],
                                   str(path):digest(path),str(mask_path):digest(mask_path),
                                   str(evidence_path):digest(evidence_path)}
        candidates.append(candidate)
    fixed=[c for c in candidates if c['seed']=='joint_fixed'] if joint_prior else []
    candidates=[c for c in candidates if c['seed']!='joint_fixed'] if joint_prior else candidates
    candidates.sort(key=lambda c:(not c['ok'],-c['silhouette_iou_median']))
    best=candidates[0]
    plausible=[c for c in candidates if c['ok'] and
               c['silhouette_iou_median'] >= best['silhouette_iou_median']-.015]
    scales=np.array([c['map_units_per_metre'] for c in plausible])
    spread=float(np.ptp(scales)/np.median(scales)) if len(scales)>1 else None
    result={**best,'applied':False,'plausible_scale_spread':spread,
            'plausible_scales':scales.tolist(),'render_round':request['round'],
            'hypotheses':[{'seed':c['seed'],'ok':c['ok'],'scale':c['map_units_per_metre'],
                           'iou':c['silhouette_iou_median'],'local':c['silhouette_local_difference'],
                           'residual_median_m':c['residual_median_m'],
                           'surface_coverage':c['surface_coverage'],'reasons':c['reasons']}
                          for c in candidates],
            'source':('Current-capture CAD rendering with uncertain commanded joints' if joint_prior else
                      'Current-capture CAD rendering; inferred joints, independent sparse-surface validation'),
            'supplied_joint_config_read':joint_prior,'board_input_read':False,
            'measured_task_or_support_size_read':False}
    result['reasons']=list(best['reasons'])
    if len(plausible)<2:
        result['reasons'].append('Fewer than two independently initialized hypotheses pass all evidence gates')
    if manifest.get('require_distinct_initializations'):
        lineages=[]
        for candidate in plausible:
            state=candidate['render_fit'].get('articulation_lineage_start')
            if not state or not all(np.isfinite(value) for value in state.values()):
                raise ValueError('Prepared worker must record its finite articulation lineage start')
            if not any(set(state)==set(old) and all(abs(state[key]-old[key])<1e-6 for key in state)
                       for old in lineages):lineages.append(state)
        result['distinct_passing_initializations']=len(lineages)
        if len(lineages)<2:
            result['reasons'].append('Passing candidates do not establish two distinct articulation initializations')
    if spread is not None and spread > .05:
        result['reasons'].append('Plausible articulated hypotheses disagree in scale by over5%')
    result['ok']=not result['reasons']
    # Bind every tested hypothesis, including failures, to the decision.
    result['input_sha256']={key:value for c in candidates+fixed for key,value in c['input_sha256'].items()}
    # The assessor may be newer than a completed worker. Bind its actual rules
    # separately, rather than presenting the worker's archived gate code as the
    # version that evaluated these candidates.
    assessor=output/'assessment_engine';assessor.mkdir(exist_ok=True)
    result['assessment_engine']=[]
    for name in ('visual_render_pipeline.py','visual_mask_fit.py','visual_robot_fit.py','calibration_scope.py',
                 'robot_visual.py','joint_prior.py','articulated_model.py','visual_model_io.py'):
        source=Path(__file__).with_name(name);snapshot=assessor/name
        if snapshot.exists() and digest(snapshot)!=digest(source):
            raise ValueError('Preserve the earlier assessment engine; prepare a new round')
        shutil.copy2(source,snapshot)
        result['input_sha256'][str(snapshot)]=digest(snapshot)
        result['assessment_engine'].append({'original':str(source),'snapshot':str(snapshot),'sha256':digest(snapshot)})
    if joint_prior:
        result['calibration_mode']='joint_prior'
        result['joint_uncertainty']=prior
        result['fixed_command_baseline']=[{key:c.get(key) for key in (
            'map_units_per_metre','silhouette_iou_median','silhouette_local_difference',
            'residual_median_m','surface_coverage','ok','reasons')} for c in fixed]
    # Verify live worker sources, then bind the exact archived bytes. A future
    # software update does not alter the completed calibration's evidence.
    review_sources(result['input_sha256'],manifest)
    result['engine_sources']=[]
    for original,snapshot in manifest.get('engine_snapshots',{}).items():
        expected=result['input_sha256'].get(original)
        if expected is None or digest(snapshot)!=expected:
            raise ValueError('Prepared engine snapshot differs from the executed source')
        result['input_sha256'].pop(original)
        result['input_sha256'][snapshot]=expected
        result['engine_sources'].append({'original':original,'snapshot':snapshot,'sha256':expected})
    decision=output/'assessment.json'
    if decision.exists():
        raise ValueError('Preserve the existing assessment; prepare a new round')
    decision.write_text(json.dumps(result,indent=2)+'\n')
    result['input_sha256'][str(decision)]=digest(decision)
    current_sources(manifest['preparation_sha256'])
    current_sources(result['input_sha256'])
    result_path=(f'splat/visual_validation_round_{request["round"]}.json' if manifest.get('validation_only')
                 else 'splat/robot_joint_fit.json' if joint_prior else 'splat/robot_visual_fit.json')
    path=session/result_path
    if path.exists():
        archive=session/'splat/visual_attempts';archive.mkdir(exist_ok=True)
        index=1
        while (archive/f'before_render_round_{request["round"]}_{index}.json').exists():index+=1
        (archive/f'before_render_round_{request["round"]}_{index}.json').write_bytes(path.read_bytes())
    if result['ok'] and apply:
        if joint_prior:
            from .robot_visual import apply_joint_result
            apply_joint_result(session,result)
        else:apply_result(session,result)
    session.write_json(result_path,result)
    print(json.dumps({key:result[key] for key in ('ok','applied','map_units_per_metre',
                    'residual_median_m','surface_coverage','silhouette_iou_median',
                    'silhouette_local_difference','plausible_scale_spread','hypotheses','reasons')},indent=2),flush=True)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session');parser.add_argument('action',choices=['prepare','assess','bind-validation'])
    parser.add_argument('--initializer')
    parser.add_argument('--joint-prior',action='store_true',help='treat this capture robot_state.json as uncertain command input')
    parser.add_argument('--occluder-suffix',help='same-capture semantic observation revision')
    parser.add_argument('--robot-suffix',help='same-capture robot-body mask revision; previous masks are preserved')
    parser.add_argument('--background-suffix',help='experimental independently labelled background masks; bounded interior uncertainty')
    parser.add_argument('--reject-frame',action='append',default=[],metavar='FRAME=REASON',
                        help='named raw-image identity/occlusion rejection; never reject by fit error')
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--validate-approved',action='store_true',help='test the current backend without replacing an approved anchor')
    parser.add_argument('--validation',help='passing current-session validation to bind to the unchanged applied pose')
    args=parser.parse_args()
    session=Session(args.session)
    if args.action=='prepare':
        if args.apply:parser.error('Application requires a completed assessment')
        if any('=' not in value for value in args.reject_frame):parser.error('--reject-frame needs FRAME=REASON')
        rejected_frames=dict(value.split('=',1) for value in args.reject_frame)
        prepare(session,args.initializer,args.validate_approved,args.occluder_suffix,args.robot_suffix,args.background_suffix,args.joint_prior,rejected_frames)
    elif args.action=='bind-validation':
        if not args.validation or args.apply:parser.error('Provide --validation without --apply')
        print(json.dumps(bind_validation(session,args.validation),indent=2))
    elif not assess(session,args.apply)['ok']:
        raise SystemExit(2)


if __name__=='__main__':main()
