"""Bounded Blender shape and pose fitting using only the twin's metric observations.

Bounded parameter fitting pattern adapted from real2sim silhouette_fit.py. Unlike that implementation, all
acceptance thresholds and local shape evidence come from calibrated_scene_reconstruction; optimizer termination is not acceptance.
"""
import json
import shutil
from contextlib import ExitStack

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from calibrated_scene_reconstruction.core.paths import resolve as resolve  # noqa: F401 -- legacy module attribute
from calibrated_scene_reconstruction.core.session_code import source_path
from calibrated_scene_reconstruction.scene.objects.object_fit import LIFT, POOR, frames_support_z, placement
from calibrated_scene_reconstruction.scene.objects.blender import evidence, store
from calibrated_scene_reconstruction.scene.objects.blender.blender_run import Blender
from calibrated_scene_reconstruction.scene.objects.blender.contract import contract, write_json


def model_path(s, obj):
    f = s.scene_graph['dynamic_objects'][obj]['fit']
    if bool(f.get('kit')) == bool(f.get('model')):
        raise ValueError(f'{obj}: choose exactly one of fit.kit and fit.model')
    return source_path(s.config_dir, f['model'])


def support_contact(s):
    """A verified modeled support replaces the provisional noisy point plane."""
    graph = s.scene_graph
    name = graph.get('task', {}).get('support')
    support = graph.get('furniture', {}).get(name, {})
    fix = support.get('fix') or {}
    if not fix.get('model'):
        return None
    from calibrated_scene_reconstruction.scene.support.blender import verify
    verify(s, name)
    return {'support': name, 'fix': fix, 'top_z': float(support['box'][1][2]),
            **({'surface_plane':fix['surface_plane']} if fix.get('surface_plane') else {})}


def round_pose_start(s, obj):
    """Reuse a pose guess only in unchanged cameras and silhouette observations.

    A failed round is a valid optimizer seed, never an acceptance certificate.
    Model geometry may change; support height and full-yaw scoring stay current.
    """
    number = s.scene_graph['dynamic_objects'][obj]['fit'].get('start_round')
    if number is None:
        return None
    if type(number) is not int or number < 1:
        raise ValueError('start_round must name a positive prior object round')
    path = s/'objects'/obj/f'round_{number}'
    previous = json.loads((path/'fit.json').read_text())
    inputs = json.loads((path/'input.json').read_text())['inputs']
    for name in ('parsing/posed_frames/cameras.npz', 'splat/align.json', f'splat/object_fit/{obj}.npz'):
        if name not in inputs or store.digest(s/name) != inputs[name]:
            raise ValueError('Pose seed observations changed: '+name)
    pose = previous['pose']
    if previous.get('round') != number or np.asarray(pose['xy']).shape != (2,) or not np.isfinite([*pose['xy'], pose['yaw_deg']]).all():
        raise ValueError('Invalid prior round pose seed')
    return pose


def _initial_pose(s, obj, extent, *, source_seed=None):
    from calibrated_scene_reconstruction.scene.capture.colmap_points import world_points
    from calibrated_scene_reconstruction.scene.objects.object_fit import local_top
    o = s.scene_graph['dynamic_objects'][obj]
    fx = o.get('fix') or {}
    path = s / 'splat/object_fit.json'
    last = (json.loads(path.read_text()) if path.exists() else {}).get(obj, {})
    here = placement(s)
    # A checked, immutable source proposal can supply the planar initializer.
    # Height and support orientation still come from the current support below.
    seed = source_seed if source_seed is not None else round_pose_start(s, obj)
    if seed is not None:
        xy, yaw = seed['xy'], seed['yaw_deg']
    elif last.get('iou_median', 0) >= POOR and fx.get('placement') == here and fx.get('xy'):
        xy, yaw = fx['xy'], fx.get('yaw_deg', 0)
    elif (s / 'splat/object_fit' / f'{obj}.start.json').exists():
        start = json.loads((s / 'splat/object_fit' / f'{obj}.start.json').read_text())
        if start['placement'] != here:
            raise ValueError('Observation start belongs to an earlier metric placement')
        xy, yaw = start['xy'], start['yaw_deg']
    else:
        from calibrated_scene_reconstruction.scene.objects.object_hull import photo_start
        z = np.load(s / 'parsing/posed_frames/cameras.npz')
        xy = photo_start(s, z, o.get('ids', []), frames_support_z(s))
        if xy is None:
            raise ValueError('No valid observation-derived placement; run object_fit observation acquisition')
        yaw = 0.0
    from .support_seed import pose as nested_support_pose
    nested=nested_support_pose(s,obj,xy,yaw)
    if nested is not None:
        if o['fit'].get('align_to_support',False):
            raise ValueError('Candidate parent supports require independently validated orientation; planar table alignment cannot be reused')
        return nested
    contact = support_contact(s)
    if contact is not None:
        top = contact['top_z']
        plane=contact.get('surface_plane')
        if plane:
            from calibrated_scene_reconstruction.scene.support.observed_plane import height
            top=height(plane,xy)
        aligned=s.scene_graph['dynamic_objects'][obj]['fit'].get('align_to_support',False)
        if type(aligned) is not bool:raise ValueError('align_to_support must be boolean')
        orientation={}
        if aligned:
            if not plane:raise ValueError('Support alignment needs a verified source plane')
            from calibrated_scene_reconstruction.assets.object_pose import support_rotation
            orientation={'support_rotation':support_rotation(plane['normal']).tolist()}
        return {'xy': list(map(float, xy)), 'yaw_deg': float(yaw), 'local_top': top,
                'support_z': top, 'contact': 'verified_modeled_support',
                **orientation,
                **({'support_plane':plane} if plane else {})}
    if s.scene_graph['dynamic_objects'][obj]['fit'].get('align_to_support',False):
        raise ValueError('Support alignment needs a verified modeled support')
    top = frames_support_z(s)
    if top is None:
        raise ValueError('No measured support plane')
    local = local_top(world_points(s)[0], *xy, float(max(extent)), top)
    top = local if local is not None else top
    return {'xy': list(map(float, xy)), 'yaw_deg': float(yaw), 'local_top': float(top), 'support_z': float(top)}


def initial_pose(s, obj, extent, *, source_seed=None):
    from .triangle_support import attach
    return attach(s, obj, _initial_pose(s, obj, extent, source_seed=source_seed))


def ground_pose(vertices,parts,pose,*,faces=None):
    if pose.get('support_triangles'):
        from .triangle_support import apply
        return apply(vertices,faces,parts,pose)
    if not pose.get('support_plane'):return pose
    from calibrated_scene_reconstruction.assets.support_placement import offset
    from calibrated_scene_reconstruction.scene.support.observed_plane import height
    from calibrated_scene_reconstruction.assets.object_pose import rotation
    R=rotation(pose)
    mesh={'V':vertices}
    if parts:mesh['links_json']=json.dumps(parts)
    plane=pose['support_plane'];surface=height(plane,pose['xy'])
    delta=offset(mesh,pose.get('support_thicknesses_m',{}),normal=plane['normal'],rotation=R)
    return {**pose,'local_top':float(surface+delta),'support_z':float(surface+delta+pose.get('down_offset',0.)),
            'support_surface_z':surface,'support_contact_offset_m':float(delta)}


def rigid_anchor(V,model_parts):
    """Rotate a compound hypothesis around its base body, not a loose flap."""
    rigid=next((p for p in model_parts or [] if p.get('kind','rigid')=='rigid'
                and 'vertex_start' in p and 'vertex_count' in p),None)
    if rigid is None:return np.zeros(2)
    points=np.asarray(V)[rigid['vertex_start']:rigid['vertex_start']+rigid['vertex_count'],:2]
    return (points.min(0)+points.max(0))/2


def reanchor_pose(pose,old_vertices,old_parts,new_vertices,new_parts):
    """Keep the primary rigid body's world centre when sampling changes."""
    delta=rigid_anchor(old_vertices,old_parts)-rigid_anchor(new_vertices,new_parts)
    from calibrated_scene_reconstruction.assets.object_pose import rotation
    return {**pose,'xy':(np.asarray(pose['xy'])+rotation(pose)[:2,:2]@delta).tolist()}


def full_resolution_seed(builder,spec,proposed,output):
    """Recover a physically admissible full model without trusting a coarse pass.

    Backtracking changes a geometry proposal, never a material limit or source
    label. Every attempted full-model failure remains in the round protocol.
    """
    defaults={name:p['init'] for name,p in spec['params'].items()};attempts=[]
    for fraction in [1.,.75,.5,.25,0.]:
        values={name:defaults[name]+fraction*(proposed[name]-defaults[name]) for name in defaults}
        try:
            path,report=builder.evaluate(values)
            attempts.append({'fraction':fraction,'params':values,'admissible':True,'mesh':str(path/'mesh.npz')})
            write_json(output,{'proposal_only':True,'full_model_attempts':attempts,'selected_fraction':fraction})
            return values,path,report
        except ValueError as error:
            attempts.append({'fraction':fraction,'params':values,'admissible':False,'error':str(error)})
            write_json(output,{'proposal_only':True,'full_model_attempts':attempts,'selected_fraction':None})
    raise ValueError('No admissible full-resolution seed; revise the model instead of accepting its coarse proposal')


def seed_body_centre(s,obj,V,parts,pose):
    """A changed loose part must not move the previous base-body pose seed."""
    number=s.scene_graph['dynamic_objects'][obj]['fit'].get('start_round')
    if number is None or not parts:return pose
    previous=s/'objects'/obj/f'round_{number}'
    with np.load(previous/'mesh.npz') as data:
        old_parts=json.loads(str(data['links_json'])) if 'links_json' in data else []
        old_anchor=rigid_anchor(data['V'],old_parts)
    if not old_parts or old_parts[0]['name']!=parts[0]['name']:return pose
    from calibrated_scene_reconstruction.assets.object_pose import rotation
    delta=old_anchor-rigid_anchor(V,parts)
    return {**pose,'xy':(np.asarray(pose['xy'])+rotation(pose)[:2,:2]@delta).tolist(),
            'pose_seed_anchor':{'round':number,'previous_mesh_sha256':store.digest(previous/'mesh.npz'),
                                'previous_fit_sha256':store.digest(previous/'fit.json'),
                                'base_link':parts[0]['name'],'local_anchor_change_m':delta.tolist()}}


def orientation_hypothesis(V, F, initial, observations, rasterizer=None, model_parts=None):
    """Choose a full-yaw image hypothesis before local shape/pose refinement."""
    candidates=[];poses={};anchor=rigid_anchor(V,model_parts)
    def rotate(point,yaw):
        from calibrated_scene_reconstruction.assets.object_pose import rotation
        return rotation({**initial,'yaw_deg':yaw})[:2,:2]@point
    centre=np.asarray(initial['xy'])+rotate(anchor,initial['yaw_deg'])
    for offset in np.arange(0.,360.,15.):
        pose={**initial,'yaw_deg':float(initial['yaw_deg']+offset)}
        pose['xy']=(centre-rotate(anchor,pose['yaw_deg'])).tolist()
        pose=ground_pose(V,model_parts,pose,faces=F)
        poses[pose['yaw_deg']]=pose
        metrics,_=evidence.score(V,F,pose,observations,rasterizer=rasterizer,model_parts=model_parts)
        candidates.append({'yaw_deg':pose['yaw_deg'],'loss':metrics['loss'],
                           'search_loss':evidence.search_objective(metrics),
                           'iou_median':metrics['iou_median'],
                           'local_difference':metrics['local_difference'],'ok':metrics['ok']})
    best=min(candidates,key=lambda item:(not item['ok'],item['search_loss']))
    return poses[best['yaw_deg']], {
        'source':'Current model geometry against all unchanged current-capture observations; no task sizes or task-test objective',
        'selected_yaw_deg':best['yaw_deg'],'rotation_anchor_local_xy':anchor.tolist(),
        'rotation_anchor_world_xy':centre.tolist(),'candidates':candidates}


def select_orientation(s,obj,V,F,initial,observations,rasterizer=None,model_parts=None):
    pose_parameters=s.scene_graph['dynamic_objects'][obj]['fit'].get('pose_parameters')
    if pose_parameters is not None and 'yaw_deg' not in pose_parameters:
        return initial,{'source':'Explicit frozen yaw from current-capture pose seed',
                        'selected_yaw_deg':initial['yaw_deg'],'candidates':[]}
    return orientation_hypothesis(V,F,initial,observations,rasterizer,model_parts)


def exact_shape_refinement(builder, spec, observations, initial, proposed, *,
                           budget, parameter_names=None, pose_parameters=None,
                           anchor_body=False, method='Nelder-Mead'):
    """Reconcile a GPU proposal with the actual acceptance raster, including shape.

    A pose-only correction cannot resolve a coverage-dependent radius or rim
    height. Keep the same mesh family, parameter bounds and every source view.
    This is still source fitting; it cannot approve materials or native tasks.
    """
    refined_spec = json.loads(json.dumps(spec))
    for name, value in proposed['params'].items():
        if name in refined_spec['params']:
            refined_spec['params'][name]['init'] = value
    start = {**initial, **proposed['pose']}
    return optimize(builder, refined_spec, observations, start, budget, None,
                    parameter_names=parameter_names, pose_parameters=pose_parameters,
                    anchor_body=anchor_body, method=method)


def optimize(builder, spec, observations, initial, max_evaluations=300, rasterizer=None,
             parameter_names=None,pose_parameters=None,anchor_body=False,checkpoint=None,
             method='Nelder-Mead'):
    if method not in {'Nelder-Mead','Powell'}:
        raise ValueError('Supported bounded source search methods are Nelder-Mead and Powell')
    names = list(spec['params']) if parameter_names is None else parameter_names
    if (not isinstance(names, list) or any(not isinstance(n, str) for n in names)
            or len(set(names)) != len(names) or set(names)-set(spec['params'])):
        raise ValueError('Optimization parameters must be distinct declared model parameter names')
    fixed_parameters = {n: p['init'] for n, p in spec['params'].items() if n not in names}
    pose_parameters=['xy','yaw_deg','down_offset'] if pose_parameters is None else pose_parameters
    if (not isinstance(pose_parameters,list) or len(set(pose_parameters))!=len(pose_parameters)
            or set(pose_parameters)-{'xy','yaw_deg','down_offset'}):raise ValueError('Choose distinct supported pose parameters')
    base_xy=np.asarray(initial['xy'],float)
    if anchor_body:
        first_path,first=builder.evaluate({n:p['init'] for n,p in spec['params'].items()})
        with np.load(first_path/'mesh.npz') as data:anchor=rigid_anchor(data['V'],first.get('parts'))
        from calibrated_scene_reconstruction.assets.object_pose import rotation
        base_xy=base_xy+rotation(initial)[:2,:2]@anchor
    contact = initial.get('contact') in {'verified_modeled_support','actual_triangle_support_proposal'}
    lo = np.array([spec['params'][n]['lo'] for n in names] +
                  [base_xy[0] - 0.05, base_xy[1] - 0.05, initial['yaw_deg'] - 90, 0. if contact else -LIFT])
    hi = np.array([spec['params'][n]['hi'] for n in names] +
                  [base_xy[0] + 0.05, base_xy[1] + 0.05, initial['yaw_deg'] + 90, 0.0])
    start = np.array([spec['params'][n]['init'] for n in names] + [*base_xy, initial['yaw_deg'], 0.0])
    for name,indices in [('xy',[-4,-3]),('yaw_deg',[-2]),('down_offset',[-1])]:
        if name not in pose_parameters:
            for index in indices:lo[index]=hi[index]=start[index]
    best, trace = {}, []
    class SourceAccepted(Exception):
        pass

    def evaluate(q):
        values = lo + np.clip(q, 0, 1) * (hi - lo)
        params = {**fixed_parameters, **dict(zip(names, values[:-4].tolist()))}
        pose = {'xy': values[-4:-2].tolist(), 'yaw_deg': float(values[-2]), 'local_top': initial['local_top'],
                'down_offset': float(values[-1]), 'support_z': float(initial['local_top'] + values[-1])}
        for key in ['support_plane','support_thicknesses_m','support_rotation','support_seed','support_seed_inputs','support_triangles']:
            if key in initial:pose[key]=initial[key]
        try:
            path, build = builder.evaluate(params)
            with np.load(path / 'mesh.npz') as mesh:
                if contact and (pose.get('support_plane') or pose.get('support_triangles')):
                    pose=ground_pose(mesh['V'],build.get('parts'),pose,faces=mesh['F'])
                elif contact and hasattr(builder,'support_offset'):
                    offset=builder.support_offset(mesh)
                    surface=initial.get('support_surface_z',initial['local_top']-initial.get('support_contact_offset_m',0.))
                    pose.update(local_top=float(surface+offset),support_z=float(surface+offset+values[-1]),
                                support_surface_z=float(surface),support_contact_offset_m=float(offset))
                if anchor_body:
                    from calibrated_scene_reconstruction.assets.object_pose import rotation
                    anchor=rigid_anchor(mesh['V'],build.get('parts'))
                    pose['xy']=(values[-4:-2]-rotation(pose)[:2,:2]@anchor).tolist()
                    if contact and (pose.get('support_plane') or pose.get('support_triangles')):
                        pose=ground_pose(mesh['V'],build.get('parts'),pose,faces=mesh['F'])
                metrics, _ = evidence.score(mesh['V'], mesh['F'], pose, observations, rasterizer=rasterizer,
                                            model_parts=build.get('parts'))
            loss = evidence.search_objective(metrics)
            # Numeric gates are constraints, not interchangeable loss terms. A
            # lower mean loss must not discard a candidate that already meets
            # every unchanged gate in favour of one violating a local gate.
            better = not best or ((bool(metrics['ok']), -loss) >
                                 (bool(best['metrics']['ok']), -best['search_loss']))
            if better:
                best.update(params=params, pose=pose, metrics=metrics, mesh=str(path / 'mesh.npz'), build=build,search_loss=loss)
            trace.append({'loss': loss,'source_loss':metrics['loss'], 'iou': metrics['iou_median'], 'local': metrics['local_difference']})
            # Passing a threshold is not an optimum: pose can partially hide
            # a dimensional error. Stop early only at an exact pixel optimum.
            # Otherwise retain the bounded refinement/convergence search.
            if rasterizer is None and metrics['ok'] and metrics['loss']<=1e-10:
                raise SourceAccepted()
            if len(trace) % 25 == 0:
                if checkpoint is not None:
                    write_json(checkpoint, {**best, 'evaluations': len(trace),
                               'proposal_only': True, 'completed': False,
                               'scope': 'Interrupted-search diagnostic only; final source checks and native validation remain required'})
                parts='; '+', '.join(f'{p["link"]} IoU {p["iou_median"]:.4f} local {p["local_difference"]:.4f}'
                                      for p in best['metrics'].get('source_parts',[])) if best['metrics'].get('source_parts') else ''
                print(f'[objects.fit] {len(trace)} evaluations; best IoU {best["metrics"]["iou_median"]:.4f}, '
                      f'local {best["metrics"]["local_difference"]:.4f}{parts}', flush=True)
            return loss
        except ValueError as exc:
            trace.append({'error': str(exc)})
            return 1e6

    span = hi - lo
    q = np.divide(start - lo, span, out=np.zeros_like(start), where=span > 0)
    # A frozen pose or fixed parameter is not an optimization dimension.
    # Keeping zero-width bounds creates duplicate simplex vertices and can
    # prevent the remaining shape coordinates from exploring independently.
    active=span>0;variable=q[active]
    def evaluate_variable(values):
        full=q.copy();full[active]=values
        return evaluate(full)
    try:
        if len(variable):
            simplex=np.tile(variable,(len(variable)+1,1))
            for i in range(len(variable)):
                simplex[i+1,i] += -.025 if variable[i]>.95 else .025
            options=({'maxfev':max_evaluations,'xatol':.0005,'fatol':.00001,
                      'initial_simplex':simplex,'adaptive':True} if method=='Nelder-Mead' else
                     {'maxfev':max_evaluations,'xtol':.0001,'ftol':.00001})
            result=minimize(evaluate_variable,variable,method=method,bounds=[(0,1)]*len(variable),options=options)
        else:
            result=OptimizeResult(success=True,message='All parameters and pose coordinates fixed',fun=evaluate(q))
    except SourceAccepted:
        result=OptimizeResult(success=True,message='Exact source pixel optimum reached',fun=best['search_loss'])
    if not best:
        raise ValueError('No valid geometry in the fit budget')
    return {**best, 'converged': bool(result.success), 'optimizer_message': str(result.message),
            'selection': 'Lowest search objective among numeric-passing candidates, else lowest search objective overall',
            'search_policy':'Raw source loss plus normalized deficits of every unchanged whole-object and functional-part IoU/local gate; invalid authored geometry rejected',
            'optimizer_terminal_loss': float(result.fun),
            'search_method':method,
            'optimized_parameters': names, 'fixed_parameters': fixed_parameters,
            'pose_parameters':pose_parameters,'pose_anchor':'primary_rigid' if anchor_body else 'model_origin',
            'evaluations': len(trace), 'trace': trace, 'ok': best['metrics']['ok']}


def run(s, obj, max_evaluations=300):
    from calibrated_scene_reconstruction.scene.objects.object_understanding import declaration, validate_build
    declaration(s, obj)
    if not 10 <= max_evaluations <= 2000:
        raise ValueError("max_evaluations must be between 10 and 2000")
    cpu_budget=s.scene_graph['dynamic_objects'][obj]['fit'].get('cpu_refine_evaluations',min(80,max_evaluations))
    if type(cpu_budget) is not int or not 0<=cpu_budget<=400:
        raise ValueError('CPU shape refinement needs a bounded0..400 evaluation budget')
    model = model_path(s, obj)
    from .representation_review import check as representation_check
    representation = representation_check(s, obj, model)
    understood_spec = contract(model)
    validate_build(s, obj, {'parts': [dict(name=n, kind=v['kind'])
                                     for n,v in understood_spec.get('links', {}).items()],
                            'joints': understood_spec.get('joints', [])}, preflight=True)
    from calibrated_scene_reconstruction.scene.objects.mechanisms import validate_model
    validate_model(s, obj, contract(model))
    from calibrated_scene_reconstruction.scene.objects.physical_priors import declaration as physical_priors
    if s.config('task.json', {}).get('task'):
        physical_priors(s, required=True)
    out, k = store.start(s, obj, model, max_evaluations)
    try:
        observations = evidence.load(s, obj,parts=True)
        from .source_pose_seed import load_snapshot as load_pose_seed
        source_seed = load_pose_seed(s, obj, out, observations=observations)
        from .projection import optional
        backend=s.scene_graph['dynamic_objects'][obj]['fit'].get('proposal_backend','auto')
        rasterizer = optional() if backend=='auto' else optional(backend)
        source_assets = json.loads((out/'input.json').read_text()).get('source_assets', {})
        assets = [*source_assets, *[out/relative for relative in source_assets.values()]]
        with ExitStack() as stack:
            full_raw=stack.enter_context(Blender(out/'model.py',s/'objects'/obj/'cache',source_files=assets))
            proposal_fixed=s.scene_graph['dynamic_objects'][obj]['fit'].get('proposal_fixed')
            if proposal_fixed is not None and (not isinstance(proposal_fixed,dict) or not proposal_fixed):
                raise ValueError('proposal_fixed must be a nonempty mapping of declared numeric FIXED sampling settings')
            full_budget=s.scene_graph['dynamic_objects'][obj]['fit'].get('full_refine_evaluations',max(10,min(60,max_evaluations//3))) if proposal_fixed else 0
            if proposal_fixed and (type(full_budget) is not int or not 10<=full_budget<=min(300,max_evaluations-10)):
                raise ValueError('Coarse search needs10..300 full_refine_evaluations and at least10 remaining proposal evaluations')
            raw_builder=(stack.enter_context(Blender(out/'model.py',s/'objects'/obj/'cache',source_files=assets,
                                                   proposal_fixed=proposal_fixed)) if proposal_fixed else full_raw)
            from .initial_state import wrap
            builder=wrap(s,obj,raw_builder,out/'initial_state')
            full_builder=wrap(s,obj,full_raw,out/'initial_state')
            first_path, first = builder.evaluate()
            initial = (initial_pose(s, obj, first['geometry']['extent_m'], source_seed=source_seed)
                       if source_seed is not None else initial_pose(s, obj, first['geometry']['extent_m']))
            with np.load(first_path/'mesh.npz') as first_mesh:
                if initial.get('support_plane') or initial.get('support_triangles'):
                    initial['support_thicknesses_m']=getattr(builder,'contract',{}).get('self_clearance_m',{})
                    initial=ground_pose(first_mesh['V'],first.get('parts'),initial,faces=first_mesh['F'])
                elif initial.get('contact')=='verified_modeled_support' and hasattr(builder,'support_offset'):
                    offset=builder.support_offset(first_mesh);surface=initial['local_top']
                    initial.update(local_top=float(surface+offset),support_z=float(surface+offset),
                                   support_surface_z=float(surface),support_contact_offset_m=float(offset))
                if source_seed is None:
                    initial=seed_body_centre(s,obj,first_mesh['V'],first.get('parts'),initial)
                initial,yaw_evidence=select_orientation(s,obj,first_mesh['V'],first_mesh['F'],initial,observations,rasterizer,
                                                          model_parts=first.get('parts'))
            write_json(out/'initial_yaw_hypotheses.json',yaw_evidence)
            write_json(out / 'initial_pose.json', initial)
            inputs = json.loads((out / 'input.json').read_text())
            parameter_names = s.scene_graph['dynamic_objects'][obj]['fit'].get('optimize_parameters')
            inputs['optimize_parameters'] = parameter_names
            contact = support_contact(s)
            if contact is not None:
                inputs['support_contact'] = contact
                for key in ('mesh', 'visual_glb'):
                    path = resolve(contact['fix'][key])
                    inputs['inputs'][str(path.relative_to(s.dir.resolve()))] = store.digest(path)
            inputs['inputs'][str((out / 'initial_pose.json').relative_to(s.dir))] = store.digest(out / 'initial_pose.json')
            inputs['inputs'][str((out/'initial_yaw_hypotheses.json').relative_to(s.dir))]=store.digest(out/'initial_yaw_hypotheses.json')
            write_json(out / 'input.json', inputs)
            store.event(s, obj, 'initialized', round=k, initial_pose=initial)
            from .proposal_stage import settings as stage_settings,observations as stage_observations
            stage=stage_settings(s,obj);spec=json.loads(json.dumps(builder.spec))
            if stage is not None:
                body_observations=stage_observations(s,obj,observations,stage['exclude_links'])
                proposal=optimize(builder,spec,body_observations,initial,stage['budget'],rasterizer,
                                  parameter_names=stage['parameters'],anchor_body=True,
                                  pose_parameters=s.scene_graph['dynamic_objects'][obj]['fit'].get('pose_parameters'))
                proposal.update(proposal_only=True,scope='Source-body initialization; no object or task acceptance')
                write_json(out/'rigid_body_proposal.json',proposal)
                for name,value in proposal['params'].items():spec['params'][name]['init']=value
                initial={**initial,**proposal['pose']}
                path,report=builder.evaluate(proposal['params'])
                with np.load(path/'mesh.npz') as mesh:
                    initial,full_yaw=select_orientation(s,obj,mesh['V'],mesh['F'],initial,observations,rasterizer,report.get('parts'))
                write_json(out/'post_body_yaw_hypotheses.json',full_yaw)
            request=s.scene_graph['dynamic_objects'][obj]['fit']
            search_observations=observations
            from .source_acceptance import policy as scoped_policy, scoped_comparison
            scopes=scoped_policy(s,obj)
            if scopes:
                scope_error=None;ignored=[]
                try:
                    with np.load(first_path/'mesh.npz') as first_mesh:
                        retained,scoped_observations,ignored=scoped_comparison(first_mesh['F'],observations,first['parts'],scopes)
                    excluded_names={n for row in scopes for n in (row.get('links') or [row['link']])}
                    rigid_retained=any(p['kind']=='rigid' and p['name'] not in excluded_names for p in first['parts'])
                    if retained.any() and rigid_retained:
                        search_observations=scoped_observations
                        search_observations[0]['_proposal_excluded_links']=sorted(excluded_names)
                except ValueError as error:
                    if not all(row.get('kind')=='qualitative_flexible' for row in scopes):raise
                    scope_error=str(error)
                if '_source_material' in observations[0]:
                    search_observations[0]['_source_material']=observations[0]['_source_material']
                write_json(out/'source_scope_search.json',{'proposal_only':True,'policy':scopes,
                    'ignored_source_pixels':ignored,'unavailable_default_scope':scope_error,
                    'scope':'Eligible independently labelled source-pose domains initialize retained rigid geometry. Missing default controls preserve full-source search. Full source metrics, physical validity and native acceptance remain required.'})
            baseline_spec=json.loads(json.dumps(spec))
            if request.get('parameter_hypotheses') is not None:
                from .proposal_stage import shape_hypotheses
                proposal,hypotheses=shape_hypotheses(builder,spec,observations,initial,
                                                   request['parameter_hypotheses'],rasterizer)
                write_json(out/'initial_parameter_hypotheses.json',hypotheses)
                for name,value in proposal['params'].items():spec['params'][name]['init']=value
                initial=proposal['pose']
            anchor=request.get('pose_anchor')
            if anchor not in {None,'primary_rigid'}:raise ValueError('Supported pose anchor is primary_rigid')
            result = optimize(builder, spec, search_observations, initial, max_evaluations-full_budget, rasterizer,
                              parameter_names=parameter_names,pose_parameters=request.get('pose_parameters'),
                              anchor_body=anchor=='primary_rigid',checkpoint=out/'fit_progress.json',
                              method=request.get('search_method','Nelder-Mead'))
            write_json(out/'proposal.json',result)
            result['proposal_rasterizer'] = 'nvdiffrast; final CPU evidence required' if rasterizer else 'CPU'
            if proposal_fixed:
                coarse=result
                values,seed_path,seed_build=full_resolution_seed(full_builder,baseline_spec,coarse['params'],out/'full_resolution_seed.json')
                with np.load(coarse['mesh']) as old,np.load(seed_path/'mesh.npz') as fine:
                    full_initial={**initial,**reanchor_pose(coarse['pose'],old['V'],coarse['build'].get('parts'),fine['V'],seed_build.get('parts'))}
                for name,value in values.items():spec['params'][name]['init']=value
                result=optimize(full_builder,spec,search_observations,full_initial,full_budget,rasterizer,
                                parameter_names=parameter_names,pose_parameters=request.get('pose_parameters'),
                                anchor_body=anchor=='primary_rigid',method=request.get('search_method','Nelder-Mead'))
                result.update(proposal_fixed=proposal_fixed,proposal_metrics=coarse['metrics'],
                              proposal_evaluations=coarse['evaluations'],full_resolution_evaluations=result['evaluations'],
                              proposal_rasterizer=coarse['proposal_rasterizer'])
                write_json(out/'full_resolution_proposal.json',result)
            if rasterizer is not None and cpu_budget>=10:
                candidate_path,candidate_build=full_builder.evaluate(result['params'])
                with np.load(candidate_path/'mesh.npz') as actual:
                    cpu_metrics,_=evidence.score(actual['V'],actual['F'],result['pose'],observations,
                                                model_parts=candidate_build.get('parts'))
                # Large structural mismatches need a revised model. Spend the
                # exact-raster budget on candidates close enough to reconcile;
                # this condition changes search scheduling, never a gate.
                near=(cpu_metrics['iou_median']>=evidence.IOU_MIN
                      and cpu_metrics['local_difference']<=3*evidence.LOCAL_MAX
                      and all(p['iou_median']>=evidence.IOU_MIN and p['local_difference']<=3*evidence.LOCAL_MAX
                              for p in cpu_metrics.get('source_parts',[])))
                if not cpu_metrics['ok'] and near:
                    exact=exact_shape_refinement(full_builder,spec,observations,initial,result,
                        budget=cpu_budget,parameter_names=parameter_names,
                        pose_parameters=request.get('pose_parameters'),anchor_body=anchor=='primary_rigid',
                        method=request.get('search_method','Nelder-Mead'))
                    exact['cpu_shape_refinement']={'evaluations':exact['evaluations'],
                        'initial_metrics':cpu_metrics,'proposal_metrics':result['metrics'],
                        'parameter_bounds_changed':False,'acceptance_limits_changed':False}
                    exact['proposal_rasterizer']=result.get('proposal_rasterizer')
                    for key in ['proposal_fixed','proposal_metrics','proposal_evaluations','full_resolution_evaluations']:
                        if key in result:exact[key]=result[key]
                    result=exact
                    write_json(out/'cpu_shape_refinement.json',result)
            path, report = full_builder.evaluate(result['params'], artifacts=True)
            validate_build(s, obj, report)
            if proposal_fixed:
                # Coarser sampling can change a compound model's recentering.
                # Preserve its actual primary rigid-body placement when the
                # complete model replaces the proposal geometry.
                with np.load(result['mesh']) as coarse,np.load(path/'mesh.npz') as fine:
                    result['proposal_pose']=dict(result['pose'])
                    result['pose']=reanchor_pose(result['pose'],coarse['V'],result['build'].get('parts'),fine['V'],report.get('parts'))
                result['full_search_metrics']=result['metrics'];result['proposal_fixed']=proposal_fixed
                result['full_model_rebuild_required']=True
            validate_model(s, obj, {'links': report.get('links') or {'model': {'kind': 'rigid'}},
                                     'surface_bindings':report.get('surface_bindings',[]),
                                     'joints': report.get('joints', [])})
        for name in ('mesh.npz', 'object.glb', 'scene.blend', 'preview_0.png', 'preview_1.png'):
            shutil.copyfile(path / name, out / name)
        snapshot_hash = contract(out / 'model.py')['sha256']
        if contract(model)['sha256'] != snapshot_hash:
            raise ValueError('Model source changed during fitting; start a new round')
        result.update(round=k, model_sha256=snapshot_hash, input_sha256=store.digest(out / 'input.json'),
                      mesh=str((out / 'mesh.npz').relative_to(s.dir)),
                      build=report, pose=result['pose'], mesh_sha256=store.digest(out / 'mesh.npz'))
        metrics = evidence.run(s, obj, out / 'mesh.npz', result['pose'], out, k)
        # A dense visible mesh can differ by subpixel coverage between the GPU
        # search and exact CPU acceptance. Refine only the permitted in-plane
        # pose, preserving support height, geometry, observations and limits.
        pose_fields = request.get('pose_parameters', ['xy', 'yaw_deg'])
        pose_fields = [name for name in (pose_fields or []) if name in ('xy', 'yaw_deg')]
        limits = metrics.get('thresholds', {})
        if (not metrics['ok'] and pose_fields
                and metrics['iou_median'] >= limits.get('iou_min', evidence.IOU_MIN)
                and metrics['local_difference'] <= 1.25 * limits.get('local_max', evidence.LOCAL_MAX)):
            from .pose_polish import run as polish
            with np.load(out/'mesh.npz') as actual:
                result['pose'], _, result['cpu_pose_polish'] = polish(
                    actual['V'], actual['F'], result['pose'], observations,
                    model_parts=report.get('parts'), parameters=pose_fields,
                    initial_metrics=metrics, checkpoint=out/'cpu_pose_progress.json')
            metrics = evidence.run(s, obj, out/'mesh.npz', result['pose'], out, k)
        metrics['opening_sheet'] = evidence.opening_sheet(s, obj, result['pose'], report.get('opening'), out)
        result.update(metrics=metrics, ok=metrics['ok'])
        if metrics.get('source_pose_controls'):
            from .source_acceptance import decision
            result['source_acceptance']=decision(s,obj,out/'mesh.npz',result)
            result['effective_ok']=result['source_acceptance']['pass']
        if representation_check(s, obj, model) != representation:
            raise ValueError('Representation decision changed during fitting')
        if representation is not None:
            result['representation_review'] = representation
        write_json(out / 'fit.json', result)
        write_json(out / 'params.json', result['params'])
        store.event(s, obj, 'fitted', round=k, ok=result['ok'], model_sha256=result['model_sha256'])
        return result, out
    except Exception as exc:
        write_json(out / 'error.json', {'ok': False, 'error': str(exc)})
        store.event(s, obj, 'failed', round=k, error=str(exc))
        raise
