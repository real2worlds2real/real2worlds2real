"""Bind material-response controls to the native task's numerical configuration.

An authored modulus or a successful collision test cannot certify elasticity.
Controls remain separate from source shape, mechanism and task acceptance.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def control_result(path,parameters,*,physics_hz,iterations,minimum_material_edge_m):
    data=json.loads(Path(path).read_text());failures=[]
    backend=data.get('backend',{})
    if (data.get('synthetic_control') is not True or data.get('positions_prescribed') is not False
            or data.get('native_physics_errors',{}).get('pass') is not True or data.get('pass') is not True):
        failures.append('No passing native uniaxial response control')
    if backend.get('name')!='isaacsim' or not str(backend.get('version','')).startswith('5.1.') or backend.get('solver')!='PGS':
        failures.append('Control did not use the validated Isaac5.1 native backend')
    protocol=data.get('parameters',{});settings=data.get('settings',{})
    if protocol.get('physics_hz')!=physics_hz or protocol.get('position_iterations')!=iterations:
        failures.append('Control timestep/solver budget differs from runtime')
    spacing=protocol.get('spacing_m')
    if (not isinstance(spacing,(float,int)) or not np.isfinite(spacing) or spacing<=0
            or minimum_material_edge_m<.9*spacing):
        failures.append('Runtime material sampling is finer than the tested resolution')
    for name in ['youngs_modulus','poissons_ratio','thickness','flexural_rigidity','areal_density','elasticity_damping','linear_damping']:
        if (name not in settings or name not in parameters
                or not np.isfinite([settings[name],parameters[name]]).all()
                or not np.isclose(settings[name],parameters[name],rtol=1e-5,atol=1e-12)):
            failures.append('Control physical parameter differs from runtime: '+name)
    if not np.isclose(settings.get('bend_damping',settings.get('elasticity_damping',np.nan)),
                      parameters.get('bend_damping',parameters.get('elasticity_damping',np.nan)),rtol=1e-5,atol=1e-12):
        failures.append('Control physical parameter differs from runtime: bend_damping')
    try:
        width=protocol.get('width_m',.04)
        if not np.isfinite(width) or width<=0:raise ValueError('Positive actual coupon width required')
        expected=protocol['traction_N']/(parameters['youngs_modulus']*parameters['thickness']*width)
        stop=protocol['seconds'];rows=[r for r in data['rows'] if r['time_s']>=stop-.2]
        values=np.asarray([r['interior_axial_strain'] for r in rows])
        complete=bool(rows and abs(data['rows'][-1]['time_s']-stop)<1e-8)
        if (not complete or not np.isfinite(values).all() or not np.isfinite(expected) or expected<=0
                or abs(np.median(values)/expected-1)>=.2 or np.ptp(values)/expected>=.1
                or not all(r['material']['pass'] for r in data['rows'])):
            failures.append('Actual settled traction response fails its continuum/validity screen')
    except (KeyError,TypeError,ValueError,ZeroDivisionError):
        failures.append('Missing or invalid complete native traction trace')
    root=Path(__file__).resolve().parents[4]
    for relative in ['apps/scene/check_surface_response.py','src/calibrated_scene_reconstruction/sim/surface_cloth.py']:
        source=root/relative
        hashes=data.get('input_sha256',{})
        if not source.is_file() or hashes.get(str(source.resolve()),hashes.get(relative))!=sha(source):
            failures.append('Native response implementation changed: '+relative)
    snapshot=Path(path).with_name('source.py')
    if not snapshot.is_file() or sha(snapshot)!=sha(root/'apps/scene/check_surface_response.py'):
        failures.append('Executed response snapshot differs from current implementation')
    return {'pass':not failures,'failures':failures,'control':str(path),
            'scope':'Native membrane response qualification only; bending, source state, contact, extraction and wiping remain separate checks.'}


def flexure_result(path,parameters,*,physics_hz,iterations,minimum_material_edge_m):
    from calibrated_scene_reconstruction.assets import flexure_reference
    data=json.loads(Path(path).read_text());failures=[];protocol=data.get('parameters',{});backend=data.get('backend',{})
    if (data.get('pass') is not True or data.get('synthetic_control') is not True
            or data.get('external_forces') is not False or data.get('gravity_enabled') is not False
            or data.get('fixed_or_kinematic_nodes')!=0 or data.get('positions_written_after_reset') is not False
            or data.get('native_physics_errors',{}).get('pass') is not True):
        failures.append('No passing unforced native flexure control')
    if backend.get('name')!='isaacsim' or not str(backend.get('version','')).startswith('5.1.') or backend.get('solver')!='PGS':
        failures.append('Flexure control backend differs from validated Isaac5.1')
    spacing=protocol.get('spacing_m')
    if (protocol.get('physics_hz')!=physics_hz or protocol.get('position_iterations')!=iterations
            or not isinstance(spacing,(int,float)) or not np.isfinite(spacing) or spacing<=0
            or minimum_material_edge_m<.9*spacing):
        failures.append('Flexure numerical profile or tested resolution differs from runtime')
    for name in ['youngs_modulus','poissons_ratio','thickness','flexural_rigidity','areal_density']:
        value=data.get('settings',{}).get(name)
        if value is None or name not in parameters or not np.isfinite([value,parameters[name]]).all() or not np.isclose(value,parameters[name],rtol=1e-5,atol=1e-12):
            failures.append('Flexure physical parameter differs from runtime: '+name)
    try:
        expected=flexure_reference.frequency(.08,parameters['flexural_rigidity'],parameters['areal_density'])
        rows=data['rows'];times=[r['time_s'] for r in rows]
        response=flexure_reference.oscillation(times,[r['modal_amplitude_m'] for r in rows])
        if (not response['pass'] or abs(response['frequency_hz']/expected-1)>=.2
                or times[-1]<max(.8,3.2/expected)-1/240 or not all(r['material']['pass'] for r in rows)):
            failures.append('Actual free-strip oscillation fails the bending-response screen')
    except (KeyError,TypeError,ValueError,ZeroDivisionError):
        failures.append('Missing or invalid full native flexure transient')
    root=Path(__file__).resolve().parents[4]
    for relative in ['apps/scene/check_surface_flexure.py','src/calibrated_scene_reconstruction/assets/flexure_reference.py','src/calibrated_scene_reconstruction/sim/surface_cloth.py']:
        source=root/relative
        if data.get('input_sha256',{}).get(str(source.resolve()))!=sha(source):
            failures.append('Native flexure implementation changed: '+relative)
    snapshot=Path(path).with_name('source.py')
    if not snapshot.is_file() or sha(snapshot)!=sha(root/'apps/scene/check_surface_flexure.py'):
        failures.append('Executed flexure snapshot differs from current implementation')
    return {'pass':not failures,'failures':failures,'control':str(path),
            'scope':'Small-amplitude bending qualification only. Real damping, natural folds, wet adhesion, source state, contact and task checks remain separate.'}


def run(ctx,contract,estimates,case):
    from .contact_response import run as contact_response
    contacts=contact_response(ctx,contract,case,estimates)
    routes={}
    for obj,spec in (contract or {}).get('objects',{}).items():
        for name,body in spec['links'].items():
            support=body.get('initial_support')
            if not support:continue
            link=obj+'/'+name
            try:
                from calibrated_scene_reconstruction.assets.metric_mesh import asset
                from calibrated_scene_reconstruction.assets.support_route import above_unthreaded
                from calibrated_scene_reconstruction.assets.surface_distance import triangle_samples
                kit=asset(ctx.sg['dynamic_objects'][obj]['fix']);parts={p['name']:p for p in kit['parts']}
                rigid=obj+'/'+support['link'];up=ctx.pose(rigid)[1]@np.asarray(support['up'])
                thickness=estimates['bodies'][link]['thickness']
                routes[link]=above_unthreaded(triangle_samples(ctx.nodes(link),parts[name]['F']),ctx.points(rigid),
                    parts[support['link']]['F'],up,tolerance=max(1e-6,2*thickness),surface_region=support['surface_region'])
            except (KeyError,ValueError,TypeError) as error:
                routes[link]={'pass':False,'failure':str(error)}
    required_pass=contacts['pass'] and all(r['pass'] for r in routes.values())
    if not contract or contract.get('material_state_contract',0)<2:
        return {'pass':required_pass,'applicable':contacts['applicable'] or bool(routes),'links':{},'contacts':contacts,'initial_support':routes}
    selected={name:view for name,view in ctx._cloth.items() if getattr(view,'backend',None)=='surface_fem'}
    if not selected:return {'pass':required_pass,'applicable':contacts['applicable'] or bool(routes),'links':{},'contacts':contacts,'initial_support':routes}
    plan=ctx.s.config('agent/material_response.json',{})
    links={};files={};root=ctx.s.dir.resolve()
    for name,view in selected.items():
        try:
            if set(plan)!={'version','controls'} or plan['version']!=2:
                raise ValueError('Surface FEM requires version2 membrane, flexure and matched release controls')
            entries=plan['controls'][name][case]
            rest=np.asarray(view.rest);edges=np.concatenate([view.faces[:,[0,1]],view.faces[:,[1,2]],view.faces[:,[2,0]]])
            minimum=float(np.linalg.norm(rest[edges[:,1]]-rest[edges[:,0]],axis=1).min())
            iterations=view.prim.GetAttribute('physxDeformableBody:solverPositionIterationCount').Get()
            parameters=estimates['bodies'][name]
            if 'areal_density' not in parameters:
                parameters={**parameters,'areal_density':parameters['native_total_mass_kg']/parameters['rest_area_m2']}
            checks={}
            for kind,checker in [('membrane',control_result),('flexure',flexure_result)]:
                entry=entries.get(kind)
                if not entry:raise ValueError('Missing native material control: '+kind)
                path=(root/entry['path']).resolve()
                if not path.is_relative_to(root/'qa') or not path.is_file() or sha(path)!=entry['sha256']:
                    raise ValueError('Current session immutable material-response result required')
                checks[kind]=checker(path,parameters,physics_hz=round(1/ctx.world.get_physics_dt()),
                                    iterations=iterations,minimum_material_edge_m=minimum)
                files[str(path.relative_to(root))]=sha(path)
            from .material_release import compare as compare_release
            releases={}
            for kind in ['release','release_zero']:
                entry=entries.get(kind)
                if not entry:raise ValueError('Missing native material control: '+kind)
                path=(root/entry['path']).resolve()
                if not path.is_relative_to(root/'qa') or not path.is_file() or sha(path)!=entry['sha256']:
                    raise ValueError('Current session immutable release result required')
                releases[kind]=path;files[str(path.relative_to(root))]=sha(path)
            checks['release']=compare_release(releases['release'],releases['release_zero'],parameters,
                physics_hz=round(1/ctx.world.get_physics_dt()),iterations=iterations,minimum_material_edge_m=minimum)
            links[name]={'pass':all(c['pass'] for c in checks.values()),'checks':checks}
        except (ValueError,KeyError,TypeError) as error:
            links[name]={'pass':False,'failures':[str(error)]}
    from calibrated_scene_reconstruction.assets.response_policy import mode
    accuracy=mode(getattr(ctx,'task',{}).get('flexible_accuracy'))
    quantitative=all(v['pass'] for v in links.values())
    return {'pass':required_pass and (quantitative or accuracy=='task_and_semantics'),
            'material_accuracy_mode':accuracy,'quantitative_response_pass':quantitative,
            'task_accepted':False,'applicable':True,'links':links,
            'contacts':contacts,'initial_support':routes,'evidence_sha256':files}
