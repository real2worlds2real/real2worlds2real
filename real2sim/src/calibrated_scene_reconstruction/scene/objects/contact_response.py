"""Qualify a native contact effect from matched, physically valid controls.

A supported result applies only to the recorded backend, representation,
resolution and numerical profile. It never approves a task or identifies a
real material. Invalid or stale controls are inconclusive, not unsupported.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def control_parameters(data,exclude=()):
    """Compare a profile by its hash-bound path, independent of CLI spelling.

    Relative paths belong to the producer's execution context, not the current
    verifier's working directory. Resolve only a unique suffix in the recorded
    absolute dependency inventory; never guess an unrecorded or ambiguous file.
    """
    result={k:v for k,v in data['parameters'].items() if k not in exclude}
    profile=result.get('profile')
    if profile is not None:
        if not isinstance(profile,str) or not profile:raise ValueError('Explicit recorded profile path required')
        raw=Path(profile);parts=raw.parts;candidates=[]
        for name in data.get('input_sha256',{}):
            path=Path(name)
            if not path.is_absolute():continue
            if (path==raw if raw.is_absolute() else path.parts[-len(parts):]==parts):candidates.append(str(path))
        if len(candidates)!=1:raise ValueError('Profile must resolve to one hash-bound control dependency')
        result['profile']=candidates[0]
    return result


def _trial(path):
    path=Path(path);data=json.loads(path.read_text());failures=[]
    if (data.get('synthetic_control') is not True or data.get('positions_prescribed') is not False
            or data.get('pass') is not True or data.get('native_physics_errors',{}).get('pass') is not True
            or data.get('material_native_readback') is not True):
        failures.append('Missing valid native control and material readback')
    backend=data.get('backend',{})
    if (backend.get('name')!='isaacsim' or not str(backend.get('version','')).startswith('5.1.')
            or backend.get('solver')!='PGS'):
        failures.append('Control is not the requested Isaac Sim5.1 native profile')
    rows=data.get('rows',[]);end=data.get('load_end')
    try:
        times=np.asarray([r['time_s'] for r in rows],float)
        if (len(times)<4 or not np.isfinite(times).all() or times[0]!=0 or not (np.diff(times)>0).all()
                or end not in rows or end['time_s']<=.2 or not all(r['physical_pass'] is True for r in rows)):
            failures.append('Incomplete or physically invalid native contact trace')
        initial=float(rows[0]['relative_x_m']);slip=abs(float(end['relative_x_m'])-initial)
        free=float(end['free_traction_displacement_m'])
        if not np.isfinite([slip,free]).all() or free<.001:
            failures.append('Control does not excite a resolved tangential response')
        gap=float(end['minimum_mid_surface_gap_m'])
        reference=float(end.get('expected_contact_gap_m',2*data.get('solid_rest_offset_m',0.)
                                if data['parameters']['support']=='self' else data.get('rest_offset_m',0.)))
        normal_speed=float(end['upper_mean_velocity_m_s'][2])
        if (not np.isfinite([gap,reference,normal_speed]).all() or reference<=0
                or gap>reference+max(.00015,.25*reference) or abs(normal_speed)>.02):
            failures.append('No settled loaded contact; airborne motion cannot qualify friction')
    except (KeyError,ValueError,TypeError,IndexError):
        failures.append('Missing contact trajectory or loaded normal-contact state');slip=free=None
    hashes=data.get('input_sha256',{})
    if not hashes:failures.append('Missing source provenance')
    for name,digest in hashes.items():
        source=Path(name)
        if not source.is_file() or sha(source)!=digest:failures.append('Control dependency changed: '+name)
    snapshot=path.with_name('source.py')
    if not snapshot.is_file() or sha(snapshot) not in hashes.values():
        failures.append('Missing executed control snapshot')
    states=path.with_name('states.npz')
    if not states.is_file():failures.append('Missing actual native states')
    return data,{'pass':not failures,'failures':failures,'slip_m':slip,'free_slip_m':free,
                 'result':str(path),'sha256':sha(path),'states_sha256':sha(states) if states.is_file() else None}


def compare(zero,nominal):
    """Separate demonstrated friction from unsupported and inconclusive tests."""
    z,zcheck=_trial(zero);n,ncheck=_trial(nominal)
    result={'pass':False,'status':'inconclusive','task_approved':False,'checks':[zcheck,ncheck],
            'scope':'Contact capability at the tested profile only; no task, material or source-pose approval'}
    if not zcheck['pass'] or not ncheck['pass']:return result
    failures=[]
    for key in ['backend','settings','spring_settings','physics_hz','position_iterations','rest_offset_m',
                'solid_rest_offset_m','spring_damping_kg_s','particle_contact_offset_m','max_neighborhood',
                'native_contact_settings','spring_model','material_edge_range_m','native_spring_readback',
                'areal_density_kg_m2','physical_thickness_m',
                'load_force_N','load_ratio_to_weight','node_count','native_mass_kg','input_sha256']:
        a,b=z.get(key),n.get(key)
        if key=='settings' and isinstance(a,dict) and isinstance(b,dict):
            a={k:v for k,v in a.items() if k!='friction'};b={k:v for k,v in b.items() if k!='friction'}
        if a!=b:failures.append('Unmatched control: '+key)
    try:
        zp=control_parameters(z,{'output','friction'});np_=control_parameters(n,{'output','friction'})
    except ValueError as error:
        result['failures']=[str(error)]
        return result
    if zp!=np_:failures.append('Control changes more than the tested friction coefficient')
    try:
        with np.load(Path(zero).with_name('states.npz')) as a,np.load(Path(nominal).with_name('states.npz')) as b:
            if 'points' in a.files:
                for key in ['rest','faces','masses']:
                    if not np.array_equal(a[key],b[key]):failures.append('Unmatched native material: '+key)
                pairs=[(a['points'][0],b['points'][0])]
            else:
                # Surface controls store each body separately. Compare the
                # actually authored material metric/mass from their USDs;
                # the FEM adapter does not claim native per-node mass readback.
                if set(a.files)!=set(b.files) or not a.files:raise ValueError('Different native bodies')
                pairs=[(a[k][0],b[k][0]) for k in a.files]
                from pxr import Usd,UsdGeom
                stages=[Usd.Stage.Open(str(Path(p).with_name('initial.usda'))) for p in [zero,nominal]]
                meshes=[{str(q.GetPath()):q for q in s.Traverse() if q.IsA(UsdGeom.Mesh)} for s in stages]
                if not meshes[0] or set(meshes[0])!=set(meshes[1]):raise ValueError('Different authored surfaces')
                for name in meshes[0]:
                    for key in ['points','faceVertexIndices','omniphysics:restShapePoints','omniphysics:mass']:
                        va,vb=[meshes[i][name].GetAttribute(key).Get() for i in [0,1]]
                        if va is None or vb is None or not np.array_equal(va,vb):
                            failures.append('Unmatched authored material: '+name+'/'+key)
            if any(x.shape!=y.shape or not np.allclose(x,y,rtol=0,atol=5e-5) for x,y in pairs):
                failures.append('Unmatched native initial positions')
    except (ValueError,KeyError,OSError):failures.append('Missing matched native states')
    try:
        mu=float(n['parameters']['friction']);ratio=float(n['load_ratio_to_weight'])
        if z['parameters']['friction']!=0 or mu<=0 or ratio<=0:raise ValueError('Invalid coefficient pair')
        for data,expected in [(z,0.),(n,mu)]:
            if not all(np.allclose(data[k],expected,rtol=0,atol=1e-7) for k in ['native_friction_initial','native_friction_final']):
                failures.append('Native friction does not match the declared coefficient')
        zero_slip=zcheck['slip_m'];nominal_slip=ncheck['slip_m'];free=zcheck['free_slip_m']
        if abs(zero_slip/free-1)>.15:failures.append('Zero friction does not reproduce the resolved free-load response')
        expected_ratio=max(0.,1-mu/ratio);observed_ratio=nominal_slip/zero_slip
        if expected_ratio>.6:failures.append('Friction load is not discriminative enough')
    except (ValueError,TypeError,KeyError,ZeroDivisionError):
        failures.append('Missing friction/load reference');expected_ratio=observed_ratio=None
    result.update(failures=failures,backend=n.get('backend'),parameters=np_,expected_slip_ratio=expected_ratio,
                  observed_slip_ratio=observed_ratio)
    if failures:return result
    if abs(observed_ratio-expected_ratio)<=.15:
        result.update(status='supported');result['pass']=True
    elif expected_ratio<=.4 and .85<=observed_ratio<=1.15:
        result['status']='unsupported_at_tested_profile'
    return result


def recorded_controls(session,objects):
    """Expose missing/stale/unsupported controls to the modeling agent early.

This reads saved native pairs. Runtime material/contact/sampling parity remains
the separate mandatory check in ``run``; a green entry is not task approval.
"""
    contract=session.config('agent/mechanisms.json',{})
    required={f'{obj}/{name}':body['required_contact_effects']
              for obj,spec in contract.get('objects',{}).items() if obj in objects
              for name,body in spec.get('links',{}).items() if body.get('required_contact_effects')}
    plan=session.config('agent/contact_response.json',{});root=session.dir.resolve();entries={}
    for link,effects in required.items():
        entries[link]={}
        for case in ['nominal','low','high']:
            rows={};entries[link][case]=rows
            for effect in effects:
                try:
                    if plan.get('version')!=1:raise ValueError('Missing version1 contact controls')
                    pair=plan['controls'][link][case][effect];paths=[]
                    for kind in ['zero','nominal']:
                        entry=pair[kind];path=(root/entry['path']).resolve()
                        if not path.is_relative_to(root/'qa') or not path.is_file() or sha(path)!=entry['sha256']:
                            raise ValueError('Missing, changed or external contact evidence')
                        paths.append(path)
                    measured=compare(*paths)
                    failures=list(measured.get('failures',[]))
                    for check in measured.get('checks',[]):failures.extend(check.get('failures',[]))
                    rows[effect]={'recorded_pair_pass':measured['pass'],'status':measured['status'],
                                  'failures':failures,'paths':[str(p.relative_to(root)) for p in paths]}
                except (KeyError,ValueError,TypeError,OSError,ImportError,RuntimeError) as error:
                    rows[effect]={'recorded_pair_pass':False,'status':'inconclusive','failures':[str(error)]}
    return {'applicable':bool(required),'recorded_controls_pass':all(row['recorded_pair_pass']
                for cases in entries.values() for rows in cases.values() for row in rows.values()),
            'runtime_profile_checked':False,'entries':entries,'task_approved':False,
            'scope':'Saved pair capability/integrity only. Current native material, numerical profile '
                    'and material-edge sampling must still match at runtime. Do not drop required '
                    'effects when an entry fails; diagnose or qualify an appropriate representation.'}


def run(ctx,contract,case,estimates=None):
    """Fail closed when a required contact law has no current matching proof."""
    required={f'{obj}/{name}':body['required_contact_effects']
              for obj,spec in (contract or {}).get('objects',{}).items()
              for name,body in spec['links'].items() if body.get('required_contact_effects')}
    if not required:return {'pass':True,'applicable':False,'links':{}}
    root=ctx.s.dir.resolve();plan=ctx.s.config('agent/contact_response.json',{});links={};files={}
    supports={'rigid_friction':'rigid','self_friction':'self','surface_pair_friction':'surface'}
    for link,effects in required.items():
        checks={}
        for effect in effects:
            try:
                if plan.get('version')!=1:raise ValueError('Declared native contact controls required')
                pair=plan['controls'][link][case][effect];paths=[]
                for kind in ['zero','nominal']:
                    entry=pair[kind];path=(root/entry['path']).resolve()
                    if not path.is_relative_to(root/'qa') or not path.is_file() or sha(path)!=entry['sha256']:
                        raise ValueError('Current immutable session contact result required')
                    paths.append(path);files[str(path.relative_to(root))]=sha(path)
                check=compare(*paths);native=json.loads(paths[1].read_text())
                if native['parameters']['support']!=supports[effect]:
                    raise ValueError('Contact control does not test the required effect')
                view=ctx._cloth[link]
                if native.get('physics_hz')!=round(1/ctx.world.get_physics_dt()):
                    raise ValueError('Contact timestep differs from runtime')
                if getattr(view,'backend',None)=='particle':
                    if native.get('backend',{}).get('representation')!='particle':
                        raise ValueError('Native contact representation differs from runtime')
                    from calibrated_scene_reconstruction.assets.particle_binding import compare_profile
                    binding=compare_profile(native,view.contact_profile(estimates['bodies'][link]))
                    checks[effect]={**check,'runtime_binding':binding}
                    continue
                if getattr(view,'backend',None)!='surface_fem':
                    raise ValueError('Unsupported native contact representation')
                if native.get('backend',{}).get('representation','surface_fem')!='surface_fem':
                    raise ValueError('Native contact representation differs from runtime')
                settings=native['settings'];prim=view.prim
                parameters=estimates['bodies'][link]
                if 'areal_density' not in parameters:
                    parameters={**parameters,'areal_density':parameters['native_total_mass_kg']/parameters['rest_area_m2']}
                for name in ['youngs_modulus','poissons_ratio','thickness','flexural_rigidity',
                             'areal_density','elasticity_damping','linear_damping']:
                    if not np.isclose(parameters[name],settings[name],rtol=1e-5,atol=1e-12):
                        raise ValueError('Contact material differs from runtime: '+name)
                for name,suffix in [('solver_position_iterations','solverPositionIterationCount'),
                                    ('collision_pair_updates','collisionPairUpdateFrequency'),
                                    ('collision_iteration_multiplier','collisionIterationMultiplier'),
                                    ('self_collision_filter_distance','selfCollisionFilterDistance'),
                                    ('max_linear_velocity','maxLinearVelocity'),
                                    ('max_depenetration_velocity','maxDepenetrationVelocity')]:
                    actual=prim.GetAttribute('physxDeformableBody:'+suffix).Get()
                    if actual is None or not np.isclose(actual,settings[name],rtol=1e-5,atol=1e-10):
                        raise ValueError('Contact numerical profile differs: '+name)
                for name,suffix in [('contact_offset','contactOffset'),('rest_offset','restOffset')]:
                    actual=prim.GetAttribute('physxCollision:'+suffix).Get()
                    if actual is None or not np.isclose(actual,settings[name],rtol=1e-5,atol=1e-10):
                        raise ValueError('Contact collision radius differs: '+name)
                from pxr import UsdShade
                material=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose='physics')[0]
                actual_mu=material.GetPrim().GetAttribute('omniphysics:dynamicFriction').Get()
                if actual_mu is None or not np.isclose(actual_mu,native['parameters']['friction'],rtol=1e-5):
                    raise ValueError('Contact friction differs from runtime material')
                # Finer sampling can change both collision filtering and friction;
                # the coarse benchmark cannot silently certify a finer mesh.
                rest=np.asarray(view.rest);faces=np.asarray(view.faces)
                edges=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
                minimum=float(np.linalg.norm(rest[edges[:,1]]-rest[edges[:,0]],axis=1).min())
                maximum=float(np.linalg.norm(rest[edges[:,1]]-rest[edges[:,0]],axis=1).max())
                if 'material_edge_range_m' in native:
                    from calibrated_scene_reconstruction.assets.contact_control_profile import resolution_matches
                    matched=resolution_matches(native['material_edge_range_m'],[minimum,maximum])
                else:
                    spacing=native['parameters'].get('spacing_m',.002)
                    matched=minimum>=.9*spacing and maximum<=1.6*spacing
                if not matched:
                    raise ValueError('Contact resolution differs from its native control')
                checks[effect]=check
            except (KeyError,TypeError,ValueError,OSError) as error:
                checks[effect]={'pass':False,'status':'inconclusive','failures':[str(error)]}
        links[link]={'pass':all(c['pass'] for c in checks.values()),'checks':checks}
    return {'pass':all(v['pass'] for v in links.values()),'applicable':True,'links':links,'evidence_sha256':files}
