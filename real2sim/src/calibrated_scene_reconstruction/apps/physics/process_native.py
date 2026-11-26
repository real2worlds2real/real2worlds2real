"""Isaac-only execution harness for agent-authored process models.

An agent module provides create(world, stage, parameters, inputs). It returns
bodies, observe(time, native_states), checks(trace), and optionally drive(time).
The harness advances physics and records the bodies; observe only derives
measurement features. No task/motion taxonomy is imposed here.
"""
import gc
import importlib.util
import json
from pathlib import Path

import numpy as np

from calibrated_scene_reconstruction.physics.process.contract import compare, digest, read, require, write
from calibrated_scene_reconstruction.physics.process.search import fit
from calibrated_scene_reconstruction.physics.process.geometry import signature


def execute(round_dir, engine_version):
    import omni.usd
    import omni.physx
    import carb
    from omni.physx.bindings import _physx as bindings
    from pxr import Usd, UsdGeom, UsdPhysics

    settings = carb.settings.get_settings()
    settings.set('/physics/suppressReadback', False)
    for name in ('SETTING_UPDATE_TO_USD', 'SETTING_UPDATE_VELOCITIES_TO_USD', 'SETTING_UPDATE_PARTICLES_TO_USD'):
        settings.set(getattr(bindings, name), True)

    out = Path(round_dir)
    plan, inputs = read(out / 'plan.json'), read(out / 'inputs.json')
    if plan['simulation'].get('runtime')=='physics_only':
        from calibrated_scene_reconstruction.sim.process_world import World,SingleRigidPrim
    else:
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleRigidPrim
    require(str(engine_version).split('.')[:2] == ['5', '1'], 'Isaac Sim 5.1 is required')
    for item in inputs.values():
        require(digest(item['path']) == item['sha256'], 'Input changed before native execution')
    filename = inputs[plan['simulation']['runner']]['path']
    spec = importlib.util.spec_from_file_location('agent_physical_process', filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    observed = dict(np.load(inputs[plan['observations']]['path'], allow_pickle=False))
    times = np.asarray(observed['t'], float)
    definitions = {b['id']: b for b in plan['model']['bodies']}
    required_checks = plan['model']['required_checks']
    dt, duration = plan['simulation']['dt'], plan['simulation']['duration']
    supplied = {k: v['path'] for k, v in inputs.items() if v['kind'] not in {'observations', 'model_review'}}

    active_world = []
    active_guards = []
    reference_geometry = None

    def evaluate(parameters, index):
        nonlocal reference_geometry
        trial = out / 'trials' / f'{index:04d}'
        trial.mkdir(parents=True, exist_ok=False)
        World.clear_instance()
        omni.usd.get_context().new_stage()
        world = World(stage_units_in_meters=1., physics_dt=dt, rendering_dt=dt,
                      backend=plan['simulation'].get('tensor_backend','numpy'),
                      device=plan['simulation'].get('device','cpu'))
        active_world.append(world)
        world.get_physics_context().enable_fabric(False)
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        handles = module.create(world, stage, dict(parameters),
                                {**supplied, '_process_context': {'physical_limits': plan['model'].get('physical_limits', {}),
                                                                 'trial_directory':str(trial)}})
        require(set(handles.get('consumed_parameters', [])) == set(parameters),
                'Model must explicitly bind every planned parameter; unused physics quantities cannot be fitted')
        if plan['model'].get('observation_definitions'):
            from calibrated_scene_reconstruction.physics.process.region_observations import validate_native
            measured=validate_native(plan['model']['observation_definitions'],handles.get('observation_definitions',{}))
            write(trial/'observation_definitions.json',measured)
        if 'effective_parameters' in handles:
            write(trial / 'effective_parameters.json', handles['effective_parameters'])
        actors = handles['bodies']
        require(set(actors) == set(definitions), 'Native bindings must cover every planned physical body')
        require(set(handles.get('driver_bodies', [])) == {k for k, b in definitions.items() if b['motion'] == 'prescribed'},
                'Declare exactly the observed, prescribed driver bodies')
        shape = signature(stage,actors,definitions)
        if reference_geometry is None:
            reference_geometry = shape
        require(shape == reference_geometry, 'Geometry changed during physics fitting; request a new reviewed model round')
        rigid = {}
        for name, binding in actors.items():
            prim = stage.GetPrimAtPath(binding['path'])
            require(bool(prim), 'Missing native actor: ' + name)
            kind = binding['kind']
            if kind == 'static':
                require(definitions[name]['motion'] == 'static'
                        and any(p.HasAPI(UsdPhysics.CollisionAPI) for p in Usd.PrimRange(prim)),
                        'A static support must have native collision geometry')
            elif kind == 'rigid':
                require(prim.HasAPI(UsdPhysics.RigidBodyAPI), 'Rigid response must be a native physics body')
                if definitions[name]['motion'] == 'dynamic':
                    require(not UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get(),
                            'A response body cannot be a prescribed kinematic animation')
                rigid[name] = SingleRigidPrim(binding['path'],reset_xform_properties=False)
            else:
                require(kind in {'particles', 'deformable'} and prim.GetAttribute('points'), 'Native point actor required')
                schemas = ' '.join(prim.GetAppliedSchemas())
                require(('ParticleSet' in schemas if kind == 'particles' else
                         any(s in schemas for s in ('Deformable', 'ParticleCloth'))),
                        'Unsimulated visual points cannot substitute for native material dynamics')
        kinetic_audit=None
        if plan['model'].get('initial_motion',{}).get('mode')=='stationary':
            from calibrated_scene_reconstruction.physics.process.initial_motion import stationary
            kinetic_audit=stationary(stage,{k:v for k,v in actors.items() if definitions[k]['motion']=='dynamic'})
            write(trial/'initial_kinetic_state.json',kinetic_audit)
        world.reset()
        if 'after_reset' in handles:
            handles['after_reset']()
        for actor in rigid.values():
            actor.initialize()
        if 'readbacks' in handles:
            values = handles['readbacks'](rigid)
            for name,item in values.items():
                require(name in parameters and np.isfinite(item['value'])
                        and np.isclose(parameters[name],item['value'],rtol=1e-4,atol=1e-8),
                        'Native parameter differs from the requested value: '+name)
            write(trial/'parameter_readbacks.json',values)
        for _ in range(round(plan['simulation'].get('settle_seconds',0.)/dt)):
            if 'settle' in handles:
                handles['settle']()
            elif 'drive' in handles:
                handles['drive'](0.)
            world.step(render=False)
            if 'after_settle_step' in handles:
                handles['after_settle_step']()
        rest_audit=None
        if plan['model'].get('initial_motion',{}).get('condition_after_settling',False):
            from calibrated_scene_reconstruction.apps.physics.process_initial_rest import quiesce
            rest_audit=quiesce(stage,{k:v for k,v in actors.items() if definitions[k]['motion']=='dynamic'},
                omni.physx.get_physx_simulation_interface(),omni.usd.get_context().get_stage_id())
            write(trial/'source_time_zero.json',rest_audit)
        write(trial / 'bindings.json', {'bodies': actors, 'driver_bodies': handles.get('driver_bodies',[]),
                                      'settle_seconds': plan['simulation'].get('settle_seconds',0.),
                                      'rest_geometry_sha256':shape})
        stage.Flatten().Export(str(trial / 'native_initial.usdc'))

        def states():
            def numpy(value):
                if hasattr(value,'detach'):value=value.detach().cpu().numpy()
                return np.asarray(value)
            values = {};xforms=UsdGeom.XformCache()
            for name, binding in actors.items():
                if name in rigid:
                    p, q = rigid[name].get_world_pose()
                    value = np.r_[numpy(p), numpy(q)]
                elif binding['kind'] == 'static':
                    value = np.asarray(xforms.GetLocalToWorldTransform(stage.GetPrimAtPath(binding['path']))).T
                else:
                    prim = stage.GetPrimAtPath(binding['path'])
                    value = np.asarray(prim.GetAttribute('points').Get(), float)
                    M = np.asarray(xforms.GetLocalToWorldTransform(prim)).T
                    value = value @ M[:3, :3].T + M[:3, 3]
                require(np.isfinite(value).all(), 'Nonfinite native state: ' + name)
                values[name] = value.copy()
            return values

        trace, predictions = {}, {'t': []}
        records, samples, step = [], 0, 0
        from calibrated_scene_reconstruction.physics.process.driver_guard import DriverGuard
        driver_check=DriverGuard(stage,[v['path'] for k,v in actors.items() if definitions[k]['motion']=='dynamic'],
                                 {k:v for k,v in rigid.items() if definitions[k]['motion']=='dynamic'})
        active_guards.append(driver_check)
        kinetic_trace={}
        max_values = plan['simulation'].get('max_trace_values', 20_000_000)
        while step * dt <= duration + dt / 2:
            t = step * dt
            if samples < len(times) and t + dt / 2 >= times[samples]:
                state = states()
                values = driver_check.run(handles['observe'],t,state)
                require('t' not in values, 'The harness owns simulation time')
                predictions['t'].append(t)
                for key, value in values.items():
                    predictions.setdefault(key, []).append(np.asarray(value, float))
                for name, value in state.items():
                    trace[f'{name}.{samples:05d}'] = value
                if 'joint_states' in handles:
                    for name, value in handles['joint_states']().items():
                        trace[f'joints.{name}.{samples:05d}'] = np.asarray(value,float).copy()
                records.append({'t': t, 'bodies': state})
                if rest_audit is not None:
                    for name,binding in actors.items():
                        if binding['kind']=='rigid' and definitions[name]['motion']=='dynamic':
                            api=UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(binding['path']))
                            kinetic_trace.setdefault(name+'.linear_velocity',[]).append(np.asarray(api.GetVelocityAttr().Get(),float))
                            kinetic_trace.setdefault(name+'.usd_angular_velocity',[]).append(np.asarray(api.GetAngularVelocityAttr().Get(),float))
                require(sum(a.size for a in trace.values()) <= max_values,
                        'Native state trace exceeds declared budget; revise the experiment explicitly')
                samples += 1
                if samples%30==0:
                    print(f'[process] trial={index} samples={samples}/{len(times)} native_t={t:.3f}',flush=True)
            if step >= round(duration / dt):
                break
            if 'drive' in handles:
                driver_check.run(handles['drive'],t)
            world.step(render=False)
            if 'after_step' in handles:driver_check.run(handles['after_step'])
            step += 1
        require(samples == len(times), 'Incomplete native observation horizon')
        predictions = {k: np.asarray(v) for k, v in predictions.items()}
        # Assign actual sample times to the nearest declared timestamp only when
        # their difference is within half a physics step; no trajectory is moved.
        require(np.max(abs(predictions['t'] - times)) <= dt / 2 + 1e-8, 'Native clock differs from observations')
        predictions['t'] = times.copy()
        checks = handles['checks'](records)
        if kinetic_audit is not None:
            checks['initial_kinetic_state']={'pass':kinetic_audit['pass'],
                'reason':'Source-stationary response initial velocities explicitly cleared on native schemas before reset; inherited values recorded. Native free settling/source agreement checked independently.'}
        if rest_audit is not None:
            checks['source_time_zero']={'pass':rest_audit['pass'],
                'reason':'After free preparation, source-stationary velocities reset using native sleep with backend poses unchanged; immediate wake-up resumes ordinary contact solving. No runtime pose or velocity clamp.'}
        require(set(required_checks) <= set(checks), 'Missing planned physical plausibility checks')
        checks['native_execution'] = {'pass': True, 'reason': 'Isaac Sim ' + str(engine_version) + '; native states recorded.'}
        trace['t'] = np.array([r['t'] for r in records])
        trace['settle_seconds'] = np.asarray(plan['simulation'].get('settle_seconds',0.))
        np.savez_compressed(trial / 'native_states.npz', **trace)
        np.savez_compressed(trial / 'predictions.npz', **predictions)
        if kinetic_trace:np.savez_compressed(trial/'kinetic_trace.npz',t=predictions['t'],**kinetic_trace)
        write(trial / 'parameters.json', parameters)
        write(trial / 'physical_checks.json', checks)
        artifacts = {str(p.relative_to(out)): digest(p) for p in trial.iterdir() if p.is_file()}
        return predictions, checks, artifacts

    def safely_evaluate(parameters, index):
        try:
            return evaluate(parameters, index)
        finally:
            for guard in active_guards:guard.close()
            active_guards.clear()
            for world in active_world:
                try:
                    world.stop()
                    world.clear()
                finally:
                    World.clear_instance()
            active_world.clear()
            gc.collect()

    def progress(row):
        with (out/'history.jsonl').open('a') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n')
        print(f'[process] trial={row["index"]} fit_loss={row.get("fit_loss")} error={row.get("error", "")}',flush=True)

    result = fit(plan, observed, safely_evaluate, progress=progress)
    result['engine'] = {'name': 'isaacsim', 'version': str(engine_version),
                        'purpose':plan['simulation'].get('purpose','validation'),'dt':plan['simulation']['dt'],
                        'runtime':plan['simulation'].get('runtime','isaac_world')}
    # Re-score the selected native prediction once, independent of optimizer state.
    if result.get('best'):
        prediction = next(p for p in result['best']['artifacts'] if p.endswith('predictions.npz'))
        result['best']['metrics'] = compare(plan, observed, dict(np.load(out / prediction)))
    write(out / 'result.json', result)
    return result
