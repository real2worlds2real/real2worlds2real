"""Play a standalone package at its authored clock, without task fixtures.

This checks native startup, not material identification or task acceptance.
By default it is required only for profiles with a Python initialization step;
--force also checks ordinary profiles. No small-step reset wrapper is installed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import traceback

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('session');parser.add_argument('usd')
parser.add_argument('--force',action='store_true')
args=parser.parse_args()
from calibrated_scene_reconstruction.scene.session import Session
from calibrated_scene_reconstruction.scene.simulation import settings
s=Session(Path(args.session).resolve());profile=settings(s)
if not args.force and not (profile or {}).get('initialization_hz'):
    print('[package.physics] Not required for this existing profile',flush=True)
    raise SystemExit(0)
if profile is None:raise ValueError('Declare simulation.json before checking package physics')
asset=Path(args.usd).resolve()
if not asset.is_file() or not asset.is_relative_to(s.dir):
    raise ValueError('An existing package in this session is required')
out=s/'qa'/('package_physics_'+asset.stem)
number=2
while out.exists():
    out=s/'qa'/('package_physics_'+asset.stem+'_r'+str(number));number+=1
out.mkdir(parents=True,exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
inputs={str(asset):sha(asset),str(Path(__file__).resolve()):sha(__file__)}
report={'pass':False,'asset':str(asset),'asset_sha256':inputs[str(asset)],
        'initialization':'Ordinary native reset at the authored clock; no Python small-step wrapper',
        'scope':'Standalone native startup/finite-state check only; task, contact accuracy and semantic approval remain separate.'}
last={}
(out/'source.py').write_bytes(Path(__file__).read_bytes())
from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402
app=SimulationApp({'headless':True})
try:
    import importlib.metadata
    import numpy as np
    import omni.usd
    import omni.physics.tensors as tensors
    from isaacsim.core.api import World
    from isaacsim.core.prims import RigidPrim, ClothPrim
    from pxr import UsdPhysics, PhysxSchema, UsdGeom
    from calibrated_scene_reconstruction.sim.physics_errors import PhysicsErrors
    from calibrated_scene_reconstruction.sim.simulation import audit
    from calibrated_scene_reconstruction.tasks.rigid.task_checks import as_numpy
    from calibrated_scene_reconstruction.scene.objects.mechanisms import fingerprint
    initial_fingerprint=fingerprint(s)
    if not omni.usd.get_context().open_stage(str(asset)):
        raise ValueError('Standalone package did not open')
    stage=omni.usd.get_context().get_stage()
    authored=audit(stage,profile)
    scenes=[p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    if len(scenes)!=1:raise ValueError('Exactly one authored native physics scene required')
    world=World(stage_units_in_meters=1.,physics_dt=1/profile['physics_hz'],rendering_dt=1/30,
                physics_prim_path=str(scenes[0].GetPath()),set_defaults=False,backend='torch',device='cuda:0')
    errors=PhysicsErrors(out/'native_physics_errors.json');errors.guard_steps(world)
    world.reset()
    if getattr(world,'_r2s2r_initialization_hz',None) is not None:
        raise ValueError('Standalone test must not depend on the Python reset wrapper')
    if not np.isclose(world.get_physics_dt(),1/profile['physics_hz'],rtol=1e-8):
        raise ValueError('Standalone runtime clock differs from the delivered profile')
    readers={};expected={};owners=[];handles=[]
    for prim in stage.Traverse():
        path=str(prim.GetPath())
        if not path.startswith('/World/Objects/'):continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            view=RigidPrim([path],name=path.replace('/','_')+'_startup',reset_xform_properties=False,disable_stablization=False);view.initialize()
            if not view.is_physics_handle_valid() or view.count!=1:raise ValueError('No actual native rigid handle: '+path)
            handles.append(view)
            readers[path]=lambda v=view: np.r_[as_numpy(v.get_world_poses()[0]).ravel(),as_numpy(v.get_world_poses()[1]).ravel(),as_numpy(v.get_velocities()).ravel()]
        elif prim.HasAPI(PhysxSchema.PhysxParticleClothAPI):
            view=ClothPrim(path,name=path.replace('/','_')+'_startup',reset_xform_properties=False);view.initialize()
            if not view.is_physics_handle_valid() or view.count!=1:raise ValueError('No actual native cloth handle: '+path)
            handles.append(view)
            readers[path]=lambda v=view: np.c_[as_numpy(v.get_world_positions())[0],as_numpy(v.get_velocities())[0]]
            expected[path]=len(UsdGeom.Mesh(prim).GetPointsAttr().Get())
        elif prim.HasAPI('OmniPhysicsSurfaceDeformableSimAPI'):
            owner=tensors.create_simulation_view('torch');owner.set_subspace_roots('/')
            view=owner.create_surface_deformable_body_view(path);owners.append(owner)
            if view is None or view.count!=1:raise ValueError('Delivered native surface body is unavailable')
            readers[path]=lambda v=view: np.c_[as_numpy(v.get_simulation_nodal_positions())[0],as_numpy(v.get_simulation_nodal_velocities())[0]]
        elif prim.HasAPI(PhysxSchema.PhysxDeformableBodyAPI):
            owner=tensors.create_simulation_view('torch');owner.set_subspace_roots('/')
            view=owner.create_soft_body_view(path);owners.append(owner)
            if view is None or view.count!=1:raise ValueError('Delivered native volume body is unavailable')
            readers[path]=lambda v=view: np.c_[as_numpy(v.get_sim_nodal_positions())[0],as_numpy(v.get_sim_nodal_velocities())[0]]
    if not readers:raise ValueError('No actual native task bodies in the delivered asset')
    dt=world.get_physics_dt();steps=round(2./dt);stride=max(1,int(.05/dt));rows=[];last={}
    started_time=world.current_time
    for step in range(steps+1):
        if step%stride==0 or step==steps:
            if not world.is_playing() or any(not v.is_physics_handle_valid() for v in handles):
                raise ValueError('Native play/handle lost; USD fallback readback cannot validate physics')
            last={path:read() for path,read in readers.items()}
            for path,value in last.items():
                if not np.isfinite(value).all() or not value.size:
                    raise ValueError('Nonfinite/empty standalone native state: '+path)
                if path in expected and value.shape!=(expected[path],6):
                    raise ValueError('Standalone particle count differs from the delivered mesh: '+path)
            rows.append({'time_s':step*dt,'finite':True})
        if step<steps:world.step(render=False)
    final=audit(stage,profile)
    errors.check();native=errors.report();precision=errors.precision_audit()
    unchanged=all(sha(p)==h for p,h in inputs.items()) and initial_fingerprint==fingerprint(s)
    advanced=world.current_time-started_time
    report.update(pass_=bool(native['pass'] and precision['pass'] and unchanged and final==authored
                            and abs(advanced-steps*dt)<1e-5),
                  simulator='Isaac Sim '+importlib.metadata.version('isaacsim'),
                  seconds=steps*dt,dt_s=dt,samples=rows,simulation_profile=final,
                  body_paths=list(readers),particle_counts=expected,native_physics=native,
                  native_handles_verified=True,runtime_advanced_s=advanced,
                  precision=precision,inputs_unchanged=unchanged,source_input_fingerprint=fingerprint(s))
    report['pass']=report.pop('pass_')
except BaseException as error:
    report.update(error=type(error).__name__+': '+str(error))
    (out/'failure.txt').write_text(traceback.format_exc());traceback.print_exc()
finally:
    report['input_sha256']=inputs
    raw={}
    for i,(path,value) in enumerate(last.items()):
        file=out/f'final_state_{i}.npy';np.save(file,value)
        raw[path]={'file':str(file.relative_to(s.dir)),'sha256':sha(file)}
    report['raw_states']=raw
    report['producer']={'path':str(Path(__file__).resolve().relative_to(Path(__file__).resolve().parents[2])),
                        'sha256':inputs[str(Path(__file__).resolve())]}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    target=s/'qa'/('package_physics_'+asset.stem+'.json')
    target.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('[package.physics] '+json.dumps({'pass':report['pass'],'error':report.get('error')}),flush=True)
    app.close()
