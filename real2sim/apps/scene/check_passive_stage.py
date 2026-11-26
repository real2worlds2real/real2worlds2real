"""Read-only unforced rigid trace from an authored native stage.

Optional exclusions are explicit causal controls, never an asset/task approval.
The original stage is opened with edits confined to an anonymous session layer.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('stage');p.add_argument('--output',required=True)
p.add_argument('--rigid',nargs='+',required=True)
p.add_argument('--exclude',nargs='*',default=[])
p.add_argument('--physics-hz',type=int,required=True)
p.add_argument('--seconds',type=float,default=.5)
a=p.parse_args()
if a.physics_hz not in [120,240,480,960,1920,3840,7680] or not .1<=a.seconds<=10:
    p.error('Bounded native duration and supported declared timestep required')
source=Path(a.stage).resolve();out=Path(a.output).resolve()
if not source.is_file() or source.suffix not in {'.usd','.usda','.usdc'}:
    raise ValueError('Existing authored native USD stage required')
if any(not name.startswith('/World/Objects/') or '..' in name for name in a.exclude):
    raise ValueError('Exclusions may only name explicit task-object prims')
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
before=sha(source);out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,out/'source.py')

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402
app=SimulationApp({'headless':True})
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from pxr import UsdPhysics, PhysxSchema  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleRigidPrim  # noqa: E402
from calibrated_scene_reconstruction.tasks.rigid.task_checks import as_numpy,rot  # noqa: E402
from calibrated_scene_reconstruction.sim.physics_errors import PhysicsErrors  # noqa: E402

result={'task_approved':False,'diagnostic_only':True,'positions_prescribed':False,
        'external_forces':False,'source_stage':str(source),'source_sha256':before,
        'excluded_prims':a.exclude,'rigid_paths':a.rigid,'physics_hz':a.physics_hz,
        'scope':'Native causal/passive diagnostic only; no source, material or scene approval.'}
errors=PhysicsErrors(out/'native_physics_errors.json')
poses=[];times=[]
try:
    omni.usd.get_context().open_stage(str(source));stage=omni.usd.get_context().get_stage()
    stage.SetEditTarget(stage.GetSessionLayer())
    scenes=[q for q in stage.Traverse() if q.IsA(UsdPhysics.Scene)]
    if len(scenes)!=1:raise ValueError('Exactly one source physics scene required')
    api=PhysxSchema.PhysxSceneAPI(scenes[0])
    profile={'hz':api.GetTimeStepsPerSecondAttr().Get(),'solver':api.GetSolverTypeAttr().Get()}
    authored_scene={str(attr.GetName()):attr.Get() for attr in scenes[0].GetAttributes()
                    if attr.HasAuthoredValueOpinion()}
    if profile['hz']!=a.physics_hz:raise ValueError('Diagnostic timestep must equal the authored stage')
    for name in a.exclude:
        if not stage.GetPrimAtPath(name) or any(r==name or r.startswith(name+'/') for r in a.rigid):
            raise ValueError('Cannot remove absent or measured rigid prim: '+name)
        stage.GetPrimAtPath(name).SetActive(False)
    world=World(stage_units_in_meters=1.,physics_dt=1/a.physics_hz,rendering_dt=1/30,
                backend='torch',device='cuda:0')
    errors.guard_steps(world)
    # World initializes its defaults even when a source physics scene exists.
    # Restore the original authored scene in the anonymous session layer before
    # native reset; otherwise a nominal PGS replay silently becomes TGS.
    restored=[]
    for name,value in authored_scene.items():
        attr=scenes[0].GetAttribute(name)
        if attr.Get()!=value:
            attr.Set(value);restored.append(name)
    result['source_scene_attributes_restored_after_world_init']=restored
    if api.GetSolverTypeAttr().Get()!=profile['solver']:
        raise ValueError('World initialization changed the source solver')
    bodies={}
    for i,name in enumerate(a.rigid):
        prim=stage.GetPrimAtPath(name)
        if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError('Measured native rigid body is missing: '+name)
        bodies[name]=SingleRigidPrim(name,name='passive_body_'+str(i))
    stage.GetSessionLayer().Export(str(out/'diagnostic_session.usda'))
    world.reset()
    for body in bodies.values():body.initialize()
    result['native_masses_kg']={name:float(np.asarray(as_numpy(body.get_mass()))) for name,body in bodies.items()}
    def record(t):
        row=[]
        for body in bodies.values():
            position,quaternion=map(as_numpy,body.get_world_pose())
            row.append(np.r_[position,rot(quaternion).ravel()])
        poses.append(row);times.append(t)
    record(0.)
    steps=round(a.seconds*a.physics_hz);stride=max(1,a.physics_hz//20)
    for step in range(steps):
        world.step(render=False)
        if (step+1)%stride==0 or step+1==steps:record((step+1)/a.physics_hz)
    errors.check();result['complete']=True
except Exception as error:
    result.update(complete=False,error=type(error).__name__+': '+str(error))
finally:
    np.savez_compressed(out/'states.npz',poses=np.asarray(poses),times=np.asarray(times),rigid_paths=a.rigid)
    result['native_physics_errors']=errors.report();result['source_unchanged']=sha(source)==before
    result['raw_trace_sha256']=sha(out/'states.npz');result['producer_sha256']=sha(__file__)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASSIVE_STAGE',json.dumps({k:result.get(k) for k in ['complete','error','source_unchanged']}),flush=True)
    errors.close();app.close()
