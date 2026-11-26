"""Diagnostic bounded-force mixed-assembly motion, without task approval.

Use a source-owned companion/setup and force proposal. This avoids image
comparison during numerical diagnosis; final source, task, semantic and package
checks still run through the normal pipeline. No robot actuation.
"""
import argparse
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('session');p.add_argument('--setup',required=True);p.add_argument('--probe',required=True)
p.add_argument('--configuration',help='Explicit configuration snapshot for a separate asset version')
p.add_argument('--physics-case',choices=['nominal','low','high'],default='nominal')
p.add_argument('--reference',required=True);p.add_argument('--output',required=True)
p.add_argument('--seconds',type=float,default=10.);p.add_argument('--settle',type=float,default=8.)
p.add_argument('--maximum-penetration-m',type=float,help='Optional fail-fast rigid contact limit; never shortens a passing horizon')
a=p.parse_args()
if not 1<=a.seconds<=30 or not 0<=a.settle<=15:p.error('Use bounded diagnostic durations')
from calibrated_scene_reconstruction.scene.session import Session
s=Session(a.session,a.configuration);out=(s.dir/a.output).resolve();setup=(s.dir/a.setup).resolve();probe=(s.dir/a.probe).resolve()
if not all(x.is_relative_to((s.dir/'qa').resolve()) for x in [out,setup,probe]):
    raise ValueError('Diagnostic files must belong to this session QA')
out.mkdir(parents=True,exist_ok=False)
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
source_bytes=Path(__file__).resolve().read_bytes()
(out/'source.py').write_bytes(source_bytes)
input_hashes={str(path):sha(path) for path in [setup,probe,Path(__file__).resolve()]}
proposal=json.loads(setup.read_text());plan=json.loads(probe.read_text())
obj=proposal['object'];moving=obj+'/'+proposal['link'];fix=s.scene_graph['dynamic_objects'][obj]['fix']
if sha(fix['mesh'])!=proposal['mesh_sha256']:raise ValueError('Setup belongs to another candidate')
plan['link']=moving

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402
app=SimulationApp({'headless':True})
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from calibrated_scene_reconstruction.assets.metric_mesh import asset  # noqa: E402
from calibrated_scene_reconstruction.assets.object_pose import rotation  # noqa: E402
from calibrated_scene_reconstruction.scene.twin.runtime_inputs import prepare_scene,prepare_background  # noqa: E402
from calibrated_scene_reconstruction.scene.twin.variants import photo_goal,prepare_variant  # noqa: E402
from calibrated_scene_reconstruction.scene.objects.mechanisms import fingerprint, declaration, sampling_tolerances  # noqa: E402
from calibrated_scene_reconstruction.sim.mechanism_stage import audit as audit_mechanisms  # noqa: E402
from calibrated_scene_reconstruction.sim.scene_stage import build_scene  # noqa: E402
from calibrated_scene_reconstruction.sim.splat_stage import add_background  # noqa: E402
from calibrated_scene_reconstruction.sim.physics_errors import PhysicsErrors  # noqa: E402
from calibrated_scene_reconstruction.sim.simulation import audit  # noqa: E402
from calibrated_scene_reconstruction.tasks.rigid.task_checks import Ctx  # noqa: E402
from calibrated_scene_reconstruction.tasks.rigid.native_assay import run  # noqa: E402

result={'task_approved':False,'diagnostic_only':True,
        'physics_case':a.physics_case,
        'scope':'Numerical motion diagnosis only; no source, semantic, task, material or delivery approval.'}
try:
    initial=fingerprint(s);errors=PhysicsErrors(out/'native_physics_errors.json')
    world=World(stage_units_in_meters=1.,physics_dt=1/120,rendering_dt=1/30,device='cuda:0',backend='torch')
    errors.guard_steps(world);world.get_physics_context().enable_gpu_dynamics(True)
    world.get_physics_context().set_broadphase_type('GPU')
    stage=omni.usd.get_context().get_stage();runtime=prepare_scene(s);runtime.physics_case=a.physics_case
    ts=build_scene(stage,runtime,photo_goal(s.dir,prepare_variant(s.dir,{})))
    contract=declaration(s)
    result['mechanism_stage']=audit_mechanisms(stage,contract,sampling_tolerances=sampling_tolerances(s)) if contract else None
    add_background(stage,prepare_background(s.dir),'zone')
    profile=audit(stage,runtime.simulation)
    result['initialization_profile']={'task_dt_s':world.get_physics_dt(),
        'initialization_hz':getattr(world,'_r2s2r_initialization_hz',None)}
    stage.GetRootLayer().Export(str(out/'authored.usda'))
    world.reset()
    result['initialization_profile']['actual_restored_dt_s']=world.get_physics_dt()
    ctx=Ctx(s,ts,world,stage,lambda *args:None);ctx.native_errors=errors
    names=[name+'/'+part['name'] for name,spec in s.scene_graph['dynamic_objects'].items()
           for part in asset(spec['fix'])['parts'] if part['kind']=='rigid']
    dt=world.get_physics_dt()
    for step in range(round(a.settle/dt)):
        world.step(render=False)
        if step%max(1,round(1/dt))==0:print(f'[assembly.diagnostic] settling {step*dt:.1f}/{a.settle:.1f}s',flush=True)
    snapshot=ctx.capture_assay_state()
    reference_obj=a.reference.split('/')[0];reference=s.scene_graph['dynamic_objects'][reference_obj]['fix']
    bp=np.r_[reference['xy'],reference['support_z']];bR=rotation(reference)
    ctx.place_assay_assembly(obj,bp+bR@np.asarray(proposal['translation_m']),bR@np.asarray(proposal['rotation']),snapshot,
                             anchor=proposal['link'])
    native,poses,times=run(ctx,names,a.seconds,out/'motion',probe_plan=plan,maximum_penetration_m=a.maximum_penetration_m)
    result.update(complete=True,finite=bool(np.isfinite(poses).all()),native=native,
                  simulation_profile=profile,input_fingerprint=initial,inputs_unchanged=initial==fingerprint(s))
except BaseException as error:
    interrupted=isinstance(error,KeyboardInterrupt)
    result.update(complete=False,finite=None if interrupted else False,
                  interrupted=interrupted,error=type(error).__name__+': '+str(error))
    state=getattr(error,'particle_state',None)
    if state:
        np.savez_compressed(out/'failed_particle_registration.npz',**state)
finally:
    result['input_sha256']=input_hashes
    result['diagnostic_source_unchanged']=all(Path(path).is_file() and sha(path)==digest
                                             for path,digest in input_hashes.items())
    (out/'diagnostic.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('[assembly.diagnostic] '+json.dumps({k:result.get(k)for k in ['complete','finite','error','task_approved']}),flush=True)
    app.close()
