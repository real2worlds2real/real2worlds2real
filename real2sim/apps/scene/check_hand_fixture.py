"""Native hand-motor preflight, with no task objects or task approval."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import traceback

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
p.add_argument('--physics-hz',type=int,default=960);p.add_argument('--finger-effort',type=float,default=.05)
p.add_argument('--phase-seconds',type=float,default=1.,help='Transition and settling interval per free-motion phase')
p.add_argument('--compensate-joint-friction',action='store_true');args=p.parse_args()
if not 240<=args.physics_hz<=7680 or not .5<=args.phase_seconds<=2.:p.error('Bounded native hand time step and settling interval required')
out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
shutil.copy2(__file__,out/'source.py');shutil.copy2(args.plan,out/'hand_input.json')

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402
app=SimulationApp({'headless':True})
import importlib.metadata  # noqa: E402
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from calibrated_scene_reconstruction.scene.capture.articulated_model import from_usd  # noqa: E402
from calibrated_scene_reconstruction.sim.hand_fixture import HandFixture  # noqa: E402
from calibrated_scene_reconstruction.sim.physics_errors import PhysicsErrors  # noqa: E402
from calibrated_scene_reconstruction.tasks.rigid.task_checks import as_numpy  # noqa: E402
from calibrated_scene_reconstruction.tasks.mechanisms.pinch_plan import pad_geometry  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

errors=PhysicsErrors(out/'native_physics_errors.json')
try:
    plan=json.loads((out/'hand_input.json').read_text());model=from_usd(plan['asset'],100)
    opened={j.name:float(plan['open_state'].get(j.name,0.)) for j in model.movable}
    closed={j.name:float(plan['closed_state'].get(j.name,0.)) for j in model.movable}
    world=World(stage_units_in_meters=1.,physics_dt=1/args.physics_hz,device='cuda:0',backend='torch');errors.guard_steps(world)
    pc=world.get_physics_context();pc.enable_gpu_dynamics(True);pc.set_broadphase_type('GPU');pc.set_solver_type('PGS')
    stage=omni.usd.get_context().get_stage();position=np.asarray(plan['p'])+[0,0,.3]
    hand=HandFixture(stage,plan['asset'],model,opened,position,plan['R'],finger_effort_Nm=args.finger_effort,
        joint_friction_compensation=args.compensate_joint_friction,
        joint_friction_prior=None if args.compensate_joint_friction else plan.get('joint_friction_prior'))
    stage.GetRootLayer().Export(str(out/'initial.usda'));world.reset();hand.initialize()
    paths=[Path(__file__),Path('src/calibrated_scene_reconstruction/sim/hand_fixture.py'),Path('src/calibrated_scene_reconstruction/tasks/mechanisms/pinch_plan.py'),Path(args.plan),*map(Path,model.dependencies)]
    hashes={str(q.resolve()):hashlib.sha256(q.read_bytes()).hexdigest() for q in paths}
    rows=[];states=[];previous=opened;steps=round(args.phase_seconds*args.physics_hz)
    for phase,goal in [('open',opened),('close',closed),('reopen',opened),('wrist_lift',opened)]:
        for step in range(steps):
            t=min(1.,(step+1)/(steps*.6));t=t*t*(3-2*t)
            state={name:previous[name]+t*(goal[name]-previous[name]) for name in goal}
            wrist=position+np.array([0,0,.04*t if phase=='wrist_lift' else 0.])
            hand.control(wrist,plan['R'],state);world.step(render=False)
            if step%(args.physics_hz//60)==0 or step==steps-1:
                q=as_numpy(hand.art.get_joint_positions());actual=dict(zip(hand.names,q.tolist()))
                wrist_q=as_numpy(hand.palm.get_world_pose()[1])
                wrist_rotation=Rotation.from_quat(wrist_q[[1,2,3,0]]).as_matrix()
                rows.append({'phase':phase,'time_s':(len(states)/60),
                    'maximum_joint_error_rad':max(abs(actual[n]-state[n]) for n in hand.names),
                    'actual':actual,'requested':state,
                    'native_target':as_numpy(hand.art._articulation_view._physics_view.get_dof_position_targets())[0].tolist(),
                    'measured_efforts':as_numpy(hand.art.get_measured_joint_efforts()).tolist(),
                    'wrist_position_error_m':float(np.linalg.norm(as_numpy(hand.palm.get_world_pose()[0])-wrist)),
                    'wrist_orientation_error_rad':float(Rotation.from_matrix(np.asarray(plan['R'])@wrist_rotation.T).magnitude()),
                    'wrist_angular_velocity_rad_s':as_numpy(hand.palm.get_angular_velocity()).tolist()})
                states.append(as_numpy(hand.art._articulation_view._physics_view.get_link_transforms())[0].copy())
        previous=goal
    errors.check();finals={phase:next(row for row in rows[::-1] if row['phase']==phase) for phase in ['open','close','reopen','wrist_lift']}
    gains=[as_numpy(value).tolist() for value in hand.art.get_articulation_controller().get_gains()]
    passed=all(row['maximum_joint_error_rad']<.05 and row['wrist_position_error_m']<.003
               and row['wrist_orientation_error_rad']<.05 and np.linalg.norm(row['wrist_angular_velocity_rad_s'])<.5
               for row in finals.values())
    result={'pass':passed,'not_task_approval':True,'backend':{'name':'isaacsim','version':importlib.metadata.version('isaacsim')},
        'rows':rows,'final_phases':finals,'hand':hand.report(),'native_physics_errors':errors.report(),
        'input_sha256':hashes,'physics_hz':args.physics_hz,'body_names':list(hand.art._articulation_view.body_names),
        'native_gains':gains,'native_max_efforts':as_numpy(hand.art.get_articulation_controller().get_max_efforts()).tolist()}
    result['dof_names']=hand.names
    armatures=as_numpy(hand.art._articulation_view.get_armatures())
    velocities=as_numpy(hand.art._articulation_view._physics_view.get_dof_max_velocities())
    friction=as_numpy(hand.art._articulation_view._physics_view.get_dof_friction_properties())
    result['native_armatures']=armatures.tolist()
    result['native_max_joint_velocities_rad_s']=velocities.tolist()
    result['native_joint_friction_properties']=friction.tolist()
    source=hand.description['source_joint_axis_properties']
    checks={'armature_preserved':bool(np.allclose(armatures[0],[source[n]['armature'] for n in hand.names],rtol=1e-5,atol=1e-10)),
            'velocity_limits_preserved':bool(np.allclose(velocities[0],np.radians([source[n]['maxJointVelocity'] for n in hand.names]),rtol=1e-5,atol=1e-6))}
    checks['source_capped_efforts']=bool(np.allclose(np.asarray(result['native_max_efforts']).ravel(),
        [hand.description['joint_effort_limits_Nm'][n] for n in hand.names],rtol=1e-5,atol=1e-8))
    f=plan.get('joint_friction_prior')
    if f and not args.compensate_joint_friction:
        checks['SI_friction_readback']=bool(np.allclose(friction[0],[[f['static_Nm'],f['dynamic_Nm'],f['viscous_Nms_rad']]]*len(hand.names),rtol=1e-5,atol=1e-8))
    result['native_property_checks']=checks
    result['pass'] &= all(checks.values())
    result['native_joint_friction']=as_numpy(hand.art._articulation_view.get_friction_coefficients()).tolist()
    contacts=[{'link':plan[key]['link'],'point':plan[key]['pad_local'],'normal':plan[key]['normal_local']}
              for key in ['thumb','index']]
    pad_report={}
    for phase in finals:
        index=next(i for i in range(len(rows)-1,-1,-1) if rows[i]['phase']==phase)
        transforms={}
        for name,pose in zip(result['body_names'],states[index]):
            tf=np.eye(4);tf[:3,:3]=Rotation.from_quat(pose[3:]).as_matrix();tf[:3,3]=pose[:3]
            transforms[name]=tf
        pad_report[phase]=pad_geometry(transforms,contacts)
    result['actual_pad_geometry']=pad_report
    result['planned_closed_pad_geometry']=pad_geometry(model.transforms(closed),contacts)
    np.savez_compressed(out/'states.npz',hand_links=states)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('HAND_FIXTURE_RESULT',json.dumps({'pass':result['pass'],'native_property_checks':checks,'maximum_final_errors_rad':{k:v['maximum_joint_error_rad'] for k,v in finals.items()}}),flush=True)
except Exception as error:
    (out/'error.json').write_text(json.dumps({'pass':False,'error':str(error),'traceback':traceback.format_exc()},indent=2)+'\n');raise
finally:
    errors.close();app.close()
