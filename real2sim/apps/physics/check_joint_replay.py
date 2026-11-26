"""Validate prepared measured joint positions against native Isaac kinematics.

One initial joint reset, then articulation drives only. No target-object motion,
recorded force, target pose fitting or physical-parameter calibration is involved.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('prepared')
parser.add_argument('output')
parser.add_argument('--robot',default='vega_1p_sharpa')
parser.add_argument('--physics-hz',type=int,default=480)
parser.add_argument('--clock-rate',type=float,default=1.)
args=parser.parse_args()
if not 120<=args.physics_hz<=960 or not .5<=args.clock_rate<=2:
    parser.error('Explicit bounded native rate and trajectory clock rate required')
output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)

from calibrated_scene_reconstruction.physics.process.trajectory_inputs import load_prepared,sample_positions  # noqa: E402
from calibrated_scene_reconstruction.physics.process.contract import digest  # noqa: E402
from calibrated_scene_reconstruction.robots.spec import load_spec  # noqa: E402
from calibrated_scene_reconstruction.core.paths import ROOT  # noqa: E402
from calibrated_scene_reconstruction.scene.capture.articulated_model import from_usd  # noqa: E402

values,receipt=load_prepared(args.prepared)
spec=copy.deepcopy(load_spec(args.robot))
names=list(map(str,values['joint_names']))
spec['home']=dict(zip(names,map(float,values['q'][0])))
inputs={str(Path(args.prepared).resolve()/'manifest.json'):digest(Path(args.prepared)/'manifest.json'),
        str(Path(__file__).resolve()):digest(__file__),str(Path(spec['usd']).resolve()):digest(spec['usd'])}
(output/'input.json').write_text(json.dumps({'inputs':inputs,'spec':spec,'physics_hz':args.physics_hz,
    'clock_rate':args.clock_rate,'clock_accepted':False,'scope':__doc__},indent=2)+'\n')

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402
app=SimulationApp({'headless':True,'width':64,'height':64,'disable_viewport_updates':True,
                   'multi_gpu':False,'limit_cpu_threads':8})
try:
    cad=from_usd(spec['usd']);expected={j.name for j in cad.movable}
    if set(names)!=expected:raise ValueError('Every native CAD degree of freedom needs exactly one measured input')
    import omni.usd
    from isaacsim.core.api import World
    from scipy.spatial.transform import Rotation
    from calibrated_scene_reconstruction.sim.robot_articulation import Robot,add_robot
    from calibrated_scene_reconstruction.sim.physics_errors import PhysicsErrors

    errors=PhysicsErrors(output/'physics_errors.json')
    world=World(stage_units_in_meters=1.,physics_dt=1/args.physics_hz,rendering_dt=1/30)
    stage=omni.usd.get_context().get_stage();add_robot(stage,spec,(0.,0.),0.)
    world.reset();robot=Robot(world,spec)
    if set(robot.names)!=set(names):raise ValueError('Native degree-of-freedom mapping differs from CAD')
    index=[names.index(n)for n in robot.names];q=values['q'][:,index]
    if np.any(q<robot.lower[None]-1e-5) or np.any(q>robot.upper[None]+1e-5):
        raise ValueError('Measured joint exceeds native limits; no clamping is allowed')
    robot.teleport(q[0])
    for _ in range(5):robot.command(q[0],np.zeros(len(index)));world.step(render=False)
    dt=1/args.physics_hz;duration=float(np.ptp(values['t'])/args.clock_rate)
    times=np.arange(int(np.floor(duration/dt))+1)*dt
    clocks=values['t'][0]+times*args.clock_rate
    targets=sample_positions(values,clocks)[:,index]
    velocities=np.gradient(targets,dt,axis=0)
    actual=[];poses=[];position=[];angle=[]
    for k,(time,target,velocity) in enumerate(zip(times,targets,velocities)):
        if k:
            robot.command(target,velocity);world.step(render=False)
        measured=robot.q();tf=robot.link_transforms();actual.append(measured);poses.append(tf)
        predicted=cad.transforms(dict(zip(robot.names,measured)))
        root=tf[robot.body_names.index(cad.root)];R0=Rotation.from_quat(root[3:]).as_matrix()
        pe=[];ae=[]
        for name,t in zip(robot.body_names,tf):
            if name not in predicted:raise ValueError('Native link is missing from CAD kinematic tree')
            p=predicted[name];pe.append(np.linalg.norm(t[:3]-(R0@p[:3,3]+root[:3])))
            ae.append(Rotation.from_matrix(Rotation.from_quat(t[3:]).as_matrix()@(R0@p[:3,:3]).T).magnitude())
        position.append(pe);angle.append(ae)
        if k%240==0:print('REPLAY',k,len(times),'q_error',float(np.max(np.abs(measured-target))),flush=True)
    actual=np.asarray(actual);position=np.asarray(position);angle=np.asarray(angle)
    np.savez_compressed(output/'trace.npz',time_s=times,trajectory_time_s=clocks,joint_names=robot.names,
        targets=targets,actual=actual,body_names=robot.body_names,body_poses=poses,
        cad_position_error_m=position,cad_rotation_error_rad=angle)
    report={'task_approved':False,'calibration_approved':False,'clock_accepted':False,
        'maximum_tracking_error_rad':float(np.max(np.abs(actual-targets))),
        'p95_tracking_error_rad':float(np.quantile(np.abs(actual-targets),.95)),
        'maximum_cad_position_error_m':float(position.max()),'maximum_cad_rotation_error_rad':float(angle.max()),
        'per_joint_maximum_error_rad':dict(zip(robot.names,np.max(np.abs(actual-targets),axis=0).tolist())),
        'scope':'Native free-space joint-axis and drive tracking diagnostic only. Clock, source posture, contacts and response calibration remain independent.'}
    report['kinematic_mapping_pass']=bool(position.max()<.002 and angle.max()<.01)
    report['drive_tracking_pass']=bool(report['maximum_tracking_error_rad']<.01)
    report['inputs_unchanged']=all(digest(path)==sha for path,sha in inputs.items())
    report['native_physics']=errors.report()
    (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    errors.close()
    print('FINAL',{k:v for k,v in report.items()if k!='per_joint_maximum_error_rad'},flush=True)
except BaseException as exc:
    import traceback
    failure={'pass':False,'error':str(exc),'type':type(exc).__name__,'traceback':traceback.format_exc()}
    (output/'failure.json').write_text(json.dumps(failure,indent=2)+'\n')
    print(failure['traceback'],flush=True)
    raise
finally:
    app.close()
