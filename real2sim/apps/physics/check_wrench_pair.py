"""Native Isaac capability assay: a force couple must not alter net translation."""
import argparse
from pathlib import Path
from importlib.metadata import version
ap=argparse.ArgumentParser(__doc__);ap.add_argument('--output',required=True);args=ap.parse_args()
from calibrated_scene_reconstruction.core.paths import ROOT  # noqa: E402
from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402
app=SimulationApp({'headless':True,'width':64,'height':64,'disable_viewport_updates':True,'multi_gpu':False,'limit_cpu_threads':8},experience=str(ROOT/'apps/physics/physics_only.kit'))
try:
 import numpy as np
 import omni.usd,omni.physx,carb
 from pxr import UsdGeom,UsdPhysics,PhysxSchema,Gf,Sdf,PhysicsSchemaTools
 from scipy.spatial.transform import Rotation
 from calibrated_scene_reconstruction.physics.process.physics_only import World,SingleRigidPrim
 from calibrated_scene_reconstruction.physics.process.wrench import circular_grip
 from calibrated_scene_reconstruction.physics.process.contract import write,digest,require
 out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
 world=World(physics_dt=1/240);stage=world.stage
 UsdPhysics.Scene(stage.GetPrimAtPath('/physicsScene')).GetGravityMagnitudeAttr().Set(0.)
 actors={};starts={};dt=world.get_physics_dt();steps=60;force=np.array([.1,0,0]);inertia=.0006;grip=.1
 for key,y in [('base',0.),('couple',1.)]:
  path='/World/'+key;prim=UsdGeom.Cube.Define(stage,path);prim.CreateSizeAttr(.06);prim.AddTranslateOp().Set(Gf.Vec3d(0,y,0));body=UsdPhysics.RigidBodyAPI.Apply(prim.GetPrim());UsdPhysics.CollisionAPI.Apply(prim.GetPrim());mass=UsdPhysics.MassAPI.Apply(prim.GetPrim());mass.CreateMassAttr(1.);mass.CreateDiagonalInertiaAttr(Gf.Vec3f(inertia));physical=PhysxSchema.PhysxRigidBodyAPI.Apply(prim.GetPrim());physical.CreateLinearDampingAttr(0.);physical.CreateAngularDampingAttr(0.);actors[key]=SingleRigidPrim(path)
 world.reset();world.step()
 for key,body in actors.items():starts[key]=body.get_world_pose()[0]
 api=omni.physx.get_physx_simulation_interface();stage_id=omni.usd.get_context().get_stage_id()
 def apply(path,F,P):api.apply_force_at_pos(stage_id,PhysicsSchemaTools.sdfPathToInt(Sdf.Path(path)),carb.Float3(*F.tolist()),carb.Float3(*P.tolist()),'Force')
 for i in range(steps):
  for key,body in actors.items():
   p,q=body.get_world_pose();R=Rotation.from_quat(q[[1,2,3,0]]).as_matrix();apply(body.path,force,p)
   if key=='couple':
    points,forces=circular_grip(p,R,[0,0,0],[1,0,0],[.03,.03],[-.02,.02],grip)
    for P,F in zip(points,forces):apply(body.path,F,P)
  world.step()
 rows={}
 for key,body in actors.items():
  p,q=body.get_world_pose();rot=Rotation.from_quat(q[[1,2,3,0]]).as_rotvec();rows[key]={'displacement_m':(p-starts[key]).tolist(),'rotation_vector_rad':rot.tolist()}
 expected=.1*dt**2*steps*(steps+1)/2;expected_angle=grip*.04/inertia*dt**2*steps*(steps+1)/2
 translation=np.array(rows['couple']['displacement_m'])-rows['base']['displacement_m'];angle=rows['couple']['rotation_vector_rad'][1]
 good=bool(np.linalg.norm(translation)<1e-6 and abs(rows['base']['displacement_m'][0]-expected)<1e-4 and abs(angle-expected_angle)<.02)
 write(out/'result.json',{'pass':good,'engine':{'name':'isaacsim','version':version('isaacsim')},'rows':rows,'paired_minus_base_translation_m':translation.tolist(),'expected_x_m':expected,'expected_couple_angle_rad':expected_angle,'scope':'Native actuator conservation/capability assay with procedural free rigid bodies. Not source-video calibration, scene accuracy, human-force measurement or task acceptance.','source_sha256':{str(Path(p).resolve()):digest(p) for p in [__file__,ROOT/'src/calibrated_scene_reconstruction/physics/process/wrench.py',ROOT/'src/calibrated_scene_reconstruction/physics/process/physics_only.py']}})
 print(rows,'pass',good,flush=True);world.clear();require(good,'Native force pair failed conservation or angular response assay')
finally:app.close()
