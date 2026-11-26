"""Native capability assay: stationary initialization must permit force wake-up."""
import argparse
from pathlib import Path
from importlib.metadata import version
ap=argparse.ArgumentParser(__doc__);ap.add_argument('--output',required=True);ap.add_argument('--particles',action='store_true');ap.add_argument('--mass-scale',type=float,default=1.);args=ap.parse_args()
from calibrated_scene_reconstruction.core.paths import ROOT  # noqa: E402
from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402
app=SimulationApp({'headless':True,'width':64,'height':64,'disable_viewport_updates':True,'multi_gpu':False,'limit_cpu_threads':8},experience=str(ROOT/'apps/physics/physics_only.kit'))
try:
 import numpy as np
 import omni.usd,omni.physx,carb
 from pxr import UsdGeom,UsdPhysics,PhysxSchema,Gf,Sdf,PhysicsSchemaTools
 from calibrated_scene_reconstruction.physics.process.physics_only import World,SingleRigidPrim
 from calibrated_scene_reconstruction.physics.process.initial_rest import quiesce
 from calibrated_scene_reconstruction.physics.process.contract import write,digest,require
 out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False);world=World(physics_dt=1/240);stage=world.stage
 UsdPhysics.Scene(stage.GetPrimAtPath('/physicsScene')).GetGravityMagnitudeAttr().Set(0.)
 prim=UsdGeom.Cube.Define(stage,'/World/Body');prim.CreateSizeAttr(.06);body=UsdPhysics.RigidBodyAPI.Apply(prim.GetPrim());body.CreateVelocityAttr(Gf.Vec3f(.1,0,0));body.CreateAngularVelocityAttr(Gf.Vec3f(0,30,0));UsdPhysics.CollisionAPI.Apply(prim.GetPrim());mass=UsdPhysics.MassAPI.Apply(prim.GetPrim());mass.CreateMassAttr(1.);mass.CreateDiagonalInertiaAttr(Gf.Vec3f(.0006));physical=PhysxSchema.PhysxRigidBodyAPI.Apply(prim.GetPrim());physical.CreateLinearDampingAttr(0.);physical.CreateAngularDampingAttr(0.)
 actors={'body':{'path':'/World/Body','kind':'rigid'}};sheet=None
 if args.particles:
  from omni.physx.scripts import particleUtils
  from pxr import Vt
  world.context.enable_gpu_dynamics(True);world.context.set_broadphase_type('GPU');world.context.set_solver_type('TGS')
  sheet=UsdGeom.Mesh.Define(stage,'/World/Sheet');sheet.CreatePointsAttr([(-.02,-.02,.2),(.02,-.02,.2),(-.02,.02,.2),(.02,.02,.2)]);sheet.CreateFaceVertexCountsAttr([3,3]);sheet.CreateFaceVertexIndicesAttr([0,1,2,1,3,2]);sheet.CreateVelocitiesAttr(Vt.Vec3fArray.FromNumpy(np.tile([.1,0,0],(4,1)).astype(np.float32)))
  particleUtils.add_physx_particle_system(stage,'/World/Particles',simulation_owner='/physicsScene',contact_offset=.001,rest_offset=.0005,particle_contact_offset=.001,solid_rest_offset=.0005)
  particleUtils.add_physx_particle_cloth(stage,'/World/Sheet',None,'/World/Particles',spring_stretch_stiffness=100.,spring_bend_stiffness=.1,spring_shear_stiffness=50.,spring_damping=.1)
  UsdPhysics.MassAPI.Apply(sheet.GetPrim()).CreateMassAttr(.001);actors['sheet']={'path':'/World/Sheet','kind':'deformable'}
 from calibrated_scene_reconstruction.physics.process.mass_bindings import apply as bind_masses,audit as audit_masses
 mass_rows=bind_masses(stage,actors,{'mass_scale':{'bodies':['body']}},{'mass_scale':args.mass_scale})
 world.reset();world.step();actor=SingleRigidPrim('/World/Body');p0=actor.get_world_pose()[0].copy();api=omni.physx.get_physx_simulation_interface();sid=omni.usd.get_context().get_stage_id()
 sheet0=None if sheet is None else np.asarray(sheet.GetPointsAttr().Get()).copy()
 audit=quiesce(stage,actors,api,sid)
 for _ in range(30):world.step()
 p1=actor.get_world_pose()[0].copy();handle=PhysicsSchemaTools.sdfPathToInt(Sdf.Path('/World/Body'));asleep=api.is_sleeping(sid,handle)
 for _ in range(30):
  p,_=actor.get_world_pose();api.apply_force_at_pos(sid,handle,carb.Float3(.1,0,0),carb.Float3(*p.tolist()),'Force');world.step()
 p2=actor.get_world_pose()[0].copy();awake=not api.is_sleeping(sid,handle)
 particle_drift=None if sheet is None else float(np.linalg.norm(np.asarray(sheet.GetPointsAttr().Get())-sheet0,axis=1).max())
 expected=.1/args.mass_scale*(1/240)**2*30*31/2
 mass_error=abs((p2[0]-p1[0])-expected)/expected
 good=bool(audit['pass'] and all(audit['native_sleeping'].values()) and all(audit['native_awake_for_contacts'].values()) and np.linalg.norm(p1-p0)<1e-7 and awake and p2[0]-p1[0]>1e-4 and not body.GetKinematicEnabledAttr().Get() and (particle_drift is None or particle_drift<1e-7) and mass_error<.02 and audit_masses(stage,mass_rows))
 write(out/'result.json',{'pass':good,'engine':{'name':'isaacsim','version':version('isaacsim')},'audit':audit,'free_drift_m':(p1-p0).tolist(),'force_displacement_m':(p2-p1).tolist(),'native_asleep_before_force':asleep,'native_awake_after_force':awake,'scope':'Native initialization/wake-up capability fixture, not source-video calibration or acceptance.','source_sha256':{str(Path(p).resolve()):digest(p) for p in [__file__,ROOT/'src/calibrated_scene_reconstruction/physics/process/initial_rest.py',ROOT/'src/calibrated_scene_reconstruction/physics/process/initial_motion.py']}})
 if sheet is not None:write(out/'particle_audit.json',{'pass':particle_drift<1e-7,'maximum_native_particle_drift_m':particle_drift,'source_stationary_initialization':True})
 write(out/'mass_audit.json',{'pass':mass_error<.02 and audit_masses(stage,mass_rows),'bindings':mass_rows,'expected_force_displacement_m':expected,'measured_force_displacement_m':float(p2[0]-p1[0]),'relative_error':float(mass_error),'scope':'Native collision-free F=ma capability assay, not video parameter identification.'})
 print('source stationary/wake-up assay',good,'particle drift',particle_drift,flush=True);world.clear();require(good,'Native stationary initializer did not preserve free wake-up')
finally:app.close()
