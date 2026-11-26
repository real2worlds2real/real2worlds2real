"""Check sampled USD readback against full native rigid/PBD readback."""
import argparse
from pathlib import Path
ap=argparse.ArgumentParser(__doc__);ap.add_argument('--output',required=True);args=ap.parse_args()
from calibrated_scene_reconstruction.core.paths import ROOT  # noqa: E402
from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402
app=SimulationApp({'headless':True,'width':64,'height':64,'disable_viewport_updates':True,'multi_gpu':False,'limit_cpu_threads':8},experience=str(ROOT/'apps/physics/physics_only.kit'))
try:
 import time,numpy as np,carb,omni.usd
 from pxr import UsdGeom,UsdPhysics,Gf,Vt
 from omni.physx.scripts import particleUtils
 from omni.physx.bindings import _physx as bindings
 from calibrated_scene_reconstruction.physics.process.physics_only import World,SingleRigidPrim
 from calibrated_scene_reconstruction.physics.process.contract import write,require,digest
 out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
 settings=carb.settings.get_settings();keys=[getattr(bindings,k) for k in ['SETTING_UPDATE_TO_USD','SETTING_UPDATE_VELOCITIES_TO_USD','SETTING_UPDATE_PARTICLES_TO_USD']]
 def readback(value):
  for key in keys:settings.set(key,value)
 results={}
 for mode in ['full','sampled']:
  readback(True);World.clear_instance();omni.usd.get_context().new_stage();world=World(physics_dt=1/240);stage=world.stage
  world.context.enable_gpu_dynamics(True);world.context.set_broadphase_type('GPU');world.context.set_solver_type('TGS')
  floor=UsdGeom.Cube.Define(stage,'/World/Floor');floor.CreateSizeAttr(1.);floor.AddScaleOp().Set(Gf.Vec3f(1,1,.02));floor.AddTranslateOp().Set(Gf.Vec3d(0,0,-.5));UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
  ball=UsdGeom.Sphere.Define(stage,'/World/Support');ball.CreateRadiusAttr(.025);ball.AddTranslateOp().Set(Gf.Vec3d(0,0,.08));UsdPhysics.CollisionAPI.Apply(ball.GetPrim())
  cube=UsdGeom.Cube.Define(stage,'/World/Body');cube.CreateSizeAttr(.03);cube.AddTranslateOp().Set(Gf.Vec3d(.2,0,.2));UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim());UsdPhysics.CollisionAPI.Apply(cube.GetPrim());UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(.1)
  x,y=np.meshgrid(np.linspace(-.05,.05,7),np.linspace(-.05,.05,7));points=np.c_[x.ravel(),y.ravel(),np.full(x.size,.2)];faces=[]
  for j in range(6):
   for i in range(6):
    a=j*7+i;faces.extend([[a,a+1,a+7],[a+1,a+8,a+7]])
  mesh=UsdGeom.Mesh.Define(stage,'/World/Sheet');mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)));mesh.CreateFaceVertexCountsAttr([3]*len(faces));mesh.CreateFaceVertexIndicesAttr(np.ravel(faces).tolist())
  particleUtils.add_physx_particle_system(stage,'/World/Particles',simulation_owner='/physicsScene',contact_offset=.001,rest_offset=.0005,particle_contact_offset=.001,solid_rest_offset=.0005,solver_position_iterations=16)
  particleUtils.add_physx_particle_cloth(stage,'/World/Sheet',None,'/World/Particles',spring_stretch_stiffness=100.,spring_bend_stiffness=.1,spring_shear_stiffness=50.,spring_damping=.1,self_collision=True,self_collision_filter=True)
  UsdPhysics.MassAPI.Apply(mesh.GetPrim()).CreateMassAttr(.001)
  world.reset();actor=SingleRigidPrim('/World/Body');records=[];started=time.perf_counter()
  for step in range(240):
   readback(mode=='full');world.api.simulate(world.dt,world.time);world.api.fetch_results();world.time+=world.dt
   if mode=='full' or (step+1)%8==0:
    readback(True);world.interface.update_transformations(False,True,True)
   if (step+1)%8==0:records.append(np.r_[actor.get_world_pose()[0],np.asarray(mesh.GetPointsAttr().Get()).ravel()])
  results[mode]={'seconds':time.perf_counter()-started,'states':np.asarray(records)};world.clear()
 readback(True);a=results['full']['states'];b=results['sampled']['states'];error=float(np.max(abs(a-b)));movement=float(np.max(abs(a[-1,3:]-points.ravel())))
 result={'pass':error<1e-6 and movement>.01,'maximum_state_difference_m':error,'native_particle_movement_m':movement,'seconds':{k:v['seconds'] for k,v in results.items()},'scope':'Procedural native rigid/PBD readback equivalence assay only; not task or video acceptance.','source_sha256':digest(__file__)}
 np.savez_compressed(out/'states.npz',full=a,sampled=b);write(out/'result.json',result);print(result,flush=True);require(result['pass'],'Sampled USD readback does not preserve native states')
finally:app.close()
