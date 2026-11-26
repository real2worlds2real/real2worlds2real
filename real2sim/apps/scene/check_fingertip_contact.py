"""Independent native contact-tool qualification; never reconstruction approval.

A free hanging surface coupon, real source-tip geometry/masses and finite rigid
forces. No container, cloth pins, nodal forces or positional cloth animation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import traceback

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--hand-plan',required=True);p.add_argument('--profile',required=True);p.add_argument('--output',required=True)
p.add_argument('--normal-force-N',type=float,required=True);p.add_argument('--zero-friction-lift',action='store_true')
p.add_argument('--tip-collision',choices=['convexHull','sdf'],default='sdf')
p.add_argument('--hold-seconds',type=float,default=1.)
p.add_argument('--size-m',type=float,nargs=2,default=[.04,.06])
p.add_argument('--normal-preload',action='store_true',help='Force-controlled closing axis, bounded by the same source wrench limits')
p.add_argument('--grasp-height-fraction',type=float,default=.5,help='Fraction of coupon height above the pad centre')
a=p.parse_args()
if not .5<=a.hold_seconds<=3 or not 0<a.normal_force_N<=2 or any(not 0<value<=.2 for value in a.size_m) or not 0<a.grasp_height_fraction<1:p.error('Bounded force, interior grasp and coupon dimensions required')
from calibrated_scene_reconstruction.tasks.mechanisms.contact_control_profile import validate
profile=validate(json.loads(Path(a.profile).read_text()))
out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
for path,name in [(Path(__file__),'source.py'),(Path(a.hand_plan),'hand_input.json'),(Path(a.profile),'profile.json')]:shutil.copy2(path,out/name)
from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402
app=SimulationApp({'headless':True})
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from pxr import UsdGeom,PhysxSchema,UsdPhysics,Vt  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from calibrated_scene_reconstruction.scene.capture.articulated_model import from_usd  # noqa: E402
from calibrated_scene_reconstruction.sim.fingertip_fixture import FingertipFixture  # noqa: E402
from calibrated_scene_reconstruction.sim.surface_cloth import author,SurfaceView  # noqa: E402
from calibrated_scene_reconstruction.sim.hand_collision import Penetration  # noqa: E402
from calibrated_scene_reconstruction.sim.surface_distance import triangle_samples  # noqa: E402
from calibrated_scene_reconstruction.sim.hand_materials import author_friction  # noqa: E402
from calibrated_scene_reconstruction.sim.physics_errors import PhysicsErrors  # noqa: E402
from calibrated_scene_reconstruction.sim.reset_timestep import reset as native_reset  # noqa: E402
from calibrated_scene_reconstruction.assets.cloth_rest import triangulate_grid  # noqa: E402
from calibrated_scene_reconstruction.assets.cloth_metric import MaterialMetric  # noqa: E402
from calibrated_scene_reconstruction.sim.cloth_contact import self_intersections  # noqa: E402
from calibrated_scene_reconstruction.tasks.rigid.task_checks import as_numpy  # noqa: E402
errors=PhysicsErrors(out/'native_physics_errors.json')
try:
 hz=profile['physics_hz'];settings=profile['settings'];h=settings['thickness']
 hand=json.loads(Path(a.hand_plan).read_text());model=from_usd(hand['asset'],100)
 world=World(stage_units_in_meters=1.,physics_dt=1/hz,device='cuda:0',backend='torch');errors.guard_steps(world)
 pc=world.get_physics_context();pc.enable_gpu_dynamics(True);pc.set_broadphase_type('GPU');pc.set_solver_type('PGS')
 stage=omni.usd.get_context().get_stage();scene=next(prim for prim in stage.Traverse()if prim.IsA(UsdPhysics.Scene))
 PhysxSchema.PhysxSceneAPI.Apply(scene).CreateEnableExternalForcesEveryIterationAttr(False)
 width,height=a.size_m;sx,sy=profile['spacing_m']
 top=height*a.grasp_height_fraction
 x=np.linspace(-width/2,width/2,max(3,round(width/sx)+1));y=np.linspace(-height+top,top,max(3,round(height/sy)+1))
 xx,yy=np.meshgrid(x,y);rest=np.c_[xx.ravel(),yy.ravel(),np.zeros(xx.size)]
 centre=np.array([0.,0.,.2]);points=np.c_[np.zeros(xx.size),xx.ravel(),yy.ravel()]+centre
 faces,chart,_=triangulate_grid(points,rest,len(y),len(x));part={'V':points,'F':faces,'cloth_rest':chart}
 mesh=UsdGeom.Mesh.Define(stage,'/World/Coupon');mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype('float32')))
 mesh.CreateFaceVertexCountsAttr([3]*len(faces));mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist());mesh.CreateSubdivisionSchemeAttr('none')
 author(stage,mesh,part,settings)
 tool=FingertipFixture(stage,hand,model,centre,np.eye(3),out,initial_gap_m=h,collision_approximation=a.tip_collision)
 stage.GetRootLayer().Export(str(out/'initial.usda'));reset=native_reset(world)
 tool.initialize(hz);view=SurfaceView(stage,'/World/Coupon',part)
 penetration=Penetration(tool.rows);metric=MaterialMetric(points,faces,chart,max_edge_strain=.25,max_area_change=.15)
 rows=[];states=[];poses=[];times=[];commands=[];failure=None;zero=None;initial=None;last=None
 duration=1.2+a.hold_seconds
 files=[__file__,a.hand_plan,a.profile,'src/calibrated_scene_reconstruction/sim/fingertip_fixture.py','src/calibrated_scene_reconstruction/sim/pinch_load.py','src/calibrated_scene_reconstruction/sim/convex_cooking.py','src/calibrated_scene_reconstruction/sim/surface_cloth.py','src/calibrated_scene_reconstruction/sim/hand_collision.py','src/calibrated_scene_reconstruction/assets/cloth_rest.py','src/calibrated_scene_reconstruction/sim/reset_timestep.py','src/calibrated_scene_reconstruction/sim/deformable_filter_pose.py','src/calibrated_scene_reconstruction/assets/deformable_filter_pose.py']
 fingerprints={str(Path(f).resolve()):hashlib.sha256(Path(f).read_bytes()).hexdigest()for f in files}
 (out/'protocol.json').write_text(json.dumps({'inputs':fingerprints,'parameters':vars(a),'cloth_forces':False,'cloth_attachments':False},indent=2)+'\n')
 def record(phase,t):
  actual=as_numpy(view.get_world_positions())[0].copy();native=tool.poses();contact=penetration.check(triangle_samples(actual,faces),native,tool.names,2*h,contact_range_m=settings['contact_offset']+tool.contact_offset_m)
  material=metric.check(actual);cross=self_intersections(actual,faces)
  native_material=as_numpy(tool.view._physics_view.get_material_properties())
  valid=bool(contact['pass'] and material['geometry_valid'] and cross['pass'])
  row={'phase':phase,'time_s':t,'contact':contact,'material':material,'self_contact':cross,'physical_pass':valid,
   'native_tip_material_properties':native_material.tolist(),'tool_mean_position':native[:,:3].mean(0).tolist(),
   'tool_forces_N':None if tool.held is None else as_numpy(tool.held[0]).tolist(),
   'tool_velocities':as_numpy(tool.view.get_velocities()).tolist()}
  rows.append(row);states.append(actual);poses.append(native);times.append(t)
  (out/'progress.json').write_text(json.dumps(row,indent=2)+'\n');return valid
 if not record('initial',0.):raise ValueError('Synthetic initial coupon invalid')
 for step in range(round(duration*hz)):
  t=(step+1)/hz
  if t<=.2:phase='clamp';target=centre
  elif t<=1.2:
   phase='lift';u=min(1.,(t-.2));u=u*u*(3-2*u);target=centre+[0,0,.08*u]
  else:phase='hold';target=centre+[0,0,.08]
  if phase=='lift' and initial is None:
   initial=as_numpy(view.get_world_positions())[0].copy()
   if a.zero_friction_lift:zero=author_friction(tool.material,0.)
  commands.append(np.r_[target,0.,a.normal_force_N,float(a.normal_preload)])
  tool.control(target,np.eye(3),0.,normal_force_limit_N=a.normal_force_N,
    normal_preload_N=a.normal_force_N if a.normal_preload else None);world.step(render=False)
  if (step+1)%(hz//20)==0 or step+1==round(duration*hz):
   if not record(phase,t):failure='Actual coupon contact/material/self validity failed';break
 last=as_numpy(view.get_world_positions())[0].copy();errors.check()
 np.savez_compressed(out/'states.npz',points=states,hand_links=poses,times=times,rest=rest,faces=faces,
  prelift_points=np.empty((0,3))if initial is None else initial,commands=commands)
 lift=None if initial is None else float(np.median(last[:,2]-initial[:,2]))
 plateau=[state for state,row in zip(states,rows)if row['phase']=='hold' and row['time_s']>=duration-.3]
 drift=max((abs(float(np.median(state[:,2]-last[:,2])))for state in plateau),default=float('inf'))
 held=bool(lift is not None and lift>.05 and abs(lift-.08)<.02 and drift<.01)
 if zero is not None:
  checked=[np.asarray(row['native_tip_material_properties'])for row in rows if row['phase']in ['lift','hold']]
  zero['native_readback_after_step']=bool(checked and all(np.all(v[...,:2]==0)for v in checked))
 result={'complete':failure is None,'physical_pass':all(r['physical_pass']for r in rows),'failure':failure,
  'synthetic_control':True,'task_approved':False,'zero_friction_lift':a.zero_friction_lift,'material_lift_m':lift,
  'lift_retained':held,'stationary_hold_drift_m':drift,'hold_seconds':a.hold_seconds,'rows':rows,'source_tool':tool.source,'bootstrap':reset,
  'normal_force_limit_N':a.normal_force_N,'zero_material_update':zero,
  'input_sha256':fingerprints,
  'native_physics_errors':errors.report(),'scope':'Independent source-tool/coupon contact only; not full-sheet, source or task approval.'}
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 print('FINGERTIP_COUPON',json.dumps({k:result[k]for k in ['complete','physical_pass','failure','material_lift_m','lift_retained']}),flush=True)
except Exception as error:
 (out/'error.json').write_text(json.dumps({'error':str(error),'traceback':traceback.format_exc()},indent=2)+'\n');print(traceback.format_exc(),flush=True)
finally:errors.close();app.close()
