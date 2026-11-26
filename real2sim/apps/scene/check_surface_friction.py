"""Native inclined-sheet friction control; never a reconstruction task pass."""
import argparse
import hashlib
import json
from pathlib import Path
import traceback

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',required=True)
p.add_argument('--cloth-friction',type=float,default=.55)
p.add_argument('--rigid-friction',type=float,default=.8)
p.add_argument('--zero-cloth-at',type=float)
p.add_argument('--zero-rigid-at',type=float)
p.add_argument('--rigid-combine',choices=['average','multiply'],default='average')
p.add_argument('--rigid-update-path',choices=['tensor','usd','rebind'],default='tensor')
a=p.parse_args()
if min(a.cloth_friction,a.rigid_friction)<0: p.error('Nonnegative friction required')
out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
(out/'protocol.json').write_text(json.dumps(vars(a),indent=2)+'\n')
from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402
app=SimulationApp({'headless':True})
import numpy as np  # noqa: E402
import torch  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from pxr import UsdGeom,UsdPhysics,UsdShade,PhysxSchema,Vt  # noqa: E402
from calibrated_scene_reconstruction.sim.surface_cloth import author,SurfaceView  # noqa: E402
from calibrated_scene_reconstruction.sim.usd import set_pose  # noqa: E402
from calibrated_scene_reconstruction.sim.physics import physics_material  # noqa: E402
from calibrated_scene_reconstruction.sim.physics_errors import PhysicsErrors  # noqa: E402
from calibrated_scene_reconstruction.assets.cloth_rest import triangulate_grid  # noqa: E402
from calibrated_scene_reconstruction.sim.hand_materials import set_friction,author_friction,summary as rigid_material_summary  # noqa: E402
errors=PhysicsErrors(out/'native_physics_errors.json')
try:
    hz=960
    world=World(stage_units_in_meters=1.,physics_dt=1/hz,device='cuda:0',backend='torch')
    errors.guard_steps(world)
    pc=world.get_physics_context();pc.enable_gpu_dynamics(True);pc.set_broadphase_type('GPU');pc.set_solver_type('PGS')
    stage=omni.usd.get_context().get_stage()
    theta=np.radians(20.);c=np.cos(theta);sn=np.sin(theta)
    rotation=np.array([[c,0,sn],[0,1,0],[-sn,0,c]])
    normal=rotation[:,2];down=rotation[:,0];centre=np.array([0.,0.,.5])
    ramp=UsdGeom.Cube.Define(stage,'/World/Ramp');ramp.CreateSizeAttr(1.)
    set_pose(ramp.GetPrim(),centre-normal*.01,[np.cos(theta/2),0,np.sin(theta/2),0])
    UsdGeom.Xformable(ramp).AddScaleOp().Set((2.,2.,.02))
    UsdPhysics.CollisionAPI.Apply(ramp.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(ramp.GetPrim()).CreateKinematicEnabledAttr(True)
    material=physics_material(stage,'/World/RampMaterial',a.rigid_friction,a.rigid_friction,0.,friction_combine=a.rigid_combine)
    UsdShade.MaterialBindingAPI.Apply(ramp.GetPrim()).Bind(material,UsdShade.Tokens.weakerThanDescendants,'physics')
    x,y=np.meshgrid(np.linspace(-.08,.08,9),np.linspace(-.08,.08,9))
    rest=np.c_[x.ravel(),y.ravel(),np.zeros(81)];faces=[]
    for row in range(8):
        for col in range(8):
            i=row*9+col;faces.extend([[i,i+1,i+10],[i,i+10,i+9]])
    faces=np.asarray(faces,np.int32)
    faces,chart,_=triangulate_grid(rest,rest,9,9)
    points=rest@rotation.T+centre+normal*.0003
    part={'V':points,'F':faces,'cloth_rest':chart}
    mesh=UsdGeom.Mesh.Define(stage,'/World/Sheet')
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr([3]*len(faces));mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
    settings=dict(youngs_modulus=4e6,poissons_ratio=.2,thickness=.0004,flexural_rigidity=1e-5,
                  elasticity_damping=.01,linear_damping=.02,friction=a.cloth_friction,
                  areal_density=.2,contact_offset=.0003,rest_offset=.0002,
                  self_collision_filter_distance=.0004,solver_position_iterations=128)
    author(stage,mesh,part,settings)
    stage.GetRootLayer().Export(str(out/'initial.usda'));world.reset()
    view=SurfaceView(stage,'/World/Sheet',part)
    mv=view.owner.create_deformable_material_view('/World/Sheet_surface_fem_material')
    rv=view.owner.create_rigid_body_view('/World/Ramp')
    if mv.count!=1:raise ValueError('Expected one actual native cloth material')
    def friction():return mv.get_dynamic_friction().detach().cpu().numpy().copy()
    if not np.allclose(friction(),a.cloth_friction,atol=1e-7):raise ValueError('Initial effective cloth friction differs')
    initial=view.get_world_positions()[0].detach().cpu().numpy().copy()
    rows=[];states=[];changed=False;rigid_changed=False
    for step in range(round(.4*hz)+1):
        t=step/hz
        if a.zero_cloth_at is not None and t>=a.zero_cloth_at and not changed:
            raw=mv.get_dynamic_friction();data=torch.zeros_like(raw)
            mv.set_dynamic_friction(data,torch.zeros(1,dtype=torch.int32,device=raw.device))
            if not np.all(friction()==0):raise ValueError('Native cloth friction setter failed')
            changed=True
        if a.zero_rigid_at is not None and t>=a.zero_rigid_at and not rigid_changed:
            if a.rigid_update_path=='tensor':
                set_friction(rv,0.)
            elif a.rigid_update_path=='usd':
                author_friction(material.GetPrim(),0.)
            else:
                zero_material=physics_material(stage,'/World/ZeroRampMaterial',0.,0.,0.,friction_combine=a.rigid_combine)
                UsdShade.MaterialBindingAPI(ramp.GetPrim()).Bind(zero_material,UsdShade.Tokens.weakerThanDescendants,'physics')
            rigid_changed=True
        if step%24==0:
            q=view.get_world_positions()[0].detach().cpu().numpy().copy()
            if not np.isfinite(q).all():raise ValueError('Nonfinite native cloth')
            rows.append({'time_s':t,'downhill_displacement_m':float((q.mean(0)-initial.mean(0))@down),
                         'cloth_friction':friction().tolist(),'rigid_material':rigid_material_summary(rv),
                         'minimum_plane_height_m':float(((q-centre)@normal).min())})
            states.append(q)
        if step<round(.4*hz):world.step(render=False)
    errors.check()
    np.savez_compressed(out/'states.npz',points=states,rest=rest,faces=faces)
    result={'completed':True,'task_approved':False,'synthetic_control':True,'rows':rows,
            'runtime_friction_changed':changed,'rigid_authored_friction':a.rigid_friction,
            'runtime_rigid_friction_changed':rigid_changed,'rigid_combine':a.rigid_combine,
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'native_physics_errors':errors.report(),
            'scope':'Inclined native surface cloth, no external forces/attachments or viscosity adapter. Compare actual displacement and native cloth material across controls; not a task certificate.'}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('FRICTION_CONTROL',json.dumps(rows[-1]),flush=True)
except Exception as error:
    (out/'error.json').write_text(json.dumps({'error':str(error),'traceback':traceback.format_exc()},indent=2)+'\n')
    raise
finally:
    errors.close();app.close()
