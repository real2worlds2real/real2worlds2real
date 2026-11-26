"""Independent native sliding control, not source or task acceptance.

Compare separate surfaces, same-body contact and a rigid support. The same-body
fixture has two disconnected components, solely to isolate its collision path;
it is never an admissible single-sheet task asset or a wet-adhesion model.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import traceback

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',required=True)
p.add_argument('--friction',type=float,required=True)
p.add_argument('--support',choices=['surface','self','rigid'],required=True)
p.add_argument('--rigid-friction',type=float)
p.add_argument('--combine',choices=['average','multiply'],default='average')
p.add_argument('--profile',help='Explicit material, solver, contact and two-axis sampling profile; never inferred from a scene name')
args=p.parse_args()
if (not 0<=args.friction<=1.5 or args.rigid_friction is not None and
        (args.support!='rigid' or not 0<=args.rigid_friction<=1.5)):
    p.error('Bounded nonnegative contact coefficients and matching rigid support required')
profile=None
if args.profile:
    from calibrated_scene_reconstruction.tasks.mechanisms.contact_control_profile import validate
    profile=validate(json.loads(Path(args.profile).read_text()))
out=Path(args.output);out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,out/'source.py')

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402
app=SimulationApp({'headless':True})

import numpy as np  # noqa: E402
import torch  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from pxr import UsdGeom,UsdPhysics,PhysxSchema,Vt,UsdShade  # noqa: E402
from calibrated_scene_reconstruction.assets.cloth_rest import triangulate_grid  # noqa: E402
from calibrated_scene_reconstruction.assets.cloth_metric import MaterialMetric  # noqa: E402
from calibrated_scene_reconstruction.sim.surface_cloth import author,SurfaceView,apply_material  # noqa: E402
from calibrated_scene_reconstruction.sim.cloth_contact import self_intersections  # noqa: E402
from calibrated_scene_reconstruction.sim.physics import physics_material  # noqa: E402
from calibrated_scene_reconstruction.sim.usd import add_box_collider  # noqa: E402
from calibrated_scene_reconstruction.sim.physics_errors import PhysicsErrors  # noqa: E402
from calibrated_scene_reconstruction.tasks.rigid.task_checks import as_numpy  # noqa: E402

errors=PhysicsErrors(out/'native_physics_errors.json')
try:
    hz=profile['physics_hz'] if profile else 7680;dt=1/hz
    thickness=profile['settings']['thickness'] if profile else .0004
    spacing=profile['spacing_m'] if profile else [.002,.002]
    world=World(stage_units_in_meters=1.,physics_dt=dt,device='cuda:0',backend='torch');errors.guard_steps(world)
    pc=world.get_physics_context();pc.enable_gpu_dynamics(True);pc.set_broadphase_type('GPU');pc.set_solver_type('PGS')
    stage=omni.usd.get_context().get_stage();scene=next(q for q in stage.Traverse() if q.IsA(UsdPhysics.Scene))
    api=PhysxSchema.PhysxSceneAPI.Apply(scene);api.CreateEnableExternalForcesEveryIterationAttr(False)
    settings={'youngs_modulus':4e6,'poissons_ratio':.2,'thickness':thickness,'flexural_rigidity':1e-5,
        'areal_density':.2,'elasticity_damping':.01,'linear_damping':.02,'friction':args.friction,
        'contact_offset':.0003,'rest_offset':.0002,'self_collision_filter_distance':.0004,
        'solver_position_iterations':128,'collision_pair_updates':16,'collision_iteration_multiplier':2,
        'max_depenetration_velocity':.1,'max_linear_velocity':10.}
    if profile:settings={**profile['settings'],'friction':args.friction}
    parts={}
    def sheet(name,width,height):
        x=np.linspace(-width/2,width/2,max(2,round(width/spacing[0])+1))
        y=np.linspace(-width/2,width/2,max(2,round(width/spacing[1])+1))
        xx,yy=np.meshgrid(x,y);rest=np.c_[xx.ravel(),yy.ravel(),np.zeros(xx.size)];points=rest+[0,0,height]
        if args.support=='self' and name=='Upper':points[:,1]*=-1
        faces,chart,_=triangulate_grid(points,rest,len(y),len(x));part={'V':points,'F':faces,'cloth_rest':chart}
        if args.support=='self':parts[name]=part;return
        mesh=UsdGeom.Mesh.Define(stage,'/World/'+name);mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype('float32')))
        mesh.CreateFaceVertexCountsAttr([3]*len(faces));mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist());mesh.CreateSubdivisionSchemeAttr('none')
        author(stage,mesh,part,settings);parts[name]=part
    if args.support in ['surface','self']:sheet('Lower',.08,.1)
    else:
        box=add_box_collider(stage,'/World/Lower',[-.04,-.04,.09],[.04,.04,.1])
        mu=args.friction if args.rigid_friction is None else args.rigid_friction
        mat=physics_material(stage,'/World/SupportMaterial',mu,mu,0.,friction_combine=args.combine)
        UsdShade.MaterialBindingAPI.Apply(box.GetPrim()).Bind(mat,UsdShade.Tokens.weakerThanDescendants,'physics')
    sheet('Upper',.04,.1+settings['rest_offset']*(2 if args.support in ['surface','self'] else 1)+.00005)
    combined_part=None;component_ids=None;whole=None
    if args.support=='self':
        from omni.physx.scripts import deformableUtils
        lower,upper=parts['Lower'],parts['Upper'];n=len(lower['V'])
        rest_upper=np.asarray(upper['cloth_rest']['points']).copy();rest_upper[:,1]+=.12
        natural=np.vstack([lower['cloth_rest']['points'],rest_upper])
        combined_part={'V':np.vstack([lower['V'],upper['V']]),'F':np.vstack([lower['F'],upper['F']+n]),
                       'cloth_rest':{'points':natural.tolist()}}
        mesh=UsdGeom.Mesh.Define(stage,'/World/Combined');mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(combined_part['V'].astype('float32')))
        mesh.CreateFaceVertexCountsAttr([3]*len(combined_part['F']));mesh.CreateFaceVertexIndicesAttr(combined_part['F'].ravel().tolist());mesh.CreateSubdivisionSchemeAttr('none')
        prim=mesh.GetPrim()
        if not deformableUtils.set_physics_surface_deformable_body(stage,prim.GetPath()):raise RuntimeError('Native disconnected-component contact fixture unavailable')
        if not prim.ApplyAPI('PhysxSurfaceDeformableBodyAPI'):raise RuntimeError('Native surface collision schema unavailable')
        prim.GetAttribute('omniphysics:restShapePoints').Set(Vt.Vec3fArray.FromNumpy(natural.astype('float32')))
        prim.GetAttribute('omniphysics:restBendAnglesDefault').Set('flatDefault')
        for name,value in [('selfCollision',True),('selfCollisionFilterDistance',settings['self_collision_filter_distance']),
            ('solverPositionIterationCount',settings['solver_position_iterations']),
            ('collisionPairUpdateFrequency',settings['collision_pair_updates']),
            ('collisionIterationMultiplier',settings['collision_iteration_multiplier']),
            ('enableSpeculativeCCD',True),('maxLinearVelocity',settings['max_linear_velocity']),
            ('maxDepenetrationVelocity',settings['max_depenetration_velocity']),
            ('settlingDamping',0.),('settlingThreshold',0.),('sleepThreshold',0.)]:prim.GetAttribute('physxDeformableBody:'+name).Set(value)
        collision=PhysxSchema.PhysxCollisionAPI.Apply(prim)
        collision.CreateContactOffsetAttr(settings['contact_offset']);collision.CreateRestOffsetAttr(settings['rest_offset'])
        apply_material(stage,prim,settings,natural,combined_part['F']);api.CreateGpuMaxDeformableSurfaceContactsAttr(1<<20)
        component_ids={'Lower':np.arange(n),'Upper':np.arange(n,len(combined_part['V']))}
    stage.GetRootLayer().Export(str(out/'initial.usda'));world.reset()
    if args.support=='self':
        whole=SurfaceView(stage,'/World/Combined',combined_part)
        class Component:
            def __init__(self,name):self.ids=component_ids[name];self.owner=whole.owner
            def get_world_positions(self):return whole.get_world_positions()[:,self.ids]
            def get_velocities(self):return whole.get_velocities()[:,self.ids]
            def get_particle_masses(self):return whole.get_particle_masses()[:,self.ids]
            def require_validated_force_profile(self):return whole.require_validated_force_profile()
        views={name:Component(name) for name in parts}
    else:views={name:SurfaceView(stage,'/World/'+name,part) for name,part in parts.items()}
    for view in views.values():view.require_validated_force_profile()
    material_view=views['Upper'].owner.create_deformable_material_view('/World/*_surface_fem_material')
    native_friction=as_numpy(material_view.get_dynamic_friction()).copy()
    if (material_view.count!=(1 if whole is not None else len(views)) or not np.isfinite(native_friction).all()
            or not np.allclose(native_friction,args.friction,rtol=0,atol=1e-7)):
        raise ValueError('Native deformable material friction differs from the declared control')
    masses={name:as_numpy(view.get_particle_masses())[0] for name,view in views.items()}
    metrics={name:MaterialMetric(part['V'],part['F'],part['cloth_rest'],max_edge_strain=.05,max_area_change=.03) for name,part in parts.items()}
    initial={name:as_numpy(view.get_world_positions())[0].copy() for name,view in views.items()}
    weights={name:mass/mass.sum() for name,mass in masses.items()}
    total_upper=float(masses['Upper'].sum());load_force=.2*total_upper*9.81
    velocity=0.;free_displacement=0.;states={name:[] for name in views};rows=[];peak_clamp=0.;pending={}
    def force(name,forces):
        if whole is not None:pending[name]=forces;return
        view=views[name];v=view.get_velocities().clone()
        v[0]+=torch.as_tensor(forces*dt/masses[name][:,None],dtype=v.dtype,device=v.device);view.set_velocities(v)
    def record(time):
        points={name:as_numpy(view.get_world_positions())[0].copy() for name,view in views.items()}
        material={name:metrics[name].check(points[name]) for name in views}
        if args.support in ['surface','self']:
            from scipy.spatial import cKDTree
            indices=cKDTree(points['Lower'][:,:2]).query(points['Upper'][:,:2])[1]
            gaps=points['Upper'][:,2]-points['Lower'][indices,2];expected_gap=2*settings['rest_offset']
            relative=float(points['Upper'][:,0].mean()-points['Lower'][:,0].mean())
            combined=np.vstack([points['Lower'],points['Upper']]);faces=np.vstack([parts['Lower']['F'],parts['Upper']['F']+len(points['Lower'])])
        else:
            gaps=points['Upper'][:,2]-.1;expected_gap=settings['rest_offset']
            relative=float(points['Upper'][:,0].mean());combined=points['Upper'];faces=parts['Upper']['F']
        cross=self_intersections(combined,faces)
        valid=bool(all(v['pass'] for v in material.values()) and cross['pass'] and gaps.min()>=expected_gap-.00015)
        rows.append({'time_s':time,'physical_pass':valid,'material':material,'crossings':cross,
            'relative_x_m':relative,'minimum_mid_surface_gap_m':float(gaps.min()),'expected_contact_gap_m':expected_gap,
            'upper_mean_velocity_m_s':as_numpy(views['Upper'].get_velocities())[0].mean(0).tolist(),
            'free_traction_displacement_m':free_displacement})
        for name in views:states[name].append(points[name])
        (out/'progress.json').write_text(json.dumps(rows[-1],indent=2)+'\n')
        return valid
    valid=record(0.)
    for step in range(round(.35*hz)):
        if not valid:break
        t=step*dt
        if args.support in ['surface','self']:
            view=views['Lower'];position=as_numpy(view.get_world_positions())[0];speed=as_numpy(view.get_velocities())[0]
            k=200.;damping=np.minimum(2*np.sqrt(k*masses['Lower'].sum())*weights['Lower'],.5*masses['Lower']/dt)
            f=k*(initial['Lower']-position)*weights['Lower'][:,None]-damping[:,None]*speed
            f[:,2]+=masses['Lower']*9.81
            magnitude=np.linalg.norm(f,axis=1);f*=np.minimum(1.,.1*weights['Lower']/np.maximum(magnitude,1e-12))[:,None]
            peak_clamp=max(peak_clamp,float(np.linalg.norm(f,axis=1).sum()));force('Lower',f)
        fx=load_force*min(1.,max(0.,(t-.2)/.01)) if .2<=t<.3 else 0.
        f=np.zeros((len(masses['Upper']),3));f[:,0]=fx*weights['Upper'];force('Upper',f)
        if whole is not None:
            native_velocity=whole.get_velocities().clone()
            for name,forces in pending.items():
                ids=component_ids[name]
                native_velocity[0,ids]+=torch.as_tensor(forces*dt/masses[name][:,None],dtype=native_velocity.dtype,device=native_velocity.device)
            whole.set_velocities(native_velocity);pending.clear()
        free_displacement+=velocity*dt+.5*fx/total_upper*dt*dt;velocity+=fx/total_upper*dt
        world.step(render=False)
        if (step+1)%(hz//40)==0:valid=record((step+1)*dt)
    errors.check();end=next((r for r in rows if abs(r['time_s']-.3)<1e-8),None)
    final_friction=as_numpy(material_view.get_dynamic_friction()).copy()
    if not np.array_equal(native_friction,final_friction):raise ValueError('Native material friction changed during the control')
    from calibrated_scene_reconstruction.tasks.mechanisms.contact_control_profile import edge_range
    ranges=[edge_range(part['cloth_rest']['points'],part['F']) for part in parts.values()]
    inputs=[Path(__file__),Path('src/calibrated_scene_reconstruction/sim/surface_cloth.py'),
            Path('src/calibrated_scene_reconstruction/tasks/mechanisms/contact_control_profile.py')]
    if args.profile:inputs.append(Path(args.profile))
    result={'pass':bool(valid and abs(rows[-1]['time_s']-.35)<1e-8),'synthetic_control':True,
        'physics_hz':hz,
        'material_edge_range_m':[min(r[0] for r in ranges),max(r[1] for r in ranges)],
        'task_approved':False,'source_approved':False,'self_contact_test':args.support=='self','positions_prescribed':False,
        'native_surface_bodies':1 if whole is not None else len(views),
        'disconnected_components_in_one_body':2 if whole is not None else None,
        'opposed_surface_normals':args.support=='self',
        'backend':{'name':'isaacsim','version':importlib.metadata.version('isaacsim'),'solver':'PGS'},
        'parameters':vars(args),'settings':settings,'load_force_N':load_force,'load_ratio_to_weight':.2,
        'peak_lower_clamp_force_N':peak_clamp,'material_native_readback':True,
        'native_friction_initial':native_friction.tolist(),'native_friction_final':final_friction.tolist(),'rows':rows,
        'load_end':end,'native_physics_errors':errors.report(),
        'input_sha256':{str(q.resolve()):hashlib.sha256(q.read_bytes()).hexdigest() for q in inputs},
        'scope':'Synthetic collision response only; matched coefficient controls required. Same-body mode intentionally has two disconnected components to isolate self-contact, not a deliverable connected cloth. No wet adhesion, material identification or task approval.'}
    np.savez_compressed(out/'states.npz',**{name:np.array(value) for name,value in states.items()})
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('PAIR_FRICTION_RESULT',result['pass'],args.support,args.friction,end,flush=True)
except Exception as error:
    (out/'error.json').write_text(json.dumps({'pass':False,'error':str(error),'traceback':traceback.format_exc()},indent=2)+'\n');raise
finally:
    errors.close();app.close()
