"""Synthetic native membrane traction control; never source or task evidence.

Opposite finite edge tractions load a free flat strip. Compare its interior
extension with small-strain uniaxial elasticity, and retain the whole transient.
No positions, kinematic constraints or material fit are imposed after reset.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import traceback

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',required=True)
parser.add_argument('--spacing-m',type=float,default=.001)
parser.add_argument('--physics-hz',type=int,choices=[240,480,960,1920,3840,7680,15360],default=1920)
parser.add_argument('--width-m',type=float,default=.04)
parser.add_argument('--length-m',type=float,default=.08)
parser.add_argument('--reference-height-m',type=float,default=.3)
parser.add_argument('--force-route',choices=['rigid_boundary','nodal_velocity'],default='rigid_boundary')
parser.add_argument('--position-iterations',type=int,default=128)
parser.add_argument('--youngs-modulus',type=float,default=4e6)
parser.add_argument('--traction-N',type=float,default=.02)
parser.add_argument('--seconds',type=float,default=1.)
parser.add_argument('--physical-priors',help='Optional declared material-prior plan, independent of any source-pose fit')
parser.add_argument('--physical-link',help='Body key in the material-prior plan')
parser.add_argument('--case',choices=['nominal','low','high'],default='nominal')
args=parser.parse_args()
if (not 0.00005<=args.spacing_m<=.004 or not 32<=args.position_iterations<=255
        or not 2*args.spacing_m<=args.width_m<=.04 or not 8*args.spacing_m<=args.length_m<=.08 or not 0<=args.reference_height_m<=2
        or not 1e5<=args.youngs_modulus<=1e10 or not 1e-7<=args.traction_N<=.05 or not .5<=args.seconds<=3.):
    parser.error('Bounded synthetic material, force and solver settings required')
out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
shutil.copy2(__file__,out/'source.py');(out/'protocol.json').write_text(json.dumps(vars(args),indent=2)+'\n')

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402
app=SimulationApp({'headless':True})

import numpy as np  # noqa: E402
import torch  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from pxr import PhysxSchema,UsdGeom,UsdPhysics,Vt  # noqa: E402
from calibrated_scene_reconstruction.assets.cloth_rest import triangulate_grid  # noqa: E402
from calibrated_scene_reconstruction.assets.cloth_metric import MaterialMetric  # noqa: E402
from calibrated_scene_reconstruction.sim.surface_cloth import author,SurfaceView  # noqa: E402
from calibrated_scene_reconstruction.sim.physics_errors import PhysicsErrors  # noqa: E402
from calibrated_scene_reconstruction.tasks.rigid.task_checks import as_numpy  # noqa: E402

errors=PhysicsErrors(out/'native_physics_errors.json')
try:
    dt=1/args.physics_hz
    world=World(stage_units_in_meters=1.,physics_dt=dt,device='cuda:0',backend='torch');errors.guard_steps(world)
    pc=world.get_physics_context();pc.enable_gpu_dynamics(True);pc.set_broadphase_type('GPU');pc.set_solver_type('PGS')
    stage=omni.usd.get_context().get_stage();scene=next(p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene))
    UsdPhysics.Scene(scene).GetGravityMagnitudeAttr().Set(0.)
    PhysxSchema.PhysxSceneAPI.Apply(scene).CreateEnableExternalForcesEveryIterationAttr(False)
    length,width=args.length_m,args.width_m
    x=np.linspace(-width/2,width/2,round(width/args.spacing_m)+1)
    y=np.linspace(-length/2,length/2,round(length/args.spacing_m)+1)
    xx,yy=np.meshgrid(x,y);rest=np.c_[xx.ravel(),yy.ravel(),np.zeros(xx.size)];points=rest+[0,0,args.reference_height_m]
    faces,chart,_=triangulate_grid(points,rest,len(y),len(x));part={'V':points,'F':faces,'cloth_rest':chart}
    mesh=UsdGeom.Mesh.Define(stage,'/World/ControlSheet');mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype('float32')))
    mesh.CreateFaceVertexCountsAttr([3]*len(faces));mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist());mesh.CreateSubdivisionSchemeAttr('none')
    settings={'youngs_modulus':args.youngs_modulus,'poissons_ratio':.2,'thickness':.0004,'flexural_rigidity':1e-5,
              'areal_density':.2,'elasticity_damping':.01,'linear_damping':.02,'friction':0.,
              'contact_offset':.0003,'rest_offset':.0002,'self_collision_filter_distance':.0004,
              'solver_position_iterations':args.position_iterations}
    prior_input=None
    if args.physical_priors:
        if not args.physical_link:raise ValueError('Explicit physical material link required')
        from calibrated_scene_reconstruction.assets.physical_priors import sample
        prior_input=out/'physics_priors_input.json';shutil.copy2(args.physical_priors,prior_input)
        material=sample(json.loads(prior_input.read_text()),args.case)['bodies'][args.physical_link]['parameters']
        for name in ['youngs_modulus','poissons_ratio','thickness','flexural_rigidity','areal_density','elasticity_damping','linear_damping']:
            settings[name]=material[name]
        if 'bend_damping' in material:settings['bend_damping']=material['bend_damping']
        h=settings['thickness'];settings.update(contact_offset=.75*h,rest_offset=h/2,self_collision_filter_distance=h)
    author(stage,mesh,part,settings)
    boundary=[]
    if args.force_route=='rigid_boundary':
        import trimesh
        from isaacsim.core.prims import SingleRigidPrim
        from calibrated_scene_reconstruction.sim.usd import set_pose
        from calibrated_scene_reconstruction.sim.surface_attachment import author_many
        for index,sign in enumerate((-1.,1.)):
            root='/World/Boundary_'+str(index);centre=np.array([0.,sign*(length/2+.002),args.reference_height_m])
            solid=trimesh.creation.box([width,.004,.004]);body=UsdGeom.Xform.Define(stage,root).GetPrim();set_pose(body,centre)
            UsdPhysics.RigidBodyAPI.Apply(body);UsdPhysics.MassAPI.Apply(body).CreateMassAttr(5e-5)
            rb=PhysxSchema.PhysxRigidBodyAPI.Apply(body);rb.CreateLinearDampingAttr(3.);rb.CreateAngularDampingAttr(3.)
            rb.CreateSolverPositionIterationCountAttr(args.position_iterations);rb.CreateSolverVelocityIterationCountAttr(1)
            rb.CreateSleepThresholdAttr(0.);rb.CreateStabilizationThresholdAttr(0.)
            shape=UsdGeom.Mesh.Define(stage,root+'/collider');shape.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(solid.vertices,np.float32)))
            shape.CreateFaceVertexCountsAttr([3]*len(solid.faces));shape.CreateFaceVertexIndicesAttr(np.asarray(solid.faces).ravel().tolist())
            UsdPhysics.CollisionAPI.Apply(shape.GetPrim());UsdPhysics.MeshCollisionAPI.Apply(shape.GetPrim()).CreateApproximationAttr('convexHull')
            ids=np.flatnonzero(abs(rest[:,1]-sign*length/2)<1e-8)
            author_many(stage,[{'path':root+'_Bond','surface':'/World/ControlSheet','rigid':root,'points':points,'faces':faces,
                'indices':ids,'rigid_geometry':{'V':solid.vertices,'F':solid.faces},'backend':'partitioned',
                'basis':'Finite native dynamic loading fixture; no world pin or prescribed material trajectory.'}])
            rigid=world.scene.add(SingleRigidPrim(root,name='boundary_'+str(index)));boundary.append((rigid,sign))
    stage.GetRootLayer().Export(str(out/'fixture.usda'));world.reset()
    view=SurfaceView(stage,'/World/ControlSheet',part)
    native_material=view.owner.create_deformable_material_view('/World/ControlSheet_surface_fem_material')
    material_readback={'youngs_modulus':as_numpy(native_material.get_youngs_modulus()).tolist(),
        'poissons_ratio':as_numpy(native_material.get_poissons_ratio()).tolist()}
    (out/'material_readback.json').write_text(json.dumps(material_readback,indent=2)+'\n')
    if args.force_route=='nodal_velocity':view.require_validated_force_profile()
    masses=as_numpy(view.get_particle_masses())[0]
    lower=np.flatnonzero(rest[:,1]<y[0]+1e-8);upper=np.flatnonzero(rest[:,1]>y[-1]-1e-8)
    # Distributed edge traction uses physical trapezoid line weights, not
    # vertex count, so endpoints own half of an interior boundary segment.
    line_weights=np.ones(len(x));line_weights[[0,-1]]=.5;line_weights/=line_weights.sum()
    ids=np.r_[lower,upper];weights=np.r_[-line_weights,line_weights]
    low_band=np.flatnonzero(abs(rest[:,1]+length/4)<args.spacing_m*.6)
    high_band=np.flatnonzero(abs(rest[:,1]-length/4)<args.spacing_m*.6)
    reference_length=rest[high_band,1].mean()-rest[low_band,1].mean()
    expected=args.traction_N/(settings['youngs_modulus']*settings['thickness']*width)
    metric=MaterialMetric(points,faces,chart,max_edge_strain=.05,max_area_change=.03)
    states=[];rows=[]
    def record(time):
        p=as_numpy(view.get_world_positions())[0].copy();v=as_numpy(view.get_velocities())[0]
        direction=p[high_band].mean(0)-p[low_band].mean(0)
        strain=float(np.linalg.norm(direction)/reference_length-1)
        rows.append({'time_s':time,'interior_axial_strain':strain,'material':metric.check(p),
                     'momentum_Ns':(v*masses[:,None]).sum(0).tolist(),
                     'kinetic_energy_J':float(.5*np.sum(masses[:,None]*v**2))})
        states.append(p);(out/'progress.json').write_text(json.dumps(rows[-1],indent=2)+'\n')
    record(0.)
    for step in range(round(args.seconds/dt)):
        load=args.traction_N*min(1.,step*dt/.25)
        if args.force_route=='rigid_boundary':
            for rigid,sign in boundary:
                position=as_numpy(rigid.get_world_pose()[0])
                rigid._rigid_prim_view.apply_forces_and_torques_at_pos(
                    forces=torch.tensor([[0.,sign*load,0.]],device='cuda:0'),
                    positions=torch.as_tensor(position[None],dtype=torch.float32,device='cuda:0'),is_global=True)
        else:
            force=np.zeros((len(ids),3));force[:,1]=load*weights
            velocity=view.get_velocities().clone();velocity[0,ids]+=torch.as_tensor(force*dt/masses[ids,None],dtype=velocity.dtype,device=velocity.device)
            view.set_velocities(velocity)
        world.step(render=False)
        if (step+1)%round(.025/dt)==0:record((step+1)*dt)
    errors.check();late=[r['interior_axial_strain'] for r in rows if r['time_s']>=args.seconds-.2]
    median=float(np.median(late));relative=abs(median/expected-1)
    result={'pass':bool(relative<.2 and all(r['material']['pass'] for r in rows)),
            'synthetic_control':True,'not_task_or_source_evidence':True,'positions_prescribed':False,
            'backend':{'name':'isaacsim','version':importlib.metadata.version('isaacsim'),'solver':'PGS'},
            'expected_small_strain':expected,'observed_late_median_strain':median,
            'relative_constitutive_error':relative,'late_strain_range':[min(late),max(late)],
            'settings':settings,'node_count':len(points),'parameters':vars(args),'rows':rows,
            'native_physics_errors':errors.report(),
            'input_sha256':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [Path(__file__),Path('src/calibrated_scene_reconstruction/sim/surface_cloth.py'),*([prior_input] if prior_input else [])]},
            'scope':'Numerical traction-response screen only; examine settling, boundary and discretization errors before adopting a solver profile or inferring any object material.'}
    np.savez_compressed(out/'states.npz',points=states,rest=rest,faces=faces,masses=masses)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('SURFACE_RESPONSE_RESULT',json.dumps({k:v for k,v in result.items() if k not in ['rows','input_sha256','native_physics_errors']}),flush=True)
except Exception as error:
    (out/'error.json').write_text(json.dumps({'pass':False,'error':str(error),'traceback':traceback.format_exc()},indent=2)+'\n');raise
finally:
    errors.close();app.close()
