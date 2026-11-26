"""Native surface-FEM mass and force regression; synthetic, never task evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import traceback
import shutil

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('case',choices=['initial','impulse','collision'])
parser.add_argument('--output',required=True)
parser.add_argument('--dt',type=float,default=1/1920)
parser.add_argument('--solver',choices=['PGS','TGS'],default='PGS')
parser.add_argument('--retained-fold',action='store_true',help='Isometric natural-curvature mapping control')
parser.add_argument('--origin',type=float,nargs=3,default=[1.,2.,.1])
parser.add_argument('--material-sheet',help='Optional flat material NPZ for an actual mesh-resolution control; no task actors')
parser.add_argument('--material-link',help='Cloth rest entry in the material-sheet NPZ')
parser.add_argument('--physical-priors',help='Optional inferred material prior plan for a matched control')
parser.add_argument('--physical-link',help='Body key in the material prior plan')
parser.add_argument('--position-iterations',type=int,default=128)
parser.add_argument('--contact-padding-m',type=float,default=0.,help='Explicit numerical contact margin, separate from material thickness')
parser.add_argument('--contact-only-material',action='store_true',
                    help='Diagnostic integration/contact transport with native elastic forces disabled')
parser.add_argument('--diagnostic-shell',action='store_true',
                    help='Synthetic transport control only: implicit continuum shell over native triangle contacts')
parser.add_argument('--shell-bending-viscosity-s',type=float,default=0.)
parser.add_argument('--shell-membrane-viscosity-s',type=float,default=0.)
parser.add_argument('--diagnostic-viscosity',action='store_true',
                    help='Synthetic internal viscosity before unchanged native elastic integration')
args=parser.parse_args()
if not 1/4000 <= args.dt <= 1/60:parser.error('dt outside diagnostic range')
if args.diagnostic_shell and not args.contact_only_material:
    parser.error('Diagnostic shell requires native elastic forces explicitly disabled')
import math
if any(not math.isfinite(v) or not 0<=v<=1 for v in [args.shell_bending_viscosity_s,args.shell_membrane_viscosity_s]):
    parser.error('Diagnostic constitutive viscosity times must be in [0,1] seconds')
if args.diagnostic_viscosity and (args.diagnostic_shell or args.contact_only_material):
    parser.error('Viscosity-only control must preserve native elasticity')
if not (args.diagnostic_shell or args.diagnostic_viscosity) and (args.shell_bending_viscosity_s or args.shell_membrane_viscosity_s):
    parser.error('Explicit synthetic shell is required for constitutive viscosity')

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402
app=SimulationApp({'headless':True})

import numpy as np  # noqa: E402
import torch  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleRigidPrim  # noqa: E402
from pxr import PhysxSchema,UsdGeom,UsdPhysics,Vt  # noqa: E402
from calibrated_scene_reconstruction.sim.surface_cloth import author,SurfaceView  # noqa: E402
from calibrated_scene_reconstruction.sim.usd import set_pose  # noqa: E402
from calibrated_scene_reconstruction.tasks.rigid.task_checks import as_numpy  # noqa: E402

out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
shutil.copy2(__file__,out/'source.py');(out/'protocol.json').write_text(json.dumps(vars(args),indent=2)+'\n')
from calibrated_scene_reconstruction.sim.physics_errors import PhysicsErrors  # noqa: E402
physics_errors=PhysicsErrors(out/'native_physics_errors.json')
import faulthandler  # noqa: E402
stack_log=(out/'python_stacks.log').open('w')
faulthandler.dump_traceback_later(60,repeat=True,file=stack_log)
try:
    world=World(stage_units_in_meters=1.,physics_dt=args.dt,device='cuda:0',backend='torch')
    physics_errors.guard_steps(world)
    pc=world.get_physics_context();pc.enable_gpu_dynamics(True);pc.set_broadphase_type('GPU');pc.set_solver_type(args.solver)
    stage=omni.usd.get_context().get_stage()
    scene=next(p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene))
    UsdPhysics.Scene(scene).GetGravityMagnitudeAttr().Set(0.)
    PhysxSchema.PhysxSceneAPI.Apply(scene).CreateEnableExternalForcesEveryIterationAttr(args.solver=='TGS')
    x,y=np.meshgrid([0,.015,.2],[0,.035,.2]);rest=np.c_[x.ravel(),y.ravel(),np.zeros(9)]
    faces=[];springs={k:[] for k in ['stretch','shear','bend']}
    for j in range(3):
        for i in range(3):
            a=j*3+i
            if i<2:springs['stretch'].append([a,a+1])
            if j<2:springs['stretch'].append([a,a+3])
            if i<2 and j<2:
                faces.extend([[a,a+1,a+4],[a,a+4,a+3]])
                springs['shear'].append([a,a+4]);springs['bend'].append([a+1,a+3])
            if i==0:springs['bend'].append([a,a+2])
            if j==0:springs['bend'].append([a,a+6])
    faces=np.array(faces);chart={'points':rest.tolist(),'springs':springs}
    if args.material_sheet:
        if args.retained_fold:raise ValueError('Mesh-resolution control cannot also change the construction reference')
        shutil.copy2(args.material_sheet,out/'material_sheet_input.npz')
        with np.load(out/'material_sheet_input.npz') as data:
            charts=json.loads(str(data['cloth_rest_json']))
            name=args.material_link or (next(iter(charts)) if len(charts)==1 else None)
            if name not in charts:raise ValueError('Explicit material-sheet link required')
            chart=charts[name];rest=np.array(chart['points']);faces=data['faces'].copy()
    points=rest+[0,0,.3]
    part={'V':points,'F':faces,'cloth_rest':chart}
    if args.retained_fold:
        side=points[:,0]>.015;t=np.radians(140);x=points[side,0]-.015
        points[side,0]=.015+x*np.cos(t);points[side,2]+=x*np.sin(t)
        part['cloth_rest']['bend_reference']={'points':points.tolist(),'source':'Synthetic isometric fold',
                                               'basis':'Natural curvature with unchanged intrinsic membrane metric'}
    mesh=UsdGeom.Mesh.Define(stage,'/World/Sheet');mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr([3]*len(faces));mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist());mesh.CreateSubdivisionSchemeAttr('none')
    theta=.37;set_pose(mesh.GetPrim(),args.origin,[np.cos(theta/2),0,0,np.sin(theta/2)])
    settings={'youngs_modulus':4e5,'poissons_ratio':.2,'thickness':.00025,'flexural_rigidity':5e-8,
        'elasticity_damping':0.,'linear_damping':0.,'friction':0.,'areal_density':.2,
        'contact_offset':.00025,'rest_offset':.000125,'self_collision_filter_distance':.00025}
    if args.physical_priors:
        if not args.physical_link:raise ValueError('Explicit physical material key required')
        shutil.copy2(args.physical_priors,out/'physics_priors_input.json')
        values=json.loads((out/'physics_priors_input.json').read_text())['bodies'][args.physical_link]['parameters']
        for name in ['youngs_modulus','poissons_ratio','thickness','flexural_rigidity','areal_density']:
            settings[name]=values[name]['nominal']
        h=settings['thickness'];settings.update(contact_offset=.75*h,rest_offset=h/2,self_collision_filter_distance=h)
    settings['solver_position_iterations']=args.position_iterations
    if not np.isfinite(args.contact_padding_m) or not 0<=args.contact_padding_m<=settings['thickness']/4:
        raise ValueError('Numerical contact padding must be between zero and one quarter of material thickness')
    if args.contact_padding_m:
        settings['rest_offset']=settings['thickness']/2+args.contact_padding_m
        settings['contact_offset']=.75*settings['thickness']+1.5*args.contact_padding_m
        settings['self_collision_filter_distance']=settings['thickness']+2*args.contact_padding_m
    author(stage,mesh,part,settings)
    if args.contact_only_material:
        physical=stage.GetPrimAtPath('/World/Sheet_surface_fem_material')
        for name in ['omniphysics:youngsModulus','omniphysics:surfaceBendStiffness',
                     'physxDeformableMaterial:elasticityDamping','physxDeformableMaterial:bendDamping']:
            physical.GetAttribute(name).Set(0.)
    if args.case=='initial':
        mesh.CreateVelocitiesAttr(Vt.Vec3fArray.FromNumpy(np.tile([.1,0,0],(len(points),1)).astype(np.float32)))
    from calibrated_scene_reconstruction.assets.cloth_rest import area as material_area
    expected_mass=settings['areal_density']*material_area(rest,faces)
    if args.case=='collision':
        sphere=UsdGeom.Sphere.Define(stage,'/World/Ball');sphere.CreateRadiusAttr(.025)
        center=np.array([.1,.1,0.])
        if args.material_sheet:
            centres=rest[faces].mean(1);target=(rest.min(0)+rest.max(0))/2
            center[:2]=centres[np.argmin(np.linalg.norm(centres-target,axis=1)),:2]
        rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        center[:2]=center[:2]@rotation.T+args.origin[:2];center[2]=args.origin[2]+.35
        set_pose(sphere.GetPrim(),center)
        UsdPhysics.CollisionAPI.Apply(sphere.GetPrim());UsdPhysics.RigidBodyAPI.Apply(sphere.GetPrim())
        UsdPhysics.MassAPI.Apply(sphere.GetPrim()).CreateMassAttr(expected_mass)
        rb=PhysxSchema.PhysxRigidBodyAPI.Apply(sphere.GetPrim())
        rb.CreateSolverPositionIterationCountAttr(64)
        rb.CreateSleepThresholdAttr(0.);rb.CreateStabilizationThresholdAttr(0.)
        rb.CreateLinearDampingAttr(0.);rb.CreateAngularDampingAttr(0.)
    stage.GetRootLayer().Export(str(out/'fixture.usda'))
    world.reset();view=SurfaceView(stage,'/World/Sheet',part)
    material_view=view.owner.create_deformable_material_view('/World/Sheet_surface_fem_material')
    parsed_material={'native_material_count':material_view.count}
    if material_view.count:
        parsed_material.update(youngs_modulus=as_numpy(material_view.get_youngs_modulus()).tolist(),
                               poissons_ratio=as_numpy(material_view.get_poissons_ratio()).tolist())
    masses=as_numpy(view.get_particle_masses())[0]
    assert np.isclose(masses.sum(),expected_mass,rtol=1e-6)
    assert np.allclose(masses,masses[0]),'Fixture verifies native uniform masses on a nonuniform geometric mesh'
    shell_report=None
    if args.diagnostic_shell or args.diagnostic_viscosity:
        # This is the qualification fixture, not permission to use an
        # unverified nodal-force adapter in a task scene. Never register it as
        # a backend or bypass SurfaceView's task-side force-profile guard.
        from calibrated_scene_reconstruction.sim import discrete_bending,discrete_membrane,discrete_dissipation,constitutive_damping
        from calibrated_scene_reconstruction.assets import cloth_projection
        from calibrated_scene_reconstruction.assets.cloth_curvature import adjacency,measured
        sources={}
        for module in [discrete_bending,discrete_membrane,discrete_dissipation,cloth_projection,constitutive_damping]:
            target=out/Path(module.__file__).name;shutil.copy2(module.__file__,target)
            sources[target.name]=hashlib.sha256(target.read_bytes()).hexdigest()
        natural=np.zeros(len(adjacency(faces)[0]))
        if chart.get('bend_reference'):
            natural=np.radians(measured(chart['bend_reference']['points'],faces)[1])
        bend=discrete_bending.DiscreteBending(rest,faces,natural,settings['flexural_rigidity'],device='cuda:0')
        membrane=discrete_membrane.DiscreteMembrane(rest,faces,
            **{k:settings[k] for k in ['youngs_modulus','poissons_ratio','thickness']},device='cuda:0')
        shell_report={'synthetic_control_only':True,'task_adapter_authorized':False,
                      'positions_written':False,'steps':0,'all_nonlinear_steps_converged':True,
                      'native_elasticity_preserved':args.diagnostic_viscosity,
                      'viscosity_times_s':{'bending':args.shell_bending_viscosity_s,'membrane':args.shell_membrane_viscosity_s},
                      'maximum_internal_linear_impulse_Ns':0.,'sources_sha256':sources}
        native_step=world.step
        def coupled_step(*step_args,**step_kwargs):
            terms=[(bend,args.shell_bending_viscosity_s),(membrane,args.shell_membrane_viscosity_s)]
            if args.diagnostic_viscosity:
                velocity,control=constitutive_damping.velocity_step(view.get_world_positions()[0],
                    view.get_velocities()[0],masses,world.get_physics_dt(),terms)
            else:
                velocity,control=bend.velocity_step(view.get_world_positions()[0],view.get_velocities()[0],
                    masses,world.get_physics_dt(),membrane=membrane,linear_solver='sparse_direct',
                    tolerance=1e-7,nonlinear_backtracking=True,nonlinear_iterations=50,viscosity_terms=terms)
            if args.diagnostic_shell and not control.get('nonlinear_converged',False):
                (out/'nonlinear_failure.json').write_text(json.dumps({'step':shell_report['steps'],**control},indent=2)+'\n')
                raise ValueError('Synthetic coupled transport failed nonlinear convergence')
            shell_report['maximum_internal_linear_impulse_Ns']=max(
                shell_report['maximum_internal_linear_impulse_Ns'],float(np.linalg.norm(control['linear_impulse_Ns'])))
            view.set_velocities(velocity.to(dtype=torch.float32)[None].contiguous())
            shell_report['steps']+=1
            if shell_report['steps']%10==0:print('SHELL_TRANSPORT_STEP',shell_report['steps'],flush=True)
            return native_step(*step_args,**step_kwargs)
        world.step=coupled_step
    trace=[];cloth_trace=[];rigid_trace=[]
    maximum_speed=0.
    if args.case in {'initial','impulse'}:
        initial_points=as_numpy(view.get_world_positions())[0].copy()
        impulse=np.array([expected_mass*(.1 if args.case=='initial' else .01),0,0])
        excitation=None
        if args.case=='impulse':
            velocity=view.get_velocities().clone();velocity.zero_()
            # A fixed impulse on one node exceeds the native speed cap after
            # mesh refinement. Use a resolved material patch at bounded speed;
            # otherwise velocity clipping masquerades as failed mass transport.
            count=max(1,int(np.ceil(.1*len(rest))))
            ids=np.argsort(np.linalg.norm(rest-rest.mean(0),axis=1))[:count]
            delta=impulse/float(masses[ids].sum())
            cap=float(mesh.GetPrim().GetAttribute('physxDeformableBody:maxLinearVelocity').Get())
            if not 0<float(np.linalg.norm(delta))<.1*cap:
                raise ValueError('Impulse excitation exceeds the resolved native speed budget')
            velocity[0,torch.as_tensor(ids,device=velocity.device)]+=torch.as_tensor(delta,dtype=velocity.dtype,device=velocity.device)
            view.set_velocities(velocity)
            excitation={'nodes':ids.tolist(),'impulse_Ns':impulse.tolist(),
                        'maximum_imposed_speed_m_s':float(np.linalg.norm(delta)),
                        'native_speed_cap_m_s':cap,'positions_written':False}
        baseline=impulse
        step_count=round(.125/args.dt)
        for step in range(step_count):
            world.step(render=False)
            velocities=as_numpy(view.get_velocities())[0]
            maximum_speed=max(maximum_speed,float(np.linalg.norm(velocities,axis=1).max()))
            momentum=(velocities*masses[:,None]).sum(0)
            trace.append(momentum)
        error=np.linalg.norm(np.array(trace)-baseline,axis=1)/np.linalg.norm(baseline)
        displacement=(as_numpy(view.get_world_positions())[0]-initial_points).T@masses/masses.sum()
        expected_displacement=impulse/expected_mass*(step_count*world.get_physics_dt())
        movement_error=np.linalg.norm(displacement-expected_displacement)/np.linalg.norm(expected_displacement)
        result={'pass':bool(error.max()<.03 and movement_error<.05),'maximum_relative_momentum_error':float(error.max()),
                'excitation':excitation,
                'final_relative_momentum_error':float(error[-1]),
                'center_of_mass_displacement_m':displacement.tolist(),'expected_displacement_m':expected_displacement.tolist(),
                'relative_center_motion_error':float(movement_error),
                'scope':'Native uniform mass on a nonuniform geometric mesh; area-lumped masses would fail this impulse/contact control.'}
    else:
        ball=SingleRigidPrim('/World/Ball');ball.initialize();ball_mass=float(as_numpy(ball.get_mass()))
        ball.set_linear_velocity(torch.tensor([0,0,-.2],dtype=torch.float32,device=world.device))
        baseline=np.array([0,0,-.2*ball_mass]);observations=[]
        for step in range(round(.5/args.dt)):
            world.step(render=False)
            velocities=as_numpy(view.get_velocities())[0]
            maximum_speed=max(maximum_speed,float(np.linalg.norm(velocities,axis=1).max()))
            cloth_momentum=(velocities*masses[:,None]).sum(0)
            rigid_momentum=as_numpy(ball.get_linear_velocity())*ball_mass
            trace.append(cloth_momentum+rigid_momentum)
            cloth_trace.append(cloth_momentum);rigid_trace.append(rigid_momentum)
            observations.append([np.linalg.norm(cloth_momentum),np.linalg.norm(2*cloth_momentum+rigid_momentum-baseline)])
        error=np.linalg.norm(np.array(trace)-baseline,axis=1)/np.linalg.norm(baseline)
        transferred=max(o[0] for o in observations)/np.linalg.norm(baseline)
        wrong_scale_error=max(o[1] for o in observations)/np.linalg.norm(baseline)
        result={'pass':bool(error.max()<.05 and transferred>.1 and wrong_scale_error>.1),
            'maximum_relative_total_momentum_error':float(error.max()),'transferred_momentum_fraction':float(transferred),
            'final_relative_total_momentum_error':float(error[-1]),
            'wrong_mass_scale_negative_control_relative_error':float(wrong_scale_error),
            'rigid_mass_readback_kg':ball_mass,'scope':'Known rigid-body mass and native collision verify authored FEM mass scale; no external gravity, damping or fixture forces.'}
    np.savez_compressed(out/'momentum.npz',trace=trace,cloth_momentum=cloth_trace,rigid_momentum=rigid_trace,
                        masses=masses,rest=rest,faces=faces,final_points=as_numpy(view.get_world_positions())[0],
                        final_velocities=as_numpy(view.get_velocities())[0])
    result.update(case=args.case,synthetic_fixture=True,not_tissue_task_evidence=True,mass_source=view.mass_source,
        solver=args.solver,contact_only_material=args.contact_only_material,
        experimental_shell=shell_report,
        parsed_material=parsed_material,physics_dt_s=args.dt,root_translation_m=args.origin,
        native_node_count=len(masses),source_sha256=hashlib.sha256(Path('src/calibrated_scene_reconstruction/sim/surface_cloth.py').read_bytes()).hexdigest())
    speed_cap=float(mesh.GetPrim().GetAttribute('physxDeformableBody:maxLinearVelocity').Get())
    result['velocity_budget']={'maximum_observed_m_s':maximum_speed,'native_cap_m_s':speed_cap,
                              'cap_reached':bool(maximum_speed>=.999*speed_cap)}
    result['pass']=bool(result['pass'] and not result['velocity_budget']['cap_reached'])
    result['material_control']={'settings':settings,'authored_mass_kg':expected_mass,
        'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                        [out/'material_sheet_input.npz',out/'physics_priors_input.json'] if p.exists()}}
    result['probe_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    physics_errors.check();result['native_physics_errors']=physics_errors.report()
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print('SURFACE_CLOTH_RESULT',json.dumps(result),flush=True)
except Exception as error:
    (out/'error.json').write_text(json.dumps({'pass':False,'error':str(error),'traceback':traceback.format_exc()},indent=2)+'\n');raise
finally:
    faulthandler.cancel_dump_traceback_later()
    stack_log.close()
    physics_errors.close()
    app.close()
