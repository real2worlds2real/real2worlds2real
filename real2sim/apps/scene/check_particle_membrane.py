"""Independent native particle membrane traction control; not a task asset.

Balanced edge tractions and mass-proportional test-fixture damping approximate
quasistatic loading. No particle positions, pins or infinite-force constraints
are imposed after reset. Distance bending is not continuum bending.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import traceback

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',required=True);p.add_argument('--spacing-m',type=float,default=.00025)
p.add_argument('--youngs-modulus',type=float,default=4e6)
p.add_argument('--thickness-m',type=float,default=.0004)
p.add_argument('--bend-stiffness',type=float,default=.8)
p.add_argument('--spring-scale',type=float,default=1.,help='Independent native membrane calibration hypothesis; continuum target and bending stay unchanged')
p.add_argument('--spring-model',choices=['balanced_square_lattice','triangle_surface'],default='balanced_square_lattice',
               help='Independent numerical topology hypothesis; changing it requires fresh material and contact controls')
p.add_argument('--physics-hz',type=int,choices=[120,240,480,960,1920,3840,7680,15360,30720],default=960,
               help='Independent native convergence experiment; higher clocks do not qualify a production profile')
p.add_argument('--linear-solver',choices=['cg','sparse_direct','cuda_direct'],default='cg')
p.add_argument('--nonlinear-iterations',type=int,default=1)
p.add_argument('--no-self-collision',action='store_true',help='Synthetic membrane isolation only; never a task setting')
p.add_argument('--constitutive',choices=['native_springs','implicit_shell'],default='native_springs',
               help='Experimental internal shell impulses retain Isaac integration/contact; no task authorization')
args=p.parse_args()
if args.spring_scale!=1. and args.constitutive!='native_springs':p.error('Spring calibration requires native springs')
if not .0002<=args.spacing_m<=.001:p.error('Resolved material spacing required')
if not 1<=args.nonlinear_iterations<=30 or (args.nonlinear_iterations>1 and args.constitutive!='implicit_shell'):
    p.error('Nonlinear iteration budget in1..30 requires the explicit internal shell')
out=Path(args.output);out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,out/'source.py')

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402
app=SimulationApp({'headless':True})

import numpy as np  # noqa: E402
import torch  # noqa: E402
import omni.usd  # noqa: E402
import omni.physics.tensors as tensors  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from omni.physx.scripts import particleUtils  # noqa: E402
from pxr import UsdGeom,UsdPhysics,PhysxSchema,UsdShade,Vt  # noqa: E402
from calibrated_scene_reconstruction.assets.cloth_rest import triangulate_grid,constraints,area  # noqa: E402
from calibrated_scene_reconstruction.assets.particle_lattice import chart,spring_proposal,calibrated_membrane_proposal  # noqa: E402
from calibrated_scene_reconstruction.assets.cloth_metric import MaterialMetric  # noqa: E402
from calibrated_scene_reconstruction.assets.cloth_mass import authoring_mass  # noqa: E402
from calibrated_scene_reconstruction.sim.physics_errors import PhysicsErrors  # noqa: E402
from calibrated_scene_reconstruction.tasks.rigid.task_checks import as_numpy  # noqa: E402

errors=PhysicsErrors(out/'native_physics_errors.json')
control_files=list(map(Path,[__file__,'src/calibrated_scene_reconstruction/assets/particle_lattice.py','src/calibrated_scene_reconstruction/assets/cloth_rest.py',
    'src/calibrated_scene_reconstruction/assets/cloth_mass.py','src/calibrated_scene_reconstruction/assets/cloth_metric.py']))
if args.constitutive=='implicit_shell':
    control_files+=list(map(Path,['src/calibrated_scene_reconstruction/sim/discrete_bending.py','src/calibrated_scene_reconstruction/sim/discrete_membrane.py',
        'src/calibrated_scene_reconstruction/assets/cloth_projection.py','src/calibrated_scene_reconstruction/assets/cloth_curvature.py']))
if args.linear_solver=='cuda_direct':
    from calibrated_scene_reconstruction.sim import cudss
    control_files+=[Path(cudss.__file__),cudss.library_path()]
input_hashes={str(q.resolve()):hashlib.sha256(q.read_bytes()).hexdigest() for q in control_files}
(out/'dependencies').mkdir()
for q in control_files:
    if q.suffix=='.py':shutil.copy2(q,out/'dependencies'/q.name)
try:
    hz=args.physics_hz;dt=1/hz;length=.04;width=.02;density=.2;force=.1;fixture_damping=300.
    proposal=spring_proposal(args.youngs_modulus,args.thickness_m,args.bend_stiffness)
    proposal['spring_model']=args.spring_model
    proposal['analytic_mapping_applies_to_selected_model']=args.spring_model=='balanced_square_lattice'
    if args.spring_scale!=1.:proposal=calibrated_membrane_proposal(proposal,args.spring_scale)
    spring_settings=proposal['settings'] if args.constitutive=='native_springs' else {
        'stretch_stiffness':0.,'shear_stiffness':0.,'bend_stiffness':0.}
    x=np.linspace(-width/2,width/2,round(width/args.spacing_m)+1)
    y=np.linspace(-length/2,length/2,round(length/args.spacing_m)+1)
    xx,yy=np.meshgrid(x,y);rest=np.c_[xx.ravel(),yy.ravel(),np.zeros(xx.size)];points=rest+[0,0,.3]
    faces,_,_=triangulate_grid(points,rest,len(y),len(x));material,chart_report=chart(rest,faces,model=args.spring_model)
    pairs,lengths,stiffness,_=constraints(material,spring_settings)
    world=World(stage_units_in_meters=1.,physics_dt=dt,device='cuda:0',backend='torch');errors.guard_steps(world)
    pc=world.get_physics_context();pc.enable_gpu_dynamics(True);pc.set_broadphase_type('GPU');pc.set_solver_type('PGS')
    stage=omni.usd.get_context().get_stage();scene=next(q for q in stage.Traverse() if q.IsA(UsdPhysics.Scene))
    UsdPhysics.Scene(scene).GetGravityMagnitudeAttr().Set(0.)
    PhysxSchema.PhysxSceneAPI.Apply(scene).CreateEnableExternalForcesEveryIterationAttr(False)
    system=particleUtils.add_physx_particle_system(stage,'/World/ParticleSystem',simulation_owner=scene.GetPath(),
        contact_offset=.001,rest_offset=.0004,solid_rest_offset=.0004,particle_contact_offset=.0005,
        solver_position_iterations=128,enable_ccd=True,max_neighborhood=256,max_depenetration_velocity=.1)
    particleUtils.add_pbd_particle_material(stage,'/World/ParticleMaterial',friction=.55,damping=0.,drag=0.,lift=0.)
    UsdShade.MaterialBindingAPI.Apply(system.GetPrim()).Bind(UsdShade.Material.Get(stage,'/World/ParticleMaterial'),UsdShade.Tokens.weakerThanDescendants,'physics')
    normals=np.zeros_like(points,dtype='float32');normals[:,2]=1.
    mesh=UsdGeom.Mesh.Define(stage,'/World/Strip')
    particleUtils.add_physx_particle_cloth_with_constraints(stage,mesh.GetPath(),system.GetPath(),
        Vt.Vec3fArray.FromNumpy(points.astype('float32')),Vt.Vec3fArray.FromNumpy(normals),
        Vt.Vec3fArray.FromNumpy(rest.astype('float32')),Vt.Vec3fArray.FromNumpy(np.zeros_like(normals)),
        Vt.IntArray(faces.ravel().tolist()),Vt.Vec2iArray.FromNumpy(pairs.astype('int32')),
        Vt.FloatArray(stiffness),Vt.FloatArray([0. if args.constitutive=='implicit_shell' else 1e-12]*len(pairs)),Vt.FloatArray(lengths.tolist()),
        self_collision=not args.no_self_collision,self_collision_filter=True)
    cooking=authoring_mass(density*area(rest,faces),rest,faces)
    UsdPhysics.MassAPI.Apply(mesh.GetPrim()).CreateMassAttr(cooking['authoring_mass_kg'])
    stage.GetRootLayer().Export(str(out/'initial.usda'));world.reset()
    owner=tensors.create_simulation_view('torch');owner.set_subspace_roots('/')
    view=owner.create_particle_cloth_view('/World/Strip');index=torch.zeros(1,dtype=torch.int32,device='cuda:0')
    position=lambda:as_numpy(view.get_positions())[0].reshape(-1,3).copy()
    initial=position()
    if not np.allclose(initial,points,rtol=0,atol=2e-7):raise ValueError('Native node order or initialization changed')
    masses=as_numpy(view.get_masses())[0].copy()
    if not np.isclose(masses.sum(),density*area(rest,faces),rtol=1e-4):raise ValueError('Native mass differs from applied-force calculation')
    native_stiffness=as_numpy(view.get_spring_stiffness()).copy()
    native_damping=as_numpy(view.get_spring_damping()).copy()
    if (native_stiffness.size!=len(stiffness) or not np.allclose(np.sort(native_stiffness.ravel()),np.sort(stiffness),rtol=1e-5)
            or not np.allclose(native_damping,0. if args.constitutive=='implicit_shell' else 1e-12,rtol=1e-5,atol=1e-15)):
        raise ValueError('Actual native spring constants differ from the material proposal')
    shell=membrane=None;internal_reports=[];target_poisson=1/3
    if args.constitutive=='implicit_shell':
        from calibrated_scene_reconstruction.sim.discrete_bending import DiscreteBending
        from calibrated_scene_reconstruction.sim.discrete_membrane import DiscreteMembrane
        from calibrated_scene_reconstruction.assets.cloth_curvature import adjacency
        target_poisson=.2
        shell=DiscreteBending(rest,faces,np.zeros(len(adjacency(faces)[0])),1e-5,device='cuda:0')
        membrane=DiscreteMembrane(rest,faces,youngs_modulus=args.youngs_modulus,
            poissons_ratio=target_poisson,thickness=args.thickness_m,device='cuda:0')
    metric=MaterialMetric(points,faces,material,max_edge_strain=.03,max_area_change=.02)
    # Trapezoidal edge quadrature gives half weight to the two free corners.
    edge_weights=np.ones(len(x));edge_weights[[0,-1]]=.5;edge_weights/=edge_weights.sum()
    upper=np.flatnonzero(np.isclose(rest[:,1],y[-1]));lower=np.flatnonzero(np.isclose(rest[:,1],y[0]))
    expected=force/(width*proposal['target_membrane_Et_N_m']);rows=[];states=[]
    def record(t):
        actual=position();centred=actual-actual.mean(0)
        strain=np.linalg.lstsq(rest[:,:2],centred[:,:2],rcond=None)[0]-np.eye(2)
        row={'time_s':t,'axial_strain':float(strain[1,1]),'transverse_strain':float(strain[0,0]),
             'shear_strain':float(strain[0,1]+strain[1,0]),'material':metric.check(actual),
             'net_momentum_kg_m_s':(masses[:,None]*as_numpy(view.get_velocities())[0].reshape(-1,3)).sum(0).tolist()}
        rows.append(row);states.append(actual);(out/'progress.json').write_text(json.dumps(row,indent=2)+'\n')
    record(0.)
    for step in range(round(1.2*hz)):
        t=(step+1)*dt;load=force*min(1.,t/.4)
        f=np.zeros_like(points);f[upper,1]=load*edge_weights;f[lower,1]=-load*edge_weights
        velocity=view.get_velocities().reshape(1,-1,3).clone()
        # Explicit, finite viscous test-fixture damping; not claimed as cloth material.
        velocity*=np.exp(-fixture_damping*dt)
        velocity[0]+=torch.as_tensor(f*dt/masses[:,None],dtype=velocity.dtype,device=velocity.device)
        if shell is not None:
            updated,internal=shell.velocity_step(view.get_positions()[0].reshape(-1,3),velocity[0],masses,dt,
                membrane=membrane,tolerance=1e-7,linear_solver=args.linear_solver,nonlinear_iterations=args.nonlinear_iterations)
            velocity=updated.to(dtype=velocity.dtype)[None];internal_reports.append(internal)
        view.set_velocities(velocity.reshape(1,-1).contiguous(),index);world.step(render=False)
        if (step+1)%(hz//40)==0:record(t)
    errors.check();late=[r for r in rows if r['time_s']>=1.]
    axial=float(np.mean([r['axial_strain'] for r in late]));transverse=float(np.mean([r['transverse_strain'] for r in late]))
    error=abs(axial/expected-1);poisson=-transverse/axial if axial>0 else None
    settled=float(np.ptp([r['axial_strain'] for r in late])/expected)
    valid=bool(all(r['material']['pass'] for r in rows))
    nonlinear_converged=(all(r.get('nonlinear_converged') is True for r in internal_reports)
                         if args.nonlinear_iterations>1 else None)
    if nonlinear_converged is False:valid=False
    source_unchanged=all(hashlib.sha256(Path(q).read_bytes()).hexdigest()==digest for q,digest in input_hashes.items())
    if not source_unchanged:valid=False
    result={'pass':bool(valid and error<.2 and poisson is not None and abs(poisson-target_poisson)<.08 and settled<.05),
        'synthetic_control':True,'task_approved':False,'source_approved':False,'continuum_bending_qualified':False,
        'backend':{'name':'isaacsim','version':importlib.metadata.version('isaacsim'),'solver':'PGS','representation':'particle'},
        'parameters':vars(args),'proposal':proposal,'chart':chart_report,'physics_hz':hz,'position_iterations':128,
        'native_spring_settings':spring_settings,'target_poissons_ratio':target_poisson,
        'experimental_internal_shell':None if shell is None else {
            'production_backend_registered':False,'native_springs_disabled':True,'positions_written':False,
            'youngs_modulus_Pa':args.youngs_modulus,'poissons_ratio':target_poisson,'flexural_rigidity_Nm':1e-5,
            'nonlinear_all_steps_converged':nonlinear_converged,
            'maximum_nonlinear_gradient_residual':max((r.get('nonlinear_relative_gradient_residual',0.) for r in internal_reports),default=0.),
            'maximum_cg_iterations':max(r['cg_iterations'] for r in internal_reports),
            'maximum_relative_solver_residual':max(r['relative_solver_residual'] for r in internal_reports),
            'maximum_internal_linear_impulse_Ns':max(np.linalg.norm(r['linear_impulse_Ns']) for r in internal_reports),
            'maximum_internal_angular_impulse_Nms':max(np.linalg.norm(r['angular_impulse_Nms']) for r in internal_reports)},
        'native_mass_kg':float(masses.sum()),'node_count':len(points),'force_N_per_edge':force,
        'native_spring_readback':True,'self_collision_enabled':not args.no_self_collision,
        'test_fixture_damping_s_inv':fixture_damping,'spring_damping_kg_s':0. if shell is not None else 1e-12,
        'expected_axial_strain':expected,'observed_axial_strain':axial,'relative_axial_error':error,
        'observed_poissons_ratio':poisson,'late_range_relative_to_target':settled,'physical_pass':valid,
        'positions_prescribed':False,'fixed_nodes':0,'rows':rows,'native_physics_errors':errors.report(),
        'input_sha256':input_hashes,'source_unchanged_during_control':source_unchanged}
    np.savez_compressed(out/'states.npz',points=states,rest=rest,faces=faces,masses=masses)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('PARTICLE_MEMBRANE_RESULT',result['pass'],axial,poisson,error,settled,flush=True)
except Exception as error:
    (out/'error.json').write_text(json.dumps({'pass':False,'error':str(error),'traceback':traceback.format_exc()},indent=2)+'\n');raise
finally:
    errors.close();app.close()
