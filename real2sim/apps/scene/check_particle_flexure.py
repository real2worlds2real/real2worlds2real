"""Unforced native particle integration with experimental internal shell forces.

This qualifies only small-amplitude bending at its recorded numerical profile.
It is not a production backend, identified real material or task acceptance.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import traceback

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
p.add_argument('--physical-priors',required=True);p.add_argument('--physical-link',required=True)
p.add_argument('--case',choices=['low','nominal','high'],default='nominal')
p.add_argument('--spacing-m',type=float,default=.001)
p.add_argument('--physics-hz',type=int,choices=[120,240,480,960,1920],default=960)
p.add_argument('--linear-solver',choices=['cg','sparse_direct','cuda_direct'],default='cg')
p.add_argument('--nonlinear-iterations',type=int,default=1)
p.add_argument('--bending-viscosity-time-s',type=float,default=0.)
p.add_argument('--membrane-viscosity-time-s',type=float,default=0.)
args=p.parse_args()
if not .0005<=args.spacing_m<=.002:p.error('Bounded material spacing required')
if not 1<=args.nonlinear_iterations<=30:p.error('Bounded nonlinear iteration budget in1..30 required')
if not all(0<=v<=1 for v in [args.bending_viscosity_time_s,args.membrane_viscosity_time_s]):
    p.error('Explicit diagnostic viscosity/modulus ratios in seconds in [0,1] required; not Isaac damping rates')
out=Path(args.output);out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,out/'source.py')
shutil.copy2(args.physical_priors,out/'physics_priors_input.json')

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402
app=SimulationApp({'headless':True})

import numpy as np  # noqa: E402
import torch  # noqa: E402
import omni.usd  # noqa: E402
import omni.physics.tensors as tensors  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from omni.physx.scripts import particleUtils  # noqa: E402
from pxr import UsdGeom,UsdPhysics,PhysxSchema,UsdShade,Vt  # noqa: E402
from calibrated_scene_reconstruction.assets.physical_priors import sample  # noqa: E402
from calibrated_scene_reconstruction.assets.cloth_rest import triangulate_grid,constraints,area  # noqa: E402
from calibrated_scene_reconstruction.assets.cloth_curvature import adjacency  # noqa: E402
from calibrated_scene_reconstruction.assets.cloth_mass import authoring_mass  # noqa: E402
from calibrated_scene_reconstruction.assets.cloth_metric import MaterialMetric  # noqa: E402
from calibrated_scene_reconstruction.assets import flexure_reference  # noqa: E402
from calibrated_scene_reconstruction.sim.discrete_bending import DiscreteBending  # noqa: E402
from calibrated_scene_reconstruction.sim.discrete_membrane import DiscreteMembrane  # noqa: E402
from calibrated_scene_reconstruction.assets.deformable_release import modal_decay  # noqa: E402
from calibrated_scene_reconstruction.sim.physics_errors import PhysicsErrors  # noqa: E402
from calibrated_scene_reconstruction.tasks.rigid.task_checks import as_numpy  # noqa: E402

errors=PhysicsErrors(out/'native_physics_errors.json')
control_files=list(map(Path,[__file__,'src/calibrated_scene_reconstruction/sim/discrete_bending.py','src/calibrated_scene_reconstruction/sim/discrete_membrane.py',
    'src/calibrated_scene_reconstruction/sim/discrete_dissipation.py','src/calibrated_scene_reconstruction/assets/deformable_release.py',
    'src/calibrated_scene_reconstruction/assets/cloth_projection.py','src/calibrated_scene_reconstruction/assets/flexure_reference.py','src/calibrated_scene_reconstruction/assets/cloth_mass.py',
    'src/calibrated_scene_reconstruction/assets/cloth_curvature.py','src/calibrated_scene_reconstruction/assets/cloth_rest.py',str(out/'physics_priors_input.json')]))
if args.linear_solver=='cuda_direct':
    from calibrated_scene_reconstruction.sim import cudss
    control_files+=[Path(cudss.__file__),cudss.library_path()]
input_hashes={str(q.resolve()):hashlib.sha256(q.read_bytes()).hexdigest() for q in control_files}
(out/'dependencies').mkdir()
for q in control_files:
    if q.suffix in {'.py','.json'}:shutil.copy2(q,out/'dependencies'/q.name)
try:
    settings=sample(json.loads((out/'physics_priors_input.json').read_text()),args.case)['bodies'][args.physical_link]['parameters']
    hz=args.physics_hz;dt=1/hz;length=.08;width=.008
    x=np.linspace(-width/2,width/2,round(width/args.spacing_m)+1)
    s=np.linspace(0,length,round(length/args.spacing_m)+1)
    shape=flexure_reference.mode(s,length);shape-=np.polyval(np.polyfit(s,shape,1),s);shape/=abs(shape).max()
    z=.0002*shape
    if (abs(np.diff(z)/np.diff(s))>=.1).any():raise ValueError('Small-amplitude flexure required')
    axial=np.r_[0.,np.cumsum(np.sqrt(np.diff(s)**2-np.diff(z)**2))]
    xx,yy=np.meshgrid(x,s-length/2);rest=np.c_[xx.ravel(),yy.ravel(),np.zeros(xx.size)]
    points=rest.copy();points[:,1]=np.repeat(axial-axial.mean(),len(x));points[:,2]=np.repeat(z,len(x))
    faces,chart,_=triangulate_grid(points,rest,len(s),len(x))
    pairs,lengths,stiffness,_=constraints(chart,{k+'_stiffness':0. for k in ['stretch','shear','bend']})
    world=World(stage_units_in_meters=1.,physics_dt=dt,device='cuda:0',backend='torch');errors.guard_steps(world)
    pc=world.get_physics_context();pc.enable_gpu_dynamics(True);pc.set_broadphase_type('GPU');pc.set_solver_type('PGS')
    stage=omni.usd.get_context().get_stage();scene=next(q for q in stage.Traverse() if q.IsA(UsdPhysics.Scene))
    UsdPhysics.Scene(scene).GetGravityMagnitudeAttr().Set(0.)
    PhysxSchema.PhysxSceneAPI.Apply(scene).CreateEnableExternalForcesEveryIterationAttr(False)
    system=particleUtils.add_physx_particle_system(stage,'/World/ParticleSystem',simulation_owner=scene.GetPath(),
        contact_offset=.001,rest_offset=.0004,solid_rest_offset=.00042,particle_contact_offset=.0005,
        solver_position_iterations=128,max_neighborhood=256,max_depenetration_velocity=.1)
    particleUtils.add_pbd_particle_material(stage,'/World/ParticleMaterial',friction=.55,damping=0.,drag=0.,lift=0.)
    UsdShade.MaterialBindingAPI.Apply(system.GetPrim()).Bind(UsdShade.Material.Get(stage,'/World/ParticleMaterial'),UsdShade.Tokens.weakerThanDescendants,'physics')
    normals=np.zeros_like(points,dtype='float32');normals[:,2]=1.
    mesh=UsdGeom.Mesh.Define(stage,'/World/FreeStrip')
    particleUtils.add_physx_particle_cloth_with_constraints(stage,mesh.GetPath(),system.GetPath(),
        Vt.Vec3fArray.FromNumpy(points.astype('float32')),Vt.Vec3fArray.FromNumpy(normals),
        Vt.Vec3fArray.FromNumpy(rest.astype('float32')),Vt.Vec3fArray.FromNumpy(np.zeros_like(normals)),
        Vt.IntArray(faces.ravel().tolist()),Vt.Vec2iArray.FromNumpy(pairs.astype('int32')),
        Vt.FloatArray(stiffness),Vt.FloatArray([0.]*len(pairs)),Vt.FloatArray(lengths.tolist()),self_collision=True,self_collision_filter=True)
    total=settings['areal_density']*area(rest,faces);cooking=authoring_mass(total,rest,faces)
    UsdPhysics.MassAPI.Apply(mesh.GetPrim()).CreateMassAttr(cooking['authoring_mass_kg'])
    stage.GetRootLayer().Export(str(out/'initial.usda'));world.reset()
    owner=tensors.create_simulation_view('torch');owner.set_subspace_roots('/')
    view=owner.create_particle_cloth_view('/World/FreeStrip');index=torch.zeros(1,dtype=torch.int32,device='cuda:0')
    position=lambda:view.get_positions()[0].reshape(-1,3)
    initial=as_numpy(position()).copy();masses=as_numpy(view.get_masses())[0].copy()
    if (not np.allclose(initial,points,rtol=0,atol=1e-7) or not np.isclose(masses.sum(),total,rtol=1e-4)
            or np.any(as_numpy(view.get_spring_stiffness())!=0) or np.any(as_numpy(view.get_spring_damping())!=0)):
        raise ValueError('Native state, mass or disabled elastic-force readback differs')
    law=DiscreteBending(rest,faces,np.zeros(len(adjacency(faces)[0])),settings['flexural_rigidity'],device='cuda:0')
    membrane=DiscreteMembrane(rest,faces,**{k:settings[k] for k in ['youngs_modulus','poissons_ratio','thickness']},device='cuda:0')
    metric=MaterialMetric(points,faces,chart,max_edge_strain=.01,max_area_change=.01)
    basis=np.repeat(shape,len(x));basis-=basis.mean();basis/=np.sqrt(np.mean(basis**2))
    expected=flexure_reference.frequency(length,settings['flexural_rigidity'],settings['areal_density'])
    seconds=max(.8,3.2/expected)
    if seconds>3.:raise ValueError('Unresolved frequency outside independent control budget')
    stride=max(1,hz//240);steps=int(np.ceil(seconds*hz/stride))*stride
    rows=[];states=[];reports=[]
    def record(step):
        actual=as_numpy(position()).copy();centred=actual-actual.mean(0);origin=rest-rest.mean(0)
        u,_,v=np.linalg.svd(centred.T@origin);rotation=u@v
        if np.linalg.det(rotation)<0:u[:,-1]*=-1;rotation=u@v
        amplitude=float((centred@rotation)[:,2]@basis/len(basis))
        velocity=as_numpy(view.get_velocities())[0].reshape(-1,3)
        q=torch.as_tensor(actual,dtype=law.dtype,device=law.device)
        row={'time_s':step/hz,'modal_amplitude_m':amplitude,'material':metric.check(actual),
             'mechanical_energy_J':float(law.energy(q)+membrane.energy(q))+.5*float((masses[:,None]*velocity**2).sum()),
             'centre_of_mass_drift_m':float(np.linalg.norm(actual.mean(0)-initial.mean(0))),
             'momentum_norm_Ns':float(np.linalg.norm((velocity*masses[:,None]).sum(0)))}
        rows.append(row);states.append(actual);(out/'progress.json').write_text(json.dumps(row,indent=2)+'\n')
    record(0)
    for step in range(steps):
        updated,report=law.velocity_step(position(),view.get_velocities()[0].reshape(-1,3),masses,dt,membrane=membrane,tolerance=1e-7,
            linear_solver=args.linear_solver,nonlinear_iterations=args.nonlinear_iterations,
            viscosity_terms=[(law,args.bending_viscosity_time_s),(membrane,args.membrane_viscosity_time_s)])
        view.set_velocities(updated.to(dtype=torch.float32).reshape(1,-1).contiguous(),index);reports.append(report)
        world.step(render=False)
        if (step+1)%stride==0:record(step+1)
    errors.check();response=flexure_reference.oscillation([r['time_s'] for r in rows],[r['modal_amplitude_m'] for r in rows])
    relative=abs(response['frequency_hz']/expected-1) if response['frequency_hz'] else None
    nonlinear_converged=(all(r.get('nonlinear_converged') is True for r in reports) if args.nonlinear_iterations>1 else None)
    source_unchanged=all(hashlib.sha256(Path(q).read_bytes()).hexdigest()==digest for q,digest in input_hashes.items())
    decay=modal_decay([r['time_s'] for r in rows],[r['modal_amplitude_m'] for r in rows],expected)
    material_damping=bool(args.bending_viscosity_time_s or args.membrane_viscosity_time_s)
    energy_growth=max(r['mechanical_energy_J'] for r in rows)/rows[0]['mechanical_energy_J']
    control_complete=bool(source_unchanged and nonlinear_converged is not False and all(r['material']['pass'] for r in rows)
        and max(r['centre_of_mass_drift_m'] for r in rows)<1e-5 and max(r['momentum_norm_Ns'] for r in rows)<1e-6 and energy_growth<1.01)
    result={'pass':bool(control_complete and not material_damping and response['pass'] and relative is not None and relative<.2),
        'control_complete':control_complete,'material_identified':False,'source_behavior_approved':False,
        'release_decay':decay,'maximum_mechanical_energy_to_initial_ratio':energy_growth,
        'internal_dissipation':{'model':'Kelvin-Voigt, coupled implicit elastic/dissipative solve','diagnostic_hypothesis_only':True,
            'bending_viscosity_time_s':args.bending_viscosity_time_s,'membrane_viscosity_time_s':args.membrane_viscosity_time_s,
            'sum_viscous_work_J':sum(r['constitutive_dissipation_J'] for r in reports)},
        'synthetic_control':True,'task_approved':False,'production_backend_registered':False,
        'backend':{'name':'isaacsim','version':importlib.metadata.version('isaacsim'),'solver':'PGS','representation':'particle'},
        'internal_constitutive_law':'experimental_implicit_shell','native_springs_disabled':True,
        'external_forces':False,'gravity_enabled':False,'positions_written_after_reset':False,'fixed_nodes':0,
        'settings':{k:settings[k] for k in ['youngs_modulus','poissons_ratio','thickness','flexural_rigidity','areal_density']},
        'material_damping_enabled':material_damping,'parameters':vars(args),'physics_hz':hz,'position_iterations':128,
        'expected_free_strip_frequency_hz':expected,'observed_oscillation':response,'relative_frequency_error':relative,
        'native_mass_kg':float(masses.sum()),'native_spring_and_mass_readback':True,'rows':rows,
        'maximum_cg_iterations':max(r['cg_iterations'] for r in reports),
        'maximum_relative_solver_residual':max(r['relative_solver_residual'] for r in reports),
        'nonlinear_all_steps_converged':nonlinear_converged,
        'native_physics_errors':errors.report(),'input_sha256':input_hashes,'source_unchanged_during_control':source_unchanged}
    np.savez_compressed(out/'states.npz',points=states,rest=rest,faces=faces,masses=masses)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('PARTICLE_FLEXURE_RESULT',result['pass'],expected,response,flush=True)
except Exception as error:
    (out/'error.json').write_text(json.dumps({'pass':False,'error':str(error),'traceback':traceback.format_exc()},indent=2)+'\n');raise
finally:
    errors.close();app.close()
