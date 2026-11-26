"""Native particle-cloth contact capability control; never a task asset.

Same-body mode intentionally uses two disconnected components. All clamp and
traction forces are finite test fixtures. Spring constants are synthetic and
are not a conversion or identification of continuum cloth properties.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import traceback

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',required=True);p.add_argument('--support',choices=['self','rigid'],required=True)
p.add_argument('--friction',type=float,required=True);p.add_argument('--rigid-friction',type=float)
p.add_argument('--combine',choices=['average','multiply'],default='average')
p.add_argument('--spacing-m',type=float,default=.0002)
p.add_argument('--constitutive',choices=['native_springs','implicit_shell'],default='native_springs')
p.add_argument('--grid-offset-fraction',type=float,choices=[0.,.5],default=0.)
p.add_argument('--profile',help='Complete native profile overrides numerical defaults; friction remains the paired control variable')
p.add_argument('--solid-rest-offset-m',type=float,default=.0004)
p.add_argument('--physics-hz',type=int,choices=[120,240,480,960,1920,3840,7680],default=960)
p.add_argument('--linear-solver',choices=['cg','sparse_direct','cuda_direct'],default='cg')
p.add_argument('--nonlinear-iterations',type=int,default=1)
p.add_argument('--rigid-ablation',action='store_true',help='At loading, reduce base friction while compensating particle-particle friction scale; capability control only')
args=p.parse_args()
profile=None
if args.profile:
    from calibrated_scene_reconstruction.tasks.mechanisms.particle_control_profile import validate
    profile=validate(json.loads(Path(args.profile).read_text()))
    if args.constitutive!='native_springs':p.error('Explicit production contact profile requires native springs')
    args.spacing_m=profile['spacing_m'];args.physics_hz=profile['physics_hz']
    args.solid_rest_offset_m=profile['contact']['solid_rest_offset']
if not 1<=args.nonlinear_iterations<=30 or (args.nonlinear_iterations>1 and args.constitutive!='implicit_shell'):
    p.error('Nonlinear iteration budget in1..30 requires the explicit internal shell')
if (not 0<=args.friction<=1.5 or args.rigid_friction is not None and not 0<=args.rigid_friction<=1.5
        or not .0001<=args.spacing_m<=.002 or (profile is None and not .0002<=args.solid_rest_offset_m<.0005)):
    p.error('Bounded contact coefficients, collision radius and mesh spacing required')
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
from scipy.spatial import cKDTree  # noqa: E402
from calibrated_scene_reconstruction.assets.cloth_rest import triangulate_grid,constraints,area  # noqa: E402
from calibrated_scene_reconstruction.assets.cloth_metric import MaterialMetric  # noqa: E402
from calibrated_scene_reconstruction.assets.cloth_mass import authoring_mass  # noqa: E402
from calibrated_scene_reconstruction.sim.physics_errors import PhysicsErrors  # noqa: E402
from calibrated_scene_reconstruction.sim.cloth_contact import self_intersections  # noqa: E402
from calibrated_scene_reconstruction.sim.physics import physics_material  # noqa: E402
from calibrated_scene_reconstruction.sim.usd import add_box_collider  # noqa: E402
from calibrated_scene_reconstruction.tasks.rigid.task_checks import as_numpy  # noqa: E402
from calibrated_scene_reconstruction.tasks.mechanisms.force_fixture import implicit_pd  # noqa: E402

errors=PhysicsErrors(out/'native_physics_errors.json')
control_files=list(map(Path,[__file__,'src/calibrated_scene_reconstruction/assets/cloth_mass.py','src/calibrated_scene_reconstruction/assets/cloth_rest.py',
    'src/calibrated_scene_reconstruction/tasks/mechanisms/force_fixture.py','src/calibrated_scene_reconstruction/assets/cloth_metric.py',
    'src/calibrated_scene_reconstruction/sim/physics_errors.py','src/calibrated_scene_reconstruction/sim/cloth_contact.py','src/calibrated_scene_reconstruction/sim/physics.py',
    'src/calibrated_scene_reconstruction/sim/usd.py','src/calibrated_scene_reconstruction/tasks/rigid/task_checks.py',
    'src/calibrated_scene_reconstruction/tasks/mechanisms/contact_control_profile.py',
    'src/calibrated_scene_reconstruction/sim/particle_friction.py',
    'src/calibrated_scene_reconstruction/scene/objects/particle_contact_binding.py']))
if profile:
    control_files+=list(map(Path,[args.profile,'src/calibrated_scene_reconstruction/tasks/mechanisms/particle_control_profile.py',
                                 'src/calibrated_scene_reconstruction/assets/particle_lattice.py']))
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
    hz=args.physics_hz;dt=1/hz;rest_offset=.0004;solid_rest=args.solid_rest_offset_m
    contact={'contact_offset':.001,'rest_offset':rest_offset,'solid_rest_offset':solid_rest,
        'particle_contact_offset':.0005,'solver_position_iterations':128,'max_neighborhood':256,
        'max_depenetration_velocity':.1}
    if profile:contact=dict(profile['contact']);rest_offset=contact['rest_offset']
    density=profile['areal_density'] if profile else .2
    physical_thickness=profile['physical_thickness_m'] if profile else .0004
    spring_damping=profile['spring_damping'] if profile else 1e-12
    world=World(stage_units_in_meters=1.,physics_dt=dt,device='cuda:0',backend='torch');errors.guard_steps(world)
    pc=world.get_physics_context();pc.enable_gpu_dynamics(True);pc.set_broadphase_type('GPU');pc.set_solver_type('PGS')
    stage=omni.usd.get_context().get_stage();scene=next(q for q in stage.Traverse() if q.IsA(UsdPhysics.Scene))
    PhysxSchema.PhysxSceneAPI.Apply(scene).CreateEnableExternalForcesEveryIterationAttr(False)
    settings={'stretch_stiffness':1200.,'shear_stiffness':400.,'bend_stiffness':.8}
    if profile:settings=dict(profile['spring_settings'])
    if args.constitutive=='implicit_shell':settings={k:0. for k in settings}
    parts={}
    def sheet(name,width,height,flip=False):
        x=np.linspace(-width/2,width/2,round(width/args.spacing_m)+1);xx,yy=np.meshgrid(x,x)
        rest=np.c_[xx.ravel(),yy.ravel(),np.zeros(xx.size)];points=rest+[0,0,height]
        if flip:points[:,1]*=-1
        faces,chart,_=triangulate_grid(points,rest,len(x),len(x))
        if profile and profile['spring_model'] in {'balanced_square_lattice','triangle_surface'}:
            from calibrated_scene_reconstruction.assets.particle_lattice import chart as balanced_chart
            chart,_=balanced_chart(rest,faces,model=profile['spring_model'])
        parts[name]={'V':points,'F':faces,'cloth_rest':chart}
    if args.support=='self':sheet('Lower',.05,.1)
    else:
        box=add_box_collider(stage,'/World/Lower',[-.025,-.025,.09],[.025,.025,.1])
        mu=args.friction if args.rigid_friction is None else args.rigid_friction
        mat=physics_material(stage,'/World/RigidMaterial',mu,mu,0.,friction_combine=args.combine)
        UsdShade.MaterialBindingAPI.Apply(box.GetPrim()).Bind(mat,UsdShade.Tokens.weakerThanDescendants,'physics')
    expected_gap=2*solid_rest if args.support=='self' else rest_offset
    sheet('Upper',.02,.1+expected_gap+.00005,args.support=='self')
    if args.support=='self':parts['Upper']['V'][:,:2]+=args.grid_offset_fraction*args.spacing_m
    vertices=[];rest_points=[];all_faces=[];all_pairs=[];all_lengths=[];all_stiffness=[];all_kinds=[];ids={};offset=0
    for name,part in parts.items():
        indices,lengths,stiffness,kinds=constraints(part['cloth_rest'],settings)
        chart=np.array(part['cloth_rest']['points'])
        if name=='Upper' and args.support=='self':chart[:,1]+=.12
        vertices.append(part['V']);rest_points.append(chart);all_faces.append(part['F']+offset)
        all_pairs.append(indices+offset);all_lengths.extend(lengths);all_stiffness.extend(stiffness);all_kinds.extend(kinds)
        ids[name]=np.arange(offset,offset+len(chart));offset+=len(chart)
    vertices=np.concatenate(vertices);rest_points=np.concatenate(rest_points);faces=np.concatenate(all_faces);pairs=np.concatenate(all_pairs)
    system=particleUtils.add_physx_particle_system(stage,'/World/ParticleSystem',simulation_owner=scene.GetPath(),
        **contact,enable_ccd=True)
    particleUtils.add_pbd_particle_material(stage,'/World/ParticleMaterial',friction=args.friction,damping=0.,drag=0.,lift=0.)
    material=UsdShade.Material.Get(stage,'/World/ParticleMaterial')
    PhysxSchema.PhysxPBDMaterialAPI(material.GetPrim()).CreateParticleFrictionScaleAttr(1.)
    UsdShade.MaterialBindingAPI.Apply(system.GetPrim()).Bind(material,UsdShade.Tokens.weakerThanDescendants,'physics')
    mesh=UsdGeom.Mesh.Define(stage,'/World/Cloth')
    normals=np.zeros_like(vertices,dtype='float32');normals[:,2]=1.
    if args.support=='self':normals[ids['Upper'],2]=-1.
    particleUtils.add_physx_particle_cloth_with_constraints(stage,mesh.GetPath(),system.GetPath(),
        Vt.Vec3fArray.FromNumpy(vertices.astype('float32')),Vt.Vec3fArray.FromNumpy(normals),
        Vt.Vec3fArray.FromNumpy(rest_points.astype('float32')),Vt.Vec3fArray.FromNumpy(np.zeros_like(vertices,dtype='float32')),
        Vt.IntArray(faces.ravel().tolist()),Vt.Vec2iArray.FromNumpy(pairs.astype('int32')),
        Vt.FloatArray(all_stiffness),Vt.FloatArray([0. if args.constitutive=='implicit_shell' else spring_damping]*len(pairs)),Vt.FloatArray(all_lengths),self_collision=True,self_collision_filter=True)
    mesh.GetPrim().SetCustomDataByKey('calibrated_scene_reconstruction:cloth_spring_kinds',json.dumps(all_kinds))
    mesh.GetPrim().SetCustomDataByKey('calibrated_scene_reconstruction:cloth_spring_model',profile['spring_model'] if profile else 'triangle_edges')
    target_mass=density*area(rest_points,faces);cooking=authoring_mass(target_mass,rest_points,faces)
    UsdPhysics.MassAPI.Apply(mesh.GetPrim()).CreateMassAttr(cooking['authoring_mass_kg'])
    # Reset performs a native warmup step before the internal elastic adapter
    # exists. Suspend gravity only for that initialization, then restore the
    # original scene gravity before every recorded physical step. No positions
    # or velocities are overwritten to conceal a warmup displacement.
    gravity=UsdPhysics.Scene(scene).GetGravityMagnitudeAttr();normal_gravity=gravity.Get()
    gravity.Set(0.);world.reset();gravity.Set(normal_gravity)
    stage.GetRootLayer().Export(str(out/'initial.usda'))
    owner=tensors.create_simulation_view('torch');owner.set_subspace_roots('/')
    view=owner.create_particle_cloth_view('/World/Cloth');index=torch.zeros(1,dtype=torch.int32,device='cuda:0')
    position=lambda:as_numpy(view.get_positions())[0].reshape(-1,3).copy()
    initial=position();dist,order=cKDTree(vertices).query(initial)
    if not np.array_equal(order,np.arange(len(vertices))) or dist.max()>.00015:raise ValueError('Native node correspondence or initialization differs from this control')
    masses=as_numpy(view.get_masses())[0].copy()
    if not np.isclose(masses.sum(),target_mass,rtol=1e-4):raise ValueError('Actual particle-cloth mass differs from the force calculation')
    def check_native_springs():
        native=as_numpy(view.get_spring_stiffness()).ravel()
        damping=as_numpy(view.get_spring_damping()).ravel()
        expected_damping=0. if args.constitutive=='implicit_shell' else spring_damping
        if (len(native)!=len(all_stiffness) or len(damping)!=len(all_stiffness)
                or not np.allclose(np.sort(native),np.sort(all_stiffness),rtol=1e-5,atol=1e-12)
                or not np.allclose(damping,expected_damping,rtol=1e-5,atol=1e-15)):
            raise ValueError('Contact control native springs differ from the requested network')
    check_native_springs()
    for key,value in contact.items():
        suffix='SolverPositionIterationCount' if key=='solver_position_iterations' else ''.join(w.title() for w in key.split('_'))
        actual=getattr(system,'Get'+suffix+'Attr')().Get()
        if actual is None or not np.isclose(actual,value,rtol=1e-5,atol=1e-12):
            raise ValueError('Authored particle contact setting changed after initialization: '+key)
    shell_laws={};shell_reports=[]
    if args.constitutive=='implicit_shell':
        from calibrated_scene_reconstruction.sim.discrete_bending import DiscreteBending
        from calibrated_scene_reconstruction.sim.discrete_membrane import DiscreteMembrane
        from calibrated_scene_reconstruction.assets.cloth_curvature import adjacency
        if np.any(as_numpy(view.get_spring_stiffness())!=0) or np.any(as_numpy(view.get_spring_damping())!=0):
            raise ValueError('Native particle elastic forces must be disabled before testing the implicit shell')
        for name,part in parts.items():
            chart=np.asarray(part['cloth_rest']['points'])
            shell_laws[name]=(DiscreteBending(chart,part['F'],np.zeros(len(adjacency(part['F'])[0])),1e-5,device='cuda:0'),
                DiscreteMembrane(chart,part['F'],youngs_modulus=4e6,poissons_ratio=.2,thickness=.0004,device='cuda:0'))
    material_view=owner.create_particle_material_view('/World/ParticleMaterial')
    native_mu=as_numpy(material_view.get_friction()).copy()
    if material_view.count!=1 or not np.allclose(native_mu,args.friction,rtol=0,atol=1e-7):raise ValueError('Actual native friction differs from declared control')
    metrics={name:MaterialMetric(part['V'],part['F'],part['cloth_rest'],max_edge_strain=.05,max_area_change=.03) for name,part in parts.items()}
    weights={name:masses[region]/masses[region].sum() for name,region in ids.items()}
    upper_mass=float(masses[ids['Upper']].sum());load=.6*upper_mass*9.81
    rows=[];states=[];free_x=0.;free_v=0.;peak_clamp=0.;switch=None
    def record(time):
        points=position();velocity=as_numpy(view.get_velocities())[0].reshape(-1,3)
        if switch is not None and time>switch['time_s']:
            base=as_numpy(material_view.get_friction())
            scale=PhysxSchema.PhysxPBDMaterialAPI(material.GetPrim()).GetParticleFrictionScaleAttr().Get()
            if (not np.allclose(base,switch['base_friction'],rtol=0,atol=1e-10)
                    or not np.isclose(scale,switch['authored_particle_friction_scale'],rtol=1e-6)):
                raise ValueError('Particle friction ablation did not persist after native stepping')
            switch['native_readback_after_step']=True
        material_state={name:metrics[name].check(points[region]) for name,region in ids.items()}
        if args.support=='self':
            nearest=cKDTree(points[ids['Lower'],:2]).query(points[ids['Upper'],:2])[1]
            gap=points[ids['Upper'],2]-points[ids['Lower'][nearest],2]
            relative=float(points[ids['Upper'],0].mean()-points[ids['Lower'],0].mean())
        else:gap=points[ids['Upper'],2]-.1;relative=float(points[ids['Upper'],0].mean())
        cross=self_intersections(points,faces)
        # At coarse sampling the staggered spheres support a smaller plane gap
        # than two aligned grids. Require a physical sheet gap as well; this
        # control may reject a collision mesh even if its friction is useful.
        minimum_gap=max(physical_thickness,np.sqrt(max(0.,(2*solid_rest)**2-args.spacing_m**2/2))) if args.support=='self' else expected_gap
        valid=bool(all(v['pass'] for v in material_state.values()) and cross['pass'] and gap.min()>=minimum_gap-.00005)
        rows.append({'time_s':time,'physical_pass':valid,'material':material_state,'crossings':cross,
            'relative_x_m':relative,'minimum_mid_surface_gap_m':float(gap.min()),'free_traction_displacement_m':free_x,
            'native_base_friction':as_numpy(material_view.get_friction()).tolist(),
            'upper_mean_velocity_m_s':velocity[ids['Upper']].mean(0).tolist()});states.append(points)
        (out/'progress.json').write_text(json.dumps(rows[-1],indent=2)+'\n');return valid
    valid=record(0.)
    for step in range(round(.275*hz)):
        if not valid:break
        t=step*dt;f=np.zeros_like(vertices)
        if args.rigid_ablation and t>=.2 and switch is None:
            from calibrated_scene_reconstruction.sim.particle_friction import ablate_rigid
            switch={'time_s':t,**ablate_rigid(material.GetPrim(),material_view)}
        if args.support=='self':
            region=ids['Lower'];points=position();speed=as_numpy(view.get_velocities())[0].reshape(-1,3)
            damping=np.minimum(2*np.sqrt(200*masses[region].sum())*weights['Lower'],.5*masses[region]/dt)
            clamp=implicit_pd(points[region],speed[region],initial[region],masses[region],
                200*weights['Lower'],damping,dt)
            clamp[:,2]+=masses[region]*9.81;mag=np.linalg.norm(clamp,axis=1)
            clamp*=np.minimum(1.,.1*weights['Lower']/np.maximum(mag,1e-12))[:,None]
            peak_clamp=max(peak_clamp,float(np.linalg.norm(clamp,axis=1).sum()));f[region]=clamp
        fx=load*min(1.,max(0.,(t-.2)/.01)) if t>=.2 else 0.;f[ids['Upper'],0]=fx*weights['Upper']
        v=view.get_velocities().reshape(1,-1,3).clone();v[0]+=torch.as_tensor(f*dt/masses[:,None],dtype=v.dtype,device=v.device)
        if shell_laws:
            actual=view.get_positions()[0].reshape(-1,3)
            for name,(law,membrane) in shell_laws.items():
                region=torch.as_tensor(ids[name],dtype=torch.long,device=v.device)
                updated,internal=law.velocity_step(actual[region],v[0,region],masses[ids[name]],dt,membrane=membrane,tolerance=1e-7,
                    linear_solver=args.linear_solver,nonlinear_iterations=args.nonlinear_iterations)
                v[0,region]=updated.to(dtype=v.dtype);shell_reports.append(internal)
        view.set_velocities(v.reshape(1,-1).contiguous(),index)
        free_x+=free_v*dt+.5*fx/upper_mass*dt*dt;free_v+=fx/upper_mass*dt
        world.step(render=False)
        if (step+1)%(hz//40)==0:valid=record((step+1)*dt)
    errors.check();final_mu=as_numpy(material_view.get_friction()).copy()
    expected=1e-6 if switch is not None else args.friction
    if not np.allclose(final_mu,expected,rtol=0,atol=1e-7):raise ValueError('Native particle friction changed unexpectedly during control')
    nonlinear_converged=(all(r.get('nonlinear_converged') is True for r in shell_reports) if args.nonlinear_iterations>1 else None)
    check_native_springs()
    from calibrated_scene_reconstruction.tasks.mechanisms.contact_control_profile import edge_range
    from calibrated_scene_reconstruction.scene.objects.particle_contact_binding import read as read_runtime_profile
    from types import SimpleNamespace
    native_profile=read_runtime_profile(SimpleNamespace(prim=mesh.GetPrim(),rest=rest_points,faces=faces,
        get_particle_masses=view.get_masses,view=SimpleNamespace(get_stretch_stiffnesses=view.get_spring_stiffness,
        get_spring_dampings=view.get_spring_damping)),{**settings,'spring_damping':0. if shell_laws else spring_damping,
        'thickness':physical_thickness,'areal_density':density,'native_total_mass_kg':target_mass})
    if not native_profile['native_spring_readback'] or not native_profile['native_mass_readback']:
        raise ValueError('Native control profile failed the production runtime readback adapter')
    source_unchanged=all(hashlib.sha256(Path(q).read_bytes()).hexdigest()==digest for q,digest in input_hashes.items())
    result={'pass':bool(source_unchanged and nonlinear_converged is not False and valid and abs(rows[-1]['time_s']-.275)<1e-8),'synthetic_control':True,
        'task_approved':False,'source_approved':False,'positions_prescribed':False,'self_contact_test':args.support=='self',
        'disconnected_components_in_one_body':2 if args.support=='self' else None,'native_particle_bodies':1,
        'backend':{'name':'isaacsim','version':importlib.metadata.version('isaacsim'),'solver':'PGS','representation':'particle'},
        'physics_hz':hz,'position_iterations':contact['solver_position_iterations'],'parameters':vars(args),'spring_settings':settings,'rest_offset_m':rest_offset,'solid_rest_offset_m':solid_rest,
        'spring_damping_kg_s':0. if shell_laws else spring_damping,'particle_contact_offset_m':contact['particle_contact_offset'],'max_neighborhood':contact['max_neighborhood'],
        'native_contact_settings':contact,'areal_density_kg_m2':density,'physical_thickness_m':physical_thickness,
        'native_spring_readback':True,'spring_model':profile['spring_model'] if profile else 'triangle_edges',
        'material_edge_range_m':edge_range(rest_points,faces),
        'runtime_profile_readback':native_profile,
        'experimental_internal_shell':None if not shell_laws else {'production_backend_registered':False,
            'native_springs_disabled':True,'positions_written':False,'youngs_modulus_Pa':4e6,'poissons_ratio':.2,
            'thickness_m':.0004,'flexural_rigidity_Nm':1e-5,
            'nonlinear_all_steps_converged':nonlinear_converged,
            'maximum_cg_iterations':max(r['cg_iterations'] for r in shell_reports),
            'maximum_relative_solver_residual':max(r['relative_solver_residual'] for r in shell_reports),
            'maximum_internal_linear_impulse_Ns':max(np.linalg.norm(r['linear_impulse_Ns']) for r in shell_reports)},
        'native_mass_kg':float(masses.sum()),'mass_cooking':cooking,'node_count':len(vertices),
        'initialization_gravity_suspended':True,'recorded_gravity_m_s2':normal_gravity,
        'load_force_N':load,'load_ratio_to_weight':.6,'peak_lower_clamp_force_N':peak_clamp,
        'lower_clamp_scheme':'bounded implicit per-node PD with explicit gravity compensation',
        'material_native_readback':True,'native_friction_initial':native_mu.tolist(),'native_friction_final':final_mu.tolist(),
        'rigid_contact_ablation':switch,
        'rows':rows,'load_end':rows[-1] if abs(rows[-1]['time_s']-.275)<1e-8 else None,
        'native_physics_errors':errors.report(),'input_sha256':input_hashes,'source_unchanged_during_control':source_unchanged,
        'scope':'Native contact capability only. Synthetic spring settings and conservative particle contact radii are not a reconstructed material or delivered cloth. Same-body mode has disconnected components solely for contact isolation.'}
    np.savez_compressed(out/'states.npz',points=states,rest=rest_points,faces=faces,masses=masses,**{name+'_indices':region for name,region in ids.items()})
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('PARTICLE_CONTACT_RESULT',result['pass'],args.support,args.friction,result['load_end']['relative_x_m'] if result['load_end'] else None,flush=True)
except Exception as error:
    (out/'error.json').write_text(json.dumps({'pass':False,'error':str(error),'traceback':traceback.format_exc()},indent=2)+'\n');raise
finally:
    errors.close();app.close()
