"""Native candidate initialization for a loose sheet in a hinged container.

An object-level finite-force fixture prepares a candidate, not task evidence.
Its released geometry must be refitted to source images and independently pass
unforced observation, material, contact, task and prior-case checks afterwards.
"""
import argparse,json,shutil,traceback
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('session');parser.add_argument('candidate');parser.add_argument('--output',required=True)
parser.add_argument('--assembly',default='wipes_dispenser');parser.add_argument('--body',default='body')
parser.add_argument('--lid',default='lid');parser.add_argument('--sheet',default='wipe')
parser.add_argument('--open-position',type=float,default=2.);parser.add_argument('--close-position',type=float,default=0.)
parser.add_argument('--physics-hz',type=int,choices=[960,1920,3840],default=1920)
parser.add_argument('--closing-seconds',type=float,default=3.)
parser.add_argument('--regularize-iterations',type=int,default=0,help='Geometry preparation before native creation only; never task evidence')
parser.add_argument('--self-separate',action='store_true',help='Finite-thickness source preparation; full triangle/native checks remain required')
parser.add_argument('--physics-plan',help='Explicit candidate material hypotheses; initialization only')
parser.add_argument('--mechanism-plan',help='Matching candidate physical representation; initialization only')
parser.add_argument('--prepared-source',help='Same-session candidate source mesh; initialization only, never an approved state')
parser.add_argument('--material-ramp-seconds',type=float,default=0.,help='Numerical preparation only: restore full declared surface stiffness before closing/release')
args=parser.parse_args()
from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402
app=SimulationApp({'headless':True})
import numpy as np  # noqa: E402
import torch  # noqa: E402
import omni.usd  # noqa: E402
import omni.physics.tensors as tensors  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleArticulation,SingleRigidPrim  # noqa: E402
from pxr import UsdPhysics,PhysxSchema,Gf,Vt  # noqa: E402
from calibrated_scene_reconstruction.scene.session import Session  # noqa: E402
from calibrated_scene_reconstruction.assets.metric_mesh import asset  # noqa: E402
from calibrated_scene_reconstruction.assets.cloth_rest import area  # noqa: E402
from calibrated_scene_reconstruction.assets.mechanisms import joint_frames  # noqa: E402
from calibrated_scene_reconstruction.sim.mechanism_stage import add_links  # noqa: E402
from calibrated_scene_reconstruction.sim.physical_priors import apply as apply_priors  # noqa: E402
from calibrated_scene_reconstruction.sim.physics import physics_material  # noqa: E402
from calibrated_scene_reconstruction.sim.usd import add_box_collider  # noqa: E402
from calibrated_scene_reconstruction.sim.simulation import apply as apply_simulation  # noqa: E402
from calibrated_scene_reconstruction.sim.cloth_contact import self_intersections  # noqa: E402
from calibrated_scene_reconstruction.sim.surface_distance import SignedSurface,triangle_samples  # noqa: E402
from calibrated_scene_reconstruction.tasks.mechanisms.cloth import MaterialMetric  # noqa: E402
from calibrated_scene_reconstruction.tasks.mechanisms.force_fixture import ForceHandle,edge_patch  # noqa: E402
from calibrated_scene_reconstruction.tasks.rigid.task_checks import as_numpy,rot  # noqa: E402
out=Path(args.output);out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,out/'source.py')
from calibrated_scene_reconstruction.sim.physics_errors import PhysicsErrors  # noqa: E402
physics_errors=PhysicsErrors(out/'native_physics_errors.json')
import faulthandler  # noqa: E402
stack_log=(out/'python_stacks.log').open('w')
faulthandler.dump_traceback_later(60,repeat=True,file=stack_log)
try:
    s=Session(args.session);info=json.loads(Path(args.candidate).read_text());obj=args.assembly
    kit=asset({'mesh':str(Path(info['path'])/'mesh.npz')});parts={p['name']:p for p in kit['parts']}
    joint=next(j for j in kit['joints'] if j['parent']==args.body and j['child']==args.lid and j['type']=='revolute')
    if not joint['limits'][0]<=args.close_position<args.open_position<=joint['limits'][1]:raise ValueError('Initialization positions exceed real limits')
    initial=float(joint.get('initial_position',0.));delta=args.open_position-initial;axis=np.asarray(joint['axis']);anchor=np.asarray(joint['anchor1'])
    rotation=Rotation.from_rotvec(axis*delta)
    parts[args.lid]['V']=(parts[args.lid]['V']-anchor)@rotation.as_matrix().T+anchor
    f0,f1=joint_frames(joint);q=(rotation*Rotation.from_quat(np.r_[f1[1:],f1[0]])).as_quat()
    joint.update(frame0=f0.tolist(),frame1=np.r_[q[3],q[:3]].tolist(),initial_position=args.open_position)
    sheet=parts[args.sheet];rest=sheet['cloth_rest'];faces=sheet['F'];radius=info['report']['params']['body_radius']
    if args.prepared_source:
        import hashlib
        source=Path(args.prepared_source).resolve()
        if not any(source.is_relative_to(root.resolve()) for root in [s.dir,s.config_dir]):
            raise ValueError('Candidate source geometry must belong to this session')
        with np.load(source) as prepared:
            if (prepared['points'].shape!=sheet['V'].shape or not np.array_equal(prepared['faces'],faces)
                    or not np.isfinite(prepared['points']).all()):raise ValueError('Prepared source changed material topology')
            sheet['V']=prepared['points'].copy()
        shutil.copy2(source,out/'source_geometry.npz')
        (out/'source_geometry.json').write_text(json.dumps({'source':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'scope':'Same-capture initialization proposal only; intrinsic metric and all fresh material/native/source/task gates retained'},indent=2)+'\n')
    if bool(args.physics_plan)!=bool(args.mechanism_plan):raise ValueError('Candidate material and representation plans must be supplied together')
    plan=json.loads(Path(args.physics_plan).read_text()) if args.physics_plan else s.config('agent/physics_priors.json')
    dec=json.loads(Path(args.mechanism_plan).read_text()) if args.mechanism_plan else s.config('agent/mechanisms.json')
    if args.physics_plan:
        shutil.copy2(args.physics_plan,out/'physics_priors.json');shutil.copy2(args.mechanism_plan,out/'mechanisms.json')
    p={k:v['nominal'] for k,v in plan['bodies'][obj+'/'+args.sheet]['parameters'].items()}
    validity=plan['bodies'][obj+'/'+args.sheet]['validity_hypothesis']
    if args.regularize_iterations:
        from calibrated_scene_reconstruction.assets.cloth_rest import project_shape
        rigid=[SignedSurface(part['V'],part['F']) for part in kit['parts'] if part['kind']=='rigid']
        separation=None
        if args.self_separate:
            from calibrated_scene_reconstruction.sim.cloth_separation import Separation
            separation=Separation(rest['points'],info.get('cloth_physics',{}).get('rest_offset',p.get('thickness',radius*.012)/2))
        def project(points):
            for surface in rigid:points=surface.project_out(points,info.get('cloth_physics',{}).get('rest_offset',p.get('thickness',radius*.012)/2))
            if separation is not None:points=separation(points)
            return points
        sheet['V'],preparation=project_shape(rest,sheet['V'],faces,iterations=args.regularize_iterations,
                                            reference_pull=0.,collision_projection=project,device='cuda:0')
        (out/'geometry_preparation.json').write_text(json.dumps(preparation,indent=2)+'\n')
        np.savez_compressed(out/'prepared_source.npz',points=sheet['V'],faces=faces)
    initial_material=MaterialMetric(sheet['V'],faces,rest,max_edge_strain=validity['max_edge_strain'],
                                  max_area_change=validity['max_area_change']).check(sheet['V'])
    (out/'authored_material.json').write_text(json.dumps(initial_material,indent=2)+'\n')
    if not initial_material['pass']:
        raise ValueError('Authored sheet exceeds material limits before native initialization')
    surface_fem=sheet.get('backend','particle')=='surface_fem'
    if not 0<=args.material_ramp_seconds<=10 or (args.material_ramp_seconds and not surface_fem):
        raise ValueError('Material continuation requires native surface FEM and a bounded0..10s duration')
    if surface_fem:
        cloth={**p,'friction':p['dynamic_friction'],'contact_offset':max(radius*.008,p['thickness']*.75),
               'rest_offset':p['thickness']/2,'self_collision_filter_distance':p['thickness'],
               'max_depenetration_velocity':.02,
               'solver_position_iterations':128,'collision_pair_updates':16,'collision_iteration_multiplier':2}
    else:
        cloth={'areal_density':p['areal_density'],'stretch_stiffness':p['stretch_stiffness'],'shear_stiffness':p['shear_stiffness'],
            'bend_stiffness':p['bend_stiffness'],'damping':p['spring_damping'],'friction':p['dynamic_friction'],
            'contact_offset':radius*.008,'rest_offset':radius*.006,'solid_rest_offset':radius*.006,
            'max_depenetration_velocity':.02,'solver_position_iterations':128}
    cloth.update(info.get('cloth_physics',{}))
    fix={'sdf':info.get('sdf',{'resolution':512,'contact_offset':radius*.004,'rest_offset':0.}),
         'link_physics':{args.sheet:cloth}}
    world=World(stage_units_in_meters=1.,physics_dt=1/args.physics_hz,rendering_dt=1/30,device='cuda:0',backend='torch')
    physics_errors.guard_steps(world)
    pc=world.get_physics_context();pc.enable_gpu_dynamics(True);pc.set_broadphase_type('GPU')
    stage=omni.usd.get_context().get_stage();apply_simulation(stage,{'physics_hz':args.physics_hz,'friction_type':'patch','solver':'PGS','basis':'Numerical candidate preparation; not source/task approval'},flexible=True)
    next(PhysxSchema.PhysxSceneAPI.Apply(p) for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)).CreateEnableCCDAttr(False)
    material=physics_material(stage,'/World/Physics',.6,.45,.05);add_box_collider(stage,'/World/Support/task_desk/surface',[-.5,-.5,-.02],[.5,.5,0])
    root=add_links(stage,'/World/Objects/'+obj,kit,fix,s.scene_graph['dynamic_objects'][obj],material)
    apply_priors(stage,plan,dec,'nominal')
    ramp_material=stage.GetPrimAtPath(str(root.GetPath())+'/'+args.sheet+'/surface_surface_fem_material') if args.material_ramp_seconds else None
    def material_fraction(value):
        if ramp_material is not None:
            ramp_material.GetAttribute('omniphysics:youngsModulus').Set(p['youngs_modulus']*value)
            ramp_material.GetAttribute('omniphysics:surfaceBendStiffness').Set(p['flexural_rigidity']/p['thickness']**3*value)
    material_fraction(.001)
    holder=UsdPhysics.FixedJoint.Define(stage,'/World/InitializationHolder');holder.CreateBody1Rel().SetTargets([str(root.GetPath())+'/'+args.body]);holder.CreateLocalPos0Attr(Gf.Vec3f(0));holder.CreateLocalPos1Attr(Gf.Vec3f(0))
    stage.GetRootLayer().Export(str(out/'initial.usda'));world.reset()
    articulation=SingleArticulation(str(root.GetPath())+'/'+args.body);articulation.initialize()
    body=SingleRigidPrim(str(root.GetPath())+'/'+args.body);body.initialize();lid=SingleRigidPrim(str(root.GetPath())+'/'+args.lid);lid.initialize()
    rigid_views={args.body:body,args.lid:lid}
    for part in kit['parts']:
        if part['kind']=='rigid' and part['name'] not in rigid_views:
            view=SingleRigidPrim(str(root.GetPath())+'/'+part['name']);view.initialize();rigid_views[part['name']]=view
    index=torch.zeros(1,dtype=torch.int32,device=world.device)
    if surface_fem:
        from calibrated_scene_reconstruction.sim.surface_cloth import SurfaceView
        native=SurfaceView(stage,str(root.GetPath())+'/'+args.sheet+'/surface',sheet)
        positions=native.get_world_positions;velocities=native.get_velocities;set_velocities=native.set_velocities
        masses=as_numpy(native.get_particle_masses())[0]
    else:
        owner=tensors.create_simulation_view('torch');owner.set_subspace_roots('/');native=owner.create_particle_cloth_view(str(root.GetPath())+'/'+args.sheet+'/surface')
        positions=lambda:native.get_positions().reshape(1,-1,3)
        velocities=lambda:native.get_velocities().reshape(1,-1,3)
        set_velocities=lambda value:native.set_velocities(value.reshape(1,-1).contiguous(),index)
        masses=as_numpy(native.get_masses())[0]
    mass=float(masses.sum());dt=world.get_physics_dt()
    expected_mass=p['areal_density']*area(rest['points'],faces)
    mass_report={'expected_mass_kg':expected_mass,'native_mass_kg':mass,'native_nodes':len(masses),
        'native_min_node_mass_kg':float(masses.min()),'native_max_node_mass_kg':float(masses.max()),
        'relative_error':abs(mass-expected_mass)/expected_mass,'not_task_approval':True}
    (out/'mass_readback.json').write_text(json.dumps(mass_report,indent=2)+'\n')
    print('NATIVE_MASS',json.dumps(mass_report),flush=True)
    assert np.isclose(mass,expected_mass,rtol=1e-5),str(mass_report)
    get=lambda:as_numpy(positions())[0].copy()
    reference=get();chart=np.asarray(rest['points']);grasp_depth=min(.01,.08*np.ptp(chart[:,1]));patch=edge_patch(chart,grasp_depth)
    handle=ForceHandle(reference[patch],masses[patch],dt=dt,held_mass=mass,force_limit=.2,speed=.02,stiffness=20.)
    target=handle.target.copy();validity=plan['bodies'][obj+'/'+args.sheet]['validity_hypothesis']
    metric=MaterialMetric(reference,faces,rest,max_edge_strain=validity['max_edge_strain'],max_area_change=validity['max_area_change'])
    surfaces={name:SignedSurface(parts[name]['V'],parts[name]['F']) for name in rigid_views}
    rows=[];release_reference=None;open_end=.3+args.material_ramp_seconds;closing_end=open_end+args.closing_seconds;total=closing_end+.5
    def record(step,phase):
        points=get();quality=metric.check(points);cross=self_intersections(points,faces,example_limit=4);penetration={}
        sample=triangle_samples(points,faces)
        for name,view in rigid_views.items():
            position,q=view.get_world_pose();R=rot(as_numpy(q));penetration[name]=surfaces[name].max_penetration((sample-as_numpy(position))@R)
        theta=float(as_numpy(articulation.get_joint_positions())[0]);velocity=float(as_numpy(articulation.get_joint_velocities())[0])
        row={'time_s':step*dt,'phase':phase,'angle_rad':theta,'angular_velocity_rad_s':velocity,'material':quality,'self_intersection':cross,'penetration_m':penetration}
        rows.append(row);np.savez_compressed(out/f'state_{step:06d}.npz',points=points,body_position=as_numpy(body.get_world_pose()[0]),body_quat=as_numpy(body.get_world_pose()[1]),lid_position=as_numpy(lid.get_world_pose()[0]),lid_quat=as_numpy(lid.get_world_pose()[1]))
        (out/'progress.json').write_text(json.dumps({'not_task_approval':True,'last_state':row},indent=2)+'\n')
        print('INITIALIZATION_STATE',row['time_s'],phase,'angle',theta,'strain',quality['maximum_principal_material_strain'],'crossings',cross['intersecting_pairs'],flush=True)
        if not cross['pass']:raise ValueError('Numerical preparation crossed sheet surfaces; retain failed candidate')
        return points
    record(0,'open')
    for step in range(round(total/dt)):
        t=step*dt;phase='open' if t<open_end else 'closing' if t<closing_end else 'released'
        if args.material_ramp_seconds:material_fraction(.001+.999*min(1.,t/args.material_ramp_seconds))
        theta=float(as_numpy(articulation.get_joint_positions())[0]);omega=float(as_numpy(articulation.get_joint_velocities())[0])
        command=max(args.close_position,args.open_position-max(0.,t-open_end))
        effort=(float(np.clip(.08*(command-theta)-.003*omega-.021*np.tanh((theta-command)/.02),-.08,.08)) if phase=='closing' else 0.)
        articulation.set_joint_efforts(torch.tensor([effort],dtype=torch.float32,device=world.device))
        if phase!='released':
            patch_points=as_numpy(positions()[0,patch]);velocity=velocities().clone();speed=as_numpy(velocity[0,patch])
            forces=handle.forces(patch_points,speed,target,load=[0,0,mass*9.81])
            velocity[0,patch]+=torch.as_tensor(forces*dt/masses[patch,None],dtype=torch.float32,device=world.device)
            set_velocities(velocity)
        elif release_reference is None:release_reference=get()
        world.step(render=False)
        if (step+1)%round(.1/dt)==0:record(step+1,phase)
    final=get();bp,bq=body.get_world_pose();br=rot(as_numpy(bq));lp,lq=lid.get_world_pose();lr=rot(as_numpy(lq));theta=float(as_numpy(articulation.get_joint_positions())[0])
    transformed=(final-as_numpy(bp))@br;lid_vertices=(parts[args.lid]['V']@lr.T+as_numpy(lp)-as_numpy(bp))@br
    np.savez_compressed(out/'candidate_state.npz',sheet_points=transformed,sheet_faces=faces,lid_vertices=lid_vertices,material_rest_points=rest['points'],body_vertices=parts[args.body]['V'])
    result={'candidate_ready_for_source_refit':bool(rows[-1]['material']['pass'] and rows[-1]['self_intersection']['pass'] and abs(theta)<.12),
        'task_pass':False,'source_fit_pass':False,'scope':'Numerical candidate initialization only; no task or delivery approval',
        'final_angle_rad':theta,'released_native_motion_m':float(np.linalg.norm(final-release_reference,axis=1).max()),
        'native_positions_prescribed_after_initialization':False,
        'force_fixture':{**handle.report(),'material_grasp_depth_m':float(grasp_depth),
                         'selected_nodes':len(patch),'selected_mass_kg':float(masses[patch].sum())},'states':rows}
    result['material_continuation']={'seconds':args.material_ramp_seconds,'final_fraction':1.,
        'scope':'Numerical candidate preparation only; full-prior independent source and task checks still required'}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('INITIALIZATION_RESULT',json.dumps({k:v for k,v in result.items() if k!='states'}),flush=True)
except Exception as error:
    (out/'error.json').write_text(json.dumps({'pass':False,'not_task_approval':True,'error':str(error),'traceback':traceback.format_exc()},indent=2)+'\n');raise
finally:
    faulthandler.cancel_dump_traceback_later()
    stack_log.close()
    physics_errors.close()
    app.close()
