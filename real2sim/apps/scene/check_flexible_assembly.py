"""Candidate-only native check of a complete mixed rigid/surface assembly.

This isolates material, bond and passive-contact failures before a task trial.
It does not approve source accuracy or replace the assembled task checks.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import traceback

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--session',required=True)
p.add_argument('--object',required=True)
p.add_argument('--unit-length-m',type=float,default=1.)
p.add_argument('--mesh',required=True)
p.add_argument('--model',required=True)
p.add_argument('--output',required=True)
p.add_argument('--seconds',type=float,default=1.)
p.add_argument('--physics-hz',type=int,default=960)
p.add_argument('--gravity',type=float,default=9.81)
p.add_argument('--floor',action='store_true')
p.add_argument('--initial-rotation',help='JSON proper rotation for candidate support-pose diagnostics; never prescribed after reset')
p.add_argument('--floor-height',type=float,default=0.)
p.add_argument('--membrane-modulus',type=float)
p.add_argument('--flexural-rigidity',type=float)
p.add_argument('--root-density',type=float)
p.add_argument('--rigid-resolution',type=int)
p.add_argument('--diagnostic-disable-root-contact',action='store_true')
p.add_argument('--settings-file')
p.add_argument('--diagnostic-shared-particle-group',type=int)
p.add_argument('--particle-registration-hz',type=int,default=0,
               help='Optional bounded small-step initialization diagnostic;0 keeps normal reset')
p.add_argument('--position-iterations',type=int)
p.add_argument('--velocity-iterations',type=int,default=1)
p.add_argument('--solver',choices=['PGS','TGS'],default='PGS')
p.add_argument('--gravity-ramp-seconds',type=float,default=0.)
p.add_argument('--impulse-velocity',type=float,default=0.)
a=p.parse_args()
if not .1<=a.seconds<=10 or a.physics_hz not in (240,480,960,1920,3840,7680,15360):p.error('Explicit bounded duration and supported step required')
if not 1<=a.velocity_iterations<=64:p.error('Explicit native velocity solver budget1..64 required')
if a.particle_registration_hz and not 480<=a.particle_registration_hz<=20000:p.error('Registration diagnostic supports480..20000Hz')
if a.gravity_ramp_seconds and not 0<a.gravity_ramp_seconds<=a.seconds-.5:p.error('Loading needs a separate constant-gravity observation phase')
from calibrated_scene_reconstruction.assets.native_units import factor
u=factor(a.unit_length_m)
out=Path(a.output);out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,out/'source.py')
from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402
app=SimulationApp({'headless':True})
import numpy as np  # noqa: E402
import torch  # noqa: E402
import omni.usd  # noqa: E402
from pxr import UsdGeom,UsdPhysics,UsdShade,Gf  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleRigidPrim  # noqa: E402
from omni.physx.scripts import physicsUtils  # noqa: E402
from calibrated_scene_reconstruction.scene.session import Session  # noqa: E402
from calibrated_scene_reconstruction.assets.metric_mesh import asset  # noqa: E402
from calibrated_scene_reconstruction.assets.cloth_metric import MaterialMetric  # noqa: E402
from calibrated_scene_reconstruction.sim.mechanism_stage import add_links  # noqa: E402
from calibrated_scene_reconstruction.sim.surface_cloth import SurfaceView  # noqa: E402
from calibrated_scene_reconstruction.sim.physics_errors import PhysicsErrors  # noqa: E402
from calibrated_scene_reconstruction.tasks.rigid.task_checks import as_numpy,rot  # noqa: E402
errors=PhysicsErrors(out/'native_physics_errors.json')
result={'pass':False,'task_approved':False,'scope':'Complete candidate isolated material/bond diagnostic; not source or task approval',
        'simulator':'Isaac Sim 5.1','positions_prescribed':False,'parameters':vars(a)}
try:
    s=Session(a.session);spec=s.scene_graph['dynamic_objects'][a.object]
    fix={**spec['fix'],'link_physics':spec['fit']['link_physics'],'mesh':str(Path(a.mesh).resolve())}
    if a.settings_file:fix['link_physics']=json.loads(Path(a.settings_file).read_text())
    # This candidate is not the previously textured canonical mesh. Use its
    # authored materials, never an appearance mapping from another topology.
    fix.pop('look',None)
    if a.rigid_resolution is not None:
        if not 64<=a.rigid_resolution<=2048:raise ValueError('Supported explicit SDF resolution required')
        for settings in fix['link_physics'].values():
            if 'sdf' in settings:settings['sdf']['resolution']=a.rigid_resolution
    if a.membrane_modulus is not None:
        if not 1e4<=a.membrane_modulus<=1e10:raise ValueError('Finite explicit material-hypothesis modulus required')
        for settings in fix['link_physics'].values():
            if 'youngs_modulus' in settings:settings['youngs_modulus']=a.membrane_modulus
    if a.flexural_rigidity is not None:
        if not 1e-9<=a.flexural_rigidity<=.01:raise ValueError('Explicit bounded bending-material hypothesis required')
        for settings in fix['link_physics'].values():
            if 'flexural_rigidity' in settings:settings['flexural_rigidity']=a.flexural_rigidity
    if a.position_iterations is not None:
        if not 32<=a.position_iterations<=255:raise ValueError('Explicit native solver budget32..255 required')
        for settings in fix['link_physics'].values():
            if 'youngs_modulus' in settings or 'stretch_stiffness' in settings:settings['solver_position_iterations']=a.position_iterations
    from calibrated_scene_reconstruction.assets import native_units
    kit=native_units.kit(asset(fix),a.unit_length_m);parts=kit['parts'];bonded=kit['surface_bindings']
    result['physical_link_settings_si']=fix['link_physics']
    fix['link_physics']=native_units.settings(fix['link_physics'],a.unit_length_m)
    world=World(stage_units_in_meters=a.unit_length_m,physics_dt=1/a.physics_hz,backend='torch',device='cuda:0');errors.guard_steps(world)
    pc=world.get_physics_context();pc.enable_gpu_dynamics(True);pc.set_broadphase_type('GPU');pc.set_solver_type(a.solver)
    stage=omni.usd.get_context().get_stage();scene=next(q for q in stage.Traverse() if q.IsA(UsdPhysics.Scene))
    UsdPhysics.Scene(scene).CreateGravityMagnitudeAttr(0. if a.gravity_ramp_seconds else a.gravity*u)
    mat=UsdShade.Material.Define(stage,'/World/PhysicsMaterial');pm=UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateDynamicFrictionAttr(.3);pm.CreateStaticFrictionAttr(.3);pm.CreateRestitutionAttr(0.)
    initial_rotation=np.eye(3)
    if a.initial_rotation:
        initial_rotation=np.asarray(json.loads(Path(a.initial_rotation).read_text())['rotation'],float)
        if (initial_rotation.shape!=(3,3) or not np.isfinite(initial_rotation).all()
                or not np.allclose(initial_rotation.T@initial_rotation,np.eye(3),atol=1e-9)
                or not np.isclose(np.linalg.det(initial_rotation),1.)):
            raise ValueError('Initial support pose needs an explicit proper rigid rotation')
    frame=json.loads(Path(a.initial_rotation).read_text()) if a.initial_rotation else {}
    gravity_direction=np.asarray(frame.get('gravity_direction_native',[0.,0.,-1.]),float)
    if gravity_direction.shape!=(3,) or not np.isfinite(gravity_direction).all() or not np.isclose(gravity_direction@gravity_direction,1.):raise ValueError('Unit native gravity direction required')
    UsdPhysics.Scene(scene).CreateGravityDirectionAttr(Gf.Vec3f(*map(float,gravity_direction)))
    result['gravity_direction_native']=gravity_direction.tolist()
    if a.floor or a.initial_rotation:
        initial_offset=0.
        if a.floor:
            from calibrated_scene_reconstruction.assets.support_placement import offset
            with np.load(a.mesh) as source_mesh:
                transformed=dict(source_mesh);transformed['V']=source_mesh['V']@initial_rotation.T
                initial_offset=offset(transformed,{name:value.get('thickness',0.)
                    for name,value in result['physical_link_settings_si'].items()})
        transform=np.eye(4);transform[:3,:3]=initial_rotation.T
        transform[3,2]=(initial_offset+a.floor_height)*u
        UsdGeom.Xform.Define(stage,'/World/Candidate').MakeMatrixXform().Set(Gf.Matrix4d(transform.tolist()))
        result['physical_initial_offset_m']=float(initial_offset)
        result['initial_rotation']=initial_rotation.tolist()
    # Candidate mass must come from its present material/volume, not the old
    # canonical assembly's aggregate estimate (which included rigid petals).
    from calibrated_scene_reconstruction.scene.objects.physical_priors import declaration
    from calibrated_scene_reconstruction.assets.physical_priors import sample
    estimates=sample(declaration(s),'nominal')['bodies']
    rigid_estimates=[estimates[a.object+'/'+x['name']]['parameters'] for x in parts if x['kind']=='rigid']
    if len(rigid_estimates)!=1 or 'density' not in rigid_estimates[0]:
        raise ValueError('This isolated diagnostic needs one density-estimated rigid root')
    object_spec={k:v for k,v in spec.items() if k not in {'mass','_mass_why'}}
    object_spec['density']=rigid_estimates[0]['density']
    if a.root_density is not None:
        if not 1<=a.root_density<=10000:raise ValueError('Explicit positive composite density required')
        object_spec['density']=a.root_density
    result['authored_link_settings']=fix['link_physics']
    result['rigid_material_estimate']=object_spec['density']
    object_spec['density']/=u**3
    root=add_links(stage,'/World/Candidate',kit,fix,object_spec,mat)
    if a.diagnostic_disable_root_contact:
        for prim in stage.Traverse():
            if prim.GetTypeName()=='OmniPhysicsElementCollisionFilter':
                target=str(prim.GetRelationship('omniphysics:src0').GetTargets()[0])
                part=next(p for p in parts if target=='/World/Candidate/'+p['name']+'/surface')
                prim.GetAttribute('omniphysics:groupElemCounts0').Set([len(part['F'])])
                prim.GetAttribute('omniphysics:groupElemIndices0').Set(list(range(len(part['F']))))
    if a.diagnostic_shared_particle_group is not None:
        from pxr import PhysxSchema
        for prim in stage.Traverse():
            if prim.HasAPI(PhysxSchema.PhysxParticleClothAPI):
                PhysxSchema.PhysxParticleAPI(prim).CreateParticleGroupAttr(a.diagnostic_shared_particle_group)
    if a.position_iterations is not None or a.velocity_iterations!=1:
        from pxr import PhysxSchema
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                api=PhysxSchema.PhysxRigidBodyAPI(prim)
                if a.position_iterations is not None:api.CreateSolverPositionIterationCountAttr(a.position_iterations)
                api.CreateSolverVelocityIterationCountAttr(a.velocity_iterations)
    if a.floor:
        physicsUtils.add_ground_plane(stage,'/World/Floor','Z',u,Gf.Vec3f(0.,0.,a.floor_height*u),Gf.Vec3f(.5))
    from calibrated_scene_reconstruction.sim.particle_surface import reset_and_register
    particle_records=[(part['name'],'/World/Candidate/'+part['name']+'/surface',part)
                      for part in parts if part['kind']=='cloth' and part.get('backend')!='surface_fem']
    stage.GetRootLayer().Export(str(out/'initial.usda'))
    if particle_records and a.particle_registration_hz:
        registered=reset_and_register(world,stage,particle_records,initial_dt=1/a.particle_registration_hz)
        result['particle_index_registration']={k:v.index_registration for k,v in registered.items()}
    else:
        world.reset();registered={}
    rigid={part['name']:SingleRigidPrim('/World/Candidate/'+part['name']) for part in parts if part['kind']=='rigid'}
    for body in rigid.values():body.initialize()
    from calibrated_scene_reconstruction.sim.rigid_partition import metadata
    from isaacsim.core.prims import RigidPrim
    partitions={}
    for name,body in rigid.items():
        info=metadata(stage.GetPrimAtPath(body.prim_path))
        if info:
            view=RigidPrim(info['carriers'],reset_xform_properties=False,name='numerical_bond_frames_'+name)
            view.initialize();partitions[name]=(view,info)
    from calibrated_scene_reconstruction.sim.particle_surface import ParticleView
    surfaces={part['name']:(SurfaceView(stage,'/World/Candidate/'+part['name']+'/surface',part)
                           if part.get('backend')=='surface_fem' else
                           registered.get(part['name']) or ParticleView(stage,'/World/Candidate/'+part['name']+'/surface',part))
              for part in parts if part['kind']=='cloth'}
    metrics={part['name']:MaterialMetric(part['V'],part['F'],part['cloth_rest'],max_edge_strain=.03,max_area_change=.03,unit_length_m=a.unit_length_m)
             for part in parts if part['kind']=='cloth'}
    def planar_support_diagnostic():
        # Diagnose the actual candidate before attributing motion to stiffness.
        # Flexible material may legitimately settle; this necessary condition
        # never replaces source comparison or the full passive/contact trial.
        from calibrated_scene_reconstruction.assets.static_support import check as support_balance
        points=[];centroid=np.zeros(3);total=0.
        for name,body in rigid.items():
            pos,quat=tuple(as_numpy(x) for x in body.get_world_pose());R=rot(quat)
            local=as_numpy(body._rigid_prim_view.get_coms()[0]).reshape(-1,3)[0]
            kg=float(as_numpy(body._rigid_prim_view.get_masses()).sum())
            total+=kg;centroid+=kg*(R@local+pos)/u
            part=next(p for p in parts if p['name']==name)
            points.append((np.asarray(part['V'])@R.T+pos)/u)
            if name in partitions:
                view,_=partitions[name];pp,qq=tuple(as_numpy(v) for v in view.get_world_poses())
                kg0=as_numpy(view.get_masses()).ravel();com0=as_numpy(view.get_coms()[0]).reshape(-1,3)
                total+=float(kg0.sum())
                centroid+=sum(m*(rot(q)@c+p)/u for m,p,q,c in zip(kg0,pp,qq,com0))
        for name,view in surfaces.items():
            p=as_numpy(view.get_world_positions())[0]/u
            kg=as_numpy(view.get_particle_masses()).reshape(-1)
            total+=float(kg.sum());centroid+=np.einsum('n,nd->d',kg,p)
            part=next(p for p in parts if p['name']==name);tri=p[np.asarray(part['F'])]
            cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);length=np.linalg.norm(cross,axis=1)
            if np.any(length<=1e-14):raise ValueError('Degenerate physical surface cannot supply a support diagnostic')
            half=result['physical_link_settings_si'][name].get('thickness',0.)/2
            normals=cross/length[:,None]
            points.extend([(tri+sign*half*normals[:,None]).reshape(-1,3) for sign in [-1,1]])
        return {**support_balance(np.concatenate(points),centroid/total,
                    plane_offset=a.floor_height,gravity=gravity_direction*a.gravity,contact_tolerance_m=.0001,friction=.3),
                'mass_kg':total,'mass_source':'Read-back native rigid masses/COMs, including numerical components; flexible adapter declared node masses, whose backend qualification remains independently required.',
                'diagnostic_only':True,'task_approved':False}
    if a.floor:result['initial_support_balance']=planar_support_diagnostic()
    inputs={}
    for name,module in list(sys.modules.items()):
        file=getattr(module,'__file__',None)
        if name.startswith('calibrated_scene_reconstruction.') and file and Path(file).suffix=='.py':
            file=Path(file).resolve();inputs[str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
    for file in [Path(__file__),Path(a.model),Path(a.mesh),s.config_dir/'scene_graph.json',s.config_dir/'agent/physics_priors.json']:
        inputs[str(file.resolve())]=hashlib.sha256(file.read_bytes()).hexdigest()
    if a.initial_rotation:
        file=Path(a.initial_rotation).resolve();inputs[str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
    if a.settings_file:
        file=Path(a.settings_file).resolve();inputs[str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
    if a.impulse_velocity:
        for body in rigid.values():
            body._rigid_prim_view.set_velocities(torch.tensor([[0.,0.,a.impulse_velocity*u,0.,0.,0.]],device='cuda:0'))
    times=[];poses=[];states={name:[] for name in surfaces};rows=[]
    def record(t):
        current={name:as_numpy(view.get_world_positions())[0].copy() for name,view in surfaces.items()}
        q={name:tuple(as_numpy(x) for x in body.get_world_pose()) for name,body in rigid.items()}
        tests={name:metrics[name].check(points) for name,points in current.items()}
        gaps={}
        weld_errors={}
        for name,(view,info) in partitions.items():
            pos,quat=q[name];R=rot(quat);pp,qq=(as_numpy(v) for v in view.get_world_poses())
            expected=np.array(info['component_centres'][1:])@R.T+pos
            weld_errors[name]={'maximum_com_gap_m':float(np.linalg.norm(pp-expected,axis=1).max()/u),
                'maximum_rotation_error_rad':float((2*np.arccos(np.clip(np.abs(qq@quat),0.,1.))).max())}
        for bond in bonded:
            part=next(x for x in parts if x['name']==bond['surface']);ids=np.asarray(bond['vertices'])
            pos,quat=q[bond['rigid']];expected=np.asarray(part['V'])[ids]@rot(quat).T+pos
            gaps[bond['surface']]=float(np.linalg.norm(current[bond['surface']][ids]-expected,axis=1).max()/u)
            result.setdefault('latest_bond_resolution_diagnostics',{})[bond['surface']]=metrics[bond['surface']].bond_diagnostic(
                current[bond['surface']],ids,expected)
        times.append(t);poses.append([np.r_[pos/u,quat] for pos,quat in q.values()])
        for name,value in current.items():states[name].append(value/u)
        rows.append({'time_s':t,'material_pass':all(x['pass'] for x in tests.values()),
                     'gravity_m_s2':float(UsdPhysics.Scene(scene).GetGravityMagnitudeAttr().Get()/u),
                     'maximum_root_error_m':max(gaps.values()),'material':tests,'root_errors_m':gaps,'weld_errors':weld_errors})
    record(0.)
    for step in range(round(a.seconds*a.physics_hz)):
        if a.gravity_ramp_seconds:
            alpha=min(1.,(step+1)/(a.gravity_ramp_seconds*a.physics_hz));fraction=alpha**3*(10-15*alpha+6*alpha*alpha)
            pc.set_gravity(-a.gravity*u*fraction)
            UsdPhysics.Scene(scene).CreateGravityDirectionAttr(Gf.Vec3f(*map(float,gravity_direction)))
        world.step(render=False)
        if (step+1)%4==0:record((step+1)/a.physics_hz)
        if (step+1)%120==0:print('step',step+1,rows[-1]['material_pass'],rows[-1]['maximum_root_error_m'],flush=True)
    if a.floor:result['final_support_balance']=planar_support_diagnostic()
    np.savez_compressed(out/'states.npz',time_s=times,rigid_names=list(rigid),rigid_poses=poses,**states)
    result.update(surface_count=len(surfaces),rows=rows,native_physics_errors=errors.report(),
                  candidate_sha256={'model.py':hashlib.sha256(Path(a.model).read_bytes()).hexdigest(),
                                    'mesh.npz':hashlib.sha256(Path(a.mesh).read_bytes()).hexdigest()})
    result['implementation_sha256']=inputs
    result['implementation_unchanged']=all(Path(p).is_file() and hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in inputs.items())
    result['gravity_loading']={'initialization_only':bool(a.gravity_ramp_seconds),'ramp_seconds':a.gravity_ramp_seconds,
                              'final_constant_gravity_m_s2':a.gravity,'no_delivered_gravity_animation':True}
    result['pass']=bool(errors.report()['pass'] and result['implementation_unchanged'] and
                        all(r['material_pass'] and r['maximum_root_error_m']<=.0001
                            and all(w['maximum_com_gap_m']<=.00001 and w['maximum_rotation_error_rad']<=.01 for w in r['weld_errors'].values()) for r in rows))
except Exception:
    result['error']=traceback.format_exc();result['native_physics_errors']=errors.report()
finally:
    result['producer']={'path':__file__,'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('RESULT',result['pass'],result.get('error',''),flush=True)
    errors.close();app.close()
