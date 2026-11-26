"""Measure native rest-distance filtering; disconnected probes are not assets."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import traceback

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
p.add_argument('--disable-filter',action='store_true')
p.add_argument('--approach',action='store_true',help='Vary cooked separation, keep declared rest offset fixed and apply one finite initial velocity impulse')
p.add_argument('--cook-material-rest',action='store_true',help='Cook separated material geometry, then author the evaluation pose once before physical time zero')
args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
if args.approach and args.cook_material_rest:
    p.error('--approach and --cook-material-rest are separate controls')
shutil.copy2(__file__,out/'source.py')
source_sha=hashlib.sha256((out/'source.py').read_bytes()).hexdigest()

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402
app=SimulationApp({'headless':True})

import numpy as np  # noqa: E402
import torch  # noqa: E402
import omni.usd  # noqa: E402
import omni.physics.tensors as tensors  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from omni.physx.scripts import particleUtils  # noqa: E402
from pxr import UsdGeom,UsdPhysics,UsdShade,PhysxSchema,Vt  # noqa: E402
from calibrated_scene_reconstruction.sim.physics_errors import PhysicsErrors  # noqa: E402
from calibrated_scene_reconstruction.sim import experimental_startup  # noqa: E402

shutil.copy2(experimental_startup.__file__,out/'experimental_startup.py')
startup_sha=hashlib.sha256((out/'experimental_startup.py').read_bytes()).hexdigest()

errors=PhysicsErrors(out/'native_physics_errors.json')
try:
    hz=7680;radius=.00042;particle_contact=.0005
    alpha=np.array([.0003,.00042,.0006,.0008,.00084,.0009,.001,.00105,.0012,.0015,.0017,
                    .002,.004,.008,.016,.024,.028,.03,.032,.04,.06,.10,.16])
    tiny=np.array([[0.,0,0],[1e-6,0,0],[0,1e-6,0]])
    points=[];rest=[];faces=[];springs=[]
    for i,value in enumerate(alpha):
        for side in range(2):
            start=len(points)*3
            base=.5*i if args.cook_material_rest else .005*i
            points.append(tiny+[base,0,.1+side*(value if args.approach else .0004)])
            rest.append(tiny+[(.5*i if args.cook_material_rest else .01*i)+side*(.0003 if args.approach else value),0,0])
            faces.append([start,start+1,start+2]);springs.extend([[start,start+1],[start+1,start+2],[start+2,start]])
    points=np.vstack(points);rest=np.vstack(rest);faces=np.asarray(faces);springs=np.asarray(springs)
    cooked_points=rest+[0.,0.,.1] if args.cook_material_rest else points.copy()
    lengths=np.linalg.norm(rest[springs[:,0]]-rest[springs[:,1]],axis=1)
    world=World(stage_units_in_meters=1.,physics_dt=1/hz,device='cuda:0',backend='torch');errors.guard_steps(world)
    pc=world.get_physics_context();pc.enable_gpu_dynamics(True);pc.set_broadphase_type('GPU');pc.set_solver_type('PGS')
    stage=omni.usd.get_context().get_stage();scene=next(q for q in stage.Traverse() if q.IsA(UsdPhysics.Scene))
    UsdPhysics.Scene(scene).GetGravityMagnitudeAttr().Set(0.)
    system=particleUtils.add_physx_particle_system(stage,'/World/System',simulation_owner=scene.GetPath(),
        contact_offset=.001,rest_offset=.0004,solid_rest_offset=radius,particle_contact_offset=particle_contact,
        solver_position_iterations=128,max_neighborhood=256,max_depenetration_velocity=.1)
    particleUtils.add_pbd_particle_material(stage,'/World/Material',friction=0.,damping=0.,drag=0.,lift=0.)
    UsdShade.MaterialBindingAPI.Apply(system.GetPrim()).Bind(UsdShade.Material.Get(stage,'/World/Material'),UsdShade.Tokens.weakerThanDescendants,'physics')
    mesh=UsdGeom.Mesh.Define(stage,'/World/Probe');normals=np.zeros_like(points);normals[:,2]=1.
    particleUtils.add_physx_particle_cloth_with_constraints(stage,mesh.GetPath(),system.GetPath(),
        Vt.Vec3fArray.FromNumpy(cooked_points.astype('float32')),Vt.Vec3fArray.FromNumpy(normals.astype('float32')),
        Vt.Vec3fArray.FromNumpy(rest.astype('float32')),Vt.Vec3fArray.FromNumpy(np.zeros_like(normals,dtype='float32')),
        Vt.IntArray(faces.ravel().tolist()),Vt.Vec2iArray.FromNumpy(springs.astype('int32')),
        Vt.FloatArray([0.]*len(springs)),Vt.FloatArray([0.]*len(springs)),Vt.FloatArray(lengths.tolist()),
        self_collision=True,self_collision_filter=not args.disable_filter)
    UsdPhysics.MassAPI.Apply(mesh.GetPrim()).CreateMassAttr(.001)
    stage.GetRootLayer().Export(str(out/'initial.usda'));world.reset()
    native_reset_time_s=world.current_time
    owner=tensors.create_simulation_view('torch');owner.set_subspace_roots('/');view=owner.create_particle_cloth_view('/World/Probe')
    if view.count!=1 or np.any(view.get_spring_stiffness().cpu().numpy()) or np.any(view.get_spring_damping().cpu().numpy()):
        raise ValueError('Native cloth or disabled spring readback differs')
    if PhysxSchema.PhysxParticleClothAPI(mesh).GetSelfCollisionFilterAttr().Get()!=(not args.disable_filter):
        raise ValueError('Native filter declaration differs')
    material=owner.create_particle_material_view('/World/Material')
    if np.any(material.get_friction().cpu().numpy()):raise ValueError('Friction must remain zero')
    cooked_readback=view.get_positions()[0].reshape(-1,3).cpu().numpy().copy()
    initialization_error=None;initial_pose_report=None
    if args.cook_material_rest:
        if not np.allclose(cooked_readback,cooked_points,atol=2e-6,rtol=0.):
            raise ValueError('Cooked material geometry changed before initial-pose authoring')
        initial_pose_report=experimental_startup.InitialMaterialPose(cooked_points,points).apply(view,time_s=world.current_time-native_reset_time_s)
        initial_pose_report['native_reset_clock_origin_s']=native_reset_time_s
        initialization_error=initial_pose_report['initial_pose_readback_maximum_error_m']
    states=[view.get_positions()[0].reshape(-1,3).cpu().numpy().copy()]
    initial_velocity=np.zeros_like(points);steps=400 if args.approach else 100
    if args.approach:
        for i in range(len(alpha)):initial_velocity[6*i+3:6*i+6,2]=-.05
        view.set_velocities(torch.as_tensor(initial_velocity,dtype=torch.float32,device='cuda:0').reshape(1,-1),
                            torch.zeros(1,dtype=torch.int32,device='cuda:0'))
    for step in range(steps):
        world.step(render=False)
        if step in [0,9,steps-1]:states.append(view.get_positions()[0].reshape(-1,3).cpu().numpy().copy())
    errors.check();rows=[]
    for i,value in enumerate(alpha):
        region=slice(6*i,6*i+6)
        free=states[0][region]+initial_velocity[region]*steps/hz
        maximum=float(np.linalg.norm(states[-1][region]-free,axis=1).max())
        rows.append({'reference_offset_m':float(value),'maximum_native_displacement_m':maximum,
                     'cooked_gap_m':float(value) if args.approach or args.cook_material_rest else .0004,
                     'declared_rest_offset_m':.0003 if args.approach else float(value),
                     'excitation_reaches_contact':bool(not args.approach or value-.05*steps/hz<2*radius),
                     'contact_response':maximum>5e-5})
    unchanged=(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==source_sha and
               hashlib.sha256(Path(experimental_startup.__file__).read_bytes()).hexdigest()==startup_sha)
    result={'complete':True,'task_approved':False,'production_backend_registered':False,'diagnostic_only':True,
        'filter_enabled':not args.disable_filter,'native_springs_disabled_readback':True,'zero_friction_readback':True,
        'no_gravity_or_external_forces':not args.approach,'initial_velocity_impulse_m_s':.05 if args.approach else 0.,
        'positions_written_after_creation':bool(args.cook_material_rest),
        'initial_pose_writes_before_recorded_t0':int(args.cook_material_rest),
        'runtime_position_writes':0,'material_rest_cooking':args.cook_material_rest,
        'initial_pose_readback_maximum_error_m':initialization_error,'physics_hz':hz,
        'initial_pose_report':initial_pose_report,'startup_source_sha256':startup_sha,
        'solid_rest_offset_m':radius,'particle_contact_offset_m':particle_contact,'rows':rows,
        'source_sha256':source_sha,'source_unchanged_during_control':unchanged,'native_physics_errors':errors.report(),
        'scope':'Native rest-distance filter response only. Pair geometry is deliberately disconnected; not material/contact/task qualification.'}
    np.savez_compressed(out/'states.npz',points=states,authored=points,cooked=cooked_readback,rest=rest,faces=faces)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('REST_FILTER_RESULT',json.dumps(rows),flush=True)
except Exception as error:
    (out/'error.json').write_text(json.dumps({'pass':False,'error':str(error),'traceback':traceback.format_exc()},indent=2)+'\n');raise
finally:errors.close();app.close()
