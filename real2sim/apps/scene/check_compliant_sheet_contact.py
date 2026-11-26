"""Synthetic compliant-contents / surface-sheet contact capability control.

Explicit force-limited platen supplies preload. A native volume body remains free,
supported by rigid boundaries, with no fixed nodes. Cloth-only finite loads
are diagnostic applied impulses, never a task grasp or a delivered holder.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import traceback

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--profile',required=True)
p.add_argument('--output',required=True)
p.add_argument('--friction',type=float,required=True)
p.add_argument('--compression-m',type=float,default=.0008)
p.add_argument('--modulus-pa',type=float,default=20000.)
p.add_argument('--preload-limit-n',type=float,default=.12)
p.add_argument('--zero-friction-after-preload',action='store_true',
               help='Matched control: prepare at nominal friction, then edit existing materials before loading')
p.add_argument('--contents-backend',choices=['beta','legacy'],default='beta')
args=p.parse_args()
if (not 0<=args.friction<=2 or not 0<=args.compression_m<=.001
        or not 1000<=args.modulus_pa<=1e6 or not 0<args.preload_limit_n<=1.):
    p.error('Bounded explicit synthetic material/preload required')
profile=json.loads(Path(args.profile).read_text())
out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
shutil.copy2(__file__,out/'source.py');shutil.copy2(args.profile,out/'profile.json')
(out/'protocol.json').write_text(json.dumps(vars(args),indent=2)+'\n')

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402
app=SimulationApp({'headless':True})
import numpy as np  # noqa: E402
import torch  # noqa: E402
import omni.usd  # noqa: E402
import omni.physics.tensors as tensors  # noqa: E402
from omni.physx.scripts import deformableUtils,deformableMeshUtils  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from pxr import Gf,PhysxSchema,UsdGeom,UsdPhysics,UsdShade,Vt  # noqa: E402
from calibrated_scene_reconstruction.assets.cloth_rest import triangulate_grid  # noqa: E402
from calibrated_scene_reconstruction.assets.cloth_metric import MaterialMetric  # noqa: E402
from calibrated_scene_reconstruction.assets.cloth_contact_tolerance import solid_tolerance  # noqa: E402
from calibrated_scene_reconstruction.sim.surface_cloth import author,SurfaceView  # noqa: E402
from calibrated_scene_reconstruction.sim import volume_fixture  # noqa: E402
from calibrated_scene_reconstruction.sim.usd import add_box_collider  # noqa: E402
from calibrated_scene_reconstruction.sim.physics import physics_material  # noqa: E402
from calibrated_scene_reconstruction.sim.simulation import apply as apply_simulation  # noqa: E402
from calibrated_scene_reconstruction.sim.physics_errors import PhysicsErrors  # noqa: E402
from calibrated_scene_reconstruction.sim.cloth_contact import self_intersections  # noqa: E402
from calibrated_scene_reconstruction.sim.surface_distance import SignedSurface,triangle_samples  # noqa: E402
from calibrated_scene_reconstruction.tasks.rigid.task_checks import as_numpy  # noqa: E402

errors=PhysicsErrors(out/'native_physics_errors.json')
try:
    hz=profile['physics_hz'];dt=1/hz
    settings={**profile['settings'],'friction':args.friction}
    solid_budget=solid_tolerance(settings['thickness'],0.)
    world=World(stage_units_in_meters=1.,physics_dt=dt,device='cuda:0',backend='torch')
    errors.guard_steps(world)
    pc=world.get_physics_context();pc.enable_gpu_dynamics(True);pc.set_broadphase_type('GPU')
    stage=omni.usd.get_context().get_stage()
    apply_simulation(stage,{'physics_hz':hz,'solver':'PGS','friction_type':'patch',
                           'basis':'Synthetic beta volume / surface contact control'},flexible=True)
    scene=UsdPhysics.Scene(next(q for q in stage.Traverse() if q.IsA(UsdPhysics.Scene)))
    scene.GetGravityMagnitudeAttr().Set(0.)
    rigid_mat=physics_material(stage,'/World/RigidMaterial',args.friction,args.friction,0.)
    boxes=[('Back',[-.014,-.03,-.003],[-.010,.03,.06]),
           ('Floor',[-.014,-.03,-.003],[.004,.03,0.]),
           ('Side0',[-.014,-.024,-.003],[.004,-.0205,.06]),
           ('Side1',[-.014,.0205,-.003],[.004,.024,.06]),
           ('Platen',[.0009,-.020,.001],[.0049,.020,.06])]
    # Moving fixtures must not start in, or sweep into, unrelated rigid walls.
    platen_lo=np.array(boxes[-1][1]);platen_hi=np.array(boxes[-1][2])
    swept_lo=platen_lo-np.array([args.compression_m,0.,0.])
    for name,lo,hi in boxes[:-1]:
        if np.all(np.minimum(platen_hi,hi)>np.maximum(swept_lo,lo)):
            raise ValueError('Preload platen intersects rigid fixture '+name)
    for name,lo,hi in boxes:
        cube=add_box_collider(stage,'/World/'+name,lo,hi)
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(rigid_mat,UsdShade.Tokens.weakerThanDescendants,'physics')
        collision=PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
        collision.CreateContactOffsetAttr(.0003);collision.CreateRestOffsetAttr(0.)
        if name=='Platen':
            UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim()).CreateKinematicEnabledAttr(False)
            UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(.1)
    base_position=(platen_lo+platen_hi)/2
    slider=UsdPhysics.PrismaticJoint.Define(stage,'/World/PreloadSlider')
    slider.CreateBody1Rel().SetTargets(['/World/Platen'])
    slider.CreateAxisAttr('X');slider.CreateLocalPos0Attr(Gf.Vec3f(*base_position))
    slider.CreateLocalPos1Attr(Gf.Vec3f(0.))
    slider.CreateLowerLimitAttr(-.002);slider.CreateUpperLimitAttr(.001)
    drive=UsdPhysics.DriveAPI.Apply(slider.GetPrim(),'linear')
    drive.CreateTypeAttr('force');drive.CreateStiffnessAttr(2000.);drive.CreateDampingAttr(30.)
    drive.CreateMaxForceAttr(args.preload_limit_n);drive.CreateTargetPositionAttr(0.)
    v,t=deformableMeshUtils.createTetraVoxelBox(8)
    v=np.asarray(v)*[.01,.04,.04]+[-.005,0.,.0204]
    t=np.asarray(t,dtype=int).reshape(-1,4)
    prim,material_path=volume_fixture.author(stage,'/World/CompliantContents',v,t,
        backend=args.contents_backend,youngs_modulus=args.modulus_pa,friction=args.friction)
    u=np.linspace(-.01,.01,51);w=np.linspace(.006,.026,51)
    uu,ww=np.meshgrid(u,w)
    points=np.c_[np.full(uu.size,.0005),uu.ravel(),ww.ravel()]
    rest=np.c_[uu.ravel(),ww.ravel(),np.zeros(uu.size)]
    faces,chart,_=triangulate_grid(points,rest,len(w),len(u))
    sheet=UsdGeom.Mesh.Define(stage,'/World/Sheet')
    sheet.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype('float32')))
    sheet.CreateFaceVertexCountsAttr([3]*len(faces));sheet.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
    sheet.CreateSubdivisionSchemeAttr('none')
    part={'V':points,'F':faces,'cloth_rest':chart,'kind':'cloth','backend':'surface_fem'}
    author(stage,sheet,part,settings)
    stage.GetRootLayer().Export(str(out/'initial.usda'))
    world.reset();errors.check()
    owner=tensors.create_simulation_view('torch');owner.set_subspace_roots('/')
    volume=volume_fixture.View(owner,prim,args.contents_backend)
    material_readback={};native_materials={}
    for name,path,expected in [('contents',material_path,args.modulus_pa),
                               ('cloth','/World/Sheet_surface_fem_material',settings['youngs_modulus'])]:
        create_material=(owner.create_soft_body_material_view if name=='contents' and args.contents_backend=='legacy'
                         else owner.create_deformable_material_view)
        native_material=create_material(path)
        native_materials[name]=native_material
        if native_material is None or native_material.count!=1:
            raise ValueError('Native material unavailable: '+name)
        observed={'youngs_modulus':float(as_numpy(native_material.get_youngs_modulus()).ravel()[0]),
                  'dynamic_friction':float(as_numpy(native_material.get_dynamic_friction()).ravel()[0])}
        material_readback[name]=observed
        if not np.isclose(observed['youngs_modulus'],expected,rtol=1e-5) or not np.isclose(
                observed['dynamic_friction'],args.friction,rtol=1e-5,atol=1e-8):
            raise ValueError('Actual native material differs from protocol: '+name+' '+str(observed))
    (out/'native_material_readback.json').write_text(json.dumps(material_readback,indent=2)+'\n')
    indices=volume.tetrahedra
    volume_rest=volume.rest
    def determinants(q):
        a=q[indices]
        return np.linalg.det(np.stack([a[:,1]-a[:,0],a[:,2]-a[:,0],a[:,3]-a[:,0]],axis=2))
    det0=determinants(volume_rest)
    if np.any(abs(det0)<1e-16):raise ValueError('Degenerate native volume cells')
    surface_faces=volume.faces
    # Native rest correspondence must preserve indexing for the surface query.
    if args.contents_backend=='beta' and (volume_rest.shape!=v.shape or not np.allclose(volume_rest,v,atol=1e-7)):
        raise ValueError('Native volume ordering changed; independent mapping required')
    view=SurfaceView(stage,'/World/Sheet',part);view.require_validated_force_profile()
    masses=view.get_particle_masses()[0];mass=float(masses.sum())
    metric=MaterialMetric(points,faces,chart,max_edge_strain=.05,max_area_change=.03)
    platen=owner.create_rigid_body_view('/World/Platen')
    if platen.count!=1:raise ValueError('Native kinematic platen missing')
    platen_target=platen.get_transforms().clone()
    initial_platen=as_numpy(platen_target)[0,:3]
    (out/'native_initial_platen.json').write_text(json.dumps({'actual':initial_platen.tolist(),
        'authored':base_position.tolist(),'difference':(initial_platen-base_position).tolist()},indent=2)+'\n')
    if (not np.allclose(initial_platen[1:],base_position[1:],atol=1e-5)
            or abs(initial_platen[0]-base_position[0])>.001):
        raise ValueError('Native platen escaped its initial slider frame: '+str(initial_platen))
    initial=as_numpy(view.get_world_positions())[0].copy()
    states=[];volume_states=[];rows=[];abort=None
    zero_update=None;preload_points=None
    def record(step,force):
        actual=as_numpy(view.get_world_positions())[0].copy()
        stock=volume.collision_points()
        j=determinants(volume.simulation_points())/det0
        material=metric.check(actual);cross=self_intersections(actual,faces,example_limit=2)
        signed=SignedSurface(stock,surface_faces)
        distance=float(signed(triangle_samples(actual,faces)).min())
        platen_pose=as_numpy(platen.get_transforms())[0].copy()
        platen_left=float(platen_pose[0]-.002)
        rigid_gap=float(platen_left-actual[:,0].max())
        row={'time_s':step*dt,'applied_cloth_force_N':force,
             'upward_cloth_displacement_m':float(np.mean(actual[:,2]-initial[:,2])),
             'cloth_material':material,'self_intersection':cross,
             'minimum_volume_jacobian':float(j.min()),'maximum_volume_jacobian':float(j.max()),
             'stock_extent_m':np.ptp(stock,axis=0).tolist(),
             'minimum_cloth_to_stock_distance_m':distance,
             'native_platen_pose_xyzw':platen_pose.tolist(),
             'native_platen_travel_m':float(base_position[0]-platen_pose[0]),
             'preload_drive_max_force_n':float(drive.GetMaxForceAttr().Get()),
             'minimum_cloth_to_platen_distance_m':rigid_gap,
             'whole_coupon_above_stock':bool(actual[:,2].min()>stock[:,2].max()+settings['thickness']/2),
             'platen_contact_thickness_pass':rigid_gap>=settings['thickness']/2-2e-6,
             'stock_contact_thickness_pass':distance>=settings['thickness']/2-2e-6}
        row['shared_solid_contact_budget_m']=solid_budget
        row['solid_contact_pass']=bool(min(distance,rigid_gap)>=-solid_budget)
        row['raw_maximum_thickness_envelope_overlap_m']=float(max(0.,settings['thickness']/2-min(distance,rigid_gap)))
        states.append(actual);volume_states.append(stock);rows.append(row)
        (out/'progress.json').write_text(json.dumps(row,indent=2)+'\n')
        return row
    record(0,0.)
    for step in range(1,round(1.1*hz)+1):
        seconds=step*dt
        compression=args.compression_m*min(1.,seconds/.15)
        drive.GetTargetPositionAttr().Set(-compression)
        if seconds>=.25 and preload_points is None:
            preload_points=as_numpy(view.get_world_positions())[0].copy()
            if args.zero_friction_after_preload:
                for path in [material_path,'/World/Sheet_surface_fem_material']:
                    mat_prim=stage.GetPrimAtPath(path)
                    if path==material_path and args.contents_backend=='legacy':
                        PhysxSchema.PhysxDeformableBodyMaterialAPI(mat_prim).CreateDynamicFrictionAttr(0.)
                    else:
                        mat_prim.GetAttribute('omniphysics:dynamicFriction').Set(0.)
                rigid_api=UsdPhysics.MaterialAPI(rigid_mat.GetPrim())
                rigid_api.GetStaticFrictionAttr().Set(0.);rigid_api.GetDynamicFrictionAttr().Set(0.)
                zero_update={'method':'existing_usd_materials','time_s':seconds,
                             'native_readback_after_step':False}
        if seconds<.25:force=0.
        elif seconds<.85:force=.03*min(1.,(seconds-.25)/.05)
        else:force=.03+.27*min(1.,(seconds-.85)/.05)
        if force:
            velocity=view.get_velocities().clone()
            velocity[:,:,2]+=force/mass*dt
            view.set_velocities(velocity)
        world.step(render=False)
        if zero_update is not None and not zero_update['native_readback_after_step']:
            coefficients={name:float(as_numpy(mat.get_dynamic_friction()).ravel()[0])
                          for name,mat in native_materials.items()}
            rigid_coefficients=as_numpy(platen.get_material_properties())[...,:2]
            if any(value!=0. for value in coefficients.values()) or np.any(rigid_coefficients!=0.):
                raise ValueError('Native friction-zero update did not persist: '+str(coefficients))
            zero_update.update(native_readback_after_step=True,deformable_coefficients=coefficients,
                               rigid_coefficients=rigid_coefficients.tolist())
        if step%round((.01 if seconds<.85 else .0025)*hz)==0:
            row=record(step,force)
            if (row['minimum_volume_jacobian']<=0 or not row['cloth_material']['geometry_valid']
                    or not row['self_intersection']['pass'] or not row['solid_contact_pass']):
                abort='Invalid contact or volume state';break
            if abs(row['upward_cloth_displacement_m'])>.035:
                abort='Declared fixture travel limit';break
    errors.check()
    np.savez_compressed(out/'states.npz',cloth=states,stock=volume_states,faces=faces,stock_faces=surface_faces,
                        tetrahedra=indices,stock_rest=volume_rest,rest=rest,preload_points=preload_points)
    result={'synthetic_control':True,'task_approved':False,'source_approved':False,
            'volume_backend':'Isaac5.1 '+args.contents_backend+' native volume FEM',
            'legacy_volume_used':args.contents_backend=='legacy','cloth_backend':'surface_fem',
            'kinematic_preload_platen':False,'force_limited_preload_platen':True,
            'preload_force_limit_n':args.preload_limit_n,'contents_rest_offset_m':0.,
            'cloth_pins':False,'volume_pins':False,
            'cloth_force_adapter':'Explicit uniform finite diagnostic impulse on qualified PGS surface adapter, not a friction grasp',
            'zero_gravity':True,'friction':args.friction,'compression_m':args.compression_m,
            'contents_modulus_pa':args.modulus_pa,'raw_material_limits_pass':all(x['cloth_material']['pass'] for x in rows),
            'native_material_readback':material_readback,
            'zero_friction_after_preload':args.zero_friction_after_preload,
            'zero_friction_update':zero_update,
            'solid_contact_policy':'Existing assets.cloth_contact_tolerance.solid_tolerance, independent of numerical padding. Raw half-thickness envelope failures retained separately.',
            'profile_sha256':hashlib.sha256(Path(args.profile).read_bytes()).hexdigest(),
            'abort':abort,'rows':rows,'native_physics_errors':errors.report(),
            'scope':'Compliant contents/sheet contact capability only. Fixed external platen and synthetic loads do not approve a free canister, inferred stock properties or task interaction.'}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('COMPLIANT_SHEET_RESULT',json.dumps({'abort':abort,'displacement_m':rows[-1]['upward_cloth_displacement_m']}),flush=True)
except Exception as error:
    (out/'error.json').write_text(json.dumps({'error':str(error),'traceback':traceback.format_exc()},indent=2)+'\n')
    raise
finally:
    errors.close();app.close()
