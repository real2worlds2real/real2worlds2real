"""Native surface bending capability control; synthetic, never task evidence."""
import argparse
import json
from pathlib import Path
import shutil
import traceback

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',required=True)
parser.add_argument('--rigidity',type=float,required=True,help='Synthetic flexural rigidity, N*m')
parser.add_argument('--reference',choices=['flat','retained','retained_reverse','signed','signed_reverse','per_triangle','force_flat','force_retained','coupled_flat','coupled_retained'],default='flat')
parser.add_argument('--physics-hz',type=int,choices=[240,480,960,1920,3840,7680],default=1920)
parser.add_argument('--solver',choices=['PGS','TGS'],default='PGS')
parser.add_argument('--translation',type=float,nargs=3,default=[1.,2.,.4],
                    help='Synthetic world translation for matched precision controls')
parser.add_argument('--binding-control',choices=['physics_only','visual_default','combined'],default='physics_only',
                    help='Independent test of a graphical default material versus the physics-purpose binding')
parser.add_argument('--position-iterations',type=int,default=128)
args=parser.parse_args()
if not 0<args.rigidity<=.01:parser.error('Positive bounded control rigidity required')
if args.solver!='PGS' and args.reference.startswith(('force_','coupled_')):
    parser.error('The nodal force adapter requires a separately validated solver profile')
out=Path(args.output);out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,out/'source.py')
(out/'protocol.json').write_text(json.dumps(vars(args),indent=2)+'\n')

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402
app=SimulationApp({'headless':True})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from pxr import PhysxSchema,UsdGeom,UsdPhysics,Vt  # noqa: E402
from calibrated_scene_reconstruction.assets.strip_reference import extruded  # noqa: E402
from calibrated_scene_reconstruction.assets.cloth_rest import triangulate_grid  # noqa: E402
from calibrated_scene_reconstruction.assets.cloth_metric import MaterialMetric  # noqa: E402
from calibrated_scene_reconstruction.sim.surface_cloth import author,SurfaceView,material_values  # noqa: E402
from calibrated_scene_reconstruction.sim.cloth_contact import self_intersections  # noqa: E402
from calibrated_scene_reconstruction.sim.physics_errors import PhysicsErrors  # noqa: E402
from calibrated_scene_reconstruction.sim.simulation import apply as apply_simulation  # noqa: E402
from calibrated_scene_reconstruction.tasks.mechanisms.cloth import nonrigid_change  # noqa: E402
from calibrated_scene_reconstruction.tasks.rigid.task_checks import as_numpy  # noqa: E402

physics_errors=PhysicsErrors(out/'native_physics_errors.json')
try:
    world=World(stage_units_in_meters=1.,physics_dt=1/args.physics_hz,device='cuda:0',backend='torch')
    physics_errors.guard_steps(world)
    pc=world.get_physics_context();pc.enable_gpu_dynamics(True);pc.set_broadphase_type('GPU')
    stage=omni.usd.get_context().get_stage()
    profile={'physics_hz':args.physics_hz,'solver':args.solver,'friction_type':'patch','basis':'Synthetic free-strip bending capability control'}
    apply_simulation(stage,profile,flexible=True)
    scene=next(p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene))
    UsdPhysics.Scene(scene).GetGravityMagnitudeAttr().Set(0.)
    PhysxSchema.PhysxSceneAPI.Apply(scene).CreateEnableCCDAttr(False)
    u=np.linspace(-.02,.02,21);v=np.linspace(0,.08,41)
    x,y=np.meshgrid(u,v);rest=np.c_[x.ravel(),y.ravel(),np.zeros(x.size)]
    angles=np.linspace(-.6,.6,len(v)-1);curved=extruded(u,v,angles)
    if not np.isfinite(args.translation).all():raise ValueError('Finite synthetic translation required')
    points=curved+args.translation
    faces,chart,_=triangulate_grid(points,rest,len(v),len(u))
    if args.reference in {'retained','retained_reverse'}:
        natural=curved.copy()
        if args.reference=='retained_reverse':natural[:,2]*=-1
        chart['bend_reference']={'points':natural.tolist(),'source':'Synthetic exact isometric cylindrical reference',
                                  'basis':'Independent geometry control, not measured tissue properties'}
    elif args.reference.startswith('signed'):
        from calibrated_scene_reconstruction.assets.cloth_curvature import measured
        pairs,degrees=measured(curved,faces)
        if args.reference=='signed_reverse':degrees=-degrees
        chart['bend_angles']={'pairs':pairs.tolist(),'degrees':degrees.tolist(),
            'source':'Synthetic cylindrical geometry with unchanged flat membrane chart',
            'basis':'Independent signed-angle capability/sign control; not tissue identification'}
    part={'V':points,'F':faces,'cloth_rest':chart,'kind':'cloth','backend':'surface_fem'}
    mesh=UsdGeom.Mesh.Define(stage,'/World/ControlSheet')
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype('float32')))
    mesh.CreateFaceVertexCountsAttr([3]*len(faces));mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
    mesh.CreateSubdivisionSchemeAttr('none')
    if args.binding_control!='physics_only':
        from pxr import UsdShade,Sdf
        visual=UsdShade.Material.Define(stage,'/World/ControlVisual')
        shader=UsdShade.Shader.Define(stage,'/World/ControlVisual/Shader');shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set((.8,.8,.8))
        visual.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(visual)
    settings={'youngs_modulus':4e6,'poissons_ratio':.2,'thickness':.0004,'flexural_rigidity':args.rigidity,
        'areal_density':.2,'elasticity_damping':0.,'linear_damping':0.,'friction':0.,
        'contact_offset':.0003,'rest_offset':.0002,'self_collision_filter_distance':.0004,
        'solver_position_iterations':args.position_iterations,'collision_pair_updates':16,'collision_iteration_multiplier':2,
        'max_depenetration_velocity':.2}
    author(stage,mesh,part,settings,diagnostic_bend_angles=args.reference.startswith('signed'))
    coupled_control=args.reference.startswith('coupled_')
    force_control=args.reference.startswith('force_') or coupled_control
    if force_control:
        # Prototype control only. Native FEM still owns membrane and contacts;
        # disable its bending before testing the separately declared hinge law.
        physical_path=mesh.GetPrim().GetRelationship('material:binding:physics').GetTargets()[0]
        physical=stage.GetPrimAtPath(physical_path)
        physical.GetAttribute('omniphysics:surfaceBendStiffness').Set(0.)
        physical.GetAttribute('physxDeformableMaterial:bendDamping').Set(0.)
        if coupled_control:
            physical.GetAttribute('omniphysics:youngsModulus').Set(0.)
            physical.GetAttribute('physxDeformableMaterial:elasticityDamping').Set(0.)
    if args.binding_control=='combined':
        binding=UsdShade.MaterialBindingAPI(mesh.GetPrim())
        physics_material=binding.ComputeBoundMaterial('physics')[0]
        binding.Bind(physics_material)
    if args.reference=='per_triangle':
        from pxr import Sdf
        prim=mesh.GetPrim()
        prim.GetAttribute('omniphysics:restShapePoints').Set(Vt.Vec3fArray.FromNumpy(curved[faces].reshape(-1,3).astype('float32')))
        prim.GetAttribute('omniphysics:restTriVtxIndices').Set(Vt.Vec3iArray.FromNumpy(np.arange(faces.size,dtype=np.int32).reshape(-1,3)))
        prim.GetAttribute('omniphysics:restBendAnglesDefault').Set('restShapeDefault')
    (out/'material_authoring.json').write_text(json.dumps({'settings':settings,'derived':material_values(settings,rest,faces)},indent=2)+'\n')
    stage.GetRootLayer().Export(str(out/'initial.usda'));world.reset()
    if args.reference=='per_triangle':
        import omni.physics.tensors as tensors
        owner=tensors.create_simulation_view('torch');owner.set_subspace_roots('/')
        probe=owner.create_surface_deformable_body_view('/World/ControlSheet')
        if probe is not None:
            np.savez_compressed(out/'native_index_probe.npz',rest=as_numpy(probe.get_rest_nodal_positions()),
                                points=as_numpy(probe.get_simulation_nodal_positions()),faces=as_numpy(probe.get_simulation_element_indices()))
    view=SurfaceView(stage,'/World/ControlSheet',part)
    initial=as_numpy(view.get_world_positions())[0].copy();masses=as_numpy(view.get_particle_masses())[0]
    bending=None;membrane=None;internal_impulses=[]
    if force_control:
        import torch
        from calibrated_scene_reconstruction.assets.cloth_curvature import measured
        from calibrated_scene_reconstruction.sim.discrete_bending import DiscreteBending
        pairs,degrees=measured(curved,faces)
        natural=np.radians(degrees) if args.reference.endswith('_retained') else np.zeros(len(pairs))
        bending=DiscreteBending(rest,faces,natural,args.rigidity,device='cuda:0',dtype=torch.float64)
        if coupled_control:
            from calibrated_scene_reconstruction.sim.discrete_membrane import DiscreteMembrane
            membrane=DiscreteMembrane(rest,faces,**{key:settings[key] for key in ['youngs_modulus','poissons_ratio','thickness']},device='cuda:0',dtype=torch.float64)
    metric=MaterialMetric(initial,faces,chart,max_edge_strain=.05,max_area_change=.03)
    states=[];rows=[];velocities=[];initial_com=(initial*masses[:,None]).sum(0)/masses.sum()
    def record(step):
        actual=as_numpy(view.get_world_positions())[0].copy();velocity=as_numpy(view.get_velocities())[0].copy()
        material=metric.check(actual);cross=self_intersections(actual,faces,example_limit=2)
        centreline=actual.reshape(len(v),len(u),3)[:,len(u)//2]
        first=centreline[2]-centreline[0];last=centreline[-1]-centreline[-3]
        angle=float(np.arccos(np.clip(np.dot(first,last)/(np.linalg.norm(first)*np.linalg.norm(last)),-1.,1.)))
        com=(actual*masses[:,None]).sum(0)/masses.sum();momentum=(velocity*masses[:,None]).sum(0)
        row={'time_s':step/args.physics_hz,'bend_angle_rad':angle,'material':material,'self_intersection':cross,
            'centre_of_mass_drift_m':float(np.linalg.norm(com-initial_com)),'momentum_norm_Ns':float(np.linalg.norm(momentum)),
            'kinetic_energy_J':float(.5*np.sum(masses[:,None]*velocity**2)),
            'nonrigid_change':nonrigid_change(initial,actual)}
        states.append(actual);velocities.append(velocity);rows.append(row)
        (out/'progress.json').write_text(json.dumps(row,indent=2)+'\n')
    record(0)
    stride=round(.01*args.physics_hz);steps=round(.2*args.physics_hz)
    for step in range(steps):
        if bending is not None:
            current=view.get_world_positions()[0];velocity=view.get_velocities()[0]
            update,internal=bending.velocity_step(current,velocity,masses,1/args.physics_hz,membrane=membrane)
            view.set_velocities(update.to(dtype=torch.float32)[None].contiguous());internal_impulses.append(internal)
        world.step(render=False)
        if (step+1)%stride==0 or step+1==steps:record(step+1)
    physics_errors.check()
    valid=all(r['material']['pass'] and r['self_intersection']['pass'] for r in rows)
    max_com=max(r['centre_of_mass_drift_m'] for r in rows);max_p=max(r['momentum_norm_Ns'] for r in rows)
    result={'synthetic_control':True,'not_tissue_task_evidence':True,'external_actuation':False,'gravity':False,
        'native_fixed_nodes':0,'native_backend':'surface_fem_with_experimental_bending' if force_control else 'surface_fem','solver':args.solver,'rigidity_Nm':args.rigidity,
        'physics_hz':args.physics_hz,
        'reference':args.reference,'material_and_self_contact_pass':valid,
        'experimental_monolithic_membrane_and_bending':coupled_control,
        'native_membrane_disabled':coupled_control,
        'binding_control':args.binding_control,
        'position_iterations':args.position_iterations,
        'world_translation_m':args.translation,
        'maximum_centre_of_mass_drift_m':max_com,'maximum_momentum_norm_Ns':max_p,
        'conservation_screen_pass':bool(max_com<=1e-5 and max_p<=1e-6),
        'initial_native_displacement_m':float(np.linalg.norm(initial-points,axis=1).max()),
        'maximum_nonrigid_change_m':max(r['nonrigid_change']['maximum_m'] for r in rows),
        'force_model':({'prototype_only':True,'native_bending_disabled':True,'positions_written':False,
            'method':'Internal discrete-shell bending impulses; native FEM membrane and contact splitting',
            'maximum_internal_linear_impulse_Ns':max(np.linalg.norm(r['linear_impulse_Ns']) for r in internal_impulses),
            'maximum_internal_angular_impulse_Nms':max(np.linalg.norm(r['angular_impulse_Nms']) for r in internal_impulses)} if force_control else None),
        'rows':rows,'native_physics_errors':physics_errors.report(),
        'scope':'Compare sampled trajectories across independently controlled rigidity/reference cases. Authored values alone do not establish implemented bending or physical parameter identification. No source/task approval.'}
    np.savez_compressed(out/'states.npz',points=states,velocities=velocities,faces=faces,rest=rest,masses=masses)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('SURFACE_BENDING_RESULT',json.dumps({k:v for k,v in result.items() if k not in ['rows','native_physics_errors']}),flush=True)
except Exception as error:
    (out/'error.json').write_text(json.dumps({'pass':False,'error':str(error),'traceback':traceback.format_exc()},indent=2)+'\n')
    raise
finally:
    physics_errors.close();app.close()
