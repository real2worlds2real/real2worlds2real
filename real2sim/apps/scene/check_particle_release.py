"""Native cloth release controls for diagnosing initial strain and damping.

Separate coupons share the same profile and authored prestrain. The default
release applies no forces or state writes after initialization and uses zero
gravity. Optional uniform gravity, exact zero-impulse writeback and finite-force
fold loading isolate the runtime adapter and coupled contact response.
This does not qualify any task asset.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import traceback

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--profile', required=True)
p.add_argument('--output', required=True)
p.add_argument('--damping', type=float, nargs='+',help='Explicit variants; loaded control defaults to the profile damping')
p.add_argument('--prestrain', type=float, default=.025)
p.add_argument('--seconds', type=float, default=.2)
p.add_argument('--shape', choices=['plane', 'cylinder', 'accordion'], default='plane')
p.add_argument('--size-m',type=float,default=.01,help='Synthetic square material size; actual --mesh size is never changed')
p.add_argument('--layer-gap-m',type=float,default=.0032,help='Resolved synthetic accordion separation')
p.add_argument('--loaded-fold',action='store_true',help='Synthetic finite-force bend/hold/release control, never source or task approval')
p.add_argument('--radius-m', type=float, default=.01)
p.add_argument('--bend-stiffness', type=float, nargs='+')
p.add_argument('--mesh',help='Actual candidate mesh.npz for unforced whole-shape isolation; no source/task approval')
p.add_argument('--cloth-link',default='wipe')
p.add_argument('--self-contact',choices=['on','off'],default='on')
p.add_argument('--bootstrap-hz',type=int,help='Explicit finer reset clock; actual warmup state is retained')
p.add_argument('--record-fps',type=int,default=160)
p.add_argument('--zero-impulse-writeback',action='store_true',
               help='Diagnostic: exercise the production source-indexed force adapter with exactly zero force every step')
p.add_argument('--gravity-m-s2',type=float,default=0.,help='Diagnostic uniform gravity in0..30; default zero, elevated values are synthetic loads')
args = p.parse_args()
from calibrated_scene_reconstruction.tasks.mechanisms.particle_control_profile import validate  # noqa: E402
profile = validate(json.loads(Path(args.profile).read_text()))
if args.damping is None:
    args.damping=[profile['spring_damping']] if args.loaded_fold else [1e-12,.001,.01,.1]
if not (0 <= args.prestrain <= .1 and 0 < args.seconds <= 2
        and 1 <= len(args.damping) <= 8 and all(0 <= d <= 10 for d in args.damping)):
    p.error('Bounded release experiment required')
if args.mesh and (args.prestrain!=0 or args.shape!='plane'):
    p.error('Actual shape isolation preserves its captured points; use zero added prestrain and no synthetic shape')
if not 0<=args.gravity_m_s2<=30:p.error('Bounded uniform diagnostic gravity required')
if not .005<=args.size_m<=.2:p.error('Bounded synthetic material size required')
if args.mesh and args.size_m!=.01:p.error('Actual mesh size cannot be rescaled by the synthetic size option')
if not 1<=args.record_fps<=profile['physics_hz'] or profile['physics_hz']%args.record_fps:
    p.error('Recording must divide the actual native clock')
if args.bootstrap_hz is not None and not profile['physics_hz']<=args.bootstrap_hz<=100000:
    p.error('Explicit finer bounded initialization clock required')
if args.shape == 'cylinder' and not .003 <= args.radius_m <= .1:
    p.error('Resolved finite cylindrical release radius required')
if args.bend_stiffness and (len(args.damping) != 1 or len(args.bend_stiffness) > 8
        or not all(0 < k <= 1e6 for k in args.bend_stiffness)):
    p.error('Bending comparison requires one fixed damping and bounded positive spring constants')
variants = ([{'damping': args.damping[0], 'bend': k} for k in args.bend_stiffness]
            if args.bend_stiffness else [{'damping': d, 'bend': profile['spring_settings']['bend_stiffness']}
                                        for d in args.damping])
if args.loaded_fold and (args.mesh or args.shape!='accordion' or len(variants)!=1 or args.seconds<.8):
    p.error('Loaded control requires one synthetic accordion variant and at least0.8seconds')
if args.loaded_fold and args.zero_impulse_writeback:p.error('Choose loaded or exact-zero force transport')
if args.shape=='accordion' and args.prestrain!=0:p.error('Accordion isolation uses zero added prestrain')
if args.shape=='accordion' and not .0003<=args.layer_gap_m<=.02:
    p.error('Explicit bounded synthetic layer spacing required')
if args.shape=='accordion' and args.size_m<3*math.pi*args.layer_gap_m:
    p.error('Synthetic material must resolve three panels and their rounded folds; increase size or reduce layer gap')
if not args.mesh and (round(args.size_m/profile['spacing_m'])+1)**2*len(variants)>500000:
    p.error('Synthetic material exceeds the explicit500000-node diagnostic budget')
out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
shutil.copy2(__file__, out/'source.py')
from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402
app = SimulationApp({'headless': True})
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import omni.physics.tensors as tensors  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from omni.physx.scripts import particleUtils  # noqa: E402
from pxr import UsdGeom, UsdPhysics, PhysxSchema, UsdShade, Vt, Gf  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from calibrated_scene_reconstruction.assets.cloth_rest import triangulate_grid, constraints, area  # noqa: E402
from calibrated_scene_reconstruction.assets.particle_lattice import chart  # noqa: E402
from calibrated_scene_reconstruction.assets.cloth_mass import authoring_mass  # noqa: E402
from calibrated_scene_reconstruction.assets.cloth_metric import MaterialMetric  # noqa: E402
from calibrated_scene_reconstruction.sim.physics_errors import PhysicsErrors  # noqa: E402
from calibrated_scene_reconstruction.tasks.rigid.task_checks import as_numpy  # noqa: E402
from calibrated_scene_reconstruction.tasks.rigid.task_checks import Ctx  # noqa: E402
from calibrated_scene_reconstruction.sim.particle_surface import ParticleView  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from calibrated_scene_reconstruction.tasks.mechanisms.force_fixture import ForceHandle,material_patch  # noqa: E402

files = [Path(__file__), Path(args.profile), *map(Path, [
    'src/calibrated_scene_reconstruction/assets/cloth_rest.py', 'src/calibrated_scene_reconstruction/assets/particle_lattice.py',
    'src/calibrated_scene_reconstruction/assets/cloth_mass.py', 'src/calibrated_scene_reconstruction/assets/cloth_metric.py',
    'src/calibrated_scene_reconstruction/assets/cloth_topology.py', 'src/calibrated_scene_reconstruction/sim/physics_errors.py',
    'src/calibrated_scene_reconstruction/sim/particle_surface.py', 'src/calibrated_scene_reconstruction/assets/particle_registration.py',
    'src/calibrated_scene_reconstruction/sim/particle_registration_output.py',
    'src/calibrated_scene_reconstruction/tasks/mechanisms/force_fixture.py','src/calibrated_scene_reconstruction/assets/cloth_fold.py',
    'src/calibrated_scene_reconstruction/assets/cloth_sampling.py','src/calibrated_scene_reconstruction/sim/cloth_contact.py',
    'src/calibrated_scene_reconstruction/sim/cloth_clearance.py','src/calibrated_scene_reconstruction/sim/cloth_shell_clearance.py',
    'src/calibrated_scene_reconstruction/tasks/mechanisms/particle_control_profile.py',
    'src/calibrated_scene_reconstruction/tasks/rigid/task_checks.py'])]
if args.mesh:files.append(Path(args.mesh))
hashes = {str(f.resolve()): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
errors = PhysicsErrors(out/'native_physics_errors.json')
try:
    hz = profile['physics_hz']; dt = 1/hz
    world = World(stage_units_in_meters=1., physics_dt=dt, device='cuda:0', backend='torch')
    errors.guard_steps(world)
    pc = world.get_physics_context(); pc.enable_gpu_dynamics(True)
    pc.set_broadphase_type('GPU'); pc.set_solver_type('PGS')
    stage = omni.usd.get_context().get_stage()
    scene = next(q for q in stage.Traverse() if q.IsA(UsdPhysics.Scene))
    UsdPhysics.Scene(scene).GetGravityDirectionAttr().Set(Gf.Vec3f(0.,0.,-1.))
    UsdPhysics.Scene(scene).GetGravityMagnitudeAttr().Set(args.gravity_m_s2)
    PhysxSchema.PhysxSceneAPI.Apply(scene).CreateEnableExternalForcesEveryIterationAttr(False)
    n = round(args.size_m/profile['spacing_m'])+1
    x = np.arange(n)*profile['spacing_m']; x -= x.mean()
    xx, yy = np.meshgrid(x, x); rest = np.c_[xx.ravel(), yy.ravel(), np.zeros(xx.size)]
    faces, material, _ = triangulate_grid(rest, rest, n, n)
    if profile['spring_model'] in {'balanced_square_lattice','triangle_surface'}:
        material, _ = chart(rest, faces,model=profile['spring_model'])
    shape_points=None
    if args.mesh:
        with np.load(args.mesh) as source:
            part=next(v for v in json.loads(str(source['links_json'])) if v['name']==args.cloth_link and v['kind']=='cloth')
            a,count,b,fc=[part[k] for k in ['vertex_start','vertex_count','face_start','triangle_count']]
            shape_points=source['V'][a:a+count].astype(float).copy()
            faces=source['F'][b:b+fc]-a
            material=json.loads(str(source['cloth_rest_json']))[args.cloth_link]
        rest=np.asarray(material['points'],float)
        if material.get('spring_model','triangle_edges')!=profile['spring_model']:
            raise ValueError('Actual material spring topology must match the explicit native profile')
        if len(rest)*len(variants)>500000:raise ValueError('Whole-shape release exceeds the explicit500000-node diagnostic budget')
    networks = [constraints(material, {**profile['spring_settings'], 'bend_stiffness': v['bend']})
                for v in variants]
    # A projection into world XY is not material strain for a folded sheet.
    # Preserve that legacy planar coupon diagnostic only for synthetic planes;
    # all shapes also receive the intrinsic, rotation-invariant surface metric.
    metric = MaterialMetric(rest, faces, material, max_edge_strain=.25, max_area_change=.15)
    target_mass = profile['areal_density']*area(rest, faces)
    cooking = authoring_mass(target_mass, rest, faces)
    authored = []
    for i, variant in enumerate(variants):
        damping = variant['damping']; pairs, lengths, stiffness, kinds = networks[i]
        path = f'/World/Coupon{i}'; system_path = path+'System'
        system = particleUtils.add_physx_particle_system(stage, system_path,
            simulation_owner=scene.GetPath(), **profile['contact'], enable_ccd=True)
        mat = system_path+'Material'
        particleUtils.add_pbd_particle_material(stage, mat, friction=.55, damping=0., drag=0., lift=0.)
        UsdShade.MaterialBindingAPI.Apply(system.GetPrim()).Bind(
            UsdShade.Material.Get(stage, mat), UsdShade.Tokens.weakerThanDescendants, 'physics')
        points = shape_points.copy() if shape_points is not None else rest.copy()
        if shape_points is None:points[:, :2] *= 1+args.prestrain
        normals = np.zeros_like(points, dtype='float32'); normals[:, 2] = 1
        if shape_points is not None:
            import trimesh
            normals=np.asarray(trimesh.Trimesh(points,faces,process=False).vertex_normals,dtype='float32')
        if args.shape == 'cylinder':
            angle = points[:, 0]/args.radius_m
            points[:, 0] = args.radius_m*np.sin(angle)
            points[:, 2] = args.radius_m*(1-np.cos(angle))
            normals[:, 0] = -np.sin(angle); normals[:, 2] = np.cos(angle)
        if args.shape=='accordion':
            from calibrated_scene_reconstruction.assets.cloth_fold import rounded_layered
            import trimesh
            points=rounded_layered(rest,panels=3,layer_gap=args.layer_gap_m)
            points[:,1]-=points[:,1].min()
            rotation=np.array([[-1.,0,0],[0,0,-1],[0,-1,0]])
            points=points@rotation.T+[0.,0.,.1+args.size_m]
            normals=np.asarray(trimesh.Trimesh(points,faces,process=False).vertex_normals,dtype='float32')
        points += ([i*(2*float(np.ptp(points,axis=0).max())+.05),0.,0.]
                   if shape_points is not None else [i*max(.1,2*args.size_m+.05),0.,.2])
        mesh = UsdGeom.Mesh.Define(stage, path)
        particleUtils.add_physx_particle_cloth_with_constraints(stage, path, system.GetPath(),
            Vt.Vec3fArray.FromNumpy(points.astype('float32')), Vt.Vec3fArray.FromNumpy(normals),
            Vt.Vec3fArray.FromNumpy(rest.astype('float32')), Vt.Vec3fArray.FromNumpy(np.zeros_like(normals)),
            Vt.IntArray(faces.ravel().tolist()), Vt.Vec2iArray.FromNumpy(pairs.astype('int32')),
            Vt.FloatArray(stiffness), Vt.FloatArray([damping]*len(pairs)), Vt.FloatArray(lengths.tolist()),
            self_collision=args.self_contact=='on', self_collision_filter=True)
        UsdPhysics.MassAPI.Apply(mesh.GetPrim()).CreateMassAttr(cooking['authoring_mass_kg'])
        authored.append(points)
    stage.GetRootLayer().Export(str(out/'initial.usda'))
    if args.bootstrap_hz is not None:world.set_simulation_dt(physics_dt=1/args.bootstrap_hz)
    world.reset()
    if args.bootstrap_hz is not None:world.set_simulation_dt(physics_dt=dt)
    gravity_actual=float(UsdPhysics.Scene(scene).GetGravityMagnitudeAttr().Get())
    gravity_direction=np.array(UsdPhysics.Scene(scene).GetGravityDirectionAttr().Get(),float)
    if not np.isclose(gravity_actual,args.gravity_m_s2,rtol=1e-6,atol=1e-10) or not np.allclose(gravity_direction,[0,0,-1],atol=1e-7):
        raise ValueError('Actual gravity differs from the isolated control declaration')
    owner = tensors.create_simulation_view('torch'); owner.set_subspace_roots('/')
    views = [owner.create_particle_cloth_view(f'/World/Coupon{i}') for i in range(len(authored))]
    transport=None
    if args.zero_impulse_writeback or args.loaded_fold:
        adapters={str(i):ParticleView(stage,f'/World/Coupon{i}',
                  {'V':points,'F':faces,'cloth_rest':material}) for i,points in enumerate(authored)}
        transport=SimpleNamespace(world=world,_cloth=adapters)
        zero_ids=np.arange(min(32,len(rest)),dtype=int)
        zero_forces=np.zeros((len(zero_ids),3))
    masses = [];contact_readback=[]
    for i, view in enumerate(views):
        pairs, lengths, stiffness, kinds = networks[i]
        points = as_numpy(view.get_positions())[0].reshape(-1, 3)
        distance, index = cKDTree(authored[i]).query(points)
        if not np.array_equal(index, np.arange(len(rest))) or distance.max() >= profile['spacing_m']*.4:
            raise ValueError('Native source correspondence ambiguous')
        m = as_numpy(view.get_masses())[0].copy(); masses.append(m)
        if not np.isfinite(m).all() or (m<=0).any():
            raise ValueError('Isolated material control cannot contain pinned or invalid native nodes')
        if not np.isclose(m.sum(), target_mass, rtol=1e-4): raise ValueError('Native mass mismatch')
        if not np.allclose(np.sort(as_numpy(view.get_spring_stiffness()).ravel()), np.sort(stiffness), rtol=1e-5):
            raise ValueError('Native stiffness mismatch')
        if not np.allclose(as_numpy(view.get_spring_damping()), variants[i]['damping'], rtol=1e-5, atol=1e-15):
            raise ValueError('Native damping mismatch')
        system=PhysxSchema.PhysxParticleSystem.Get(stage,f'/World/Coupon{i}System')
        observed={}
        for key,value in profile['contact'].items():
            suffix=('SolverPositionIterationCount' if key=='solver_position_iterations'
                    else ''.join(word.title() for word in key.split('_')))
            observed[key]=getattr(system,'Get'+suffix+'Attr')().Get()
            if observed[key] is None or not np.isclose(observed[key],value,rtol=1e-5,atol=1e-12):
                raise ValueError('Contact setting changed after native reset: '+key)
        contact_readback.append(observed)
    if any(q.HasAPI(UsdPhysics.CollisionAPI) or q.HasAPI(UsdPhysics.RigidBodyAPI)
           or q.GetTypeName()=='PhysxPhysicsAttachment' for q in stage.Traverse()):
        raise ValueError('Material isolation control contains an unexpected rigid partner or attachment')
    grips=[]
    if args.loaded_fold:
        actual=as_numpy(views[0].get_positions())[0].reshape(-1,3)
        for fraction,width,share,load_share,translation,limit in [
                ([.5,.02],args.size_m,.75,1.,[0.,-.018,-.025],.1),
                ([.5,.98],min(.016,args.size_m/3),.25,0.,[-.012,.018,-.003],.03)]:
            ids=material_patch(rest,fraction,[width,min(.008,args.size_m/8)])
            handle=ForceHandle(actual[ids],masses[0][ids],dt=dt,held_mass=float(masses[0].sum()*share),
                               force_limit=limit,speed=.08,stiffness=20.,ramp_seconds=.05)
            grips.append({'ids':ids,'handle':handle,'initial':handle.target.copy(),
                          'translation':np.array(translation),'load_share':load_share,'last_force_N':0.,
                          'material_fraction':fraction,'size_m':[width,min(.008,args.size_m/8)]})
    rows = []; states = []; velocities = []
    external_impulses=np.zeros((len(authored),3));initial_momenta=[]
    gravity_impulse=np.array([[0.,0.,-float(m.astype(float).sum())*gravity_actual*dt] for m in masses])
    def record(t):
        batch = []; ps = []; vs = []
        for i, view in enumerate(views):
            pairs, lengths, stiffness, kinds = networks[i]
            points = as_numpy(view.get_positions())[0].reshape(-1, 3).copy()
            velocity = as_numpy(view.get_velocities())[0].reshape(-1, 3).copy()
            if not np.isfinite(points).all() or not np.isfinite(velocity).all():
                raise ValueError('Nonfinite native release')
            delta = np.linalg.norm(points[pairs[:, 1]]-points[pairs[:, 0]], axis=1)-lengths
            momentum=(masses[i].astype(float)[:,None]*velocity.astype(float)).sum(0)
            if not rows:initial_momenta.append(momentum.copy())
            planar = shape_points is None and args.shape == 'plane'
            strain = (np.linalg.lstsq(rest[:, :2], (points-points.mean(0))[:, :2], rcond=None)[0]-np.eye(2)
                      if planar else None)
            batch.append({'damping_kg_s': variants[i]['damping'], 'bend_stiffness_N_m': variants[i]['bend'],
                'rms_nonplanarity_m': float(np.linalg.svd(points-points.mean(0), compute_uv=False)[-1]/np.sqrt(len(points))),
                'affine_strain': strain.tolist() if planar else None,
                'affine_strain_scope': 'World XY planar coupon diagnostic only; not used for folded or curved shapes',
                'material': metric.check(points),
                'authored_spring_energy_J': float((.5*np.asarray(stiffness)*delta**2).sum()),
                'kinetic_energy_J': float((.5*masses[i][:, None]*velocity**2).sum()),
                'maximum_speed_m_s': float(np.linalg.norm(velocity, axis=1).max()),
                'net_momentum':momentum.tolist(),
                'expected_external_impulse_Ns':external_impulses[i].tolist(),
                'momentum_balance_residual_Ns':(momentum-initial_momenta[i]-external_impulses[i]).tolist(),
                'momentum_balance_scope':'Native linear momentum versus initial momentum plus known gravity and finite grip impulses; diagnostic, no conservation tolerance or task approval is implied'})
            if args.loaded_fold:
                from calibrated_scene_reconstruction.assets.cloth_sampling import rigid_coverage
                from calibrated_scene_reconstruction.sim.cloth_contact import self_intersections
                from calibrated_scene_reconstruction.sim.cloth_clearance import measure as finite_clearance
                sampling=rigid_coverage(points,faces,profile['contact'])
                crossings=self_intersections(points,faces)
                clearance=finite_clearance(points,faces,rest,profile['physical_thickness_m'],
                                            device='cuda:0',local_bending_review=True)
                batch[-1].update(native_sampling=sampling,self_intersection=crossings,
                                 finite_sheet_clearance=clearance,
                                 physical_pass=bool(batch[-1]['material']['geometry_valid'] and sampling['pass']
                                                    and crossings['pass'] and clearance['pass']),
                                 grips=[{'target':g['handle'].target.tolist(),
                                         'tracking_error_m':float(np.linalg.norm(
                                             np.average(points[g['ids']],axis=0,weights=g['handle'].weights)-g['handle'].target)),
                                         'last_sum_force_N':g['last_force_N']} for g in grips])
            ps.append(points); vs.append(velocity)
        rows.append({'time_s': t, 'coupons': batch}); states.append(ps); velocities.append(vs)
        (out/'progress.json').write_text(json.dumps(rows[-1], indent=2)+'\n')
        return all(row.get('physical_pass',True) for row in batch)
    physical_valid=record(0.)
    for step in range(round(args.seconds*hz)):
        if args.loaded_fold and not physical_valid:break
        if transport is not None:
            if args.loaded_fold:
                time=step*dt
                if time<.65:
                    adapter=transport._cloth['0']
                    points=as_numpy(adapter.get_world_positions())[0]
                    velocity=as_numpy(adapter.get_velocities())[0]
                    ids=[];forces=[];limits=[]
                    for grip in grips:
                        selected=grip['ids'];handle=grip['handle']
                        goal=grip['initial']+np.clip((time-.15)/.5,0,1)*grip['translation']
                        force=handle.forces(points[selected],velocity[selected],goal,
                            load=[0,0,float(masses[0].sum()*args.gravity_m_s2*grip['load_share'])])
                        grip['last_force_N']=float(np.linalg.norm(force,axis=1).sum())
                        ids.append(selected);forces.append(force);limits.append(handle.force_limit*float(handle.weights.max()))
                    Ctx.apply_nodal_forces(transport,'0',np.concatenate(ids),np.concatenate(forces),force_limit_N=max(limits))
                    external_impulses[0]+=np.concatenate(forces).sum(0)*dt
                else:
                    for grip in grips:grip['last_force_N']=0.
            else:
                for i in range(len(authored)):
                    Ctx.apply_nodal_forces(transport,str(i),zero_ids,zero_forces,force_limit_N=1e-8)
        external_impulses+=gravity_impulse
        world.step(render=False)
        if (step+1) % (hz//args.record_fps) == 0 or step+1 == round(args.seconds*hz):
            physical_valid=record((step+1)*dt)
    errors.check()
    unchanged = all(hashlib.sha256(Path(f).read_bytes()).hexdigest() == h for f, h in hashes.items())
    np.savez_compressed(out/'states.npz', points=states, velocities=velocities, rest=rest, faces=faces, masses=masses)
    result = {'complete':abs(rows[-1]['time_s']-round(args.seconds*hz)/hz)<1e-8, 'source_unchanged': unchanged, 'task_approved': False,
        'material_identified': False, 'scope': 'Synthetic material diagnostic only; optional finite grips are test fixtures, never an asset. Authored spring energy is not native solver energy.',
        'rigid_collision_partners':0,'fixed_nodes':0,'cloth_attachments':0,
        'profile': profile, 'parameters': vars(args), 'variants': variants, 'native_readback_verified': True,
        'contact_settings_after_reset':contact_readback,
        'gravity_m_s2_after_reset':gravity_actual,'gravity_direction_after_reset':gravity_direction.tolist(),
        'positions_written_after_reset': False, 'velocities_written_after_reset': bool(args.zero_impulse_writeback or args.loaded_fold),
        'zero_impulse_writeback':bool(args.zero_impulse_writeback), 'external_impulses_applied':bool(args.loaded_fold),
        'loaded_fold_control':bool(args.loaded_fold),'physical_state_pass':physical_valid if args.loaded_fold else None,
        'all_fixture_impulses_end_time_s':.65 if args.loaded_fold else None,
        'fold_fixture_plan':({'motion_start_s':.15,'motion_end_s':.65,'positions_prescribed':False,
            'grips':[{'material_fraction':g['material_fraction'],'size_m':g['size_m'],
                       'initial_world':g['initial'].tolist(),'translation_m':g['translation'].tolist(),
                       'gravity_compensation_share':g['load_share'],'handle':g['handle'].report()} for g in grips]}
            if args.loaded_fold else None),
        'whole_shape_diagnostic':bool(args.mesh),'self_contact_enabled':args.self_contact=='on',
        'gravity_disabled_for_constitutive_isolation':args.gravity_m_s2==0, 'input_sha256': hashes,
        'native_physics_errors': errors.report(), 'rows': rows}
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print('PARTICLE_RELEASE_COMPLETE', unchanged, flush=True)
except Exception as error:
    (out/'error.json').write_text(json.dumps({'complete': False, 'error': str(error), 'traceback': traceback.format_exc()}, indent=2)+'\n')
    raise
finally:
    errors.close(); app.close()
