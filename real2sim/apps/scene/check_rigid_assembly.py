"""Rigid-assembly diagnostic before the complete task is ready.

Uses current exported physical surfaces and priors on a disclosed planar
support. It cannot approve source appearance, full-scene stability or the task.
Optional bounded tool forces and one independently reconstructed companion
allow contact experiments without changing production geometry or poses.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('session');parser.add_argument('object')
parser.add_argument('--output',required=True)
parser.add_argument('--seconds',type=float,default=2.)
parser.add_argument('--case',choices=['nominal','low','high'],default='nominal')
parser.add_argument('--support',choices=['plane','modeled'],default='plane',
                    help='Use the current verified support mesh, or a disclosed diagnostic slab')
parser.add_argument('--support-collision-edge-m',type=float,
                    help='Diagnostic conforming subdivision of the identical modeled support surface')
parser.add_argument('--gpu-dynamics',action='store_true',help='Explicit native GPU solver diagnostic, not adopted for delivery')
parser.add_argument('--support-source-solids',action='store_true',help='Diagnose exact convex authored support solids, without convexifying concave parts')
parser.add_argument('--sdf-resolution',type=int,default=None,
                    help='Diagnostic-only precision hypothesis; delivery requires separate reauthoring and validation')
parser.add_argument('--physics-hz', type=int, choices=[120,240,480,960,1920,3840],
                    help='Explicit diagnostic timestep; never changes the delivery profile')
parser.add_argument('--rigid-iterations', type=int, nargs=2, metavar=('POSITION','VELOCITY'),
                    help='Diagnostic rigid solver iteration counts; not adopted into delivery')
parser.add_argument('--collision-approximation',choices=['sdf','convexDecomposition'],default=None,
                    help='Diagnostic-only collider hypothesis; never approves physical shape fidelity')
parser.add_argument('--link-physics-plan',help='Diagnostic per-link collision plan; does not modify delivery configuration')
parser.add_argument('--force-probe-plan', help='Optional bounded translation force, followed by full release; diagnostic only')
parser.add_argument('--companion-plan', help='One current rigid task link, placed once by a disclosed rigid transform for a contact experiment')
candidate_args=parser.add_mutually_exclusive_group()
candidate_args.add_argument('--geometry-round',type=int,
                    help='Inspect a preserved candidate before export; source failures remain failures')
candidate_args.add_argument('--candidate-directory',
                    help='Session QA snapshot with model.py, mesh.npz and hashed fit.json; diagnostic only')
args=parser.parse_args()
if not 0<args.seconds<=30:parser.error('Use a finite diagnostic interval in0..30s')
if args.rigid_iterations and any(not 1 <= value <= 256 for value in args.rigid_iterations):
    parser.error('Rigid iteration diagnostic counts must be in1..256')

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402
app=SimulationApp({'headless':True})
import sys  # noqa: E402
_original_exception_hook=sys.excepthook
def _failed_check(kind,error,traceback):
    # Kit shutdown can otherwise turn an uncaught Python exception into exit0.
    _original_exception_hook(kind,error,traceback)
    app.app.post_quit(2)
sys.excepthook=_failed_check
import numpy as np  # noqa: E402
import trimesh  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Usd,UsdPhysics,PhysxSchema,UsdShade  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleRigidPrim, RigidPrim  # noqa: E402
from calibrated_scene_reconstruction.scene.session import Session  # noqa: E402
from calibrated_scene_reconstruction.core.paths import resolve  # noqa: E402
from calibrated_scene_reconstruction.assets.metric_mesh import asset,sdf_settings  # noqa: E402
from calibrated_scene_reconstruction.assets.physical_priors import sample,validate  # noqa: E402
from calibrated_scene_reconstruction.sim.mechanism_stage import add_links  # noqa: E402
from calibrated_scene_reconstruction.sim.physics import physics_material,bind_colliders  # noqa: E402
from calibrated_scene_reconstruction.sim.usd import set_pose,add_box_collider  # noqa: E402
from calibrated_scene_reconstruction.sim.simulation import apply as apply_profile  # noqa: E402
from calibrated_scene_reconstruction.sim.simulation import apply_rigid_iterations,audit as audit_profile  # noqa: E402
from calibrated_scene_reconstruction.scene.simulation import settings  # noqa: E402
from calibrated_scene_reconstruction.tasks.rigid.task_checks import as_numpy,rot  # noqa: E402
from calibrated_scene_reconstruction.sim.physics_errors import PhysicsErrors  # noqa: E402

s=Session(args.session);out=(s/args.output).resolve()
if not out.is_relative_to(s.dir.resolve()):raise ValueError('Output must remain in this session')
out.mkdir(parents=True,exist_ok=False)
(out/'runner.py').write_bytes(Path(__file__).read_bytes())
spec=s.scene_graph['dynamic_objects'][args.object];fix=json.loads(json.dumps(spec.get('fix') or {}));candidate=None
if not fix and args.geometry_round is None and args.candidate_directory is None:
    raise ValueError('Choose an exported asset or an explicit preserved candidate; no unreviewed fallback geometry')
if args.geometry_round is not None or args.candidate_directory is not None:
    if args.geometry_round is not None:
        if args.geometry_round<1:raise ValueError('Positive candidate round required')
        directory=s/'objects'/args.object/f'round_{args.geometry_round}'
    else:
        directory=(s/args.candidate_directory).resolve()
        if not directory.is_relative_to((s/'qa').resolve()):
            raise ValueError('A standalone diagnostic candidate must remain under this session QA')
    fitted=json.loads((directory/'fit.json').read_text())
    for name,expected in [('mesh.npz',fitted['mesh_sha256']),('model.py',fitted['model_sha256'])]:
        if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Candidate snapshot changed: '+name)
    candidate_pose={key:fitted['pose'][key] for key in ('xy','yaw_deg','support_z')}
    if 'support_rotation' in fitted['pose']:
        candidate_pose['support_rotation']=fitted['pose']['support_rotation']
    fix.update(mesh=str((directory/'mesh.npz').resolve()),round=args.geometry_round,
               params=fitted['build']['params'],model_sha256=fitted['model_sha256'],
               opening=fitted['build'].get('opening'),**candidate_pose)
    fix.pop('look',None)  # Never bind an atlas from a different candidate surface.
    candidate={'round':args.geometry_round,'directory':str(directory),'source_checks_pass':bool(fitted['ok']),
               'fit_sha256':hashlib.sha256((directory/'fit.json').read_bytes()).hexdigest(),
               'model_snapshot':str(directory/'model.py'),'delivery_approved':False}
    # A first candidate has no export's numerical collider defaults yet. Keep
    # the diagnostic settings explicit without inventing a production export.
    fix['sdf']=sdf_settings(fix.get('sdf'))
kit=asset(fix)
if args.sdf_resolution is not None:
    fix['sdf']=sdf_settings({**fix['sdf'],'resolution':args.sdf_resolution})
if args.collision_approximation is not None:
    for part in kit['parts']:
        fix.setdefault('link_physics',{}).setdefault(part['name'],{})['collision_approximation']=args.collision_approximation
if args.link_physics_plan:
    plan_path=Path(args.link_physics_plan)
    overrides=json.loads(plan_path.read_text())
    (out/'link_physics_plan.json').write_bytes(plan_path.read_bytes())
if not kit.get('links') or any(p['kind']!='rigid' for p in kit['parts']):
    raise ValueError('This diagnostic requires explicit rigid links')
profile=settings(s)
if profile is None:raise ValueError('Declare the delivered simulation profile first')
if args.physics_hz is not None:
    profile = {**profile, 'physics_hz': args.physics_hz,
               'basis': str(profile.get('basis', ''))+'; explicit diagnostic timestep override, not adopted'}
if args.gpu_dynamics:profile={**profile,'gpu_dynamics':True}
if args.rigid_iterations:profile={**profile,'rigid_iterations':args.rigid_iterations}
contract=s.config('agent/mechanisms.json');plan=s.config('agent/physics_priors.json')
physical=sample(validate(plan,contract),args.case)
if kit.get('joints'):
    raise ValueError('This unforced independent-part diagnostic does not resolve joint collision exclusions')
physical_names = {part['name']: args.object+'/'+part['name'] for part in kit['parts']}
companion_record = None
if args.companion_plan:
    import copy
    companion_path = Path(args.companion_plan)
    companion_record = json.loads(companion_path.read_text())
    required = {'object', 'link', 'name', 'rotation', 'translation_m', 'mesh_sha256', 'basis'}
    if set(companion_record) != required or not companion_record['basis']:
        raise ValueError('Companion requires exact source link/hash and disclosed experimental placement')
    cobj = companion_record['object']; cname = companion_record['name']
    if (cobj == args.object or cobj not in s.scene_graph['dynamic_objects']
            or not isinstance(cname, str) or not cname.isidentifier() or cname in physical_names):
        raise ValueError('Choose another current task object and a unique rigid link name')
    cfix = s.scene_graph['dynamic_objects'][cobj]['fix']
    if hashlib.sha256(resolve(cfix['mesh']).read_bytes()).hexdigest() != companion_record['mesh_sha256']:
        raise ValueError('Companion source mesh changed')
    ckit = asset(cfix)
    if ckit.get('joints') or len(ckit['parts']) != 1 or ckit['parts'][0]['kind'] != 'rigid':
        raise ValueError('Companion diagnostic currently requires one unjointed rigid link')
    part = copy.deepcopy(ckit['parts'][0]); old_name = part['name']
    if old_name != companion_record['link']:
        raise ValueError('Companion link differs from the current reconstructed asset')
    from calibrated_scene_reconstruction.tasks.mechanisms.rigid_probe import rigid_transform
    R, t = rigid_transform(companion_record['rotation'], companion_record['translation_m'])
    # Keep the exact source collider coordinates. Baking an assay rotation into
    # vertices changes SDF sampling and no longer tests the delivered collider.
    part['name'] = cname
    kit['parts'].append(part)
    physical_names[cname] = cobj+'/'+old_name
    if physical_names[cname] not in physical['bodies']:
        raise ValueError('Companion has no current physical prior')
    csettings = copy.deepcopy(cfix.get('link_physics', {}).get(old_name, {}))
    csettings.setdefault('sdf', cfix['sdf'])
    fix.setdefault('link_physics', {})[cname] = csettings
    (out/'companion_plan.json').write_bytes(companion_path.read_bytes())
    companion_record = {**companion_record, 'source_fix': cfix,
                        'transformation_applied_once_before_reset': True,
                        'placement_on_body_transform':True}
if args.link_physics_plan:
    if set(overrides)-{p['name'] for p in kit['parts']}:raise ValueError('Collision plan contains unknown links')
    for name,value in overrides.items():fix.setdefault('link_physics',{}).setdefault(name,{}).update(value)
probe_plan = None
if args.force_probe_plan:
    from calibrated_scene_reconstruction.tasks.mechanisms.rigid_probe import validate as validate_probe, TranslationProbe
    probe_path = Path(args.force_probe_plan)
    probe_plan = validate_probe(json.loads(probe_path.read_text()),
                                [part['name'] for part in kit['parts']], args.seconds)
    (out/'force_probe_plan.json').write_bytes(probe_path.read_bytes())
from calibrated_scene_reconstruction.assets.rigid_initial_state import measure as initial_rigid_state  # noqa: E402
initial_parts=[dict(part) for part in kit['parts']]
if companion_record:
    initial_parts[-1]['V']=initial_parts[-1]['V']@R.T+t
initial_state=initial_rigid_state(initial_parts,spacing_m=.0005,device='cuda:0')
(out/'initial_rigid_state.json').write_text(json.dumps(initial_state,indent=2)+'\n')
if not initial_state['pass']:
    (out/'result.json').write_text(json.dumps({'diagnostic_only':True,'task_approved':False,
        'simulation_executed':False,'initial_rigid_state':initial_state,'candidate':candidate,
        'mesh_sha256':hashlib.sha256(resolve(fix['mesh']).read_bytes()).hexdigest(),
        'reason':'Authored independent solids interpenetrate; native settling cannot approve the initial state'},indent=2)+'\n')
    app.app.post_quit(2)
    app.close()
    raise SystemExit(2)
world=World(stage_units_in_meters=1.,physics_dt=1/profile['physics_hz'],backend='numpy')
stage=omni.usd.get_context().get_stage();errors=PhysicsErrors(out/'native_errors.json');errors.guard_steps(world)
apply_profile(stage,profile)
support_name=s.scene_graph['task']['support']
support_params=physical['supports'][support_name]['parameters']
mat=physics_material(stage,'/World/Materials/probe_support',support_params['static_friction'],
                     support_params['dynamic_friction'],support_params['restitution'])
top=fix['support_z'];static_support=None
if args.support=='modeled':
    from calibrated_scene_reconstruction.scene.support.runtime_inputs import models as support_models
    from calibrated_scene_reconstruction.sim.support_stage import add_support
    supported=support_models(s)[support_name];support_path='/World/Support/'+support_name
    if args.support_collision_edge_m is not None:
        supported={**supported,'collision_max_edge_m':args.support_collision_edge_m}
    solids_record=None
    if args.support_source_solids or 'collision_solids' in supported:
        from calibrated_scene_reconstruction.assets.convex_source_solids import load as load_convex_solids
        support_fix=s.scene_graph['furniture'][support_name]['fix']
        solids_path=resolve(support_fix['mesh']).parent/'source_solids.npz'
        support_fit=json.loads((solids_path.parent/'fit.json').read_text())
        solids_sha=hashlib.sha256(solids_path.read_bytes()).hexdigest()
        if solids_sha!=support_fit.get('source_solids_sha256'):raise ValueError('Support source solids changed')
        supported={**supported,'collision_solids':load_convex_solids(solids_path)}
        solids_record={'path':str(solids_path),'sha256':solids_sha,'cells':len(supported['collision_solids'])}
    add_support(stage,support_path,supported);support_collision_path=support_path+'/collision'
    support=stage.GetPrimAtPath(support_collision_path)
    support_fix=s.scene_graph['furniture'][support_name]['fix']
    support_mesh=resolve(support_fix['mesh'])
    static_support={'name':support_name,'mesh':str(support_mesh),'mesh_sha256':hashlib.sha256(support_mesh.read_bytes()).hexdigest(),
                    'xy':supported['xy'],'yaw_deg':supported['yaw_deg'],'collision_path':support_collision_path,
                    'collision_max_edge_m':supported.get('collision_max_edge_m'),'source_solids':solids_record}
else:
    support_collision_path='/World/Support/diagnostic_plane'
    support=add_box_collider(stage,support_collision_path,[-2,-2,top-.05],[2,2,top]).GetPrim()
UsdShade.MaterialBindingAPI.Apply(support).Bind(mat,UsdShade.Tokens.strongerThanDescendants,'physics')
support_collision_paths=[str(p.GetPath()) for p in stage.Traverse()
                         if str(p.GetPath()).startswith(support_collision_path) and p.HasAPI(UsdPhysics.CollisionAPI)]
if not support_collision_paths:raise ValueError('Actual support has no native colliders')
path='/World/Objects/'+args.object
initial_mass=0.
for part in kit['parts']:
    parameters=physical['bodies'][physical_names[part['name']]]['parameters']
    volume=float(trimesh.Trimesh(part['V'],part['F'],process=True).volume)
    initial_mass+=float(parameters['mass'] if 'mass' in parameters else parameters['density']*volume)
spec={**spec,'mass':initial_mass,'_mass_why':'Current sampled material priors and actual source-solid volumes; per-link values applied before any simulation step.'}
root=add_links(stage,path,kit,fix,spec,mat,look_base=s.dir)
from scipy.spatial.transform import Rotation  # noqa: E402
from calibrated_scene_reconstruction.assets.object_pose import rotation as object_rotation  # noqa: E402
q=Rotation.from_matrix(object_rotation(fix)).as_quat()[[3,0,1,2]]
set_pose(root,[*fix['xy'],top],q)
if companion_record:
    from scipy.spatial.transform import Rotation
    cq=Rotation.from_matrix(np.asarray(companion_record['rotation'])).as_quat()[[3,0,1,2]]
    set_pose(stage.GetPrimAtPath(path+'/'+companion_record['name']),companion_record['translation_m'],cq)
mass_report={}
for part in kit['parts']:
    ref=physical_names[part['name']];p=physical['bodies'][ref]['parameters']
    volume=float(trimesh.Trimesh(part['V'],part['F'],process=True).volume)
    mass=float(p['mass'] if 'mass' in p else p['density']*volume)
    prim=stage.GetPrimAtPath(path+'/'+part['name']);UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
    from calibrated_scene_reconstruction.sim.source_moments import author as source_moments
    source_moments(prim,part['V'],part['F'],mass)
    rb=PhysxSchema.PhysxRigidBodyAPI.Apply(prim);rb.CreateLinearDampingAttr(p['linear_damping']);rb.CreateAngularDampingAttr(p['angular_damping'])
    material=physics_material(stage,path+'/'+part['name']+'_material',p['static_friction'],p['dynamic_friction'],p['restitution'],friction_combine='average')
    bind_colliders(stage,[str(prim.GetPath())],material);mass_report[ref]=mass
apply_rigid_iterations(stage,profile)
actual_profile=audit_profile(stage,profile)
input_record={'diagnostic_only':True,'fix':fix,'profile':profile,'case':args.case,
              'authored_profile_audit':actual_profile,
              'backend':{'name':'isaacsim','version':importlib.metadata.version('isaacsim')},
              'candidate':candidate,
              'mesh_sha256':hashlib.sha256(resolve(fix['mesh']).read_bytes()).hexdigest(),
              'physics_priors':plan,'source':str(Path(__file__).resolve()),
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'support':('Verified modeled support; full scene review remains required' if static_support else
                         'Diagnostic finite horizontal slab at reconstructed support height; full task support remains separately required'),
              'static_support':static_support,
              'external_actuation':probe_plan is not None,'task_approved':False,'mass_kg':mass_report,
              'force_probe':probe_plan,
              'companion':companion_record,
              'support_parameters':support_params,
              'diagnostic_overrides':{'sdf_resolution':args.sdf_resolution,'collision_approximation':args.collision_approximation,
                                      'gpu_dynamics':args.gpu_dynamics,
                                      'physics_hz': args.physics_hz,
                                      'rigid_iterations': args.rigid_iterations,
                                      'link_physics_plan':args.link_physics_plan},
              'delivery_profile_validated':False}
(out/'input.json').write_text(json.dumps(input_record,indent=2)+'\n')
stage.GetRootLayer().Export(str(out/'initial.usda'))
if Usd.Stage.Open(str(out/'initial.usda')) is None:
    raise ValueError('Serialized native stage could not be reopened')
world.reset();bodies=[]
initial_precision=errors.precision_audit()
if not initial_precision['pass']:
    (out/'result.json').write_text(json.dumps({'diagnostic_only':True,'task_approved':False,
        'simulation_executed':False,'initial_rigid_state':initial_state,
        'precision_audit':initial_precision,
        'reason':'Native cooking did not preserve requested collision precision; revise the profile before motion'},indent=2)+'\n')
    raise ValueError('Native cooking precision failed before dynamic trial')
for part in kit['parts']:
    body=SingleRigidPrim(path+'/'+part['name']);body.initialize();bodies.append(body)
probe = None; probe_index = None; applied_forces = []; applied_torques = []; probe_contacts = None; contact_trace = []
contact_peaks = {}
body_paths = [b.prim_path for b in bodies]
all_filters = body_paths+support_collision_paths
all_contact_peaks = {}; all_contact_trace = []
from calibrated_scene_reconstruction.tasks.rigid.contact_pairs import depths as pair_depths
import inspect
pair_code=Path(inspect.getfile(pair_depths))
(out/'contact_pairs_engine.py').write_bytes(pair_code.read_bytes())
input_record['contact_pairs_engine_sha256']=hashlib.sha256(pair_code.read_bytes()).hexdigest()
if probe_plan is not None:
    probe_index = next(i for i, part in enumerate(kit['parts']) if part['name'] == probe_plan['link'])
    native_mass = float(as_numpy(bodies[probe_index]._rigid_prim_view.get_masses()).reshape(-1)[0])
    probe_com_local = as_numpy(bodies[probe_index]._rigid_prim_view.get_coms()[0]).reshape(-1, 3)[0]
    if not np.isfinite(probe_com_local).all():
        raise ValueError('Native probe centre of mass unavailable')
    native_inertia = as_numpy(bodies[probe_index]._rigid_prim_view.get_inertias()).reshape(-1, 3, 3)[0]
    minimum_inertia = float(np.linalg.eigvalsh((native_inertia+native_inertia.T)/2).min())
    probe = TranslationProbe(probe_plan, native_mass, 1/profile['physics_hz'], native_inertia)
    input_record['force_probe_native_mass_kg'] = native_mass
    input_record['force_probe_native_minimum_inertia_kg_m2'] = minimum_inertia
    input_record['force_probe_native_inertia_local_kg_m2'] = native_inertia.tolist()
    input_record['force_probe_native_com_local_m'] = probe_com_local.tolist()
    input_record['force_application_point'] = 'Actual native world centre of mass; explicit force position'
    import inspect
    probe_code = Path(inspect.getfile(TranslationProbe))
    (out/'force_probe_engine.py').write_bytes(probe_code.read_bytes())
    input_record['force_probe_engine_sha256'] = hashlib.sha256(probe_code.read_bytes()).hexdigest()
    from calibrated_scene_reconstruction.tasks.mechanisms.force_fixture import implicit_pd
    force_code = Path(inspect.getfile(implicit_pd))
    (out/'force_fixture_engine.py').write_bytes(force_code.read_bytes())
    input_record['force_fixture_engine_sha256'] = hashlib.sha256(force_code.read_bytes()).hexdigest()
    # Create native sensor views before any dynamics, since initialization can
    # restore authored transforms. They never reset a running experiment.
    filters = [path+'/'+part['name'] for i,part in enumerate(kit['parts']) if i != probe_index]
    filters.extend(support_collision_paths)
    contact_peaks = {key: {'maximum_penetration_m': 0., 'time_s': None} for key in filters}
stabilization_before=world.get_physics_context().is_stablization_enabled()
probe_contacts = RigidPrim(body_paths,
                           contact_filter_prim_paths_expr=[all_filters for _ in body_paths], max_contact_count=32768,
                           reset_xform_properties=False, disable_stablization=False, name='assembly_contacts')
probe_contacts.initialize()
if list(probe_contacts.prim_paths)!=body_paths:
    raise ValueError('Native contact view reordered body paths; explicit pose/contact mapping required')
input_record['contact_monitor_changes_stabilization']=False
stabilization_after=world.get_physics_context().is_stablization_enabled()
if stabilization_before != stabilization_after:
    raise ValueError('Contact instrumentation changed the physical stabilization setting')
input_record['native_stabilization']={'before_sensor':stabilization_before,'after_sensor':stabilization_after}
from calibrated_scene_reconstruction.tasks.mechanisms.contact import contact_separations
# Bind collision recipes and support authoring before any dynamics. A later
# implementation edit cannot silently qualify an older native trajectory.
implementation={}
for module_name,module in list(sys.modules.items()):
    file=getattr(module,'__file__',None)
    if module_name.startswith('calibrated_scene_reconstruction.') and file and Path(file).suffix=='.py':
        file=Path(file).resolve()
        implementation[str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
implementation[str(Path(__file__).resolve())]=hashlib.sha256((out/'runner.py').read_bytes()).hexdigest()
input_record['implementation_sha256']=implementation
(out/'input.json').write_text(json.dumps(input_record, indent=2)+'\n')
states=[];times=[];points=[];physics_poses=[]
for step in range(round(args.seconds*profile['physics_hz'])+1):
    positions,orientations=(as_numpy(x) for x in probe_contacts.get_world_poses())
    physics_poses.append(np.c_[positions,orientations])
    if step%max(1,profile['physics_hz']//30)==0:
        poses=[tuple(as_numpy(x) for x in b.get_world_pose()) for b in bodies]
        states.append(np.array([np.r_[p,q] for p,q in poses]));times.append(step/profile['physics_hz'])
        points.append(np.concatenate([part['V']@rot(q).T+p for part,(p,q) in zip(kit['parts'],poses)]))
    if step<round(args.seconds*profile['physics_hz']):
        if probe is not None:
            body = bodies[probe_index]
            bp, bq = (as_numpy(x) for x in body.get_world_pose())
            bq = np.asarray(bq, float)
            if not np.isfinite(bq).all() or np.linalg.norm(bq) < 1e-12:
                raise ValueError('Finite native rigid orientation required')
            bq = bq/np.linalg.norm(bq)
            com_world = bp+rot(bq)@probe_com_local
            force = probe.force(step/profile['physics_hz'], com_world,
                                as_numpy(body.get_linear_velocity()))
            torque = probe.torque(step/profile['physics_hz'], rot(bq), as_numpy(body.get_angular_velocity()))
            body._rigid_prim_view.apply_forces_and_torques_at_pos(
                forces=np.asarray([force], np.float32), torques=np.asarray([torque], np.float32),
                positions=np.asarray([com_world], np.float32), is_global=True)
            applied_forces.append(force.copy())
            applied_torques.append(torque.copy())
        world.step(render=False)
        if probe_contacts is not None:
            data = probe_contacts.get_contact_force_data(dt=1/profile['physics_hz'])
            counts = as_numpy(data[4]).reshape(len(bodies), len(all_filters))
            starts = as_numpy(data[5]).reshape(len(bodies), len(all_filters))
            depths=pair_depths(as_numpy(data[3]),counts,starts,body_paths,all_filters)
            now=(step+1)/profile['physics_hz']
            changed=[pair for pair,depth in depths.items()
                     if depth>all_contact_peaks.get(pair,{}).get('maximum_penetration_m',0.)]
            if changed:
                peak_poses=[np.r_[as_numpy(b.get_world_pose()[0]),as_numpy(b.get_world_pose()[1])].tolist() for b in bodies]
                for pair in changed:
                    all_contact_peaks[pair]={'maximum_penetration_m':depths[pair],
                                             'time_s':now,'poses':peak_poses}
                    candidates=[]
                    for i,body_path in enumerate(body_paths):
                        for j,other in enumerate(all_filters):
                            if tuple(sorted([body_path,other]))==pair and counts[i,j]:
                                candidates.extend(range(int(starts[i,j]),int(starts[i,j]+counts[i,j])))
                    separation=as_numpy(data[3]).reshape(-1)
                    index=min(candidates,key=lambda k:separation[k])
                    all_contact_peaks[pair]['native_contact_sample']={
                        'normal_force_N':float(as_numpy(data[0]).reshape(-1)[index]),
                        'point_m':as_numpy(data[1]).reshape(-1,3)[index].tolist(),
                        'normal':as_numpy(data[2]).reshape(-1,3)[index].tolist(),
                        'separation_m':float(separation[index])}
            all_contact_trace.append([now,max(depths.values(),default=0.)])
            if probe is not None:
                for key in filters:
                    depth=depths[tuple(sorted([body_paths[probe_index],key]))]
                    if depth>contact_peaks[key]['maximum_penetration_m']:
                        contact_peaks[key]={'maximum_penetration_m':depth,'time_s':now}
                distances=contact_separations(as_numpy(data[3]),counts[probe_index],starts[probe_index])
                contact_trace.append([now,len(distances),max(0.,-float(distances.min())) if len(distances) else 0.])
states=np.asarray(states);points=np.asarray(points)
np.savez_compressed(out/'states.npz',poses=states,points=points,times_s=times,names=[p['name'] for p in kit['parts']])
np.savez_compressed(out/'physics_poses.npz',poses=np.asarray(physics_poses),
                    times_s=np.arange(len(physics_poses))/profile['physics_hz'],
                    names=[p['name'] for p in kit['parts']])
np.savez_compressed(out/'all_contacts.npz',time_maximum_penetration_m=np.asarray(all_contact_trace))
(out/'contact_peak_poses.json').write_text(json.dumps({'body_paths':body_paths,
    'peaks':[{'pair':list(pair),**row} for pair,row in all_contact_peaks.items()]},indent=2)+'\n')
if probe is not None:
    np.savez_compressed(out/'forces.npz', forces_N=np.asarray(applied_forces),
                        torques_Nm=np.asarray(applied_torques),
                        times_s=np.arange(len(applied_forces))/profile['physics_hz'],
                        contacts_time_count_penetration_m=np.asarray(contact_trace))
native=errors.report();precision=errors.precision_audit()
changed=[file for file,expected in implementation.items()
         if not Path(file).is_file() or hashlib.sha256(Path(file).read_bytes()).hexdigest()!=expected]
(out/'implementation_audit.json').write_text(json.dumps({'pass':not changed,'changed':changed},indent=2)+'\n')
(out/'native_errors.json').write_text(json.dumps(native,indent=2)+'\n')
errors.close()
report={'diagnostic_only':True,'task_approved':False,'external_actuation':probe_plan is not None,
        'implementation_unchanged':not changed,
        'simulation_executed':True,'initial_rigid_state':initial_state,
        'native_physics_pass':native['pass'],'precision_audit':precision,
        'finite_states':bool(np.isfinite(states).all()),'bodies':{},'native_errors':str(out/'native_errors.json')}
report['all_rigid_contact_pairs']={'maximum_penetration_m':max(row[1] for row in all_contact_trace),
    'pairs':[{'pair':list(pair),**{k:v for k,v in row.items() if k!='poses'}} for pair,row in all_contact_peaks.items()],
    'peak_states':'contact_peak_poses.json','trace':'all_contacts.npz'}
if probe is not None:
    report['force_probe'] = {'plan': probe_plan, 'peak_force_N': probe.peak_force,
                             'contact_pair_peaks': contact_peaks,
                             'peak_torque_Nm': probe.peak_torque,
                             'maximum_native_contact_penetration_m': max(row[2] for row in contact_trace),
                             'positions_prescribed': False, 'orientations_prescribed': False,
                             'torques_applied': 'orientation_hold' in probe_plan,
                             'release_forces_zero': bool(np.all(np.asarray(applied_forces)[
                                 np.arange(len(applied_forces))/profile['physics_hz'] >=
                                 probe_plan['start_s']+probe_plan['duration_s']] == 0)),
                             'release_torques_zero': bool(np.all(np.asarray(applied_torques)[
                                 np.arange(len(applied_torques))/profile['physics_hz'] >=
                                 probe_plan['start_s']+probe_plan['duration_s']] == 0)),
                             'scope': 'Bounded tool-force diagnostic; removal/insertion and contact quality require separate measurements.'}
offset=0
for index,part in enumerate(kit['parts']):
    count=len(part['V']);motion=np.linalg.norm(points[:,offset:offset+count]-points[0,offset:offset+count],axis=2)
    report['bodies'][part['name']]={'maximum_surface_motion_m':float(motion.max()),'final_surface_motion_p95_m':float(np.percentile(motion[-1],95)),'minimum_world_z_m':float(points[:,offset:offset+count,2].min())};offset+=count
(out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
app.close()
