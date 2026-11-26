"""QA: the twin's task objects support the task -- the agent's own check of the session's task, run in the twin.

    tools/run_isaac.sh apps/scene/task_check.py sessions/<name> [--background zone|splat|mesh]

The check is the agent's: configs/<name>/agent/task_check.py, written for the task in task.json (what the task's end
state is, how to set it up, what counts as done), defines `check(ctx)` and returns {"pass", "what", ...}; the tools
it has are sim.task_checks.Ctx (poses, placing, PhysX, a kit's opening, renders) -- no criteria are fixed here.
A session with a task and no check script writes {"pass": false, "why": "no task check written"}: the report
flags it. Writes qa/task_check.json (and whatever images the script renders), prints one PASS / FAIL line.
"""
import argparse
import importlib.util
import importlib.metadata
import json
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("session")
ap.add_argument('--configuration',help='Explicit versioned configuration snapshot for isolated candidate validation')
ap.add_argument("--background", default="mesh", choices=["mesh", "splat", "zone", "blender"])
ap.add_argument('--physics-case', default='all', choices=['all', 'nominal', 'low', 'high'])
ap.add_argument('--physics-first', default='nominal', choices=['nominal', 'low', 'high'],
                help='For all-case trials, run a previously failing boundary first; all three are still required')
ap.add_argument('--trial-output', default=None)
args = ap.parse_args()

if args.physics_case == 'all':
    from calibrated_scene_reconstruction.scene.session import Session
    from calibrated_scene_reconstruction.core.session_code import source_path
    if source_path(Session(args.session,args.configuration).config_dir, 'physics_priors.json').exists():
        from calibrated_scene_reconstruction.scene.objects.physics_trials import run
        run(Session(args.session,args.configuration), sys.executable, str(Path(__file__).resolve()), args.background,
            first=args.physics_first)
        sys.exit(0)

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402

from calibrated_scene_reconstruction.scene.session import Session  # noqa: E402
from calibrated_scene_reconstruction.core.session_code import source_path  # noqa: E402
from calibrated_scene_reconstruction.scene.twin.variants import photo_goal, prepare_variant  # noqa: E402
from calibrated_scene_reconstruction.scene.twin.runtime_inputs import prepare_background, prepare_scene  # noqa: E402
from calibrated_scene_reconstruction.sim.scene_stage import build_scene, configure_render  # noqa: E402
from calibrated_scene_reconstruction.tasks.rigid.task_checks import Ctx, as_numpy  # noqa: E402
from calibrated_scene_reconstruction.sim.usd import add_camera  # noqa: E402
from calibrated_scene_reconstruction.scene.objects.mechanisms import declaration, fingerprint, evidence_files, sampling_tolerances  # noqa: E402
from calibrated_scene_reconstruction.assets.mechanisms import check_results, parameter_constraints  # noqa: E402
from calibrated_scene_reconstruction.sim.mechanism_stage import audit as audit_mechanisms  # noqa: E402
from calibrated_scene_reconstruction.scene.objects.physics_trials import preserve  # noqa: E402
from calibrated_scene_reconstruction.scene.objects.physical_priors import declaration as physical_priors  # noqa: E402
from calibrated_scene_reconstruction.sim.physics_errors import PhysicsErrors  # noqa: E402

s = Session(args.session,args.configuration)
(s / 'qa/review').mkdir(parents=True, exist_ok=True)
(s / 'qa/review/S5.ran').touch()
physics_errors=PhysicsErrors(s.dir/(args.trial_output or 'qa')/'native_physics_errors.json')
(s / "qa/task_check.json").unlink(missing_ok=True)
script = source_path(s.config_dir, "task_check.py")
if not (s.config("task.json", {}) or {}).get("task"):
    s.write_json("qa/task_check.json", {"pass": None, "why": "no task given"})
    print("[task_check] SKIP: no task given", flush=True)
    app.close()
    sys.exit(0)
if not script.exists():
    s.write_json("qa/task_check.json", {"pass": False, "why": f"no task check written ({script.name})"})
    print(f"[task_check] FAIL: a task is given and no check is written (configs/{s.dir.name}/agent/task_check.py)",
          flush=True)
    app.close()
    sys.exit(0)

from calibrated_scene_reconstruction.scene.twin.task_provenance import snapshot as task_snapshot  # noqa: E402
task_inputs = task_snapshot(s)
configure_render(mode="rt", auto_exposure=args.background != "zone")
mechanism_contract = declaration(s)
backend_version = importlib.metadata.version('isaacsim')
if mechanism_contract and not backend_version.startswith('5.1.'):
    raise ValueError('This mechanism contract requires validation in the primary Isaac Sim 5.1 backend')
if mechanism_contract:
    parameter_constraints(mechanism_contract, {n: (o.get('fix') or {}).get('params', {})
                                              for n, o in s.scene_graph['dynamic_objects'].items()})
has_soft = mechanism_contract and any(link['kind'] != 'rigid' for obj in mechanism_contract['objects'].values()
                                     for link in obj['links'].values())
world = World(stage_units_in_meters=1.0, physics_dt=1 / 120, rendering_dt=1 / 30,
              device='cuda:0' if has_soft else 'cpu', backend='torch' if has_soft else 'numpy')
physics_errors.guard_steps(world)
if has_soft:
    world.get_physics_context().enable_gpu_dynamics(True)
    world.get_physics_context().set_broadphase_type('GPU')
stage = omni.usd.get_context().get_stage()
runtime = prepare_scene(s)
runtime.physics_case = args.physics_case if args.physics_case != 'all' else 'nominal'
ts = build_scene(stage, runtime, photo_goal(s, prepare_variant(s, {})))
mechanism_stage = audit_mechanisms(stage, mechanism_contract,sampling_tolerances=sampling_tolerances(s)) if mechanism_contract else None
from calibrated_scene_reconstruction.sim.simulation import audit as audit_simulation  # noqa: E402
simulation_profile = audit_simulation(stage, runtime.simulation)
if args.background in ("splat", "zone"):
    from calibrated_scene_reconstruction.sim.splat_stage import add_background

    add_background(stage, prepare_background(s), args.background)
world.reset()
W, H = 960, 720
K = np.array([[W / 2 / np.tan(np.radians(25)), 0, W / 2], [0, W / 2 / np.tan(np.radians(25)), H / 2], [0, 0, 1]])
state = {}


def render(look, eye, out):
    if "cam" not in state:
        state["cam"] = add_camera(stage, "/World/Cams/task_check", K, W, H, eye=eye, target=look)
        rp = rep.create.render_product(str(state["cam"].GetPath()), (W, H))
        state["ann"] = rep.AnnotatorRegistry.get_annotator("rgb")
        state["ann"].attach(rp)
    from calibrated_scene_reconstruction.sim.usd import look_at_quat, set_pose

    set_pose(state["cam"].GetPrim(), eye, look_at_quat(eye, look))
    for _ in range(20):
        world.step(render=True)
    img = np.ascontiguousarray(np.asarray(state["ann"].get_data())[..., :3])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img[:, :, ::-1])
    return str(out)


spec = importlib.util.spec_from_file_location("session_task_check", script)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
input_fingerprint = fingerprint(s)
try:
    ctx = Ctx(s, ts, world, stage, render)
except Exception as error:
    state=getattr(error,'particle_state',None)
    if state:
        destination=s.dir/(args.trial_output or 'qa')
        destination.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(destination/'failed_particle_registration.npz',**state)
    raise
ctx.authored_mechanism_audit = mechanism_stage
initial_articulations=ctx.initial_articulation_state()
initial_destination=s.dir/(args.trial_output or 'qa')
initial_destination.mkdir(parents=True,exist_ok=True)
(initial_destination/'initial_articulation_state.json').write_text(json.dumps(initial_articulations,indent=2)+'\n')
if not initial_articulations['pass']:raise ValueError('Native reset changed the authored initial articulation state')
if not (mechanism_contract or {}).get('observation_contract'):
    for _ in range(30):world.step(render=False)
ctx.native_errors = physics_errors
ctx.trial_output = args.trial_output
ctx.physics_case = args.physics_case
from calibrated_scene_reconstruction.tasks.rigid.check_script import validate as validate_script  # noqa: E402
validate_script(script, ctx)
actual_physics = {}
if physical_priors(s) is not None:
    for ref, estimate in ts.physical_estimates['bodies'].items():
        if 'native_total_mass_kg' in estimate:
            masses = as_numpy(ctx._cloth[ref].get_particle_masses())
            kg = float(masses.sum())
            if (not np.isfinite(masses).all() or (masses <= 0).any()
                    or not np.isclose(kg, estimate['native_total_mass_kg'], rtol=1e-4)):
                raise ValueError('Actual native cloth mass does not match the material estimate')
            actual_physics[ref] = {'mass_kg': kg, 'nodes': int(masses.size),
                                   'kind': 'native_surface_fem' if getattr(ctx._cloth[ref],'backend',None)=='surface_fem' else 'native_particle_cloth',
                                   'mass_source':getattr(ctx._cloth[ref],'mass_source','Native particle mass readback')}
            if getattr(ctx._cloth[ref],'backend',None)=='surface_fem':
                from pxr import PhysxSchema, UsdShade
                from calibrated_scene_reconstruction.assets.surface_contact_profile import audit as audit_surface_contacts
                prim=ctx._cloth[ref].prim;collision=PhysxSchema.PhysxCollisionAPI(prim)
                material,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
                actual={'rest_offset':collision.GetRestOffsetAttr().Get(),
                        'contact_offset':collision.GetContactOffsetAttr().Get(),
                        'self_filter_distance':prim.GetAttribute('physxDeformableBody:selfCollisionFilterDistance').Get()}
                actual_physics[ref]['surface_contacts']=audit_surface_contacts(estimate['thickness'],
                    estimate['sampled_surface_contacts'],material.GetPrim().GetAttribute('omniphysics:surfaceThickness').Get(),actual)
        if 'actual_mass_kg' in estimate:
            body = ctx.body(ref)
            kg,inertia = ctx.rigid_mass_properties(ref)
            if (not np.isclose(kg, estimate['actual_mass_kg'], rtol=1e-4)
                    or not np.isfinite(inertia).all() or np.linalg.eigvalsh(inertia).min() <= 0):
                raise ValueError('Actual runtime mass/inertia does not match the physical estimate')
            actual_physics[ref] = {'mass_kg': kg, 'inertia_kg_m2': inertia.tolist()}
from calibrated_scene_reconstruction.tasks.mechanisms.observation_match import run as check_observations  # noqa: E402
from calibrated_scene_reconstruction.scene.objects.material_response import run as check_material_response  # noqa: E402
material_response=check_material_response(ctx,mechanism_contract,getattr(ts,'physical_estimates',None),args.physics_case)
observation_output = (s.dir/args.trial_output/'physical_observations') if args.trial_output else s.dir/'qa/physical_observation_trial'
observed_behavior = check_observations(ctx, mechanism_contract, observation_output)
from calibrated_scene_reconstruction.scene.acceptance import declaration as acceptance_plan, evaluate as accept, may_attempt  # noqa: E402
acceptance_plan(ctx.task)
observation_defaults = {}
observation_physics_valid = True
for observed in observed_behavior['mechanisms'].values():
    for record in observed['checks'].values():
        observation_defaults['observation:'+record['observation_id']] = record
        samples = record.get('sampled_response', {}).get('samples', [])
        observation_physics_valid &= bool(samples) and all(
            row['result'].get('actual_material_state', {}).get('pass') is True for row in samples)
failed_observations = [name for name,value in observation_defaults.items() if value['pass'] is not True]
attempt_requested = may_attempt(ctx.task, failed_observations, physical_valid=observation_physics_valid)
res = mod.check(ctx) if (observed_behavior['pass'] or attempt_requested) and material_response['pass'] else {
    'pass': False, 'what': 'Native material-response qualification or observed passive source state failed; task not attempted',
    'mechanisms': {m['id']: {'pass': False, 'representation': m['representation'], 'checks': {}, 'evidence_files': []}
                   for m in mechanism_contract['mechanisms']}}
res['physical_observation_validation'] = observed_behavior
res['native_material_response']=material_response
question_defaults={'mechanism_question:'+q['id']:{'pass':q['status']=='resolved','question':q}
                   for q in (mechanism_contract or {}).get('open_questions',[]) if q['role']!='secondary'}
input_acceptance = accept(ctx.task, {'task_operation': {'pass': res.get('pass') is True},**question_defaults,
                                   **observation_defaults}, res.get('input_criteria', {}))
res['input_acceptance'] = input_acceptance
res['pass'] = input_acceptance['pass'] and material_response['pass']
for criterion in res.get('input_criteria', {}).values():
    res.setdefault('mechanisms', {}).setdefault('input_criteria', {'evidence_files': []})['evidence_files'].extend(
        criterion.get('evidence_files', []))
for name, observed in observed_behavior['mechanisms'].items():
    mechanism = res.setdefault('mechanisms', {}).setdefault(name, {})
    mechanism.setdefault('checks', {}).update(observed['checks'])
    mechanism.setdefault('evidence_files', []).extend(observed['evidence_files'])
    if 'operation_pass' in mechanism:
        mechanism['pass']=bool(mechanism['operation_pass'] and all(row['pass'] for row in mechanism['checks'].values()))
res = {"pass": bool(res.get("pass")), "script": f"configs/{s.dir.name}/{script.relative_to(s.config_dir).as_posix()}", **res}
res.update(validation_scope='object_model',robot_execution_required=bool(ctx.task.get('robot_execution_required',False)),
           hand_interaction_required=bool(ctx.task.get('hand_interaction_required',False)))
physics_errors.check()
res['native_physics_errors']=physics_errors.report()
res['native_collision_precision']=physics_errors.precision_audit()
res['pass']=bool(res['pass'] and res['native_collision_precision']['pass'])
res['task_input_provenance'] = task_inputs
res['task_inputs_unchanged'] = task_snapshot(s) == task_inputs
res['pass'] = bool(res['pass'] and res['task_inputs_unchanged'])
if mechanism_contract is not None:
    relaxed_observations = [name.split(':',1)[1] for name,row in input_acceptance['defaults'].items()
                            if name.startswith('observation:') and row['relaxed']]
    relaxed_questions = [name.split(':',1)[1] for name,row in input_acceptance['defaults'].items()
                         if name.startswith('mechanism_question:') and row['relaxed']]
    mechanism_checks = check_results(mechanism_contract, res.get('mechanisms'),
                                     relaxed_observations=relaxed_observations,relaxed_questions=relaxed_questions)
    res.update(mechanism_validation=mechanism_checks, input_fingerprint=input_fingerprint,
               mechanism_stage=mechanism_stage, backend={'name': 'isaacsim', 'version': backend_version})
    res['simulation_profile'] = simulation_profile
    res['mechanism_evidence_sha256'] = evidence_files(s, res.get('mechanisms', {}))
    if any(m['role'] != 'secondary' for m in mechanism_contract['mechanisms']) and not res['mechanism_evidence_sha256']:
        res['pass'] = False
        res['mechanism_validation'] = {'pass': False, 'failures': ['No saved mechanism evidence']}
    res['pass'] = res['pass'] is True and mechanism_checks['pass'] and fingerprint(s) == input_fingerprint
if physical_priors(s) is not None:
    res['physical_estimates'] = ts.physical_estimates
    res['physical_estimates']['runtime_rigid_state'] = actual_physics
    if not args.trial_output:
        raise ValueError('Physical-prior validation must run through all trial cases, with preserved trial outputs')
    destination = (s.dir/args.trial_output).resolve()
    if not destination.is_relative_to((s.dir/'qa/physics_trials').resolve()):
        raise ValueError('Trial outputs must stay in qa/physics_trials')
    preserve(s, res, destination)
else:
    s.write_json("qa/task_check.json", res)
print(f"[task_check] {'PASS' if res['pass'] else 'FAIL'}: {res.get('what', '')}", flush=True)
physics_errors.close()
app.close()
