"""Candidate-level native articulated-hand cloth contact assay.

This preserves diagnostics before official object export. It never grants a
default waiver or delivery approval by itself. No cloth forces or attachments.
"""
import argparse
import hashlib
import math
import json
from pathlib import Path
import shutil
import traceback

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('session');p.add_argument('--candidate',required=True);p.add_argument('--hand-plan',required=True)
p.add_argument('--assembly',required=True);p.add_argument('--body-link',default='body');p.add_argument('--lid-link',default='lid')
p.add_argument('--cloth-link',default='wipe');p.add_argument('--output',required=True)
p.add_argument('--case',choices=['nominal','low','high'],default='nominal')
p.add_argument('--physics-plan',help='QA-owned diagnostic material hypothesis; cannot grant input acceptance')
p.add_argument('--mechanism-plan',help='QA-owned matching representation hypothesis; requires --physics-plan and remains diagnostic')
p.add_argument('--rigid-collision-plan',help='QA-owned diagnostic map of rigid link names to sdf/convexDecomposition; cannot grant input acceptance')
p.add_argument('--fixture-proof',help='Current-session passing free-hand control result; defaults to the hand plan reference')
p.add_argument('--simulation-plan',help='QA-owned numerical profile hypothesis; cannot grant input acceptance')
p.add_argument('--bootstrap-hz',type=int,help='Diagnostic real native reset clock, up to1000000Hz; no state overwrites or input acceptance')
p.add_argument('--sample-hz',type=int,default=20,
    help='Recorded-state/check rate5..60Hz; changing the20Hz default is diagnostic and cannot grant acceptance')
p.add_argument('--experimental-native-viscosity-proof',help='Matched native-elastic transport proof; diagnostic runtime only')
p.add_argument('--viscosity-linear-solver',choices=['sparse_direct','cuda_direct'],default='sparse_direct')
p.add_argument('--diagnose-thickness-residual',action='store_true',
    help='Continue past finite-thickness residuals for diagnosis; retain failures and prohibit acceptance. Actual crossings, material and solid failures still stop.')
p.add_argument('--raw-thickness-diagnostic',action='store_true',
    help='With --diagnose-thickness-residual, retain the conservative raw distance failures without expensive local-bend certification; never acceptance evidence')
p.add_argument('--zero-friction-lift',action='store_true',help='Matched negative control: FEM at lift start; qualified particles after lift and before isolated hold')
p.add_argument('--particle-friction-proof',help='Six current native PBD controls; particle-wide intervention requires full-sheet isolation during hold')
p.add_argument('--replay-motion',help='Saved positive motion.json; required for a matched zero-friction control')
p.add_argument('--stop-after',choices=['opening','approach','pinch','lift','release'],default='release')
p.add_argument('--require-contact-free-approach',action='store_true',
    help='Optional planning constraint: reject expected deformable contact during approach; native contact checks always apply')
p.add_argument('--closure-descent-m',type=float,default=None,
    help='Final descent distance selected from source geometry; executed with open fingers by default')
p.add_argument('--closure-mode',choices=['open_descent','coupled'],default='open_descent',
    help='Descend with fingers open before closing; coupled preserves the explicit legacy simultaneous motion')
p.add_argument('--open-descent-seconds',type=float,default=.5)
p.add_argument('--closure-contact-link',action='append',default=[],help='Declared expected rigid contact during finite-force closure; actual penetration checks remain mandatory')
p.add_argument('--closure-contact-basis',help='Task/source basis for expected closure contact')
p.add_argument('--inside-finger',choices=['either','thumb','index'],default='either',
    help='Optional source-hand grasp proposal constraint; both finger assignments remain available')
p.add_argument('--minimum-normal-alignment',type=float,default=None,
    help='Minimum pinch-axis/local-sheet-normal cosine for proposals; 0 retains unfiltered legacy proposals. Native task checks are unchanged')
p.add_argument('--pinch-material-axis',type=float,nargs=3,help='Longitudinal direction in the intrinsic sheet chart; recomputed from actual material near each candidate grasp')
p.add_argument('--pinch-material-frame',choices=['spatial','opening_plane'],default='spatial')
p.add_argument('--material-contact-mode',choices=['same_component','single_patch'],default='same_component')
p.add_argument('--pinch-axis',type=float,nargs=3,help='Source-reviewed longitudinal pinch direction in the body frame, projected perpendicular to body up')
p.add_argument('--minimum-pinch-axis-alignment',type=float,default=.94,help='Actual projected pad-line/source-axis cosine; requires a body or material axis')
p.add_argument('--pinch-settle-seconds',type=float,default=.4,
    help='Bounded finite-drive hold at the end of closure before contact evaluation; 0 preserves the immediate legacy evaluation')
p.add_argument('--track-pinch-material',action='store_true',
    help='Bounded wrist-only feedback during closure; record every command for exact zero-friction replay')
p.add_argument('--opening-hold-seconds',type=float,default=.5,
    help='Finite open-lid hold in the test motion; task conditions and matched control still apply')
p.add_argument('--opening-seconds',type=float,default=1.5,
    help='Finite lid transition in the test motion; actual physical checks remain mandatory')
p.add_argument('--approach-seconds',type=float,default=1.5,
    help='Finite hand approach duration; actual motor tracking and contacts remain required')
p.add_argument('--pinch-seconds',type=float,default=1.,
    help='Finite finger-closure duration; actual source effort and velocity limits remain active')
p.add_argument('--closed-hold-seconds',type=float,default=0.,
    help='Diagnostic closed-contact conditioning:0.5s finite lid-drive ramp followed by this hold; cannot grant acceptance')
p.add_argument('--closed-angle-rad',type=float,
    help='Explicit source-supported closure target required for closed-contact conditioning')
args=p.parse_args()
if bool(args.closure_contact_link)!=bool(args.closure_contact_basis and args.closure_contact_basis.strip()):
    p.error('Expected closure contacts require both link names and a source/task basis')
if args.minimum_normal_alignment is None:
    args.minimum_normal_alignment=0. if args.pinch_axis is not None or args.pinch_material_axis is not None else .7
if args.raw_thickness_diagnostic and not args.diagnose_thickness_residual:
    p.error('--raw-thickness-diagnostic requires --diagnose-thickness-residual and cannot grant acceptance')
if not 5<=args.sample_hz<=60:p.error('Recorded-state/check sample rate must be between5and60Hz')
if args.pinch_axis is not None and args.pinch_material_axis is not None:p.error('Choose one source-axis convention')
if args.pinch_material_axis is not None and (not all(math.isfinite(v) for v in args.pinch_material_axis) or sum(v*v for v in args.pinch_material_axis)<1e-12):p.error('Resolved finite chart axis required')
if args.mechanism_plan and not args.physics_plan:p.error('Representation hypotheses require matching explicit material priors')
if not 0<args.minimum_pinch_axis_alignment<=1:p.error('Pinch-axis cosine must be within(0,1]')
if args.pinch_axis is not None and (not all(math.isfinite(v) for v in args.pinch_axis) or math.hypot(*args.pinch_axis[:2])<1e-8):p.error('Source pinch axis must have a finite resolved horizontal projection')
if not 0<=args.minimum_normal_alignment<=1:p.error('Normal alignment cosine must be within0..1')
if not 0<=args.pinch_settle_seconds<=2.:p.error('Pinch settling must be between0and2seconds')
if not math.isfinite(args.open_descent_seconds) or not .04<=args.open_descent_seconds<=5:
    p.error('Open descent duration must be between0.04and5seconds')
if args.closure_descent_m is not None and not 0<=args.closure_descent_m<=.03:p.error('Closure descent must be between0and30mm')
if not 0<=args.opening_hold_seconds<=5.:p.error('Open-lid hold must be between0and5seconds')
if any(not math.isfinite(v) or not .25<=v<=5. for v in
       [args.opening_seconds,args.approach_seconds,args.pinch_seconds]):
    p.error('Finite opening, approach and pinch durations must be between0.25and5seconds')
if not 0<=args.closed_hold_seconds<=3.:p.error('Closed-lid hold must be between0and3seconds')
if (args.closed_hold_seconds>0)!=(args.closed_angle_rad is not None):p.error('Closed conditioning requires both a positive hold and an explicit angle')
if args.zero_friction_lift and not args.replay_motion:p.error('Zero-friction control requires the positive --replay-motion')
out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
shutil.copy2(__file__,out/'source.py');shutil.copy2(args.candidate,out/'candidate_input.json');shutil.copy2(args.hand_plan,out/'hand_input.json')

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402
app=SimulationApp({'headless':True})

from copy import deepcopy  # noqa: E402
from types import SimpleNamespace  # noqa: E402
import importlib.metadata  # noqa: E402
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from pxr import Gf,PhysxSchema,Usd,UsdGeom,UsdPhysics,UsdShade  # noqa: E402
from calibrated_scene_reconstruction.scene.session import Session  # noqa: E402
from calibrated_scene_reconstruction.scene.acceptance_evidence import subject as acceptance_subject  # noqa: E402
from calibrated_scene_reconstruction.scene.capture.articulated_model import from_usd  # noqa: E402
from calibrated_scene_reconstruction.assets.metric_mesh import asset  # noqa: E402
from calibrated_scene_reconstruction.assets.physical_priors import sample  # noqa: E402
from calibrated_scene_reconstruction.assets.mechanisms import revolute_position,joint_frames  # noqa: E402
from calibrated_scene_reconstruction.assets.cloth_metric import MaterialMetric  # noqa: E402
from calibrated_scene_reconstruction.assets.response_policy import mode as response_mode,material_verdict  # noqa: E402
from calibrated_scene_reconstruction.assets.cloth_container_state import check as container_state  # noqa: E402
from calibrated_scene_reconstruction.sim.mechanism_stage import add_links  # noqa: E402
from calibrated_scene_reconstruction.sim.physical_priors import apply as apply_priors  # noqa: E402
from calibrated_scene_reconstruction.sim.physics import physics_material  # noqa: E402
from calibrated_scene_reconstruction.sim.usd import add_box_collider  # noqa: E402
from calibrated_scene_reconstruction.sim.simulation import apply as apply_simulation,audit as audit_simulation  # noqa: E402
from calibrated_scene_reconstruction.sim.hand_fixture import HandFixture  # noqa: E402
from calibrated_scene_reconstruction.sim.hand_materials import summary as native_hand_materials,author_friction  # noqa: E402
from calibrated_scene_reconstruction.sim.hand_collision import colliders as hand_colliders,Penetration,PlannedClearance,OpenApproachClearance,PlannedPinchContact,ApproachSamples,OpenPinchReadiness  # noqa: E402
from calibrated_scene_reconstruction.sim.physics_errors import PhysicsErrors  # noqa: E402
from calibrated_scene_reconstruction.sim.cloth_contact import self_intersections  # noqa: E402
from calibrated_scene_reconstruction.sim.cloth_clearance import measure as finite_clearance  # noqa: E402
from calibrated_scene_reconstruction.sim.surface_distance import SignedSurface,triangle_samples  # noqa: E402
from calibrated_scene_reconstruction.tasks.rigid.task_checks import Ctx,as_numpy  # noqa: E402
from calibrated_scene_reconstruction.tasks.mechanisms.hand_cloth import rim_target,lift_measurement,same_hand_plan,opening_assignment,pinch_contact,clearance_offsets,pinch_axis_alignment,material_tangent,capture_target  # noqa: E402
from calibrated_scene_reconstruction.tasks.mechanisms.hand_motion import decode as decode_motion,orientation_path  # noqa: E402
from calibrated_scene_reconstruction.tasks.mechanisms.closure_tracking import ClosureTracking  # noqa: E402
from calibrated_scene_reconstruction.tasks.mechanisms.hand_contact_isolation import material_samples  # noqa: E402
from calibrated_scene_reconstruction.tasks.mechanisms.contact import contact_separations

errors=PhysicsErrors(out/'native_physics_errors.json')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
try:
    s=Session(Path(args.session).resolve());info=json.loads((out/'candidate_input.json').read_text());hand=json.loads((out/'hand_input.json').read_text())
    if info.get('requires_experimental_particle_shell_runtime'):
        raise ValueError('Experimental shell candidate cannot be tested through a substituted native FEM runtime')
    proof_path=Path(args.fixture_proof).resolve() if args.fixture_proof else (s.dir/hand['native_fixture_check']).resolve()
    if not proof_path.is_relative_to(s.dir/'qa'):
        raise ValueError('Free-hand control proof must belong to this session QA')
    fixture_proof=json.loads(proof_path.read_text())
    if fixture_proof.get('pass') is not True:
        raise ValueError('Actual free-hand control must pass before an object assay')
    if not same_hand_plan(json.loads(proof_path.with_name('hand_input.json').read_text()),hand):
        raise ValueError('Free-hand control used a different hand plan')
    if (not fixture_proof.get('native_property_checks') or not all(fixture_proof['native_property_checks'].values())
            or not np.isclose(fixture_proof['hand']['finger_effort_limit_Nm'],hand['finger_effort_limit_Nm'])
            or fixture_proof['hand'].get('joint_friction_prior')!=hand.get('joint_friction_prior')):
        raise ValueError('Free-hand native properties or actuator settings differ from this assay')
    for name,expected in fixture_proof['input_sha256'].items():
        if sha(name)!=expected:raise ValueError('Free-hand control inputs changed: '+name)
    mesh=Path(info['path'])/'mesh.npz'
    if not mesh.resolve().is_relative_to(s.dir.resolve()/'qa'):raise ValueError('Use a current-session candidate')
    if info['report'].get('initial_state_admissibility',{}).get('pass') is not True:
        raise ValueError('Blender-rebuilt authored material/self/solid admissibility required')
    if info.get('requires_additional_preparation') or info.get('preparation_only') or info.get('requires_material_restore'):
        raise ValueError('An actuated construction seed needs completed released preparation before a task assay')
    g=s.scene_graph;dec=s.config('agent/mechanisms.json');priors=s.config('agent/physics_priors.json')
    accuracy_mode=response_mode(s.config('task.json',{}).get('flexible_accuracy'))
    hypothesis_files=[];collision_hypothesis={}
    profile=s.config('simulation.json')
    for label,value in [('physics',args.physics_plan),('mechanisms',args.mechanism_plan),('rigid_collision',args.rigid_collision_plan),
                        ('simulation',args.simulation_plan)]:
        if value is None:continue
        path=Path(value).resolve()
        if not path.is_relative_to(s.dir/'qa'):raise ValueError('Diagnostic hypotheses must belong to current session QA')
        data=json.loads(path.read_text());hypothesis_files.append(path)
        shutil.copy2(path,out/(label+'_hypothesis.json'))
        if label=='physics':priors=data
        elif label=='mechanisms':dec=data
        elif label=='simulation':profile=data
        else:collision_hypothesis=data
    if args.experimental_native_viscosity_proof:
        hypothesis_files.append(Path(args.experimental_native_viscosity_proof).resolve())
    timing={'opening':args.opening_seconds,'approach':args.approach_seconds,'pinch':args.pinch_seconds}
    diagnostic_hypothesis=bool(hypothesis_files or args.diagnose_thickness_residual or args.sample_hz!=20
        or args.closed_hold_seconds or args.bootstrap_hz is not None)
    if args.particle_friction_proof and not Path(args.particle_friction_proof).resolve().is_relative_to(s.dir/'qa'):
        raise ValueError('Particle friction proof must belong to the current session QA')
    spec=g['dynamic_objects'][args.assembly];requested=spec['fit']
    values=sample(priors,args.case)['bodies'][args.assembly+'/'+args.cloth_link]['parameters']
    reference_values=sample(priors,'nominal')['bodies'][args.assembly+'/'+args.cloth_link]['parameters']
    hz=int(profile['physics_hz'])
    if fixture_proof.get('physics_hz')!=hz or profile.get('solver')!='PGS':
        raise ValueError('Free-hand control timestep and validated PGS solver must match the object assay')
    # Numerical padding is authored against nominal thickness. Apply the
    # requested prior case once, through the same stage adapter as delivery.
    # Building with sampled thickness first would misinterpret nominal offsets
    # as a new reference, rejecting high cases or padding low cases twice.
    from calibrated_scene_reconstruction.assets.cloth_native_profile import resolve as native_cloth_profile
    cloth_backend=dec['objects'][args.assembly]['links'][args.cloth_link].get('backend','particle')
    declared_cloth=requested.get('link_physics',{}).get(args.cloth_link,{})
    if args.mechanism_plan and cloth_backend=='particle':
        # A complete candidate particle profile is mandatory when exploring a
        # different backend. Do not fill missing radii with unrelated FEM
        # settings or silently reinterpret continuum constants as springs.
        declared_cloth=info.get('cloth_physics',{})
    settings,candidate_numerical_changes=native_cloth_profile(cloth_backend,reference_values,
        declared_cloth,info.get('cloth_physics',{}))
    if cloth_backend=='particle':
        # PBD ignores the rigid partner's friction coefficient in the installed
        # contact controls. The FEM hand-material ablation cannot certify it.
        # Unqualified positive probes remain diagnostic. A separately
        # qualified PBD intervention must still pass the native matched pair.
        diagnostic_hypothesis |= not bool(args.particle_friction_proof)
        if args.zero_friction_lift and not args.particle_friction_proof:
            raise ValueError('Particle hand/cloth acceptance requires a qualified PBD-specific friction ablation')
    elif args.particle_friction_proof:
        raise ValueError('Particle friction proof cannot be applied to another cloth backend')
    if info.get('sdf',requested['sdf'])!=requested['sdf']:
        candidate_numerical_changes['sdf']={'declared':requested['sdf'],'candidate':info['sdf']}
    for link,link_settings in info.get('link_physics',{}).items():
        if link_settings!=requested.get('link_physics',{}).get(link,{}):
            candidate_numerical_changes['link_physics/'+link]={'declared':requested.get('link_physics',{}).get(link,{}),'candidate':link_settings}
    # A candidate may explore a contact profile, but cannot qualify an asset
    # whose declared delivery settings would be different from the tested ones.
    diagnostic_hypothesis |= bool(candidate_numerical_changes)
    fix={'model':True,'mesh':str(mesh.resolve()),'opening':info['report']['opening'],
        'sdf':deepcopy(info.get('sdf',requested['sdf'])),
        'link_physics':{**deepcopy(requested.get('link_physics',{})),**deepcopy(info.get('link_physics',{})),args.cloth_link:settings}}
    for link,approximation in collision_hypothesis.items():
        if dec['objects'][args.assembly]['links'].get(link,{}).get('kind')!='rigid' or approximation not in {'sdf','convexDecomposition'}:
            raise ValueError('Diagnostic collision plan must name declared rigid links and supported approximations')
        fix['link_physics'].setdefault(link,{})['collision_approximation']=approximation
    spec['fix']=fix;kit=asset(fix)
    parts={row['name']:row for row in kit['parts']};sheet=parts[args.cloth_link]
    support=dec['objects'][args.assembly]['links'][args.cloth_link].get('initial_support')
    initial_support=None
    if support:
        from calibrated_scene_reconstruction.assets.material_support import check as support_state
        base=parts[support['link']]
        initial_support=support_state(sheet['V'],sheet['cloth_rest']['points'],sheet['F'],base['V'],base['F'],support,
            thickness_m=reference_values['thickness'],tolerance_m=1e-6)
        (out/'initial_support_route.json').write_text(json.dumps(initial_support,indent=2)+'\n')
        if not initial_support['pass']:raise ValueError('Candidate contradicts the declared initial support/contact relation')
    replay_bytes=None
    if args.replay_motion:
        replay_path=Path(args.replay_motion).resolve()
        if not replay_path.is_relative_to(s.dir/'qa'):raise ValueError('Replay motion must belong to current-session QA')
        replay_bytes=replay_path.read_bytes()
        decode_motion(json.loads(replay_bytes),subject=acceptance_subject(s,args.assembly,mesh,args.case),
                      candidate=sha(args.candidate),hand=sha(args.hand_plan),node_count=len(sheet['V']),closure_mode=args.closure_mode,material_axis_frame=args.pinch_material_frame)
    radius=float(np.ptp(parts[args.body_link]['V'],axis=0)[:2].mean()/2)
    initial_distribution=container_state(sheet['V'],sheet['cloth_rest']['points'],sheet['F'],fix['opening'],hand['initial_material_distribution'])
    (out/'initial_material_distribution.json').write_text(json.dumps(initial_distribution,indent=2)+'\n')
    if not initial_distribution['pass']:raise ValueError('Candidate does not have the specified initial interior/exterior cloth distribution')
    model=from_usd(hand['asset'],100);open_state={j.name:float(hand['open_state'].get(j.name,0.)) for j in model.movable}
    closed_state={j.name:float(hand['closed_state'].get(j.name,0.)) for j in model.movable}
    opened_transforms=model.transforms(open_state)
    opened_pads=np.array([opened_transforms[hand[f]['link']][:3,:3]@np.asarray(hand[f]['pad_local'])
                          +opened_transforms[hand[f]['link']][:3,3] for f in ['thumb','index']])
    def initial_assignment(point,radial_rotation):
        pads=opened_pads@(radial_rotation@np.asarray(hand['R'])).T+point+radial_rotation@np.asarray(hand['p'])
        return (opening_assignment(pads,fix['opening'],radius,args.inside_finger)['pass']
                and (args.pinch_axis is None and args.pinch_material_axis is None or pinch_axis_alignment(pads,args.pinch_axis if args.pinch_axis is not None else radial_rotation[:,0],up=None if args.pinch_material_axis is not None else [0,0,1],minimum=args.minimum_pinch_axis_alignment)['pass']))
    # Initialize the hand above the scene in the requested finger assignment.
    # Otherwise an index-inside attempt starts thumb-inside and asks a finite
    # wrist controller to turn about180degrees while already descending.
    selection=rim_target(sheet['V'],sheet['cloth_rest']['points'],sheet['F'],fix['opening'],outer_radius=radius,
                         feasible=initial_assignment if args.inside_finger!='either' or args.pinch_axis is not None or args.pinch_material_axis is not None else None,pinch_axis=args.pinch_axis,material_axis=args.pinch_material_axis)
    target=selection['point'];rotation=selection['rotation']@np.asarray(hand['R']);offset=selection['rotation']@np.asarray(hand['p'])
    source_hinge=next(j for j in kit['joints'] if j['type']=='revolute')
    sweep=float(np.linalg.norm(parts[args.lid_link]['V']-source_hinge['anchor0'],axis=1).max())
    hover_height=max(float(hand['approach_height_m']),sweep+.04)
    wrist_at=target+offset;wrist_hover=wrist_at+[0,0,hover_height]
    class LocalSession(Session):
        @property
        def scene_graph(self):return deepcopy(g)
        def config(self,name,*default):
            if name=='agent/mechanisms.json':return deepcopy(dec)
            if name=='agent/physics_priors.json':return deepcopy(priors)
            if name=='simulation.json':return deepcopy(profile)
            return super().config(name,*default)
    local=LocalSession(s.dir)
    world=World(stage_units_in_meters=1.,physics_dt=1/hz,rendering_dt=1/30,device='cuda:0',backend='torch');errors.guard_steps(world)
    pc=world.get_physics_context();pc.enable_gpu_dynamics(True);pc.set_broadphase_type('GPU')
    stage=omni.usd.get_context().get_stage();apply_simulation(stage,profile,flexible=True)
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):PhysxSchema.PhysxSceneAPI.Apply(prim).CreateEnableCCDAttr(False)
    desk=add_box_collider(stage,'/World/Support/task_desk/surface',[-.5,-.5,-.02],[.5,.5,0.])
    # The source inventory is required for every hand trial, including a
    # diagnostic without a particle-friction proof. Do not leave this fixture
    # on the schema's unresolved automatic contact-offset sentinel.
    collision=PhysxSchema.PhysxCollisionAPI.Apply(desk.GetPrim())
    collision.CreateContactOffsetAttr(.001);collision.CreateRestOffsetAttr(0.)
    material=physics_material(stage,'/World/Looks/TestPhysics',.6,.45,0.)
    root=add_links(stage,'/World/Objects/'+args.assembly,kit,fix,spec,material)
    prior_application=apply_priors(stage,priors,dec,args.case)
    sampled_contacts=prior_application['bodies'][args.assembly+'/'+args.cloth_link].get('sampled_surface_contacts')
    if sampled_contacts:
        settings.update(values,friction=values['dynamic_friction'],contact_offset=sampled_contacts['contact_offset'],
            rest_offset=sampled_contacts['rest_offset'],self_collision_filter_distance=sampled_contacts['self_filter_distance'])
    sampled_particles=prior_application['bodies'][args.assembly+'/'+args.cloth_link].get('sampled_particle_contacts')
    if sampled_particles:settings.update(sampled_particles)
    if args.experimental_native_viscosity_proof:
        if sheet.get('backend')!='surface_fem':raise ValueError('Native viscosity requires surface FEM')
        surface=stage.GetPrimAtPath(str(root.GetPath())+'/'+args.cloth_link+'/surface')
        physical=UsdShade.MaterialBindingAPI(surface).ComputeBoundMaterial('physics')[0].GetPrim()
        physical.GetAttribute('physxDeformableMaterial:elasticityDamping').Set(0.)
        physical.GetAttribute('physxDeformableMaterial:bendDamping').Set(0.)
        surface.GetAttribute('physxDeformableBody:linearDamping').Set(0.)
    holder=UsdPhysics.FixedJoint.Define(stage,'/World/TestFixtures/container_holder')
    holder.CreateBody1Rel().SetTargets([str(root.GetPath())+'/'+args.body_link]);holder.CreateLocalPos0Attr(Gf.Vec3f(0));holder.CreateLocalPos1Attr(Gf.Vec3f(0))
    hinges=[prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.RevoluteJoint)
            and str(prim.GetPath()).startswith(str(root.GetPath()))]
    if len(hinges)!=1:raise ValueError('Name an assembly with one attached flip hinge')
    hinge=UsdPhysics.RevoluteJoint(hinges[0]);joint=next(j for j in kit['joints'] if j['type']=='revolute')
    if args.closed_angle_rad is not None and (not np.isfinite(args.closed_angle_rad)
            or not joint['limits'][0]<=args.closed_angle_rad<=joint['limits'][1]):
        raise ValueError('Diagnostic closure target must respect the declared hinge limits')
    drive=UsdPhysics.DriveAPI.Apply(hinge.GetPrim(),'angular');drive.CreateTypeAttr('force')
    drive.CreateStiffnessAttr(.2*np.pi/180);drive.CreateDampingAttr(.02*np.pi/180);drive.CreateMaxForceAttr(.2)
    initial_angle=float(joint.get('initial_position',0.));drive.CreateTargetPositionAttr(float(np.degrees(initial_angle)))
    fixture=HandFixture(stage,hand['asset'],model,open_state,wrist_hover,rotation,
        finger_effort_Nm=hand['finger_effort_limit_Nm'],joint_friction_prior=hand.get('joint_friction_prior'))
    hand_surfaces=hand_colliders(stage,fixture.path);hand_penetration=Penetration(hand_surfaces)
    planned_hand=PlannedClearance(hand_surfaces,model,open_state,closed_state)
    planned_open_hand=OpenApproachClearance(hand_surfaces,model,open_state)
    planned_pinch=PlannedPinchContact(hand_surfaces,model,closed_state,[hand[f]['link'] for f in ['thumb','index']])
    open_capture=OpenPinchReadiness(hand_surfaces,[hand[f]['link'] for f in ['thumb','index']])
    from isaacsim.core.prims import RigidPrim
    hand_body_paths=[fixture.path+'/'+name for name in model.transforms(open_state)]
    from calibrated_scene_reconstruction.sim.collision_inventory import source_groups
    collision_owners={name:str(root.GetPath())+'/'+name for name,part in parts.items() if part['kind']=='rigid'}
    collision_owners['task_desk']=str(desk.GetPath())
    # This inventory describes rigid obstacles. Surface FEM cloth also has a
    # CollisionAPI, but is validated separately as the deforming material.
    non_rigid_roots=[str(root.GetPath())+'/'+name for name,part in parts.items() if part['kind']!='rigid']
    source_inventory=source_groups(stage,collision_owners,excluded_roots=[fixture.path,*non_rigid_roots])
    container_body_paths=sorted({collider['body_path'] for name in parts if name in source_inventory
                                 for collider in source_inventory[name]['colliders']})
    # PhysX requires one filter group per sensor expression, including explicit
    # body paths. A flat list creates a single group for all 33 expressions.
    rigid_contacts=RigidPrim(hand_body_paths,
        contact_filter_prim_paths_expr=[container_body_paths[:] for _ in hand_body_paths],
        max_contact_count=32768,reset_xform_properties=False,name='hand_container_contacts')
    sources=[Path(__file__),Path('src/calibrated_scene_reconstruction/sim/reset_timestep.py'),Path('src/calibrated_scene_reconstruction/sim/mechanism_stage.py'),Path('src/calibrated_scene_reconstruction/sim/physical_priors.py'),
             Path('src/calibrated_scene_reconstruction/tasks/mechanisms/material_grasp.py'),
             Path('src/calibrated_scene_reconstruction/sim/contact_carrier.py'),Path('src/calibrated_scene_reconstruction/sim/source_moments.py'),
             Path('src/calibrated_scene_reconstruction/assets/revolved_source_cells.py'),
             Path('src/calibrated_scene_reconstruction/assets/cloth_contact_tolerance.py'),
             Path('src/calibrated_scene_reconstruction/assets/articulation_initial_state.py'),Path('src/calibrated_scene_reconstruction/sim/collision_inventory.py'),*hypothesis_files,
             Path('src/calibrated_scene_reconstruction/sim/hand_fixture.py'),Path('src/calibrated_scene_reconstruction/tasks/mechanisms/hand_cloth.py'),
             Path('src/calibrated_scene_reconstruction/tasks/mechanisms/hand_motion.py'),
             Path('src/calibrated_scene_reconstruction/tasks/mechanisms/closure_tracking.py'),
             Path('src/calibrated_scene_reconstruction/tasks/mechanisms/hand_contact_isolation.py'),
             Path('src/calibrated_scene_reconstruction/tasks/mechanisms/particle_hand_acceptance.py'),
             Path('src/calibrated_scene_reconstruction/sim/hand_materials.py'),
             Path('src/calibrated_scene_reconstruction/assets/response_policy.py'),
             Path('src/calibrated_scene_reconstruction/assets/joint_state.py'),
             Path('src/calibrated_scene_reconstruction/assets/cloth_native_profile.py'),
             Path('src/calibrated_scene_reconstruction/sim/particle_surface.py'),Path('src/calibrated_scene_reconstruction/sim/particle_contacts.py'),
             Path('src/calibrated_scene_reconstruction/sim/particle_registration_output.py'),Path('src/calibrated_scene_reconstruction/assets/particle_registration.py'),
             Path('src/calibrated_scene_reconstruction/assets/particle_contacts.py'),Path('src/calibrated_scene_reconstruction/assets/particle_contact_profile.py'),
             Path('src/calibrated_scene_reconstruction/assets/cloth_rest.py'),Path('src/calibrated_scene_reconstruction/assets/cloth_mass.py'),
             Path('src/calibrated_scene_reconstruction/assets/surface_contact_profile.py'),
             Path('src/calibrated_scene_reconstruction/assets/cloth_metric.py'),Path('src/calibrated_scene_reconstruction/sim/simulation.py'),
             Path('src/calibrated_scene_reconstruction/assets/support_route.py'),
             Path('src/calibrated_scene_reconstruction/sim/cloth_clearance.py'),Path('src/calibrated_scene_reconstruction/sim/native_viscosity_diagnostic.py'),
             Path('src/calibrated_scene_reconstruction/sim/cloth_shell_clearance.py'),Path('src/calibrated_scene_reconstruction/sim/cloth_contact.py'),
             Path('src/calibrated_scene_reconstruction/sim/surface_cloth.py'),Path('src/calibrated_scene_reconstruction/sim/hand_collision.py'),
             Path('src/calibrated_scene_reconstruction/assets/cloth_container_state.py'),Path(args.hand_plan),mesh,*map(Path,model.dependencies),
             Path('src/calibrated_scene_reconstruction/assets/material_support.py'),Path('src/calibrated_scene_reconstruction/assets/support_route.py'),
             proof_path,*map(Path,fixture_proof['input_sha256']),proof_path.with_name('states.npz'),proof_path.with_name('hand_input.json')]
    if args.particle_friction_proof:
        sources += list(map(Path,[args.particle_friction_proof,
            'src/calibrated_scene_reconstruction/sim/particle_grasp.py','src/calibrated_scene_reconstruction/sim/particle_friction.py',
            'src/calibrated_scene_reconstruction/scene/objects/particle_hand_proof.py',
            'src/calibrated_scene_reconstruction/scene/objects/particle_contact_binding.py',
            'src/calibrated_scene_reconstruction/scene/objects/particle_friction_ablation.py']))
    protocol={'candidate_sha256':sha(args.candidate),'input_sha256':{str(q.resolve()):sha(q) for q in sources},
        'diagnostic_hypothesis':diagnostic_hypothesis,'initial_articulation_state':None,
        'candidate_numerical_changes':candidate_numerical_changes,
        'material_accuracy_mode':accuracy_mode,
        'require_contact_free_approach':args.require_contact_free_approach,
        'closure_descent_policy':'source_clearance' if args.closure_descent_m is None else args.closure_descent_m,
        'closure_mode':args.closure_mode,'open_descent_seconds':args.open_descent_seconds,
        'expected_closure_contacts':{'links':args.closure_contact_link,'basis':args.closure_contact_basis,
            'native_penetration_checks_relaxed':False},
        'inside_finger_policy':args.inside_finger,
        'source_pinch_axis':args.pinch_axis,'source_material_axis':args.pinch_material_axis,'minimum_pinch_axis_alignment':args.minimum_pinch_axis_alignment,
        'pinch_material_frame':args.pinch_material_frame,
        'material_contact_mode':args.material_contact_mode,
        'minimum_normal_alignment':args.minimum_normal_alignment,
        'pinch_settle_seconds':args.pinch_settle_seconds,
        'pinch_material_tracking':{'enabled':args.track_pinch_material,
            'maximum_shift_m':.008,'speed_m_s':.01,'observation_hz':240.,
            'feedback':'Actual native finger-pad world transforms; bounded residual correction',
            'cloth_actuation':False,'matched_control':'Exact recorded wrist targets, no feedback recomputation'},
        'diagnose_thickness_residual':args.diagnose_thickness_residual,
        'raw_thickness_diagnostic':args.raw_thickness_diagnostic,
        'opening_hold_seconds':args.opening_hold_seconds,
        'closed_conditioning':None if not args.closed_hold_seconds else {
            'target_angle_rad':args.closed_angle_rad,'ramp_s':.5,'hold_s':args.closed_hold_seconds,
            'lid_effort_limit_Nm':.2,'cloth_actuation':False,'delivery_approved':False},
        'experimental_material_operator':None if not args.experimental_native_viscosity_proof else {
            'name':'native_elasticity_with_internal_viscosity',
            'transport_proof_sha256':sha(args.experimental_native_viscosity_proof),
            'linear_solver':args.viscosity_linear_solver,
            'native_damping_overrides':{'elasticityDamping':0.,'bendDamping':0.,'linearDamping':0.},
            'external_cloth_actuation':False,'task_or_delivery_qualified':False},
        'subject_sha256':acceptance_subject(s,args.assembly,mesh,args.case),
        'simulation':profile,'simulation_audit':audit_simulation(stage,profile),
        'transition_seconds':timing,
        'requested_state_sample_hz':args.sample_hz,
        'case':args.case,'zero_friction_lift':args.zero_friction_lift,
        'friction_intervention_phase':'hold_after_lift' if args.particle_friction_proof else 'lift',
        'particle_friction_proof':None if not args.particle_friction_proof else str(Path(args.particle_friction_proof).resolve()),
        'container_holder':True,'cloth_forces':False,'cloth_attachments':False,
        'assembly':args.assembly,'body_link':args.body_link,'lid_link':args.lid_link,'cloth_link':args.cloth_link,
        'candidate_mesh_sha256':sha(mesh),'authored_physics':{k:v for k,v in fix.items() if k in ['sdf','link_physics']},
        'initial_material_distribution':initial_distribution,'lid_sweep_radius_m':sweep,'hover_height_m':hover_height,
        'initial_support_route':initial_support,
        'physical_prior_application':prior_application,
        'free_hand_control_proof':str(proof_path.relative_to(s.dir)),
        'selection':{k:v.tolist() if hasattr(v,'tolist') else v for k,v in selection.items()},
        'task':s.config('task.json'),'priors':priors,'mechanisms':dec}
    motion={k:v for k,v in protocol.items() if k!='zero_friction_lift'}
    motion_hash=hashlib.sha256(json.dumps(motion,sort_keys=True).encode()).hexdigest()
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');stage.GetRootLayer().Export(str(out/'initial.usda'))
    # Minimize native warmup before the experimental material operator exists.
    if args.experimental_native_viscosity_proof:world.set_simulation_dt(physics_dt=1e-6,rendering_dt=1/30)
    from calibrated_scene_reconstruction.sim.reset_timestep import reset as native_reset
    protocol['native_initialization']=native_reset(world,bootstrap_hz=args.bootstrap_hz)
    fixture.initialize();rigid_contacts.initialize();ctx=Ctx(local,SimpleNamespace(objects={args.assembly:root}),world,stage,None)
    protocol['native_hand_joint_names']=fixture.names;protocol['hand_root_path']=fixture.path
    initial_joints=ctx.initial_articulation_state()
    protocol['initial_articulation_state']=initial_joints
    (out/'initial_articulation_state.json').write_text(json.dumps(initial_joints,indent=2)+'\n')
    if not initial_joints['pass']:raise ValueError('Native reset changed the authored initial articulation state')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    particle_control=None;isolation_history=[];precontrol=None;control_wrist=None
    if args.particle_friction_proof:
        from calibrated_scene_reconstruction.sim.particle_grasp import ParticleGraspControl
        particle_control=ParticleGraspControl(ctx._cloth[args.assembly+'/'+args.cloth_link],values,
            args.particle_friction_proof,physics_hz=hz,solver=profile['solver'])
        protocol['particle_contact_qualification']=particle_control.proof
        protocol['input_sha256'].update(particle_control.proof['input_sha256'])
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    native_viscosity_report=None
    if args.experimental_native_viscosity_proof:
        world.set_simulation_dt(physics_dt=1/hz,rendering_dt=1/30)
        from calibrated_scene_reconstruction.sim.native_viscosity_diagnostic import install as install_viscosity
        native_viscosity_report=install_viscosity(world,ctx._cloth[args.assembly+'/'+args.cloth_link],
            proof_path=args.experimental_native_viscosity_proof,session_dir=s.dir,out=out,
            faces=sheet['F'],rest=sheet['cloth_rest'],material=values,cloth=settings,
            solver=profile['solver'],physics_hz=hz,position_iterations=settings['solver_position_iterations'],
            linear_solver=args.viscosity_linear_solver)
    hand_material_view=fixture.art._articulation_view._physics_view
    initial_native_material=native_hand_materials(hand_material_view)
    if (not np.allclose(initial_native_material['static_friction'],fixture.description['friction'],atol=1e-7)
            or not np.allclose(initial_native_material['dynamic_friction'],fixture.description['friction'],atol=1e-7)):
        raise ValueError('Native hand shape materials differ from the authored positive friction')
    ctx.native_errors=errors
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.types import ArticulationAction
    lid_motor=SingleArticulation(str(root.GetPath())+'/'+args.body_link,name='lid_test_motor');lid_motor.initialize()
    if len(lid_motor.dof_names)!=1:raise ValueError('The lid test fixture requires exactly one native hinge DOF')
    sdf_error=0.
    for name,part in parts.items():
        if part['kind']!='rigid':continue
        collider=stage.GetPrimAtPath(str(root.GetPath())+'/'+name+'/surface')
        resolution=PhysxSchema.PhysxSDFMeshCollisionAPI(collider).GetSdfResolutionAttr().Get()
        if resolution:
            actual=errors.sdf_resolution(collider.GetPath(),int(resolution))
            sdf_error=max(sdf_error,2*float(np.ptp(part['V'],axis=0).max())/actual['effective_resolution_bound'])
    from calibrated_scene_reconstruction.assets.cloth_contact_tolerance import solid_tolerance as contact_tolerance
    solid_tolerance=contact_tolerance(float(values['thickness']),sdf_error)
    validity=priors['bodies'][args.assembly+'/'+args.cloth_link]['validity_hypothesis']
    metric=MaterialMetric(sheet['V'],sheet['F'],sheet['cloth_rest'],max_edge_strain=validity['max_edge_strain'],max_area_change=validity['max_area_change'])
    surfaces={name:SignedSurface(part['V'],part['F']) for name,part in parts.items() if part['kind']=='rigid'}
    isolation_margins={}
    if particle_control is not None:
        from calibrated_scene_reconstruction.sim.collision_inventory import source_groups
        owners={name:str(root.GetPath())+'/'+name for name in surfaces}
        owners['task_desk']=str(desk.GetPath())
        inventory=source_groups(stage,owners,excluded_roots=[fixture.path,*non_rigid_roots])
        (out/'non_hand_collision_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
        for name,group in inventory.items():
            isolation_margins[name]=float(group['maximum_contact_offset_m']+settings['contact_offset']+solid_tolerance)
    def isolation_sample():
        from calibrated_scene_reconstruction.tasks.mechanisms.hand_contact_isolation import clearance as isolate
        import torch
        # Reduce the full native sheet on GPU. The two bounds enclose all its
        # triangles, not only the pinched patch. No solver state is changed.
        native=ctx._cloth[args.assembly+'/'+args.cloth_link].get_world_positions()[0]
        bounds=as_numpy(torch.stack([native.amin(0),native.amax(0)]))
        rigid={'task_desk':[[-.5,-.5,-.02],[.5,.5,0.]]}
        for name in surfaces:
            position,rotation=ctx.pose(args.assembly+'/'+name)
            vertices=parts[name]['V']@rotation.T+position
            rigid[name]=[vertices.min(0),vertices.max(0)]
        report=isolate(bounds,rigid,isolation_margins,expected_colliders=isolation_margins)
        report['time_s']=time;isolation_history.append(report)
        return report
    body_name=args.assembly+'/'+args.body_link;lid_name=args.assembly+'/'+args.lid_link;cloth_name=args.assembly+'/'+args.cloth_link
    frames=joint_frames(joint);rows=[];states=[];links=[];times=[];phases={};time=0.;physical_pass=True
    grasp_regions=None;approach_patch=None
    contact_groups={finger:[hand[finger]['link']]for finger in ['thumb','index']}
    rigid_history={name:[] for name in surfaces}
    target_angle=min(2.,float(joint['limits'][1])-.1);current_wrist=wrist_hover.copy();current_state=dict(open_state)
    current_hinge_angle=initial_angle;current_rotation=rotation.copy()
    hand_names=list(fixture.art._articulation_view.body_names)
    def actual_angle():
        _,a=ctx.pose(body_name);_,b=ctx.pose(lid_name);return float(revolute_position(a,b,*frames))
    def record(phase):
        global zero_readback
        points=ctx.nodes(cloth_name).copy();material=metric.check(points);crossing=self_intersections(points,sheet['F'])
        if sheet.get('backend','particle')=='particle':
            from calibrated_scene_reconstruction.assets.cloth_sampling import rigid_coverage
            particle=PhysxSchema.PhysxParticleAPI(ctx._cloth[cloth_name].prim)
            system=PhysxSchema.PhysxParticleSystem(stage.GetPrimAtPath(particle.GetParticleSystemRel().GetTargets()[0]))
            dynamic_sampling=rigid_coverage(points,sheet['F'],{
                'rest_offset':system.GetRestOffsetAttr().Get(),
                'contact_offset':system.GetContactOffsetAttr().Get()})
        else:dynamic_sampling={'pass':True,'applicable':False,'backend':'surface_fem'}
        response=material_verdict(material,accuracy_mode)
        clearance=finite_clearance(points,sheet['F'],sheet['cloth_rest']['points'],values['thickness'],device='cuda:0',
                                   local_bending_review=accuracy_mode=='task_and_semantics' and not args.raw_thickness_diagnostic)
        if args.raw_thickness_diagnostic:
            clearance['diagnostic_raw_screen_only']=True
            clearance['local_bend_certificate_performed']=False
        if not clearance.get('raw_mid_surface_clearance_pass',clearance['pass']) and sheet.get('backend','particle')=='surface_fem':
            from calibrated_scene_reconstruction.sim.cloth_clearance import rest_filter_diagnostic
            chart=sheet['cloth_rest'];natural=chart.get('bend_reference',{}).get('points',chart['points'])
            clearance['native_filter_diagnostic']=rest_filter_diagnostic(
                natural,sheet['F'],clearance,settings['self_collision_filter_distance'])
        samples=triangle_samples(points,sheet['F']);penetration={}
        for name,surface in surfaces.items():
            p,r=ctx.pose(args.assembly+'/'+name);penetration[name]=surface.max_penetration((samples-p)@r)
        tf=as_numpy(fixture.art._articulation_view._physics_view.get_link_transforms())[0].copy()
        contact_range=settings['contact_offset']+fixture.description['collision_contact_offset_m']
        hand_contact=hand_penetration.check(samples,tf,hand_names,solid_tolerance,contact_range_m=contact_range)
        selected_contact=hand_penetration.check(material_samples(points,sheet['F'],selection['material_patch']),
                                                tf,hand_names,solid_tolerance,contact_range_m=contact_range)
        contact_data=rigid_contacts.get_contact_force_data(dt=1/hz)
        separations=contact_separations(as_numpy(contact_data[3]),as_numpy(contact_data[4]),as_numpy(contact_data[5]))
        rigid_depth=float(max(0.,-separations.min())) if len(separations) else 0.
        # Geometric contact witnesses accompany the causal test. Native surface
        # FEM contact impulses are not exposed by the rigid pair-force API.
        pads=[]
        for finger in ['thumb','index']:
            pose=tf[hand_names.index(hand[finger]['link'])]
            pads.append(Rotation.from_quat(pose[3:]).apply(hand[finger]['pad_local'])+pose[:3])
        distances=[float(np.linalg.norm(points-pad,axis=1).min()) for pad in pads]
        bp,br=ctx.pose(body_name)
        distribution=container_state((points-bp)@br,sheet['cloth_rest']['points'],sheet['F'],fix['opening'],{})
        wrist_position,wrist_q=[as_numpy(v) for v in fixture.palm.get_world_pose()]
        wrist_rotation=Rotation.from_quat(wrist_q[[1,2,3,0]]).as_matrix()
        non_thickness_valid=bool(response['pass'] and crossing['pass'] and dynamic_sampling['pass']
                                and max(penetration.values())<=solid_tolerance and hand_contact['pass'] and rigid_depth<=solid_tolerance)
        valid=bool(non_thickness_valid and clearance['pass'])
        row={'time_s':time,'phase':phase,'material':material,'material_response':response,'self_contact':crossing,
             'finite_sheet_clearance':clearance,'native_particle_sampling':dynamic_sampling,
             'non_thickness_physical_pass':non_thickness_valid,
             'solid_penetration_m':penetration,'physical_pass':valid,'lid_angle_rad':actual_angle(),
             'finger_pads':np.asarray(pads).tolist(),'nearest_cloth_to_pad_m':distances,
             'wrist_target':current_wrist.tolist(),'wrist_actual':wrist_position.tolist(),
             'wrist_position_error_m':float(np.linalg.norm(current_wrist-wrist_position)),
             'wrist_orientation_error_rad':float(Rotation.from_matrix(current_rotation@wrist_rotation.T).magnitude()),
             'wrist_rotation_target':current_rotation.tolist(),'finger_joint_targets_rad':dict(current_state),
             'lid_target_rad':current_hinge_angle,
             'hand_contact':hand_contact,'material_distribution':distribution,
             'selected_patch_contact':selected_contact,
             'hand_joint_positions_rad':as_numpy(fixture.art.get_joint_positions()).tolist(),
             'hand_joint_velocities_rad_s':as_numpy(fixture.art.get_joint_velocities()).tolist(),
             'rigid_poses':{name:np.r_[ctx.pose(args.assembly+'/'+name)[0],
                                      ctx.pose(args.assembly+'/'+name)[1].ravel()].tolist()
                            for name in surfaces},
             'body_pose':np.r_[bp,br.ravel()].tolist(),'native_hand_container_penetration_m':rigid_depth,
             'native_hand_container_contact_count':len(separations),
             'hinge_motor_position_rad':as_numpy(lid_motor.get_joint_positions()).tolist(),
             'hinge_measured_effort_Nm':as_numpy(lid_motor.get_measured_joint_efforts()).tolist()}
        row['selected_material_point']=points[selection['material_node']].tolist()
        row['selected_material_node']=selection['material_node']
        if phase=='initial' and support:
            from calibrated_scene_reconstruction.assets.material_support import check as support_state
            sp,sr=ctx.pose(args.assembly+'/'+support['link']);base=parts[support['link']]
            row['initial_support_state']=support_state((points-sp)@sr,sheet['cloth_rest']['points'],sheet['F'],
                base['V'],base['F'],support,thickness_m=reference_values['thickness'],tolerance_m=solid_tolerance)
            non_thickness_valid=bool(non_thickness_valid and row['initial_support_state']['pass'])
            row['non_thickness_physical_pass']=non_thickness_valid
            valid=bool(valid and row['initial_support_state']['pass']);row['physical_pass']=valid
        if grasp_regions is not None:
            from calibrated_scene_reconstruction.tasks.mechanisms.material_grasp import retained_regions
            row['grasp_region_contacts']=retained_regions(points,sheet['F'],grasp_regions,hand_penetration,
                tf,hand_names,solid_tolerance,contact_range,contact_groups)
        if closure_tracker is not None:
            row['closure_tracking_correction_m']=closure_tracker.correction.tolist()
        if prelift is not None and phase in ['lift','hold','release']:
            row['lift_progress']=lift_measurement(prelift,points,selection['material_patch'],wrist_before,wrist_position)
        if args.zero_friction_lift and particle_control is None and phase in ['lift','hold','release']:
            m=native_hand_materials(hand_material_view)
            row['native_finger_friction_zero']=bool(np.all(np.asarray(m['static_friction'])==0)
                                                   and np.all(np.asarray(m['dynamic_friction'])==0))
            if not row['native_finger_friction_zero']:raise ValueError('Native zero-friction setting did not persist during the control')
            native_zero['native_readback_after_step']=True
            zero_readback=True
        if particle_control is not None and particle_control.receipt is not None:
            row['native_particle_friction_zero']=particle_control.observe_after_step()
            zero_readback=True
        native_rigid={name:ctx.pose(args.assembly+'/'+name) for name in rigid_history}
        for name,(position,orientation) in native_rigid.items():
            rigid_history[name].append(np.r_[position,orientation.ravel()])
        rows.append(row);states.append(points);links.append(tf);times.append(time)
        (out/'progress.json').write_text(json.dumps(row,indent=2)+'\n')
        return valid
    dt=1/hz;stride=max(1,hz//args.sample_hz)
    class InvalidState(RuntimeError):pass
    class MeasuredControlLoss(RuntimeError):pass
    def advance(phase,seconds,goal,state_goal,hinge_goal,*,settle_seconds=0.):
        global time,current_wrist,current_state,physical_pass,current_hinge_angle,current_rotation,closure_tracker
        start=current_wrist.copy();q0=dict(current_state);angle0=current_hinge_angle
        turn=orientation_path(current_rotation,rotation)
        ramp_steps=round(seconds*hz)
        steps=ramp_steps+round(settle_seconds*hz);phases[phase]=[time,time+steps*dt]
        tracking=phase=='pinch' and args.track_pinch_material
        if tracking:
            recorded=motion_data.get('pinch_tracking') if replay_bytes is not None else None
            if replay_bytes is not None and recorded is None:
                raise ValueError('Tracked closure requires the exact positive command trace')
            closure_tracker=ClosureTracking(physics_hz=hz,steps=steps,replay=recorded)
            observation_stride=max(1,round(hz/240))
        print('HAND_CLOTH_PHASE',phase,flush=True)
        for step in range(steps):
            fraction=min(1.,(step+1)/ramp_steps);smooth=fraction*fraction*(3-2*fraction)
            current_wrist=start+smooth*(np.asarray(goal)-start)
            current_rotation=turn(smooth)
            current_state={name:q0[name]+smooth*(state_goal[name]-q0[name]) for name in q0}
            if tracking:
                observed=None
                if replay_bytes is None and step%observation_stride==0:
                    from calibrated_scene_reconstruction.tasks.mechanisms.closure_tracking import native_correction
                    transforms=as_numpy(fixture.art._articulation_view._physics_view.get_link_transforms())[0]
                    poses=[transforms[hand_names.index(hand[f]['link'])] for f in ['thumb','index']]
                    pads=np.array([Rotation.from_quat(pose[3:]).apply(hand[f]['pad_local'])+pose[:3]
                                   for f,pose in zip(['thumb','index'],poses)])
                    material=ctx.nodes(cloth_name)[selection['material_node']]
                    observed=native_correction(closure_tracker.correction,material,pads,
                        selection.get('tracking_standoff_world',[0.,0.,selection['fixture_standoff_m']]))
                current_wrist=closure_tracker.step(step,current_wrist,observed)
            current_hinge_angle=float(angle0+smooth*(hinge_goal-angle0))
            lid_motor.apply_action(ArticulationAction(joint_positions=np.array([current_hinge_angle],np.float32)))
            fixture.control(current_wrist,current_rotation,current_state);world.step(render=False);time+=dt
            if particle_control is not None and phase=='hold':
                if not isolation_sample()['pass']:
                    raise InvalidState('Whole cloth is not isolated from all non-hand rigid colliders')
                if args.zero_friction_lift and precontrol is not None:
                    # End a resolved slip before the falling tail can reach a
                    # support. Video-rate sampling alone can miss this short
                    # isolated interval; physical checks still run at the end.
                    patch=selection['material_patch']
                    now=ctx.nodes(cloth_name)
                    wrist=as_numpy(fixture.palm.get_world_pose()[0])
                    slip=float(wrist[2]-control_wrist[2]-np.median(now[patch,2]-precontrol[patch,2]))
                    if slip>.015:
                        physical_pass=record(phase) and physical_pass
                        rows[-1]['post_intervention_downward_slip_m']=slip
                        if not physical_pass:raise InvalidState('Physical validity failed at measured particle-control slip')
                        raise MeasuredControlLoss('Selected material slipped after isolated particle-friction intervention')
            if (step+1)%stride==0 or step==steps-1:
                physical_pass=record(phase) and physical_pass
                if not physical_pass and not (args.diagnose_thickness_residual and rows[-1]['non_thickness_physical_pass']):
                    raise InvalidState('Actual cloth material/self/solid validity failed in '+phase)
                if phase=='lift' and prelift is not None:
                    patch=selection['material_patch']
                    material_lift=float(np.median(states[-1][patch,2]-prelift[patch,2]))
                    wrist_lift=float(np.asarray(rows[-1]['wrist_actual'])[2]-wrist_before[2])
                    if wrist_lift-material_lift>.015:
                        if args.zero_friction_lift and particle_control is None:raise MeasuredControlLoss('Selected material slipped downward relative to the actual fingers')
                        raise InvalidState('Positive friction grasp lost the selected material region')
                if particle_control is not None and phase=='hold' and precontrol is not None:
                    patch=selection['material_patch']
                    slip=float(np.asarray(rows[-1]['wrist_actual'])[2]-control_wrist[2]
                               -np.median(states[-1][patch,2]-precontrol[patch,2]))
                    rows[-1]['post_intervention_downward_slip_m']=slip
                    if slip>.015:
                        if args.zero_friction_lift:raise MeasuredControlLoss('Selected material slipped after isolated particle-friction intervention')
                        raise InvalidState('Positive friction hold lost the selected material region')
    prelift=None;lift=None;failure=None;zero_readback=False;control_termination=None;native_zero=None;prelift_contact=None
    closure_tracker=None;motion_data=None;preclosure_selection=None
    try:
        physical_pass=record('initial')
        if not physical_pass and not (args.diagnose_thickness_residual and rows[-1]['non_thickness_physical_pass']):
            raise InvalidState('Native initialization invalid')
        if args.closed_hold_seconds:
            advance('closing',.5,wrist_hover,open_state,args.closed_angle_rad)
            advance('closed_hold',args.closed_hold_seconds,wrist_hover,open_state,args.closed_angle_rad)
        advance('opening',timing['opening'],wrist_hover,open_state,target_angle)
        if args.opening_hold_seconds:
            advance('lid_hold',args.opening_hold_seconds,wrist_hover,open_state,target_angle)
        if actual_angle()<1.5:raise InvalidState('Finite lid actuation did not leave a clear open aperture')
        # Resolve the actual post-opening cloth, rather than grasping a stale
        # authored point after the sheet has slipped or fallen away.
        bp,br=ctx.pose(body_name)
        planned_poses={name:ctx.pose(args.assembly+'/'+name) for name in surfaces}
        if not set(args.closure_contact_link)<=set(surfaces):raise InvalidState('Expected closure contact names must identify current rigid parts')
        closing_surfaces={n:v for n,v in surfaces.items() if n not in args.closure_contact_link}
        opened_material_samples=ApproachSamples(triangle_samples(ctx.nodes(cloth_name),sheet['F']))
        planned_checks=[];accepted_grasp={}
        standoff_values,placement_resolution=clearance_offsets(values['thickness'])
        protocol['placement_resolution']=placement_resolution
        def feasible_grasp(point,radial_rotation):
            r=br@radial_rotation@np.asarray(hand['R'])
            p=point@br.T+bp+br@radial_rotation@np.asarray(hand['p'])
            pads=(opened_pads@r.T+p-bp)@br
            straddle=opening_assignment(pads,fix['opening'],radius,args.inside_finger)
            axis_check=(None if args.pinch_axis is None and args.pinch_material_axis is None else
                        pinch_axis_alignment(pads,args.pinch_axis if args.pinch_axis is not None else radial_rotation[:,0],up=None if args.pinch_material_axis is not None and args.pinch_material_frame=='spatial' else [0,0,1],minimum=args.minimum_pinch_axis_alignment))
            if not straddle['pass'] or axis_check is not None and not axis_check['pass']:
                planned_checks.append({'point_local':point.tolist(),'rotation_local':radial_rotation.tolist(),
                    'pass':False,'opening_straddle':straddle,'source_axis_alignment':axis_check,'proposal_only':True})
                (out/'grasp_clearance.json').write_text(json.dumps(planned_checks,indent=2)+'\n')
                return False
            # Standoff must preserve predicted closed-pad capture. Expected
            # deformable contact on approach is allowed under native finite
            # drives, unless the caller explicitly requires a free path.
            turn=orientation_path(current_rotation,r);fractions=np.linspace(0,1,33)
            rotations=np.array([turn(t) for t in fractions])
            for standoff in standoff_values:
                goal=p+np.array([0.,0.,standoff])
                capture=planned_pinch.check(point@br.T+bp,goal,r)
                if not capture['pass']:continue
                endpoint=planned_hand.check_closed_pose(goal,r,closing_surfaces,planned_poses,.0005)
                if not endpoint['pass']:
                    planned_checks.append({'point_local':point.tolist(),'rotation_local':radial_rotation.tolist(),
                        'wrist_standoff_m':standoff,'opening_straddle':straddle,'closed_capture':capture,
                        'closed_rigid':endpoint,'pass':False,'proposal_only':True})
                    continue
                descents=[args.closure_descent_m] if args.closure_descent_m is not None else [.002,.004,.006,.008,.010,.015,.020,.025,.030]
                for descent in descents:
                    precontact=goal+np.array([0.,0.,descent])
                    approach=planned_open_hand.check(opened_material_samples,
                        current_wrist+fractions[:,None]*(precontact-current_wrist),rotations,margin=.0005)
                    check={'pass':False,'proposal_only':True,'closed_capture':capture,'open_approach':approach,
                        'deformable_contact_expected':not approach['pass'],'native_contact_validation_required':True}
                    # Open descent may contact material under finite drives.
                    # Rigid clearance and actual capture remain mandatory.
                    if approach['pass'] or (not args.require_contact_free_approach and
                            (args.closure_mode=='open_descent' or args.closure_descent_m is not None)):
                        rigid_approach=planned_hand.check_open_path(
                            current_wrist+fractions[:,None]*(precontact-current_wrist),rotations,surfaces,planned_poses,.0005)
                        check['approach_rigid']=rigid_approach
                        if rigid_approach['pass']:
                            if args.closure_mode=='open_descent':
                                descent_path=planned_hand.check_open_path(np.linspace(precontact,goal,17),
                                    np.tile(r,(17,1,1)),surfaces,planned_poses,.0005)
                                material_path=planned_open_hand.check(opened_material_samples,
                                    np.linspace(precontact,goal,17),np.tile(r,(17,1,1)),margin=.0005)
                                rigid=planned_hand.check(goal,r,closing_surfaces,planned_poses,.0005)
                                check.update({'pass':descent_path['pass'] and rigid['pass'] and
                                    (material_path['pass'] or not args.require_contact_free_approach),
                                    'open_descent_rigid':descent_path,'open_descent_material':material_path,'closing_rigid':rigid})
                            else:
                                alpha=np.linspace(0,1,len(planned_hand.points))
                                closing_positions=precontact+alpha[:,None]*(goal-precontact)
                                rigid=planned_hand.check_poses(closing_positions,np.tile(r,(len(alpha),1,1)),closing_surfaces,planned_poses,.0005)
                                check.update({'pass':rigid['pass'],'closing_rigid':rigid})
                    planned_checks.append({'point_local':point.tolist(),'rotation_local':radial_rotation.tolist(),
                        'wrist_standoff_m':standoff,'closure_descent_m':descent,'opening_straddle':straddle,**check})
                    (out/'grasp_clearance.json').write_text(json.dumps(planned_checks,indent=2)+'\n')
                    if check['pass']:
                        accepted_grasp.update(position=goal,standoff_m=standoff,closure_descent_m=descent)
                        return True
            return False
        if replay_bytes is None:
            try:
                opened_selection=rim_target((ctx.nodes(cloth_name)-bp)@br,sheet['cloth_rest']['points'],sheet['F'],fix['opening'],
                    outer_radius=radius,feasible=feasible_grasp,roll_angles=np.radians([0,15,-15,30,-30,45,-45,60,-60]),
                    align_surface=True,minimum_normal_alignment=args.minimum_normal_alignment,pinch_axis=args.pinch_axis,material_axis=args.pinch_material_axis,material_axis_frame=args.pinch_material_frame)
            except ValueError as error:
                raise InvalidState('Post-opening grasp proposal failed: '+str(error)) from error
            selection=opened_selection;target=selection['point']@br.T+bp
            rotation=br@selection['rotation']@np.asarray(hand['R'])
            wrist_at=accepted_grasp['position'];selection['fixture_standoff_m']=accepted_grasp['standoff_m']
            closure_descent=accepted_grasp['closure_descent_m']
            saved_motion={'version':1,'subject_sha256':protocol['subject_sha256'],
                'candidate_sha256':sha(args.candidate),'hand_sha256':sha(args.hand_plan),
                'selection':{k:v.tolist() if hasattr(v,'tolist') else v for k,v in selection.items()},
                'wrist_at_world':wrist_at.tolist(),'rotation_world':rotation.tolist(),'closure_descent_m':closure_descent,
                'closure_mode':args.closure_mode,'open_descent_seconds':args.open_descent_seconds,'pinch_material_frame':args.pinch_material_frame}
            (out/'motion.json').write_text(json.dumps(saved_motion,indent=2)+'\n')
        else:
            opened_selection,wrist_at,rotation=decode_motion(json.loads(replay_bytes),subject=protocol['subject_sha256'],
                candidate=sha(args.candidate),hand=sha(args.hand_plan),node_count=len(sheet['V']),closure_mode=args.closure_mode,material_axis_frame=args.pinch_material_frame)
            if args.closure_mode=='open_descent' and json.loads(replay_bytes).get('open_descent_seconds')!=args.open_descent_seconds:
                raise InvalidState('Replay open-descent duration differs from the positive motion')
            actual_point=(ctx.nodes(cloth_name)[opened_selection['material_node']]-bp)@br
            if np.linalg.norm(actual_point-opened_selection['point'])>.001:
                raise InvalidState('Replayed material target differs from positive opening state by more than1mm')
            selection=opened_selection
            closure_descent=float(json.loads(replay_bytes).get('closure_descent_m',0.))
            if not 0<=closure_descent<=.03:raise InvalidState('Replayed closure descent out of bounds')
            (out/'motion.json').write_bytes(replay_bytes)
        motion_data=json.loads((out/'motion.json').read_text())
        protocol['executed_motion_sha256']=sha(out/'motion.json')
        motion_hash=hashlib.sha256(json.dumps({k:v for k,v in protocol.items() if k!='zero_friction_lift'},sort_keys=True).encode()).hexdigest()
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        if args.stop_after!='opening':
            advance('approach',timing['approach'],wrist_at+np.array([0.,0.,closure_descent]),open_state,target_angle)
        if args.stop_after not in ['opening','approach']:
            if args.closure_mode=='open_descent':
                advance('open_descent',args.open_descent_seconds,wrist_at,open_state,target_angle)
                actual_tf=as_numpy(fixture.art._articulation_view._physics_view.get_link_transforms())[0]
                actual_points=ctx.nodes(cloth_name);pads=np.asarray(rows[-1]['finger_pads']);bp,br=ctx.pose(body_name)
                if replay_bytes is None:
                    chart=np.asarray(sheet['cloth_rest']['points']);patch=selection['material_patch']
                    patch_radius=float(np.linalg.norm(chart[patch]-chart[selection['material_node']],axis=1).max())
                    def feasible_capture(index,region):
                        if not open_capture.check(actual_points[index],pads,actual_tf,hand_names)['pass']:return False
                        if args.pinch_axis is None and args.pinch_material_axis is None:return True
                        axis=(args.pinch_axis if args.pinch_axis is not None else
                            material_tangent((actual_points-bp)@br,chart,region,args.pinch_material_axis))
                        return pinch_axis_alignment((pads-bp)@br,axis,up=None if args.pinch_material_axis is not None and
                            args.pinch_material_frame=='spatial' else [0,0,1],minimum=args.minimum_pinch_axis_alignment)['pass']
                    try:
                        associated=capture_target(actual_points,chart,sheet['F'],pads.mean(0),patch_radius=patch_radius,
                            search_radius=max(2*patch_radius,float(np.linalg.norm(pads[1]-pads[0]))),feasible=feasible_capture)
                    except ValueError as error:raise InvalidState('Preclosure material association failed: '+str(error)) from error
                    associated['point']=(associated['point']-bp)@br
                    preclosure_selection={**selection,**associated,'selected_at_time_s':time,
                        'tracking_standoff_world':(pads.mean(0)-actual_points[associated['material_node']]).tolist()}
                    motion_data['preclosure_selection']={k:v.tolist() if hasattr(v,'tolist') else v for k,v in preclosure_selection.items()}
                    (out/'motion.json').write_text(json.dumps(motion_data,indent=2)+'\n')
                else:
                    saved=motion_data.get('preclosure_selection')
                    if saved is None:raise InvalidState('Open-descent replay requires the recorded preclosure material association')
                    preclosure_selection,_,_=decode_motion({**motion_data,'selection':saved},subject=protocol['subject_sha256'],
                        candidate=sha(args.candidate),hand=sha(args.hand_plan),node_count=len(sheet['V']),closure_mode=args.closure_mode,material_axis_frame=args.pinch_material_frame)
                    if np.linalg.norm((actual_points[preclosure_selection['material_node']]-bp)@br-preclosure_selection['point'])>.001:
                        raise InvalidState('Replayed preclosure material differs from the positive state by more than1mm')
                selection=preclosure_selection
                protocol['executed_motion_sha256']=sha(out/'motion.json')
                motion_hash=hashlib.sha256(json.dumps({k:v for k,v in protocol.items()if k!='zero_friction_lift'},sort_keys=True).encode()).hexdigest()
                (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
                capture=open_capture.check(actual_points[selection['material_node']],pads,actual_tf,hand_names)
                actual_joints=as_numpy(fixture.art.get_joint_positions())
                joint_error=max(abs(actual_joints[list(fixture.art.dof_names).index(name)]-value) for name,value in open_state.items())
                readiness={'pass':bool(capture['pass'] and joint_error<.05),
                    'actual_open_pad_capture':capture,'maximum_open_joint_error_rad':float(joint_error),
                    'scope':'Actual open hand at descent end; current material must be inside the open jaw corridor. Not a grasp or friction proof.'}
                (out/'preclosure_readiness.json').write_text(json.dumps(readiness,indent=2)+'\n')
                if not readiness['pass']:raise InvalidState('Open descent did not establish a ready open-hand capture pose; closure was not commanded')
            if args.pinch_axis is not None or args.pinch_material_axis is not None:
                bp,br=ctx.pose(body_name)
                expected_axis=(args.pinch_axis if args.pinch_axis is not None else
                    material_tangent((ctx.nodes(cloth_name)-bp)@br,sheet['cloth_rest']['points'],selection['material_patch'],args.pinch_material_axis))
                measured=pinch_axis_alignment((np.asarray(rows[-1]['finger_pads'])-bp)@br,expected_axis,up=None if args.pinch_material_axis is not None and args.pinch_material_frame=='spatial' else [0,0,1],minimum=args.minimum_pinch_axis_alignment)
                (out/'preclosure_axis_check.json').write_text(json.dumps(measured,indent=2)+'\n')
                if not measured['pass']:raise InvalidState('Actual finger opposition misses the source-required pinch axis before closure')
            advance('pinch',timing['pinch'],wrist_at,closed_state,target_angle,settle_seconds=args.pinch_settle_seconds)
            if args.pinch_axis is not None or args.pinch_material_axis is not None:
                bp,br=ctx.pose(body_name)
                expected_axis=(args.pinch_axis if args.pinch_axis is not None else
                    material_tangent((ctx.nodes(cloth_name)-bp)@br,sheet['cloth_rest']['points'],selection['material_patch'],args.pinch_material_axis))
                measured=pinch_axis_alignment((np.asarray(rows[-1]['finger_pads'])-bp)@br,expected_axis,up=None if args.pinch_material_axis is not None and args.pinch_material_frame=='spatial' else [0,0,1],minimum=args.minimum_pinch_axis_alignment)
                (out/'pinch_axis_check.json').write_text(json.dumps(measured,indent=2)+'\n')
                if not measured['pass']:raise InvalidState('Actual closed finger opposition misses the source-required material direction')
            if args.track_pinch_material:wrist_at=current_wrist.copy()
            prelift=ctx.nodes(cloth_name).copy();wrist_before=as_numpy(fixture.palm.get_world_pose()[0]).copy()
            approach_patch=np.asarray(selection['material_patch']).copy()
            if args.material_contact_mode=='same_component':
                from calibrated_scene_reconstruction.tasks.mechanisms.material_grasp import contact_regions
                native=as_numpy(fixture.art._articulation_view._physics_view.get_link_transforms())[0]
                radius=float(np.linalg.norm(np.asarray(sheet['cloth_rest']['points'])[approach_patch]-
                    np.asarray(sheet['cloth_rest']['points'])[selection['material_node']],axis=1).max())
                grasp_regions=contact_regions(prelift,sheet['cloth_rest']['points'],sheet['F'],rows[-1]['finger_pads'],
                    native,hand_names,hand_penetration.rows,contact_groups,
                    contact_range_m=settings['contact_offset']+fixture.description['collision_contact_offset_m'],
                    patch_radius=radius/2,boundary_margin=radius)
                if not grasp_regions['pass']:raise InvalidState('Actual opposing interior material regions were not captured')
                if replay_bytes is None:
                    motion_data['grasp_regions']=grasp_regions
                    (out/'motion.json').write_text(json.dumps(motion_data,indent=2)+'\n')
                else:
                    from calibrated_scene_reconstruction.tasks.mechanisms.material_grasp import retained_regions
                    saved_regions=motion_data.get('grasp_regions')
                    if not isinstance(saved_regions,dict) or not retained_regions(prelift,sheet['F'],saved_regions,
                            hand_penetration,native,hand_names,solid_tolerance,
                            settings['contact_offset']+fixture.description['collision_contact_offset_m'],contact_groups)['pass']:
                        raise InvalidState('Matched replay did not capture the recorded source material regions')
                    grasp_regions=saved_regions
                selection={**selection,'material_patch':np.asarray(grasp_regions['material_patch'],int)}
                # Record the transition with the resolved source-indexed regions
                # before the friction intervention; no native step or write.
                physical_pass=record('pinch') and physical_pass
                protocol['executed_motion_sha256']=sha(out/'motion.json')
                motion_hash=hashlib.sha256(json.dumps({k:v for k,v in protocol.items()if k!='zero_friction_lift'},sort_keys=True).encode()).hexdigest()
                (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
            prelift_contact=pinch_contact(prelift,selection['material_patch'],rows[-1]['finger_pads'],
                                          rows[-1]['selected_patch_contact']['near_surface_samples'])
            if not prelift_contact['pass']:
                raise InvalidState('Finger closure did not establish two-distal-finger contact with the selected material')
        if args.stop_after in ['lift','release']:
            if args.zero_friction_lift and particle_control is None:
                hand_mat=stage.GetPrimAtPath(fixture.path+'/FingerPhysics');hm=UsdPhysics.MaterialAPI(hand_mat)
                native_zero=author_friction(hand_mat,0.)
                cloth_prim=stage.GetPrimAtPath(str(root.GetPath())+'/'+args.cloth_link+'/surface')
                cm=UsdShade.MaterialBindingAPI(cloth_prim).ComputeBoundMaterial('physics')[0].GetPrim()
                friction=cm.GetAttribute('omniphysics:dynamicFriction')
                mode=PhysxSchema.PhysxMaterialAPI(hand_mat).GetFrictionCombineModeAttr().Get()
                if not (hm.GetDynamicFrictionAttr().Get()==0. and mode=='multiply'
                        and np.isclose(friction.Get(),values['dynamic_friction'])):
                    raise ValueError('Zero-friction authoring changed the contact combination or cloth material')
                # Do not call the tensor material setter: in Isaac5.1 it loses
                # the effective multiply policy for surface contacts. Check
                # actual hand materials after native steps in record().
            from calibrated_scene_reconstruction.tasks.mechanisms.hand_motion import translation_seconds
            advance('lift',translation_seconds(hand['lift_height_m'],hand['lift_speed_m_s']),
                wrist_at+[0,0,hand['lift_height_m']],closed_state,target_angle)
            if particle_control is not None:
                if not isolation_sample()['pass']:
                    raise InvalidState('Cannot isolate finger friction while the full sheet is near another collider')
                precontrol=ctx.nodes(cloth_name).copy();control_wrist=as_numpy(fixture.palm.get_world_pose()[0]).copy()
                if args.zero_friction_lift:native_zero=particle_control.switch(time)
            advance('hold',hand['hold_s'],current_wrist,closed_state,target_angle)
            lift=lift_measurement(prelift,ctx.nodes(cloth_name),selection['material_patch'],wrist_before,as_numpy(fixture.palm.get_world_pose()[0]))
            if args.stop_after=='release':advance('release',.75,current_wrist,open_state,target_angle)
    except MeasuredControlLoss as error:
        lift=lift_measurement(prelift,ctx.nodes(cloth_name),selection['material_patch'],wrist_before,as_numpy(fixture.palm.get_world_pose()[0]))
        control_termination={'reason':'measured_slip','description':str(error),'time_s':time,
                             'relative_vertical_slip_m':lift['vertical_slip_m']}
    except InvalidState as error:failure=str(error)
    if closure_tracker is not None:
        if replay_bytes is None:
            motion_data['pinch_tracking']=closure_tracker.report()
            (out/'motion.json').write_text(json.dumps(motion_data,indent=2)+'\n')
        protocol['executed_motion_sha256']=sha(out/'motion.json')
        motion_hash=hashlib.sha256(json.dumps({k:v for k,v in protocol.items() if k!='zero_friction_lift'},sort_keys=True).encode()).hexdigest()
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    errors.check()
    np.savez_compressed(out/'states.npz',points=states,hand_links=links,times=times,faces=sheet['F'],
        rest=np.asarray(sheet['cloth_rest']['points']),material_patch=selection['material_patch'],
        approach_material_patch=selection['material_patch'] if approach_patch is None else approach_patch,
        precontrol_points=np.empty((0,3)) if precontrol is None else precontrol,
        prelift_points=np.empty((0,3)) if prelift is None else prelift,
        **{name+'_poses':np.asarray(poses) for name,poses in rigid_history.items()})
    if particle_control is not None:
        (out/'isolation.json').write_text(json.dumps(isolation_history,indent=2)+'\n')
    result={'pass':False,'candidate_assay':True,'task_approved':False,'native_states_read':True,
        'material_accuracy_mode':accuracy_mode,
        'quantitative_material_pass':all(row['material']['pass'] for row in rows),
        'physical_plausibility_review_required':any(not row['material']['pass'] for row in rows),
        'diagnostic_hypothesis':diagnostic_hypothesis,
        'experimental_native_viscosity':native_viscosity_report,
        'backend':{'name':'isaacsim','version':importlib.metadata.version('isaacsim')},
        'failure':failure,'complete':failure is None,'physical_pass':physical_pass,'lift':lift,
        'prelift_contact':prelift_contact,
        'grasp_regions':grasp_regions,
        'friction_zero_readback':zero_readback,'motion_sha256':motion_hash,'phases':phases,'rows':rows,
        'native_finger_material_initial':initial_native_material,'native_zero_friction':native_zero,
        'particle_control':None if particle_control is None else {
            'qualification':particle_control.proof,'wrist_at_intervention':None if control_wrist is None else control_wrist.tolist(),
            'isolation_sha256':sha(out/'isolation.json'),'isolation_steps':len(isolation_history),
            'isolation_pass':bool(isolation_history and all(row['pass'] for row in isolation_history))},
        'control_termination':control_termination,'completed_physics_time_s':time,
        'hand_body_names':hand_names,'hand':fixture.report(),'native_physics_errors':errors.report(),
        'postopening_selection':None if 'opened_selection' not in globals() else {k:v.tolist() if hasattr(v,'tolist') else v for k,v in opened_selection.items()},
        'preclosure_selection':None if preclosure_selection is None else {k:v.tolist() if hasattr(v,'tolist') else v for k,v in preclosure_selection.items()},
        'evidence_files':[str(q.relative_to(s.dir)) for q in [out/'states.npz',out/'protocol.json',out/'initial.usda']],
        'scope':'Actual native hand/object candidate diagnostics; matched friction control and full criterion validation required before acceptance.'}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('HAND_CLOTH_RESULT',json.dumps({k:result[k] for k in ['complete','physical_pass','failure','lift']}),flush=True)
except Exception as error:
    if 'states' in globals():np.savez_compressed(out/'failed_states.npz',points=states,hand_links=links,times=times)
    (out/'error.json').write_text(json.dumps({'pass':False,'error':str(error),'traceback':traceback.format_exc()},indent=2)+'\n');raise
finally:
    errors.close();app.close()
