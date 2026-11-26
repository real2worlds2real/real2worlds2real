"""Unforced native candidate probe; no source-fit, scale, or task approval."""
import argparse
from pathlib import Path
import json
import shutil
import traceback
import hashlib
import math
import time as walltime

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('session');parser.add_argument('candidate');parser.add_argument('--output',required=True)
parser.add_argument('--assembly',default='wipes_dispenser');parser.add_argument('--seconds',type=float,default=.5)
parser.add_argument('--physics-hz',type=int,choices=[240,480,960,1920,3840,7680,15360,30720],default=960,
                    help='Explicit native convergence experiment; a higher clock alone does not qualify a production profile')
parser.add_argument('--regularize-iterations',type=int,default=0,help='Source geometry preparation only; never task approval')
parser.add_argument('--self-separate',action='store_true',help='Maintain physical point separation during source preparation; actual triangle/native checks remain required')
parser.add_argument('--physics-plan',help='Explicit candidate material hypotheses; diagnostic only')
parser.add_argument('--mechanism-plan',help='Matching candidate native representation; diagnostic only')
parser.add_argument('--understanding-plan',help='Explicit candidate semantic/physical inventory; defaults to current session inventory')
parser.add_argument('--experimental-bending',help='Explicit diagnostic internal bending history; never task/export approval')
parser.add_argument('--experimental-particle-shell',action='store_true',help='Diagnostic continuum shell forces over native particle contacts; not a delivery backend')
parser.add_argument('--experimental-native-viscosity-proof',help='Passing same-mesh synthetic transport control for a diagnostic internal damping step; never task approval')
parser.add_argument('--bending-viscosity-time-s',type=float)
parser.add_argument('--membrane-viscosity-time-s',type=float)
parser.add_argument('--cook-material-rest',action='store_true',help='Diagnostic particle shell: cook intrinsic material geometry, author initial pose once before t0')
parser.add_argument('--linear-solver',choices=['cg','sparse_direct','cuda_direct'],default='sparse_direct')
parser.add_argument('--nonlinear-iterations',type=int,default=12)
parser.add_argument('--defer-experimental-startup',action='store_true',help='Diagnostic monolithic shell only: isolate native reset warmup until the elastic adapter is installed, then restore all contacts/gravity before recording')
parser.add_argument('--case',choices=['low','nominal','high'],default='nominal')
parser.add_argument('--no-contact',action='store_true',help='Diagnostic negative control only')
parser.add_argument('--no-rigid-contact',action='store_true',help='Keep actual sheet self-contact; remove rigid contacts for diagnosis only')
parser.add_argument('--no-gravity',action='store_true',help='Diagnostic isolation only')
parser.add_argument('--position-iterations',type=int,default=None,
                    help='Explicit native iteration override; otherwise retain the candidate profile or128')
parser.add_argument('--solver',choices=['PGS','TGS'],default='PGS')
parser.add_argument('--max-depenetration-velocity',type=float,default=.02,
                    help='Explicit numerical contact-correction speed; not a material estimate')
parser.add_argument('--pair-updates',type=int,default=16)
parser.add_argument('--collision-subiterations',type=int,default=2)
parser.add_argument('--self-filter-distance',type=float,default=None,help='Diagnostic native neighborhood-filter hypothesis; all self-crossing checks remain active')
parser.add_argument('--sdf-resolution',type=int,help='Explicit diagnostic rigid-contact resolution; effective native bounds are still checked')
parser.add_argument('--max-linear-velocity',type=float,help='Explicit native speed profile; -1 requests the installed surface automatic per-vertex safeguard')
parser.add_argument('--bootstrap-hz',type=int,help='Explicit diagnostic positive warmup timestep before an external constitutive adapter can attach; no state overwrites')
parser.add_argument('--topology-filter-rings',type=int,choices=[1,2,3],help='Diagnostic explicit local face-neighbour filter; native capability must be independently verified')
parser.add_argument('--diagnostic-actions',action='store_true',help='Record finite-force open/extract/close preview; NOT a task pass and failed quality stays recorded')
parser.add_argument('--preparation-plan',help='Current-capture finite-force cloth packing/pose construction plan; not source or task approval')
parser.add_argument('--preparation-isolation',action='store_true',
                    help='Explicit counterfactual preparation prefix with a gravity/contact ablation; never a released-state proposal')
parser.add_argument('--record-fps',type=int,default=20)
parser.add_argument('--retention-plan',help='Explicit material-distribution and released observation requirements; separate from physical validity and task approval')
parser.add_argument('--aggregate-pair-capacity',type=int,help='Explicit diagnostic native GPU aggregate-pair budget,1024..16777216')
parser.add_argument('--surface-contact-capacity',type=int,help='Explicit diagnostic native deformable-surface contact budget,1024..16777216')
parser.add_argument('--grasp-width',type=float,help='Bounded material-local extraction patch width in metres; default min(10mm,8%% of sheet width)')
parser.add_argument('--grasp-fraction',type=float,default=1.,help='Material edge patch centre in0..1; a fixture hypothesis, not a detected source grasp')
parser.add_argument('--legacy-full-edge-grasp',action='store_true',help='Diagnostic legacy full material-width strip; can pull several folded plies together')
parser.add_argument('--continue-invalid-diagnostic',action='store_true',
                    help='Keep recording after physical validity fails; diagnostic only, never approval')
parser.add_argument('--surface-link',default='wipe',help='Explicit physical surface link; candidate/plan ownership is validated before native authoring')
args=parser.parse_args()
surface_link=args.surface_link
if not surface_link.isidentifier():parser.error('Surface link must be a local physical-link name')
if surface_link!='wipe' and (args.diagnostic_actions or args.retention_plan):
    parser.error('Explicit task/retention fixture required for a different surface')
for name in ['candidate','physics_plan','mechanism_plan','understanding_plan','preparation_plan','retention_plan',
             'experimental_bending','experimental_native_viscosity_proof']:
    value=getattr(args,name)
    if value and not Path(value).is_file():parser.error(name.replace('_','-')+' file missing: '+value)
if Path(args.output).exists():parser.error('Output already exists; preserve the old trial and choose a new output')
if args.preparation_isolation and (not args.preparation_plan or not (args.no_contact or args.no_rigid_contact or args.no_gravity)):
    parser.error('Preparation isolation requires a plan and an explicit gravity/contact ablation')
if args.experimental_native_viscosity_proof and (args.experimental_particle_shell or args.experimental_bending
        or args.diagnostic_actions or args.no_gravity or args.no_contact or args.no_rigid_contact):
    parser.error('Native internal-viscosity diagnostic requires unchanged native elasticity and real contacts')
viscosity=[args.bending_viscosity_time_s,args.membrane_viscosity_time_s]
if any(v is not None for v in viscosity) and (not args.experimental_particle_shell
        or any(v is None or not math.isfinite(v) or not 0<=v<=1 for v in viscosity)):
    parser.error('Both explicit diagnostic viscosity times in [0,1]s require the experimental particle shell')
if args.cook_material_rest and not args.experimental_particle_shell:
    parser.error('Material contact cooking requires the explicit diagnostic particle shell')
if not 2<=args.nonlinear_iterations<=50:parser.error('Nonlinear iteration budget in2..50 required')
if args.experimental_particle_shell and (args.no_gravity or args.no_contact or args.no_rigid_contact):
    parser.error('Particle shell startup restores full gravity/contact; use separate validated component controls for ablations')
if not math.isfinite(args.seconds):
    parser.error('Finite native duration required')
if round(args.seconds*args.physics_hz)<1 or not 1<=args.record_fps<=args.physics_hz:
    parser.error('At least one actual native step and a valid record frequency required')
if args.aggregate_pair_capacity is not None and not 1024<=args.aggregate_pair_capacity<=16777216:
    parser.error('Native aggregate pair capacity must be1024..16777216')
if args.surface_contact_capacity is not None and not 1024<=args.surface_contact_capacity<=16777216:
    parser.error('Native surface contact capacity must be1024..16777216')

target_preflight=None
if args.preparation_plan:
    import numpy as np
    from calibrated_scene_reconstruction.tasks.mechanisms.preparation_geometry import initial_targets
    candidate_meta=json.loads(Path(args.candidate).read_text())
    candidate_mesh=Path(candidate_meta['path'])/'mesh.npz'
    preparation_preflight=json.loads(Path(args.preparation_plan).read_text())
    if preparation_preflight.get('candidate_mesh_sha256')!=hashlib.sha256(candidate_mesh.read_bytes()).hexdigest():
        parser.error('Material preparation belongs to a different source mesh')
    try:
        releases=[float(g['active_s'][1]) for g in preparation_preflight.get('grips',[])]
        if preparation_preflight.get('hinge'):
            h=preparation_preflight['hinge']
            releases.append(float(h.get('active_s',[0,h['keyframes'][-1][0]])[1]))
        if not releases or not np.isfinite(releases).all() or min(releases)<=0:
            raise ValueError('Finite positive preparation release times required')
    except (KeyError,IndexError,TypeError,ValueError) as error:
        parser.error('Invalid preparation release schedule: '+str(error))
    if not args.preparation_isolation and args.seconds<max(releases)+.5:
        parser.error('Record at least0.5s with every temporary cloth and hinge force released after preparation')
    with np.load(candidate_mesh) as data:
        rest_meta=json.loads(str(data['cloth_rest_json']))
        source_parts=[]
        for part in json.loads(str(data['links_json'])):
            a,n=part['vertex_start'],part['vertex_count']
            source_parts.append({**part,'V':data['V'][a:a+n],
                                 'cloth_rest':rest_meta.get(part['name'])})
        target_preflight=initial_targets(preparation_preflight,source_parts)
    if target_preflight.get('applicable') and not target_preflight['pass']:
        parser.error('Source preparation targets use a different model frame: '+json.dumps(target_preflight))

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402
app=SimulationApp({'headless':True})

from copy import deepcopy  # noqa: E402
from types import SimpleNamespace  # noqa: E402
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from pxr import PhysxSchema,UsdPhysics,Gf  # noqa: E402
from calibrated_scene_reconstruction.scene.session import Session  # noqa: E402
from calibrated_scene_reconstruction.assets.metric_mesh import asset  # noqa: E402
from calibrated_scene_reconstruction.assets.physical_priors import sample  # noqa: E402
from calibrated_scene_reconstruction.sim.mechanism_stage import add_links  # noqa: E402
from calibrated_scene_reconstruction.sim.physical_priors import apply as apply_priors  # noqa: E402
from calibrated_scene_reconstruction.sim.physics import physics_material  # noqa: E402
from calibrated_scene_reconstruction.sim.usd import add_box_collider  # noqa: E402
from calibrated_scene_reconstruction.sim.simulation import apply as apply_simulation,audit as audit_simulation  # noqa: E402
from calibrated_scene_reconstruction.sim.cloth_contact import self_intersections  # noqa: E402
from calibrated_scene_reconstruction.sim.surface_distance import SignedSurface,triangle_samples  # noqa: E402
from calibrated_scene_reconstruction.tasks.rigid.task_checks import Ctx,as_numpy  # noqa: E402
from calibrated_scene_reconstruction.tasks.mechanisms.cloth import MaterialMetric  # noqa: E402

out=Path(args.output);out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,out/'source.py')
if target_preflight is not None:
    (out/'initial_target_preflight.json').write_text(json.dumps(target_preflight,indent=2)+'\n')
from calibrated_scene_reconstruction.sim.physics_errors import PhysicsErrors  # noqa: E402
physics_errors=PhysicsErrors(out/'native_physics_errors.json')
# Native progress and PhysicsErrors are the probe diagnostics. Avoid a watchdog
# walking all Python stacks while Kit is stepping: two interrupted native runs
# ended during that walk, including an incomplete/corrupt frame dump. This does
# not alter simulation settings or any physical acceptance check.
try:
    s=Session(args.session);candidate_bytes=Path(args.candidate).read_bytes()
    from calibrated_scene_reconstruction.assets import response_policy
    task_configuration=s.config('task.json',{})
    material_accuracy=response_policy.mode(task_configuration.get('flexible_accuracy'))
    (out/'task_configuration.json').write_text(json.dumps(task_configuration,indent=2)+'\n')
    (out/'candidate_input.json').write_bytes(candidate_bytes)
    candidate_sha256=hashlib.sha256(candidate_bytes).hexdigest()
    info=json.loads(candidate_bytes);g=s.scene_graph;obj=args.assembly
    mesh_input=(Path(info['path'])/'mesh.npz').resolve()
    if (not Path(args.candidate).resolve().is_relative_to(s.dir.resolve())
            or not mesh_input.is_relative_to(s.dir.resolve())):
        raise ValueError('Native candidate and mesh must belong to this current capture')
    if bool(args.physics_plan)!=bool(args.mechanism_plan):raise ValueError('Candidate representation and material hypotheses must be supplied together')
    dec=json.loads(Path(args.mechanism_plan).read_text()) if args.mechanism_plan else s.config('agent/mechanisms.json')
    plan=json.loads(Path(args.physics_plan).read_text()) if args.physics_plan else s.config('agent/physics_priors.json')
    selected_surface=dec.get('objects',{}).get(obj,{}).get('links',{}).get(surface_link,{})
    if selected_surface.get('kind') not in {'cloth','deformable'}:
        raise ValueError('Selected link is not a declared physical surface: '+obj+'/'+surface_link)
    if not any(p.get('name')==surface_link and p.get('kind') in {'cloth','deformable'} for p in info['report'].get('parts',[])):
        raise ValueError('Selected physical surface is absent from the actual candidate')
    understanding_path=Path(args.understanding_plan) if args.understanding_plan else s.config_dir/'agent/object_understanding.json'
    understanding_bytes=understanding_path.read_bytes()
    understanding_plan=json.loads(understanding_bytes)
    understood=understanding_plan.get('objects',{}).get(obj,{})
    (out/'object_understanding.json').write_bytes(understanding_bytes)
    from calibrated_scene_reconstruction.scene.objects.object_understanding import validate_build
    validate_build(s,obj,info['report'],understanding_plan=understanding_plan)
    if args.physics_plan:
        shutil.copy2(args.physics_plan,out/'physics_priors.json');shutil.copy2(args.mechanism_plan,out/'mechanisms.json')
    else:
        (out/'physics_priors.json').write_text(json.dumps(plan,indent=2)+'\n')
        (out/'mechanisms.json').write_text(json.dumps(dec,indent=2)+'\n')
    def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    probe_input={'version':1,'input':str(Path(args.candidate).resolve()),'candidate_sha256':candidate_sha256,
                 'mesh':str(mesh_input),'mesh_sha256':sha(mesh_input),
                 'arguments':vars(args),'source_sha256':sha(out/'source.py'),
                 'physics_priors_sha256':sha(out/'physics_priors.json'),
                 'mechanisms_sha256':sha(out/'mechanisms.json')}
    probe_input['understanding_sha256']=sha(out/'object_understanding.json')
    probe_input['task_configuration_sha256']=sha(out/'task_configuration.json')
    probe_input['validity_sources']={str(Path(p).resolve()):sha(p) for p in
        ['src/calibrated_scene_reconstruction/sim/cloth_contact.py','src/calibrated_scene_reconstruction/sim/cloth_clearance.py',
         'src/calibrated_scene_reconstruction/sim/cloth_shell_clearance.py','src/calibrated_scene_reconstruction/assets/cloth_metric.py',
         'src/calibrated_scene_reconstruction/assets/cloth_sampling.py','src/calibrated_scene_reconstruction/assets/response_policy.py','src/calibrated_scene_reconstruction/assets/spring_energy.py',
         'src/calibrated_scene_reconstruction/assets/cloth_failure_location.py','src/calibrated_scene_reconstruction/assets/cloth_timestep.py',
         'src/calibrated_scene_reconstruction/tasks/mechanisms/passive_extent.py']}
    probe_input['validity_sources'].update({str(Path(p).resolve()):sha(p) for p in
        ['src/calibrated_scene_reconstruction/sim/mechanism_stage.py','src/calibrated_scene_reconstruction/sim/contact_carrier.py',
         'src/calibrated_scene_reconstruction/sim/rigid_partition.py','src/calibrated_scene_reconstruction/assets/rigid_partition.py',
         'src/calibrated_scene_reconstruction/assets/inertia_metric.py','src/calibrated_scene_reconstruction/sim/particle_attachment.py',
         'src/calibrated_scene_reconstruction/sim/source_moments.py','src/calibrated_scene_reconstruction/sim/physical_priors.py',
         'src/calibrated_scene_reconstruction/assets/revolved_source_cells.py',
         'src/calibrated_scene_reconstruction/assets/cloth_contact_tolerance.py',
         'src/calibrated_scene_reconstruction/assets/cloth_rest.py','src/calibrated_scene_reconstruction/assets/articulation_initial_state.py',
         'src/calibrated_scene_reconstruction/scene/objects/particle_contact_binding.py',
         'src/calibrated_scene_reconstruction/tasks/rigid/task_checks.py','src/calibrated_scene_reconstruction/sim/contact_ablation.py',
         'src/calibrated_scene_reconstruction/sim/soft_surface.py','src/calibrated_scene_reconstruction/assets/tetra_boundary.py',
         'src/calibrated_scene_reconstruction/sim/surface_distance.py','src/calibrated_scene_reconstruction/assets/volume_contact.py',
         'src/calibrated_scene_reconstruction/sim/volume_body.py','src/calibrated_scene_reconstruction/sim/volume_initial_state.py',
         'src/calibrated_scene_reconstruction/assets/volume_initial_state.py','src/calibrated_scene_reconstruction/sim/deformable_filter_pose.py',
         'src/calibrated_scene_reconstruction/assets/deformable_filter_pose.py']})
    from calibrated_scene_reconstruction.tasks.mechanisms import passive_response
    passive_producer=Path(passive_response.__file__).resolve()
    shutil.copy2(passive_producer,out/'passive_response.py')
    probe_input['passive_response_source']={'path':str(passive_producer),'sha256':sha(passive_producer)}
    response_producer=Path(response_policy.__file__).resolve()
    shutil.copy2(response_producer,out/'response_policy.py')
    probe_input['response_policy_source']={'path':str(response_producer),'sha256':sha(response_producer)}
    preparation_plan=None
    retention_plan=None
    if args.retention_plan:
        retention_path=Path(args.retention_plan).resolve()
        if not retention_path.is_relative_to(s.dir.resolve()):
            raise ValueError('Retention requirements must belong to this capture')
        retention_plan=json.loads(retention_path.read_text())
        shutil.copy2(retention_path,out/'retention_input.json')
        from calibrated_scene_reconstruction.assets import cloth_retention
        retention_source=Path(cloth_retention.__file__).resolve()
        shutil.copy2(retention_source,out/'cloth_retention.py')
        probe_input['retention_sources']={str(retention_path):sha(retention_path),
            str(retention_source):sha(retention_source)}
    if args.preparation_plan:
        path=Path(args.preparation_plan).resolve()
        if (not path.is_relative_to(s.dir.resolve()) or args.diagnostic_actions or args.experimental_bending
                or ((args.no_gravity or args.no_contact or args.no_rigid_contact) and not args.preparation_isolation)):
            raise ValueError('Native preparation needs current-capture plan, real gravity/contact and separate fixture mode')
        preparation_plan=json.loads(path.read_text())
        if preparation_plan.get('candidate_mesh_sha256')!=probe_input['mesh_sha256']:
            raise ValueError('Material preparation belongs to a different source mesh')
        shutil.copy2(path,out/'preparation_input.json')
        from calibrated_scene_reconstruction.tasks.mechanisms import preparation,force_fixture,preparation_geometry,joint_drive_fixture,surface_contact_state
        from calibrated_scene_reconstruction.tasks.rigid import task_checks
        binding={}
        for module in [preparation,force_fixture,preparation_geometry,joint_drive_fixture,surface_contact_state,task_checks]:
            name='preparation_'+Path(module.__file__).name;shutil.copy2(module.__file__,out/name);binding[name]=sha(out/name)
        binding['preparation_input.json']=sha(out/'preparation_input.json')
        probe_input['preparation_files_sha256']=binding
    if args.experimental_bending:
        descriptor=Path(args.experimental_bending).resolve()
        if not descriptor.is_relative_to(s.dir.resolve()):raise ValueError('Current-capture bending descriptor required')
        shutil.copy2(descriptor,out/'experimental_bending_input.json')
        probe_input['experimental_bending_sha256']=sha(descriptor)
        history=json.loads(descriptor.read_text())
        natural=(s.dir/history['natural_file']).resolve()
        if not natural.is_relative_to(s.dir.resolve()) or sha(natural)!=history['natural_sha256']:
            raise ValueError('Current-capture natural curvature changed')
        shutil.copy2(natural,out/'experimental_bending_natural.npz')
        from calibrated_scene_reconstruction.sim import experimental_bending,discrete_bending,cohesive_interface,discrete_membrane
        from calibrated_scene_reconstruction.assets import cloth_projection,cloth_curvature
        prototype={}
        for module in [experimental_bending,discrete_bending,cloth_projection,cloth_curvature,cohesive_interface,discrete_membrane]:
            name='experimental_'+Path(module.__file__).name
            shutil.copy2(module.__file__,out/name)
            prototype[name]=sha(out/name)
        if args.defer_experimental_startup:
            from calibrated_scene_reconstruction.sim import experimental_startup
            name='experimental_'+Path(experimental_startup.__file__).name
            shutil.copy2(experimental_startup.__file__,out/name);prototype[name]=sha(out/name)
        prototype['experimental_bending_input.json']=sha(descriptor)
        prototype['experimental_bending_natural.npz']=sha(natural)
        if history.get('cohesion'):
            interface=(s.dir/history['cohesion']['file']).resolve()
            if not interface.is_relative_to(s.dir.resolve()) or sha(interface)!=history['cohesion']['sha256']:
                raise ValueError('Experimental initial cohesive interfaces changed')
            shutil.copy2(interface,out/'experimental_cohesive_interfaces.npz')
            prototype['experimental_cohesive_interfaces.npz']=sha(interface)
        probe_input['experimental_files_sha256']=prototype
    if args.defer_experimental_startup and (not args.experimental_bending
            or not history.get('monolithic_membrane',False)
            or args.no_contact or args.no_rigid_contact or args.no_gravity):
        raise ValueError('Deferred startup requires the explicit monolithic diagnostic and full gravity/contact during recording')
    (out/'probe_input.json').write_text(json.dumps(probe_input,indent=2)+'\n')
    # Keep the original file above for provenance; all subsequent property
    # reads need the same expanded material families used by native sampling.
    from calibrated_scene_reconstruction.assets.physical_priors import validate as validate_priors
    plan=validate_priors(plan,dec)
    estimate=sample(plan,args.case);obj=args.assembly;spec=g['dynamic_objects'][obj]
    from calibrated_scene_reconstruction.scene.objects.candidate_response import probe_contact_length
    with np.load(mesh_input) as authored_mesh:
        radius,length_basis=probe_contact_length(authored_mesh,info['report'])
    (out/'numerical_contact_length.json').write_text(json.dumps(length_basis,indent=2)+'\n')
    material=estimate['bodies'][obj+('/'+surface_link)]['parameters']
    surface_fem=dec['objects'][obj]['links'][surface_link].get('backend','particle')=='surface_fem'
    if args.experimental_particle_shell:
        if args.experimental_bending or args.defer_experimental_startup:
            raise ValueError('Select one explicit experimental constitutive path')
        surface_fem=False
        cloth={'areal_density':material['areal_density'],'stretch_stiffness':1e-12,'shear_stiffness':1e-12,
            'bend_stiffness':1e-12,'damping':1e-12,'friction':material['dynamic_friction'],
            'solver_position_iterations':args.position_iterations or 128,'contact_offset':.001,'rest_offset':.0004,'solid_rest_offset':.00042}
    elif surface_fem:
        cloth={**material,'friction':material['dynamic_friction'],'contact_offset':max(radius*.008,material['thickness']*.75),
            'rest_offset':material['thickness']/2,'self_collision_filter_distance':args.self_filter_distance or material['thickness'],
            'solver_position_iterations':args.position_iterations or 128,'collision_pair_updates':args.pair_updates,
            'collision_iteration_multiplier':args.collision_subiterations}
    else:
        cloth={'areal_density':material['areal_density'],'stretch_stiffness':material['stretch_stiffness'],
        'shear_stiffness':material['shear_stiffness'],'bend_stiffness':material['bend_stiffness'],
        'damping':material['spring_damping'],'friction':material['dynamic_friction'],
        'solver_position_iterations':args.position_iterations or 128,'contact_offset':radius*.008,'rest_offset':radius*.006,
        'solid_rest_offset':radius*.006}
    cloth.update(info.get('cloth_physics',{}))
    if args.position_iterations is not None:cloth['solver_position_iterations']=args.position_iterations
    if not np.isfinite(args.max_depenetration_velocity) or not 0<args.max_depenetration_velocity<=10:
        raise ValueError('Finite contact-correction speed in0..10m/s required')
    cloth['max_depenetration_velocity']=args.max_depenetration_velocity
    if args.max_linear_velocity is not None:
        if not np.isfinite(args.max_linear_velocity) or args.max_linear_velocity==0:
            raise ValueError('Finite nonzero numerical speed setting required')
        if args.max_linear_velocity<0 and (not surface_fem or args.max_linear_velocity!=-1):
            raise ValueError('Only the explicit surface-FEM automatic setting -1 is supported')
        cloth['max_linear_velocity']=args.max_linear_velocity
    fix={'model':True,'mesh':str(Path(info['path'])/'mesh.npz'),'opening':info['report'].get('opening'),
         'sdf':info.get('sdf',{'resolution':512,'contact_offset':radius*.004,'rest_offset':0.}),
         'link_physics':{**deepcopy(info.get('link_physics',{})),surface_link:cloth}}
    if args.sdf_resolution is not None:
        from calibrated_scene_reconstruction.assets.metric_mesh import sdf_settings
        fix['sdf']=sdf_settings({**fix['sdf'],'resolution':args.sdf_resolution})
    # Author the declared per-link material estimates before creating bodies.
    # The later prior pass still installs/readbacks complete native properties.
    for link, request in dec['objects'][obj]['links'].items():
        if request['kind'] != 'rigid':continue
        parameters=estimate['bodies'][obj+'/'+link]['parameters']
        settings=fix['link_physics'].setdefault(link,{})
        for key in ('mass','density'):
            if key in parameters:settings[key]=parameters[key]
        if 'mass' in parameters:settings['_mass_why']='Sampled current session physical prior, case '+args.case
    spec['fix']=fix;kit=asset(fix);parts={part['name']:part for part in kit['parts']};sheet=parts[surface_link]
    if sha(mesh_input)!=probe_input['mesh_sha256']:raise ValueError('Candidate mesh changed during native initialization')
    if args.regularize_iterations:
        from calibrated_scene_reconstruction.assets.cloth_rest import project_shape
        surfaces=[SignedSurface(part['V'],part['F']) for part in kit['parts'] if part['kind']=='rigid']
        separation=None
        if args.self_separate:
            from calibrated_scene_reconstruction.sim.cloth_separation import Separation,separate_triangles
            separation=Separation(sheet['cloth_rest']['points'],cloth['rest_offset'])
        def project(points):
            for surface in surfaces:points=surface.project_out(points,cloth['rest_offset'])
            if separation is not None:points=separation(points)
            if args.self_separate:
                points=separate_triangles(points,sheet['F'],2*cloth['rest_offset'],device='cuda:0')
            return points
        sheet['V'],preparation=project_shape(sheet['cloth_rest'],sheet['V'],sheet['F'],iterations=args.regularize_iterations,
                                            reference_pull=0.,collision_projection=project,device='cuda:0')
        (out/'geometry_preparation.json').write_text(json.dumps(preparation,indent=2)+'\n')
        np.savez_compressed(out/'prepared_source.npz',points=sheet['V'],faces=sheet['F'])
        from calibrated_scene_reconstruction.assets.cloth_state import check as prepared_state_check
        with np.load(mesh_input) as data:prepared_mesh={key:data[key].copy() for key in data.files}
        metadata=json.loads(str(prepared_mesh['links_json']));link=next(p for p in metadata if p['name']==surface_link)
        a,n=link['vertex_start'],link['vertex_count'];prepared_mesh['V'][a:a+n]=sheet['V']
        prepared_contract={'material_state_contract':dec.get('material_state_contract',2),
            'links':{surface_link:plan['bodies'][obj+('/'+surface_link)]['validity_hypothesis']},
            'maximum_authored_solid_penetration_m':1e-6}
        admissibility=prepared_state_check(prepared_mesh,prepared_contract,device='cuda:0')
        (out/'prepared_admissibility.json').write_text(json.dumps(admissibility,indent=2)+'\n')
        if not admissibility['pass']:
            raise ValueError('Prepared source geometry failed material/self/solid admissibility; no native state created')
    class LocalSession(Session):
        @property
        def scene_graph(self):return deepcopy(g)
        def config(self,name,*default):
            if name=='agent/mechanisms.json':return deepcopy(dec)
            if name=='agent/physics_priors.json':return deepcopy(plan)
            return super().config(name,*default)
    local=LocalSession(s.dir)
    world=World(stage_units_in_meters=1.,physics_dt=1/args.physics_hz,rendering_dt=1/30,device='cuda:0',backend='torch')
    physics_errors.guard_steps(world)
    pc=world.get_physics_context();pc.enable_gpu_dynamics(True);pc.set_broadphase_type('GPU')
    stage=omni.usd.get_context().get_stage()
    profile={'physics_hz':args.physics_hz,'friction_type':'patch','solver':args.solver,'basis':'Native candidate probe, no source approval'}
    declared=s.config('simulation.json',{})
    for key in ['gpu_max_deformable_surface_contacts','gpu_found_lost_aggregate_pairs_capacity']:
        if key in declared:profile[key]=declared[key]
    if args.aggregate_pair_capacity is not None:
        profile['gpu_found_lost_aggregate_pairs_capacity']=args.aggregate_pair_capacity
    if args.surface_contact_capacity is not None:
        profile['gpu_max_deformable_surface_contacts']=args.surface_contact_capacity
    apply_simulation(stage,profile,flexible=True)
    for scene in stage.Traverse():
        if scene.IsA(UsdPhysics.Scene):PhysxSchema.PhysxSceneAPI.Apply(scene).CreateEnableCCDAttr(False)
    mat=physics_material(stage,'/World/Looks/ProbePhysics',.6,.45,.05)
    add_box_collider(stage,'/World/Support/task_desk/surface',[-.5,-.5,-.02],[.5,.5,0])
    print('CANDIDATE_PHASE author_links',flush=True)
    root=add_links(stage,'/World/Objects/'+obj,kit,fix,spec,mat)
    from calibrated_scene_reconstruction.assets.physical_priors import probe_scope
    probe_plan, probe_dec, scope_receipt = probe_scope(plan, dec, [obj],
        supports=['task_desk'] if 'task_desk' in plan.get('supports', {}) else [])
    (out/'physical_prior_scope.json').write_text(json.dumps(scope_receipt,indent=2)+'\n')
    particle_startup=None
    if args.experimental_particle_shell:
        if sheet.get('backend','particle')!='particle':raise ValueError('Rebuild the candidate as an explicitly diagnostic particle cloth')
        from pxr import Vt
        from calibrated_scene_reconstruction.sim import particle_shell,discrete_bending,discrete_membrane,discrete_dissipation,experimental_startup
        from calibrated_scene_reconstruction.assets import cloth_projection
        files={}
        for module in [particle_shell,discrete_bending,discrete_membrane,discrete_dissipation,cloth_projection,experimental_startup]:
            name='particle_shell_'+Path(module.__file__).name;shutil.copy2(module.__file__,out/name);files[name]=sha(out/name)
        if args.linear_solver=='cuda_direct':
            from calibrated_scene_reconstruction.sim import cudss
            name='particle_shell_cudss.py';shutil.copy2(cudss.__file__,out/name);files[name]=sha(out/name)
            probe_input['cudss_library']={'path':str(cudss.library_path().resolve()),'sha256':sha(cudss.library_path())}
        probe_input['experimental_particle_shell_files']=files
        (out/'probe_input.json').write_text(json.dumps(probe_input,indent=2)+'\n')
        # The ordinary rigid prior path remains unchanged. This diagnostic
        # defers only the cloth law and reports its omitted dissipation.
        rigid_dec=deepcopy(probe_dec);rigid_plan=deepcopy(probe_plan)
        rigid_refs={f'{name}/{link}' for name,spec_ in rigid_dec['objects'].items()
                    for link,body in spec_['links'].items() if body['kind']=='rigid'}
        for name,spec_ in rigid_dec['objects'].items():spec_['links']={n:v for n,v in spec_['links'].items() if f'{name}/{n}' in rigid_refs}
        rigid_dec['mechanisms']=[m for m in rigid_dec['mechanisms'] if set(m['links'])<=rigid_refs]
        rigid_plan['bodies']={n:v for n,v in rigid_plan['bodies'].items() if n in rigid_refs}
        apply_priors(stage,rigid_plan,rigid_dec,args.case)
        primitive=stage.GetPrimAtPath(str(root.GetPath())+('/'+surface_link+'/surface'));api=PhysxSchema.PhysxParticleClothAPI(primitive)
        count=len(api.GetSpringIndicesAttr().Get());api.CreateSpringStiffnessesAttr(Vt.FloatArray([0.]*count));api.CreateSpringDampingsAttr(Vt.FloatArray([0.]*count))
        system_path=PhysxSchema.PhysxParticleAPI(primitive).GetParticleSystemRel().GetTargets()[0]
        system=PhysxSchema.PhysxParticleSystem.Get(stage,system_path)
        system.CreateParticleContactOffsetAttr(.0005);system.CreateMaxNeighborhoodAttr(256)
        # Reserve from actual candidate point neighborhoods, independent of
        # the task/video identity. Native overflow monitoring remains active.
        from scipy.spatial import cKDTree
        neighbors=cKDTree(sheet['V']).query_ball_point(sheet['V'],.001,return_length=True)
        if neighbors.max()>256:raise ValueError('Candidate exceeds the validated particle neighborhood budget')
        capacity=1<<max(20,(int(np.sum(neighbors-1))-1).bit_length())
        if capacity>1<<25:raise ValueError('Candidate particle contact capacity exceeds the diagnostic memory budget')
        for scene in stage.Traverse():
            if scene.IsA(UsdPhysics.Scene):
                api_scene=PhysxSchema.PhysxSceneAPI.Apply(scene)
                api_scene.CreateGpuMaxParticleContactsAttr(max(capacity,int(api_scene.GetGpuMaxParticleContactsAttr().Get())))
        (out/'particle_contact_capacity.json').write_text(json.dumps({'candidate_points':len(sheet['V']),
            'maximum_initial_neighbors':int(neighbors.max()),'initial_pair_upper_bound':int(np.sum(neighbors-1)//2),
            'reserved_contacts':capacity,'task_approved':False},indent=2)+'\n')
        particle_startup=particle_shell.DeferredParticleContacts(stage,primitive,
            material_rest=sheet['cloth_rest']['points'] if args.cook_material_rest else None)
    else:
        prior_readback=apply_priors(stage,probe_plan,probe_dec,args.case)
        (out/'physical_prior_readback.json').write_text(json.dumps(prior_readback,indent=2)+'\n')
    if args.experimental_native_viscosity_proof:
        if not surface_fem:raise ValueError('Native viscosity diagnostic requires surface FEM')
        from pxr import UsdShade
        surface=stage.GetPrimAtPath(str(root.GetPath())+('/'+surface_link+'/surface'))
        physical=UsdShade.MaterialBindingAPI(surface).ComputeBoundMaterial('physics')[0].GetPrim()
        physical.GetAttribute('physxDeformableMaterial:elasticityDamping').Set(0.)
        physical.GetAttribute('physxDeformableMaterial:bendDamping').Set(0.)
        surface.GetAttribute('physxDeformableBody:linearDamping').Set(0.)
    if args.topology_filter_rings:
        if not surface_fem:raise ValueError('Topology filter capability diagnostic requires surface FEM')
        from calibrated_scene_reconstruction.assets.cloth_neighborhood import face_pairs
        from pxr import Sdf,Vt
        topology=face_pairs(sheet['F'],rings=args.topology_filter_rings)
        np.savez_compressed(out/'native_topology_filter.npz',pairs=topology)
        filtered=stage.DefinePrim(str(root.GetPath())+'/DiagnosticLocalTopologyFilter','OmniPhysicsElementCollisionFilter')
        collider=str(root.GetPath())+('/'+surface_link+'/surface')
        for side in [0,1]:
            filtered.CreateRelationship(f'omniphysics:src{side}').SetTargets([collider])
            filtered.CreateAttribute(f'omniphysics:groupElemCounts{side}',Sdf.ValueTypeNames.UIntArray).Set(Vt.UIntArray([1]*len(topology)))
            filtered.CreateAttribute(f'omniphysics:groupElemIndices{side}',Sdf.ValueTypeNames.UIntArray).Set(Vt.UIntArray(topology[:,side].tolist()))
        filtered.CreateAttribute('omniphysics:filterEnabled',Sdf.ValueTypeNames.Bool).Set(True)
        (out/'topology_filter_capability.json').write_text(json.dumps({'diagnostic_only':True,
            'rings':args.topology_filter_rings,'pairs':len(topology),'native_implementation_verified':False,
            'scope':'Topologically local faces only; all nonlocal layer collisions and every actual crossing measurement retained. Schema authoring alone is not capability evidence.'},indent=2)+'\n')
    if args.experimental_bending:
        if not surface_fem:raise ValueError('Experimental bending needs the native surface membrane/contact adapter')
        from pxr import UsdShade
        surface=stage.GetPrimAtPath(str(root.GetPath())+('/'+surface_link+'/surface'))
        physical=UsdShade.MaterialBindingAPI(surface).ComputeBoundMaterial('physics')[0].GetPrim()
        if not history.get('native_bending',False):
            physical.GetAttribute('omniphysics:surfaceBendStiffness').Set(0.)
            physical.GetAttribute('physxDeformableMaterial:bendDamping').Set(0.)
        if history.get('monolithic_membrane',False):
            if history.get('native_bending',False):raise ValueError('Contradictory monolithic and native bending diagnostic')
            physical.GetAttribute('omniphysics:youngsModulus').Set(0.)
            physical.GetAttribute('physxDeformableMaterial:elasticityDamping').Set(0.)
    if (args.no_contact or args.no_rigid_contact) and not surface_fem:
        # Native system-wide switch covers every rigid partner, including
        # fixed contact carriers and partition cells. Keep their authored
        # provenance/geometry intact. Disabling rigid CollisionAPI alone also
        # failed to remove particle self contact in the old --no-contact mode.
        surface=stage.GetPrimAtPath(str(root.GetPath())+('/'+surface_link+'/surface'))
        particle=PhysxSchema.PhysxParticleAPI(surface)
        system_paths=particle.GetParticleSystemRel().GetTargets()
        if len(system_paths)!=1:raise ValueError('One diagnostic particle contact system required')
        system=PhysxSchema.PhysxParticleSystem(stage.GetPrimAtPath(system_paths[0]))
        system.CreateNonParticleCollisionEnabledAttr(False)
        if args.no_contact:particle.CreateSelfCollisionAttr(False)
        (out/'contact_ablation.json').write_text(json.dumps({
            'diagnostic_only':True,'task_approved':False,'system':str(system_paths[0]),
            'non_particle_collision_enabled':system.GetNonParticleCollisionEnabledAttr().Get(),
            'self_collision_enabled':particle.GetSelfCollisionAttr().Get(),
            'rigid_collision_geometry_changed':False},indent=2)+'\n')
    elif args.no_rigid_contact or args.no_contact:
        # A surface-FEM simulation mesh must retain its collider API/enabled
        # state or native creation removes the entire body. Isolate contacts
        # by disabling rigid partners and native sheet self-contact instead.
        from calibrated_scene_reconstruction.sim.contact_ablation import rigid_partners
        # Fixed collision carriers can be siblings of their visual owner.
        # Scan this single-assembly diagnostic stage, preserving every actual
        # flexible owner instead of relying on logical /part/surface paths.
        ablation=rigid_partners(stage,['/World'],exclude_roots=[
            str(root.GetPath())+'/'+part['name'] for part in kit['parts'] if part['kind']!='rigid'])
        if args.no_contact:
            sheet_collision=stage.GetPrimAtPath(str(root.GetPath())+('/'+surface_link+'/surface')).GetAttribute('physxDeformableBody:selfCollision')
            sheet_collision.Set(False)
            if sheet_collision.Get() is not False:raise ValueError('Native self-contact ablation did not read back disabled state')
            ablation['self_collision_enabled']=False
        (out/'contact_ablation.json').write_text(json.dumps(ablation,indent=2)+'\n')
    if args.no_gravity:
        next(UsdPhysics.Scene(p) for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)).GetGravityMagnitudeAttr().Set(0.)
    if args.diagnostic_actions:
        if args.no_gravity or args.no_contact or args.no_rigid_contact or args.seconds<6.4:
            raise ValueError('Action preview requires real gravity/contact and enough time for the complete fixture')
        holder=UsdPhysics.FixedJoint.Define(stage,'/World/DiagnosticFixtures/container_holder')
        holder.CreateBody1Rel().SetTargets([str(root.GetPath())+'/body'])
        holder.CreateLocalPos0Attr(Gf.Vec3f(0));holder.CreateLocalPos1Attr(Gf.Vec3f(0))
    if preparation_plan and preparation_plan.get('fixed_base'):
        name=preparation_plan['fixed_base']
        if not any(p['name']==name and p['kind']=='rigid' for p in kit['parts']):
            raise ValueError('Preparation base fixture must name a current rigid link')
        holder=UsdPhysics.FixedJoint.Define(stage,'/World/DiagnosticFixtures/preparation_holder')
        holder.CreateBody1Rel().SetTargets([str(root.GetPath())+'/'+name])
        holder.CreateLocalPos0Attr(Gf.Vec3f(0));holder.CreateLocalPos1Attr(Gf.Vec3f(0))
    stage.GetRootLayer().Export(str(out/'initial.usda'))
    startup=None
    if args.defer_experimental_startup:
        from calibrated_scene_reconstruction.sim.experimental_startup import DeferredStartup,FilteredPairs
        gravity=next(UsdPhysics.Scene(p) for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)).GetGravityMagnitudeAttr()
        entries=[(gravity,0.),(stage.GetPrimAtPath(str(root.GetPath())+('/'+surface_link+'/surface')).GetAttribute('physxDeformableBody:selfCollision'),False)]
        surface=stage.GetPrimAtPath(str(root.GetPath())+('/'+surface_link+'/surface'))
        relation=UsdPhysics.FilteredPairsAPI.Apply(surface).CreateFilteredPairsRel()
        filtered=FilteredPairs(relation)
        rigid_paths=['/World/Support/task_desk/surface',*[str(root.GetPath())+'/'+part['name'] for part in kit['parts'] if part['kind']=='rigid']]
        entries.append((filtered,list(dict.fromkeys(filtered.Get()+rigid_paths))))
        startup=DeferredStartup(entries);startup.suspend()
        stage.GetRootLayer().Export(str(out/'experimental_startup_suspended.usda'))
    if args.bootstrap_hz is not None:
        if not args.physics_hz<=args.bootstrap_hz<=1000000:
            raise ValueError('Diagnostic bootstrap frequency must be between the actual profile and1MHz')
        world.set_simulation_dt(physics_dt=1/args.bootstrap_hz)
    print('CANDIDATE_PHASE reset',flush=True);world.reset()
    native_reset_time_s=world.current_time
    if args.bootstrap_hz is not None:world.set_simulation_dt(physics_dt=1/args.physics_hz)
    print('CANDIDATE_PHASE native_state',flush=True)
    assert audit_simulation(stage,profile)['solver']==args.solver
    initial_pose_report=particle_startup.author_initial_pose(world,str(root.GetPath())+('/'+surface_link+'/surface'),native_reset_time_s=native_reset_time_s) if particle_startup else None
    if initial_pose_report is not None:
        (out/'initial_material_pose.json').write_text(json.dumps(initial_pose_report,indent=2)+'\n')
    ctx=Ctx(local,SimpleNamespace(objects={obj:root}),world,stage,None)
    initial_joints=ctx.initial_articulation_state()
    (out/'initial_articulation_state.json').write_text(json.dumps(initial_joints,indent=2)+'\n')
    if not initial_joints['pass'] and not args.continue_invalid_diagnostic:
        raise ValueError('Native reset changed the authored initial articulation state')
    ctx.native_errors=physics_errors
    initial=ctx.nodes(obj+('/'+surface_link)).copy();faces=sheet['F'];rest=sheet['cloth_rest']
    if surface_fem:
        from calibrated_scene_reconstruction.assets.membrane_energy import estimate as estimate_membrane
        parameters={key:material[key] for key in ['youngs_modulus','poissons_ratio','thickness','areal_density']}
        stored_energy={'authored':estimate_membrane(rest['points'],sheet['V'],faces,**parameters),
                       'native_initial':estimate_membrane(rest['points'],initial,faces,**parameters)}
        (out/'initial_elastic_energy.json').write_text(json.dumps(stored_energy,indent=2)+'\n')
    mass_view=ctx._cloth[obj+('/'+surface_link)]
    if not surface_fem and not args.experimental_particle_shell:
        from calibrated_scene_reconstruction.scene.objects.particle_contact_binding import read as read_particle_profile
        # Legacy diagnostic priors can omit physical thickness; retain that
        # uncertainty while still auditing their actual springs and masses.
        actual_profile=read_particle_profile(mass_view,{**material,'thickness':material.get('thickness')})
        (out/'native_particle_profile.json').write_text(json.dumps(actual_profile,indent=2)+'\n')
        if not actual_profile['native_spring_readback'] or not actual_profile['native_mass_readback']:
            raise ValueError('Whole-cloth native springs or masses differ from the candidate material before preparation')
    if hasattr(mass_view,'index_registration'):
        (out/'particle_index_registration.json').write_text(json.dumps(mass_view.index_registration,indent=2)+'\n')
        probe_input['particle_registration_sources']={str(Path(p).resolve()):sha(p) for p in
            ['src/calibrated_scene_reconstruction/sim/particle_surface.py','src/calibrated_scene_reconstruction/sim/particle_registration_output.py',
             'src/calibrated_scene_reconstruction/assets/particle_registration.py']}
        (out/'probe_input.json').write_text(json.dumps(probe_input,indent=2)+'\n')
    native_viscosity_report=None
    if args.experimental_native_viscosity_proof:
        from calibrated_scene_reconstruction.sim.native_viscosity_diagnostic import install as install_viscosity
        native_viscosity_report=install_viscosity(world,mass_view,
            proof_path=args.experimental_native_viscosity_proof,session_dir=s.dir,out=out,
            faces=faces,rest=rest,material=material,cloth=cloth,solver=args.solver,
            physics_hz=args.physics_hz,position_iterations=int(cloth['solver_position_iterations']),linear_solver=args.linear_solver)
    preparation_controller=None
    if preparation_plan:
        from calibrated_scene_reconstruction.tasks.mechanisms.preparation import MaterialPreparation
        preparation_controller=MaterialPreparation(ctx,obj,kit,preparation_plan)
        if args.seconds<preparation_controller.release_time+.5 and not args.preparation_isolation:
            raise ValueError('Record at least0.5s with every temporary cloth and hinge force released after preparation')
    experimental_report=None
    particle_shell_report=None
    if args.experimental_particle_shell:
        from calibrated_scene_reconstruction.sim.particle_shell import install as install_particle_shell
        dissipation=None if viscosity[0] is None else dict(zip(['bending_viscosity_time_s','membrane_viscosity_time_s'],viscosity))
        particle_shell_report=install_particle_shell(world,stage,str(root.GetPath())+('/'+surface_link+'/surface'),sheet,material,linear_solver=args.linear_solver,prediction_output=out/'first_internal_prediction.npz',nonlinear_iterations=args.nonlinear_iterations,dissipation=dissipation)
        particle_shell_report['initialization']=particle_startup.restore()
        particle_shell_report['initial_material_pose']=initial_pose_report
        difference=float(np.linalg.norm(ctx.nodes(obj+('/'+surface_link))-initial,axis=1).max())
        particle_shell_report['initialization']['maximum_restore_state_change_m']=difference
        if difference>1e-7:raise ValueError('Restoring particle contacts changed the initial material state')
    if args.experimental_bending:
        from calibrated_scene_reconstruction.sim.experimental_bending import load as load_bending,install as install_bending
        config,natural=load_bending(s.dir,args.experimental_bending,sheet['cloth_rest']['points'],faces)
        if config.get('link')!=obj+('/'+surface_link):raise ValueError('Bending history belongs to a different physical link')
        cohesion=None
        if config.get('cohesion'):
            import torch
            from calibrated_scene_reconstruction.sim.cohesive_interface import CohesiveInterface
            settings=config['cohesion']
            if not settings.get('source') or not settings.get('basis'):
                raise ValueError('Finite-strength material interfaces need explicit source and inferred construction basis')
            with np.load(out/'experimental_cohesive_interfaces.npz') as data:
                cohesion=CohesiveInterface(sheet['V'],data['indices'],data['barycentric'],data['area'],
                    **settings['parameters'],device='cuda:0',dtype=torch.float64)
        membrane_parameters=({key:material[key] for key in ['youngs_modulus','poissons_ratio','thickness']}
                             if config.get('monolithic_membrane',False) else None)
        experimental_report=install_bending(world,mass_view,sheet['cloth_rest']['points'],faces,natural,material['flexural_rigidity'],cohesion=cohesion,native_bending=config.get('native_bending',False),membrane_parameters=membrane_parameters)
    startup_report=None
    if startup is not None:
        startup_report=startup.restore()
        physics_errors.check()
        delta=float(np.linalg.norm(ctx.nodes(obj+('/'+surface_link))-initial,axis=1).max())
        startup_report['maximum_restore_state_change_m']=delta
        (out/'experimental_startup.json').write_text(json.dumps(startup_report,indent=2)+'\n')
        if delta>1e-7:raise ValueError('Restoring native environment unexpectedly changed the initialized sheet state')
        stage.GetRootLayer().Export(str(out/'experimental_startup_restored.usda'))
    if surface_fem:
        mass_model={'direct_native_nodal_readback':False,'solver':args.solver,
                    'force_profile_validated':args.solver=='PGS',
                    'basis':'Uniform nodal model and authored total mass validated by PGS impulse/contact controls; TGS remains unverified or failed'}
        uniform_mass=True if args.solver=='PGS' else None
    else:
        nodal_masses=as_numpy(mass_view.get_particle_masses())[0]
        uniform_mass=bool(np.allclose(nodal_masses,nodal_masses[0],rtol=1e-5,atol=0.))
        mass_model={'direct_native_nodal_readback':True,'solver':args.solver,
                    'minimum_nodal_mass_kg':float(nodal_masses.min()),'maximum_nodal_mass_kg':float(nodal_masses.max()),
                    'total_mass_kg':float(nodal_masses.sum())}
    from calibrated_scene_reconstruction.assets.constitutive_features import usage
    constitutive=usage(rest,'surface_fem' if surface_fem else 'particle')
    (out/'native_constitutive_authoring.json').write_text(json.dumps(constitutive,indent=2)+'\n')
    authored_motion=float(np.linalg.norm(initial-sheet['V'],axis=1).max())
    from calibrated_scene_reconstruction.assets.cloth_sampling import rigid_coverage,initial_overlaps,native_probe_coverage
    native_solid_radius=None
    if not surface_fem:
        particle=PhysxSchema.PhysxParticleAPI(ctx._cloth[obj+('/'+surface_link)].prim)
        contact_system=PhysxSchema.PhysxParticleSystem(stage.GetPrimAtPath(particle.GetParticleSystemRel().GetTargets()[0]))
        native_solid_radius=contact_system.GetSolidRestOffsetAttr().Get()
        native_rigid_contact={'rest_offset':contact_system.GetRestOffsetAttr().Get(),
                              'contact_offset':contact_system.GetContactOffsetAttr().Get()}
        if native_solid_radius is None or not np.isfinite(native_solid_radius) or native_solid_radius<=0:
            raise ValueError('Actual finite native particle self-contact radius required')
    # Use the same semantic sampling policy as the scene route. The isolated
    # probe must not silently impose primary-task precision on an explicitly
    # secondary flexible detail. Physical coverage stays visible in the report;
    # source, actual contact, released state and task checks remain independent.
    primary_links={link for item in dec.get('mechanisms',[]) if item.get('role')=='primary'
                   for link in item.get('links',[])}
    secondary_sampling=response_policy.secondary_sampling_tolerances(kit['parts'],understood,material_accuracy)
    sampling_gap=(0. if obj+('/'+surface_link) in primary_links else secondary_sampling.get(surface_link,0.))
    sampling=({'backend':'native_surface_fem','particle_sampling_required':False} if surface_fem else
              rigid_coverage(sheet['V'],faces,native_rigid_contact,maximum_gap_m=sampling_gap))
    if not surface_fem:
        sampling['initial_nonlocal_envelopes']=initial_overlaps(initial,rest['points'],native_solid_radius)
        sampling['pass'] &= sampling['initial_nonlocal_envelopes']['pass']
    (out/'authored_sampling.json').write_text(json.dumps(sampling,indent=2)+'\n')
    sampling_policy=native_probe_coverage(sampling,secondary_approximation=bool(np.any(np.asarray(sampling_gap)>0)))
    validity=plan['bodies'][obj+('/'+surface_link)]['validity_hypothesis']
    metric=MaterialMetric(initial,faces,rest,max_edge_strain=validity['max_edge_strain'],max_area_change=validity['max_area_change'])
    surfaces={part['name']:SignedSurface(part['V'],part['F']) for part in kit['parts'] if part['kind']=='rigid'}
    from calibrated_scene_reconstruction.sim.soft_surface import from_view
    soft_surfaces={part['name']:from_view(ctx._soft[obj+'/'+part['name']],
        stage.GetPrimAtPath(str(root.GetPath())+'/'+part['name']+'/surface'))
        for part in kit['parts'] if part['kind'] in {'deformable','rope'}}
    soft_states={name:[] for name in soft_surfaces}
    soft_sim_states={name:[] for name in soft_surfaces}
    soft_vel_states={name:[] for name in soft_surfaces}
    volume_reference=None
    if soft_surfaces:
        reference_file=out/'native_volume_references.npz'
        np.savez_compressed(reference_file,**{name+'_'+key:value for name,surface in soft_surfaces.items()
                                             for key,value in surface.reference().items()})
        volume_reference={'file':reference_file.name,'sha256':sha(reference_file),
            'scope':'Original local material references and native tetrahedral indices; current collision/simulation points and simulation velocities are recorded separately in world coordinates. No rest-shape or native state writes; export/reload remains unqualified.'}
    surface_prim=stage.GetPrimAtPath(str(root.GetPath())+('/'+surface_link+'/surface'))
    if surface_fem:
        native_rest=PhysxSchema.PhysxCollisionAPI(surface_prim).GetRestOffsetAttr().Get()
    else:
        system=PhysxSchema.PhysxParticleAPI(surface_prim).GetParticleSystemRel().GetTargets()[0]
        native_rest=PhysxSchema.PhysxParticleSystem(stage.GetPrimAtPath(system)).GetRestOffsetAttr().Get()
    if native_rest is None or not np.isfinite(native_rest) or native_rest<=0:
        raise ValueError('Read actual native rest contact thickness before evaluating solid penetration')
    sdf_error=0.;sdf_resolution_records=[]
    for part in kit['parts']:
        if part['kind']!='rigid':continue
        prim=stage.GetPrimAtPath(str(root.GetPath())+'/'+part['name']+'/surface')
        resolution=PhysxSchema.PhysxSDFMeshCollisionAPI(prim).GetSdfResolutionAttr().Get()
        if resolution:
            resolved=physics_errors.sdf_resolution(prim.GetPath(),int(resolution));sdf_resolution_records.append(resolved)
            sdf_error=max(sdf_error,2*float(np.ptp(part['V'],axis=0).max())/resolved['effective_resolution_bound'])
    from calibrated_scene_reconstruction.assets.cloth_contact_tolerance import solid_tolerance
    penetration_tolerance=solid_tolerance(material.get('thickness'),sdf_error,legacy_rest_offset=float(native_rest))
    (out/'native_contact_settings.json').write_text(json.dumps({
        'rest_offset_m':float(native_rest),'sdf_discretization_m':sdf_error,
        'physical_thickness_m':material.get('thickness'),
        'penetration_budget_basis':'Physical thickness and rigid discretization; numerical padding cannot increase the limit when thickness is declared',
        'sdf_resolution_records':sdf_resolution_records,
        'maximum_depenetration_velocity_m_s':args.max_depenetration_velocity,
        'maximum_solid_penetration_m':penetration_tolerance,'backend':'surface_fem' if surface_fem else 'particle'},indent=2)+'\n')
    from calibrated_scene_reconstruction.assets.mechanisms import joint_frames,revolute_position
    moving_joint=next((j for j in kit['joints'] if j['type']=='revolute' and j['parent']=='body' and j['child']=='lid'),None)
    if moving_joint is not None:frames0,frames1=joint_frames(moving_joint)
    if args.diagnostic_actions and moving_joint is None:
        raise ValueError('The declared opening fixture needs its actual body/lid revolute connection')
    def lid_angle():
        return (float(revolute_position(ctx.pose(obj+'/body')[1],ctx.pose(obj+'/lid')[1],frames0,frames1))
                if moving_joint is not None else None)
    handle=None
    if args.diagnostic_actions:
        from calibrated_scene_reconstruction.tasks.mechanisms.force_fixture import ForceHandle,edge_patch
        chart=np.asarray(rest['points'])
        if args.legacy_full_edge_grasp and args.grasp_width is not None:
            raise ValueError('Choose a local material patch or the explicit legacy strip control')
        grasp_width=(None if args.legacy_full_edge_grasp else
                     min(.01,.08*np.ptp(chart[:,0])) if args.grasp_width is None else args.grasp_width)
        patch=edge_patch(chart,min(.01,.08*np.ptp(chart[:,1])),width=grasp_width,centre_fraction=args.grasp_fraction)
        (out/'force_patch.json').write_text(json.dumps({
            'indices':patch.tolist(),'material_min':chart[patch].min(0).tolist(),
            'material_max':chart[patch].max(0).tolist(),
            'selection':'Explicit legacy full-width strip' if args.legacy_full_edge_grasp else 'Single compact material-coordinate edge patch',
            'basis':'Diagnostic finite-force fixture hypothesis; not an observed grasp or a gripper-access certificate',
            'material_width_m':grasp_width,'centre_fraction':args.grasp_fraction,
            'native_positions_prescribed':False,'task_approved':False},indent=2)+'\n')
        masses=as_numpy(ctx._cloth[obj+('/'+surface_link)].get_particle_masses())[0]
        handle=ForceHandle(initial[patch],masses[patch],dt=world.get_physics_dt(),held_mass=float(masses.sum()),
                           force_limit=1.5,speed=.12,stiffness=20.)
        target=handle.target.copy()+[.04,0,.35]
    def phase(time):
        if preparation_controller is not None:return 'preparation_released' if preparation_controller.released(time) else 'material_pose_preparation'
        return ('initial_unforced' if time<.3 else 'open_lid' if time<1.9 else 'extract_sheet' if time<4.8 else 'close_lid') if args.diagnostic_actions else 'unforced'
    if args.record_fps<1 or args.physics_hz%args.record_fps:raise ValueError('Record uniformly spaced actual native states')
    states=[];cloth_velocities=[];report=[];body=[];lid=[];rigid_states={name:[] for name in surfaces}
    def record(step):
        check_started=walltime.perf_counter()
        def checkpoint(stage):
            (out/'check_progress.json').write_text(json.dumps({
                'native_time_s':step/args.physics_hz,'stage':stage,
                'elapsed_check_s':walltime.perf_counter()-check_started,
                'physical_validity_approved':False,'task_approved':False},indent=2)+'\n')
        checkpoint('read_native_state')
        points=ctx.nodes(obj+('/'+surface_link))
        velocity=as_numpy(ctx._cloth[obj+('/'+surface_link)].get_velocities())[0].copy()
        # Preserve the actual state before expensive checks. A completed
        # diagnostic/validity report is still required to use it as evidence.
        np.savez_compressed(out/'unchecked_native_state.npz',points=points,velocities=velocity,
                            faces=faces,rest=rest['points'],native_time_s=step/args.physics_hz,
                            physical_validity_approved=False,task_approved=False)
        checkpoint('material_metric')
        row=metric.check(points)
        checkpoint('self_intersections')
        cross=self_intersections(points,faces,example_limit=4)
        checkpoint('particle_sampling')
        dynamic_sampling=({'pass':True,'applicable':False,'backend':'surface_fem'} if surface_fem else
                          rigid_coverage(points,faces,native_rigid_contact,maximum_gap_m=sampling_gap))
        from calibrated_scene_reconstruction.sim.cloth_clearance import measure as finite_clearance
        checkpoint('finite_thickness_clearance')
        layer_clearance=(finite_clearance(points,faces,rest['points'],material['thickness'],device='cuda:0',
                                          local_bending_review=material_accuracy=='task_and_semantics')
                         if 'thickness' in material else None)
        checkpoint('solid_penetration')
        probes=triangle_samples(points,faces);penetration={}
        for name,surface in surfaces.items():
            position,rotation=ctx.pose(obj+'/'+name);penetration[name]=surface.max_penetration((probes-position)@rotation)
        soft_checks={}
        for name,surface in soft_surfaces.items():
            soft_checks[name]=surface.refresh()
            from calibrated_scene_reconstruction.assets.volume_contact import rigid_contacts
            soft_checks[name]['rigid_contacts']=rigid_contacts(surface.points,surface.faces,surfaces,
                {n:ctx.pose(obj+'/'+n) for n in surfaces},penetration_tolerance)
            penetration[name]=surface.distance.max_penetration(probes)
            soft_states[name].append(surface.points.copy())
            soft_sim_states[name].append(surface.simulation_points.copy())
            soft_vel_states[name].append(as_numpy(surface.view.get_simulation_mesh_nodal_velocities())[0].copy())
        states.append(points);cloth_velocities.append(velocity)
        legacy_poses={}
        for name,values in [('body',body),('lid',lid)]:
            if name in surfaces:
                position,orientation=ctx.pose(obj+'/'+name)
                values.append(np.r_[position,orientation.ravel()]);legacy_poses[name]=values[-1]
        for name in surfaces:
            position,rotation=ctx.pose(obj+'/'+name);rigid_states[name].append(np.r_[position,rotation.ravel()])
        report.append({'time_s':step/args.physics_hz,'phase':phase(step/args.physics_hz),'lid_angle_rad':lid_angle(),
            'material':row,'material_response':response_policy.material_verdict(row,material_accuracy),
            'native_particle_sampling':dynamic_sampling,
            'self_intersection':cross,'finite_sheet_clearance':layer_clearance,'penetration_m':penetration,
            'native_soft_solid_checks':soft_checks,
            'solid_contact_pass':all(value<=penetration_tolerance for value in penetration.values()) and all(
                value['rigid_contacts']['pass'] for value in soft_checks.values()),
            'maximum_solid_penetration_m':penetration_tolerance,
            'maximum_motion_m':float(np.linalg.norm(points-initial,axis=1).max())})
        checkpoint('response_and_energy')
        if material.get('thickness'):
            from calibrated_scene_reconstruction.assets.cloth_timestep import measure as timestep_measure
            report[-1]['time_resolution']=timestep_measure(velocity,args.physics_hz,material['thickness'],float(native_rest))
        if not surface_fem and not args.experimental_particle_shell:
            from calibrated_scene_reconstruction.assets.spring_energy import measure as spring_energy
            report[-1]['initialization_energy_diagnostic']=spring_energy(
                points,rest,cloth,mass_kg=float(as_numpy(ctx._cloth[obj+('/'+surface_link)].get_particle_masses()).sum()))
        if preparation_controller is not None:
            report[-1]['construction_tracking']=preparation_controller.observe(step/args.physics_hz,points)
        if not cross['pass']:
            from calibrated_scene_reconstruction.assets.cloth_failure_location import locate
            report[-1]['crossing_locations']=locate(points,faces,rest['points'],cross)
        (out/'progress.json').write_text(json.dumps({'not_task_approval':True,'last_state':report[-1]},indent=2)+'\n')
        np.savez_compressed(out/f'state_{step:06d}.npz',points=points,velocities=velocity,**legacy_poses,
                            **{name+'_pose':values[-1] for name,values in rigid_states.items()},
                            **{name+'_surface_points':values[-1] for name,values in soft_states.items()},
                            **{name+'_simulation_points':values[-1] for name,values in soft_sim_states.items()},
                            **{name+'_simulation_velocities':values[-1] for name,values in soft_vel_states.items()},
                            **{name+'_surface_faces':surface.faces for name,surface in soft_surfaces.items()})
        print('PASSIVE_STATE',step/args.physics_hz,'strain',row['maximum_principal_material_strain'],
              'crossings',cross['intersecting_pairs'],'motion_m',report[-1]['maximum_motion_m'],flush=True)
        checkpoint('recorded')
        return bool(report[-1]['material_response']['pass'] and cross['pass'] and (layer_clearance is None or layer_clearance['pass'])
                    and report[-1]['solid_contact_pass'] and sampling_policy['pass'] and dynamic_sampling['pass'])
    valid=record(0);stopped_on_invalid=not valid and not args.continue_invalid_diagnostic
    requested_steps=round(args.seconds*args.physics_hz)
    stopped_on_checkpoint=False
    for step in range(requested_steps):
        if stopped_on_invalid:break
        if preparation_controller is not None and preparation_controller.step(step/args.physics_hz) is False:
            # Keep the last actual state at the failed preparation boundary.
            # This is an unmet construction condition, not a contact failure.
            if abs(report[-1]['time_s']-step/args.physics_hz)>1e-10:
                valid=record(step)
                if not valid and not args.continue_invalid_diagnostic:stopped_on_invalid=True
            stopped_on_checkpoint=True
            break
        if args.diagnostic_actions:
            time=step/args.physics_hz
            desired=0. if time<.3 else min(2.,(time-.3)*1.25) if time<4.8 else max(0.,2.-(time-4.8)*1.25)
            theta=lid_angle();omega=as_numpy(ctx.body(obj+'/lid').get_angular_velocity())
            axis=ctx.pose(obj+'/body')[1]@np.asarray(moving_joint['axis'])
            effort=float(np.clip(.08*(desired-theta)-.003*np.dot(omega,axis),-.2,.2))
            ctx.apply_joint_effort(obj,'body','lid',effort,effort_limit=.2)
            if time>=1.9:
                view=ctx._cloth[obj+('/'+surface_link)];points=ctx.nodes(obj+('/'+surface_link));velocities=as_numpy(view.get_velocities())[0,patch]
                forces=handle.forces(points[patch],velocities,target,load=[0,0,float(masses.sum())*9.81])
                ctx.apply_nodal_forces(obj+('/'+surface_link),patch,forces,force_limit_N=1.5*float(handle.weights.max()))
        world.step(render=False)
        if (step+1)%max(1,args.physics_hz//20)==0:
            (out/'physics_progress.json').write_text(json.dumps({
                'time_s':(step+1)/args.physics_hz,'last_fully_checked_time_s':report[-1]['time_s'],
                'phase':phase((step+1)/args.physics_hz),'task_approved':False},indent=2)+'\n')
        if (step+1)%(args.physics_hz//args.record_fps)==0 or step+1==requested_steps:
            valid=record(step+1)
            if not valid and not args.continue_invalid_diagnostic:stopped_on_invalid=True
    result={'not_source_or_task_approval':True,'case':args.case,'input':str(args.candidate),'provisional_scale':True,
        'complete':abs(report[-1]['time_s']-requested_steps/args.physics_hz)<1e-8,
        'candidate_sha256':candidate_sha256,
        'mesh_sha256':probe_input['mesh_sha256'],
        'probe_input_sha256':sha(out/'probe_input.json'),
        'diagnostic_contact_disabled':args.no_contact,'diagnostic_gravity_disabled':args.no_gravity,
        'diagnostic_rigid_contact_disabled':args.no_rigid_contact,
        'diagnostic_preparation_isolation':args.preparation_isolation,
        'initial_native_vs_authored_displacement_m':authored_motion,'authored_particle_sampling':sampling,
        'initial_articulation_state':initial_joints,
        'native_sampling_policy':sampling_policy,
        'numerical_profile':profile,'native_cloth_parameters':cloth,
        'solver_position_iterations':int(cloth['solver_position_iterations']),
        'native_uniform_mass':uniform_mass,'native_mass_model':mass_model,
        'external_actuation':bool(args.diagnostic_actions or preparation_controller is not None),'task_approved':False,
        'initial_state_approved':False,
        'material_preparation':preparation_controller.report() if preparation_controller is not None else None,
        'diagnostic_bootstrap_hz':args.bootstrap_hz,
        'constitutive_authoring':constitutive,
        'experimental_bending':experimental_report,
        'experimental_particle_shell':particle_shell_report,
        'experimental_native_viscosity':native_viscosity_report,
        'experimental_startup':startup_report,
        'experimental_startup_sha256':sha(out/'experimental_startup.json') if startup_report is not None else None,
        'geometry_preparation':({'method':'Intrinsic material projection before native creation, never a native state overwrite',
            'prepared_source_sha256':sha(out/'prepared_source.npz'),
            'admissibility_sha256':sha(out/'prepared_admissibility.json')} if args.regularize_iterations else None),
        'requested_duration_s':args.seconds,'completed_duration_s':report[-1]['time_s'],
        'stopped_after_physical_failure':stopped_on_invalid,
        'stopped_after_preparation_checkpoint_failure':stopped_on_checkpoint,
        'force_fixture':handle.report() if handle is not None else None,
        'material_accuracy_mode':material_accuracy,
        'quantitative_material_pass':all(r['material']['pass'] for r in report),
        'physical_plausibility_review_required':not all(r['material']['pass'] for r in report),
        'material_and_self_contact_pass':all(
            r['material_response']['pass'] and r['self_intersection']['pass'] and
            (r['finite_sheet_clearance'] is None or r['finite_sheet_clearance']['pass']) for r in report),
        'physical_state_pass':bool(initial_joints['pass'] and sampling_policy['pass'] and all(
            r['material_response']['pass'] and r['self_intersection']['pass'] and r['solid_contact_pass'] and
            r['native_particle_sampling']['pass'] and
            (r['finite_sheet_clearance'] is None or r['finite_sheet_clearance']['pass']) for r in report)),
        'states':report}
    from calibrated_scene_reconstruction.scene.acceptance_evidence import mesh_digest
    result['physical_mesh_sha256']=mesh_digest(mesh_input)
    result['native_volume_reference']=volume_reference
    np.savez_compressed(out/'states.npz',points=states,velocities=cloth_velocities,faces=faces,rest=rest['points'],body=body,lid=lid,
                        **{name+'_poses':values for name,values in rigid_states.items()},
                        **{name+'_surface_points':values for name,values in soft_states.items()},
                        **{name+'_simulation_points':values for name,values in soft_sim_states.items()},
                        **{name+'_simulation_velocities':values for name,values in soft_vel_states.items()},
                        **{name+'_surface_faces':surface.faces for name,surface in soft_surfaces.items()})
    physics_errors.check();result['native_physics_errors']=physics_errors.report()
    if retention_plan is not None:
        from calibrated_scene_reconstruction.assets.cloth_retention import measure as retained_material
        preparation=result['material_preparation']
        released=preparation['all_force_release_time_s'] if preparation is not None else 0.
        retention=retained_material(states,rest['points'],faces,rigid_states['body'],
            [row['time_s'] for row in report],info['report']['opening'],retention_plan,
            released_at_s=released,physical_pass=result['physical_state_pass'],complete=result['complete'])
        assisted=bool(args.diagnostic_actions or args.no_gravity or args.no_contact or args.no_rigid_contact
            or args.preparation_isolation or args.regularize_iterations or args.self_separate
            or args.experimental_bending or args.experimental_particle_shell
            or args.experimental_native_viscosity_proof
            or preparation_plan and (preparation_plan.get('grips') or preparation_plan.get('fixed_base')))
        receipt=(preparation or {}).get('hinge_release_receipt') or {}
        cleared=(preparation is None or preparation.get('hinge_effort_cleared') is True
            and receipt.get('readback_after_release') is True
            and (preparation.get('hinge_control')!='native_drive' or receipt.get('native_drive_gains_zero') is True))
        retention['checks'].update(no_cloth_or_body_assistance=not assisted,
            actual_drive_release=bool(cleared),native_error_check=result['native_physics_errors']['pass'])
        if 'minimum_lid_angle_rad' in retention_plan:
            limit=retention_plan['minimum_lid_angle_rad']
            if type(limit) not in (int,float) or not math.isfinite(limit) or limit<=0:
                raise ValueError('Explicit positive minimum open-lid angle required')
            observed=[row for row in report if row['time_s']>=retention_plan['observation_start_s']-1e-9]
            retention['checks']['lid_remains_open']=bool(observed and all(row['lid_angle_rad']>=limit for row in observed))
        support_request=dec['objects'][obj]['links'][surface_link].get('initial_support')
        if support_request is not None:
            from calibrated_scene_reconstruction.assets.material_support import check as initial_support_check
            support_name=support_request['link']
            host=next(part for part in kit['parts'] if part['name']==support_name)
            host_pose=np.asarray(rigid_states[support_name][0])
            local=(states[0]-host_pose[:3])@host_pose[3:].reshape(3,3)
            retention['initial_support']=initial_support_check(local,rest['points'],faces,
                host['V'],host['F'],support_request,thickness_m=float(cloth['thickness']),tolerance_m=1e-6)
            retention['checks']['native_initial_support']=retention['initial_support']['pass']
        retention['pass']=all(retention['checks'].values())
        result['material_retention']=retention
    if preparation_controller is not None:
        preparation=result['material_preparation']
        result['preparation_goals_retained_pass']=bool(
            preparation['checkpoints_pass'] and preparation['final_release_conditions']['pass'] and preparation['released_conditions_retained'])
    passive_indices=[]
    if not result['external_actuation']:
        passive_indices=list(range(len(report)))
    elif preparation_controller is not None and preparation_controller.released(report[-1]['time_s']):
        passive_indices=[i for i,row in enumerate(report) if row['phase']=='preparation_released']
    if (not args.no_gravity and not args.no_contact and not args.no_rigid_contact
            and len(passive_indices)>1):
        result['rigid_passive_response']=passive_response.evaluate(understood,kit['parts'],
            {name:[values[i] for i in passive_indices] for name,values in rigid_states.items()},
            [report[i]['time_s'] for i in passive_indices],
            flexible_states={name:[values[i]for i in passive_indices]for name,values in {**soft_states,surface_link:states}.items()},
            observations={surface_link:{
                'self_intersections':{'unit':'count','values':[report[i]['self_intersection']['intersecting_pairs']for i in passive_indices]},
                'solid_penetration_m':{'unit':'m','values':[max(report[i]['penetration_m'].values(),default=0.)for i in passive_indices]}}})
        result['passive_response_interval_s']=[report[passive_indices[0]]['time_s'],report[passive_indices[-1]]['time_s']]
        result['passive_response_after_preparation']=preparation_controller is not None
        result['unforced_rigid_and_contact_pass']=bool(result['complete'] and result['physical_state_pass']
            and result['native_physics_errors']['pass'] and result['rigid_passive_response']['pass'])
        result['passive_scope']=('Unforced rigid motion and sampled contact on diagnostic support only. '
            'A flexible part can leave its intended support without violating these checks. '
            'Required initial material distribution, source/semantic response, full scene/support '
            'and task checks must pass separately; this probe does not approve the initial state.')
    (out/'native_physics_errors.json').write_text(json.dumps(result['native_physics_errors'],indent=2)+'\n')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('CANDIDATE_RESULT',json.dumps({key:result[key] for key in
        ['not_source_or_task_approval','completed_duration_s','physical_state_pass','stopped_after_physical_failure',
         'stopped_after_preparation_checkpoint_failure']}),flush=True)
except (Exception,KeyboardInterrupt) as error:
    partial={}
    try:
        registration=getattr(error,'particle_state',None)
        output=getattr(error,'native_output',None)
        if output is not None:
            np.savez_compressed(out/'failed_particle_output.npz',**output)
            partial['native_particle_output']='failed_particle_output.npz'
        if registration is not None:
            np.savez_compressed(out/'failed_particle_registration.npz',
                source=registration['source'],native=registration['native'])
            partial['particle_registration']={'path':registration['path'],
                'states':'failed_particle_registration.npz','states_changed':False}
        if globals().get('ctx') is not None and globals().get('sheet') is not None:
            # Preserve the actual failure state even if diagnostic recording
            # was sparse. This is read-only geometry, never accepted evidence.
            rigid_names=[part['name'] for part in kit['parts'] if part['kind']=='rigid']
            terminal={name+'_poses':np.r_[p,R.ravel()] for name in rigid_names
                      for p,R in [ctx.pose(obj+'/'+name)]}
            np.savez_compressed(out/'terminal_native_geometry.npz',points=ctx.nodes(obj+('/'+surface_link)),
                                faces=sheet['F'],rest=sheet['cloth_rest']['points'],**terminal)
            partial['terminal_native_geometry']={'file':'terminal_native_geometry.npz',
                'native_time_s':float(world.current_time),'task_approved':False,'physical_validity_approved':False}
        if isinstance(globals().get('report'),list) and report and globals().get('states'):
            count=min(len(report),len(states),len(body),len(lid),*[len(v) for v in rigid_states.values()])
            if count:
                np.savez_compressed(out/'partial_states.npz',points=states[:count],velocities=cloth_velocities[:count],faces=faces,rest=rest['points'],
                    body=body[:count],lid=lid[:count],**{name+'_poses':values[:count] for name,values in rigid_states.items()},
                    **{name+'_surface_points':values[:count] for name,values in soft_states.items()},
                    **{name+'_simulation_points':values[:count] for name,values in soft_sim_states.items()},
                    **{name+'_simulation_velocities':values[:count] for name,values in soft_vel_states.items()},
                    **{name+'_surface_faces':surface.faces for name,surface in soft_surfaces.items()})
                saved={'complete':False,'physical_state_pass':False,'task_approved':False,
                    'failure':str(error) or type(error).__name__,'completed_samples':count,
                    'states':report[:count],
                    'sampled_physical_validity':all(r['material']['pass'] and r['self_intersection']['pass'] and r['solid_contact_pass'] for r in report[:count]),
                    'scope':'Incomplete diagnostic prefix only; cannot be promoted as a complete released trace.'}
                (out/'partial_result.json').write_text(json.dumps(saved,indent=2)+'\n')
                partial.update(partial_result='partial_result.json',partial_states='partial_states.npz')
    except Exception as preservation_error:
        partial={'partial_trace_preservation_error':str(preservation_error)}
    (out/'error.json').write_text(json.dumps({'pass':False,'error':str(error) or type(error).__name__,
        'error_type':type(error).__name__,'origin':'interrupted' if isinstance(error,KeyboardInterrupt) else 'native_error',
        'traceback':traceback.format_exc(),**partial},indent=2)+'\n');raise
finally:
    physics_errors.close()
    app.close()
