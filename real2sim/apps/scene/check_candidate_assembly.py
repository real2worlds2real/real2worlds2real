"""Native multi-object candidate assay, including multiple flexible surfaces.

Explicit unapproved candidates let construction and task diagnostics advance
without manufacturing a source pass or export receipt. Every selected object's
physical links remain present. The agent callback chooses bounded assays in
the actual modeled support; the harness itself never approves a scene.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('session');parser.add_argument('--plan',required=True);parser.add_argument('--output',required=True)
parser.add_argument('--diagnostic-unapproved-source',action='store_true')
args=parser.parse_args()
from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402
app=SimulationApp({'headless':True})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from calibrated_scene_reconstruction.scene.session import Session  # noqa: E402
from calibrated_scene_reconstruction.assets.metric_mesh import asset,sdf_settings  # noqa: E402
from calibrated_scene_reconstruction.assets.physical_priors import probe_scope,sample  # noqa: E402
from calibrated_scene_reconstruction.assets.response_policy import secondary_sampling_tolerances  # noqa: E402
from calibrated_scene_reconstruction.assets.object_pose import rotation  # noqa: E402
from calibrated_scene_reconstruction.sim.mechanism_stage import add_links,audit  # noqa: E402
from calibrated_scene_reconstruction.sim.physical_priors import apply as apply_priors  # noqa: E402
from calibrated_scene_reconstruction.sim.simulation import apply as simulation,apply_rigid_iterations,audit as audit_simulation  # noqa: E402
from calibrated_scene_reconstruction.sim.physics_errors import PhysicsErrors  # noqa: E402
from calibrated_scene_reconstruction.sim.physics import physics_material,bind_colliders  # noqa: E402
from calibrated_scene_reconstruction.sim.usd import set_pose,quat_from_matrix  # noqa: E402
from calibrated_scene_reconstruction.sim.support_stage import add_support  # noqa: E402
from calibrated_scene_reconstruction.scene.support.runtime_inputs import models as support_models  # noqa: E402
from calibrated_scene_reconstruction.scene.objects.object_understanding import validate_build  # noqa: E402
from calibrated_scene_reconstruction.tasks.rigid.task_checks import Ctx  # noqa: E402
from calibrated_scene_reconstruction.tasks.rigid.assay_status import summarize as summarize_assay  # noqa: E402
from calibrated_scene_reconstruction.scene.objects.blender.candidate_parameters import complete as build_parameters  # noqa: E402


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')


out=None;errors=None
try:
    s=Session(args.session);plan_path=Path(args.plan)
    if not plan_path.is_file() and not plan_path.is_absolute():plan_path=s.config_dir/plan_path
    plan_path=plan_path.resolve();plan=json.loads(plan_path.read_text())
    out=(s/args.output).resolve()
    if out.exists() or not out.is_relative_to((s/'qa').resolve()):raise ValueError('Choose a new current-session QA output')
    if not plan.get('basis') or not isinstance(plan.get('candidates'),dict) or not plan['candidates']:
        raise ValueError('Explicit candidate scope and hypothesis basis required')
    out.mkdir(parents=True)
    graph=copy.deepcopy(s.scene_graph);configuration={name:s.config(name) for name in ['agent/mechanisms.json','agent/physics_priors.json','agent/object_understanding.json','task.json']}
    configuration['scene_graph.json']=copy.deepcopy(graph)
    for name in plan.get('additional_configs',[]):
        if name in configuration:raise ValueError('Duplicate additional config')
        path=(s.config_dir/name).resolve()
        if not path.is_relative_to(s.config_dir.resolve()):raise ValueError('Config escaped current capture')
        configuration[name]=json.loads(path.read_text())
    inputs={str(plan_path):digest(plan_path),str(Path(__file__).resolve()):digest(__file__)}
    def bind(path):
        path=Path(path).resolve()
        if not any(path.is_relative_to(root.resolve()) for root in [s.dir,s.config_dir]):
            raise ValueError('Candidate input must belong to current capture')
        inputs[str(path)]=digest(path);return path
    representation_receipt=None
    if plan.get('representation_hypothesis') is not None:
        if plan.get('physics_prior_proposal') is not None or plan.get('case','nominal')!='nominal':
            raise ValueError('A same-geometry representation control cannot also resample material priors')
        from calibrated_scene_reconstruction.scene.objects import candidate_representation
        proposal_path=bind(s.config_dir/plan['representation_hypothesis'])
        proposal=json.loads(proposal_path.read_text())
        write(out/'source_configuration.json',configuration)
        configuration,representation_receipt=candidate_representation.apply(configuration,proposal,list(plan['candidates']))
        write(out/'representation_hypothesis.json',{'proposal':proposal,**representation_receipt})
        for path in [out/'source_configuration.json',out/'representation_hypothesis.json',Path(candidate_representation.__file__).resolve()]:
            inputs[str(path)]=digest(path)
    if plan.get('physics_prior_proposal') is not None:
        if plan.get('case','nominal')!='nominal':
            raise ValueError('An isolated prior proposal cannot also resample every low/high parameter')
        from calibrated_scene_reconstruction.assets import physical_prior_proposal
        proposal_path=bind(s.config_dir/plan['physics_prior_proposal'])
        proposal=json.loads(proposal_path.read_text())
        write(out/'source_physics_priors.json',configuration['agent/physics_priors.json'])
        candidate_physics,proposal_receipt=physical_prior_proposal.apply(configuration['agent/physics_priors.json'],proposal)
        if any(row['group']=='bodies' and row['body'].split('/')[0] not in plan['candidates']
               for row in proposal_receipt['changes']):
            raise ValueError('Diagnostic material proposal must refer to a selected candidate object')
        configuration['agent/physics_priors.json']=candidate_physics
        write(out/'physical_prior_proposal.json',{'proposal':proposal,**proposal_receipt})
        for path in [out/'source_physics_priors.json',out/'physical_prior_proposal.json',Path(physical_prior_proposal.__file__).resolve()]:
            inputs[str(path)]=digest(path)
    write(out/'configuration.json',configuration)
    inputs[str(out/'configuration.json')]=digest(out/'configuration.json')
    check=bind(s.config_dir/plan['check'])
    support=graph['task']['support'];physical,mechanisms,scope=probe_scope(configuration['agent/physics_priors.json'],configuration['agent/mechanisms.json'],list(plan['candidates']),supports=[support])
    from calibrated_scene_reconstruction.core.capabilities import entries, invoke
    for name in entries('probe_disclosure'):
        invoke('probe_disclosure', name, scope)
    estimate=sample(physical,plan.get('case','nominal'));profile=plan['simulation']
    graph['dynamic_objects']={};candidates={};has_flexible=False;sampling={}
    for name,request in plan['candidates'].items():
        if name not in configuration['scene_graph.json']['dynamic_objects']:raise ValueError('Unknown candidate object')
        if ('round' in request)==('descriptor' in request):raise ValueError('Choose one exact object round or candidate descriptor')
        if 'round' in request:
            if type(request['round']) is not int or request['round']<1:raise ValueError('Positive candidate round required')
            folder=s/f"objects/{name}/round_{request['round']}";f=json.loads(bind(folder/'fit.json').read_text())
            mesh=bind(folder/'mesh.npz');model=bind(folder/'model.py')
            if digest(mesh)!=f['mesh_sha256'] or digest(model)!=f['model_sha256']:raise ValueError('Round snapshot changed')
            from calibrated_scene_reconstruction.scene.objects.blender.candidate_source import round_status
            from calibrated_scene_reconstruction.scene.objects.blender.raster_contract import sources as raster_sources
            for source in raster_sources():inputs[str(source)]=digest(source)
            manifest_path=folder/'input.json'
            manifest=json.loads(bind(manifest_path).read_text()) if manifest_path.is_file() else None
            source_status=round_status(f,manifest,digest(manifest_path) if manifest is not None else None)
            report=f['build'];pose=f['pose'];source_pass=source_status['source_geometry_pass']
        else:
            descriptor=bind(s/request['descriptor']);f=json.loads(descriptor.read_text());folder=Path(f['path']).resolve()
            mesh=bind(folder/'mesh.npz');report=f['report'];pose=f['pose'];source_pass=False
            source_status={'source_geometry_pass':False,'object_approved':False,
                           'scope':'Explicit unapproved descriptor; source measurement and complete OBJ review remain pending.'}
        if not source_pass and not args.diagnostic_unapproved_source:raise ValueError('Unapproved geometry requires explicit diagnostic flag')
        validate_build(s,name,report,understanding_plan=configuration['agent/object_understanding.json'])
        original=copy.deepcopy(configuration['scene_graph.json']['dynamic_objects'][name])
        fix={'model':original['fit']['model'],'mesh':str(mesh),'opening':report.get('opening'),'params':build_parameters(report,f.get('params')),**pose,
             'sdf':sdf_settings(request.get('sdf')),'link_physics':copy.deepcopy(request.get('link_physics',{}))}
        kit=asset(fix)
        if set(mechanisms['objects'][name]['links'])!={p['name'] for p in kit['parts']}:
            raise ValueError('Candidate omitted a declared physical link')
        for part in kit['parts']:
            declaration=mechanisms['objects'][name]['links'][part['name']]
            if part['kind']!=declaration['kind'] or (part['kind']=='cloth' and
                    part.get('backend','particle')!=declaration.get('backend','particle')):
                raise ValueError('Actual candidate physical representation differs from its declaration')
        if representation_receipt and name in representation_receipt['reference_candidates']:
            reference_path=bind(s/representation_receipt['reference_candidates'][name])
            reference=json.loads(reference_path.read_text());reference_mesh=bind(Path(reference['path'])/'mesh.npz')
            if report.get('initial_state_admissibility',{}).get('pass') is not True:
                raise ValueError('Representation control requires a fresh valid actual material-state build')
            comparison=candidate_representation.same_geometry(asset({'mesh':str(reference_mesh)}),kit,reference['pose'],pose)
            write(out/(name+'_representation_geometry.json'),comparison)
        for part in kit['parts']:
            settings=fix['link_physics'].setdefault(part['name'],{});values=estimate['bodies'][name+'/'+part['name']]['parameters']
            if part['kind']=='rigid':
                for key in ['density','mass']:
                    if key in values:settings[key]=values[key]
                if 'mass' in values:settings['_mass_why']='Current sampled declared material prior; not inferred from successful task'
            else:
                has_flexible=True
                # Material properties come from priors; contact/solver settings
                # remain explicit per-link hypotheses in the candidate plan.
                for key in ['density','mass','areal_density','thickness','stretch_stiffness','shear_stiffness','bend_stiffness','youngs_modulus','poissons_ratio',
                            'flexural_rigidity','elasticity_damping','linear_damping','bend_damping']:
                    if key in values:settings[key]=values[key]
                if 'spring_damping' in values:settings['damping']=values['spring_damping']
                if 'dynamic_friction' in values:settings['friction']=values['dynamic_friction']
        original['fix']=fix;graph['dynamic_objects'][name]=original
        understood=configuration['agent/object_understanding.json']['objects'][name]
        sampling.update({name+'/'+link:value for link,value in secondary_sampling_tolerances(kit['parts'],understood,configuration['task.json'].get('flexible_accuracy')).items()})
        from calibrated_scene_reconstruction.assets.cloth_route_capability import inspect as cloth_route_capability
        route_capabilities={part['name']:cloth_route_capability(part.get('backend','particle'),part['cloth_rest'],part['F'])
                            for part in kit['parts'] if part['kind']=='cloth'}
        candidates[name]={'kit':kit,'fix':fix,'report':report,'source_geometry_pass':source_pass,
                          'cloth_route_capabilities':route_capabilities,
                          'source_status':source_status,'mesh':str(mesh)}
    # Audit separate source solids before native relaxation can push an
    # invalid delivered overlap away. Flexible bonds have their own audit.
    from calibrated_scene_reconstruction.assets.rigid_initial_state import measure as initial_rigid_contacts
    rigid_sources=[]
    for name,data in candidates.items():
        pose=data['fix'];R=rotation(pose);position=np.r_[pose['xy'],pose['support_z']]
        for part in data['kit']['parts']:
            if part['kind']=='rigid':
                rigid_sources.append({'name':name+'/'+part['name'],'V':part['V']@R.T+position,'F':part['F']})
    if len(rigid_sources)>1:
        initial=initial_rigid_contacts(rigid_sources,spacing_m=.0005,tolerance_m=.0001,device='cpu')
        write(out/'authored_rigid_pair_screen.json',initial)
        if not initial['pass']:raise ValueError('Authored independent rigid source solids overlap before native initialization')
    world=World(stage_units_in_meters=1.,physics_dt=1/profile['physics_hz'],rendering_dt=1/30,device='cuda:0',backend='torch')
    world.get_physics_context().enable_gpu_dynamics(True);world.get_physics_context().set_broadphase_type('GPU')
    stage=omni.usd.get_context().get_stage();errors=PhysicsErrors(out/'native_errors.json');errors.guard_steps(world)
    simulation(stage,profile,flexible=has_flexible)
    support_values=estimate['supports'][support]['parameters']
    material=physics_material(stage,'/World/Materials/candidate_support',support_values['static_friction'],support_values['dynamic_friction'],support_values['restitution'])
    support_fix=graph['furniture'][support]['fix']
    for key in ['mesh','visual_glb']:bind(support_fix[key])
    if support_fix.get('source_solids'):bind(support_fix['source_solids']['path'])
    for texture in support_fix.get('textures',[]):bind(texture['path'])
    add_support(stage,'/World/Support/'+support,support_models(s)[support]);bind_colliders(stage,['/World/Support/'+support],material)
    objects={}
    for name,data in candidates.items():
        print('[candidate.assembly] author',name,flush=True)
        root=add_links(stage,'/World/Objects/'+name,data['kit'],data['fix'],graph['dynamic_objects'][name],material,look_base=s.dir)
        pose=data['fix'];set_pose(root,[*pose['xy'],pose['support_z']],quat_from_matrix(rotation(pose)));objects[name]=root
    prior_report=apply_priors(stage,physical,mechanisms,plan.get('case','nominal'));apply_rigid_iterations(stage,profile)
    authored_mechanisms=audit(stage,mechanisms,sampling_tolerances=sampling)
    write(out/'authoring_audit.json',{'mechanisms':authored_mechanisms,'simulation':audit_simulation(stage,profile),'physical_priors':prior_report,'scope':scope,
        'cloth_route_capabilities':{name:data['cloth_route_capabilities'] for name,data in candidates.items()}})
    stage.GetRootLayer().Export(str(out/'initial.usda'))
    world.reset();precision=errors.precision_audit()
    if not precision['pass']:raise ValueError('Native cooking changed requested precision')
    class CandidateSession:
        scene_graph=graph
        def config(self,name,*unused):
            if name not in configuration:raise ValueError('Callback must declare config before simulation: '+name)
            return copy.deepcopy(configuration[name])
        def __getattr__(self,name):return getattr(s,name)
        def __truediv__(self,name):return s/name
    ctx=Ctx(CandidateSession(),SimpleNamespace(objects=objects,inputs=SimpleNamespace(simulation=profile),
                physical_estimates=prior_report),world,stage,lambda *unused:None)
    ctx.native_errors=errors;ctx.trial_output=str(out.relative_to(s.dir.resolve()))
    ctx.authored_mechanism_audit=authored_mechanisms
    ctx.candidate_plan=copy.deepcopy(plan)
    native_initial=ctx.initial_articulation_state();write(out/'native_initial.json',native_initial)
    if not native_initial['pass']:raise ValueError('Native articulation initial state changed')
    module_spec=importlib.util.spec_from_file_location('candidate_agent_check',check);module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    import sys
    from calibrated_scene_reconstruction.core.paths import ROOT
    for imported in list(sys.modules.values()):
        file=getattr(imported,'__file__',None)
        if file and Path(file).is_file() and Path(file).resolve().is_relative_to(ROOT/'src/calibrated_scene_reconstruction'):inputs[str(Path(file).resolve())]=digest(file)
    write(out/'inputs.json',inputs)
    result=summarize_assay(module.check(ctx));errors.check()
    changed=[name for name,expected in inputs.items() if not Path(name).is_file() or digest(name)!=expected]
    if changed:raise ValueError('Candidate inputs changed during assay: '+', '.join(changed))
    write(out/'result.json',{'diagnostic_only':True,'task_approved':False,'source_geometry_pass':{name:d['source_geometry_pass'] for name,d in candidates.items()},
          'source_status':{name:d['source_status'] for name,d in candidates.items()},
                           'native_experiment':result,'native_physics':errors.report(),'native_precision':errors.precision_audit(),'scope':scope})
    print(json.dumps({'native_checks_pass':result['raw_assay_pass'],
                      'status':result['assay_status']['status'],'object_approved':False,
                      'task_approved':False,'output':str(out)}),flush=True)
except BaseException as error:
    import traceback
    # Kit shutdown may terminate before Python prints a pending exception.
    # Preserve a readable failure even when setup failed before creating out.
    traceback.print_exc()
    if out is not None and out.is_dir():write(out/'failure.json',{'pass':False,'task_approved':False,'error':str(error),'traceback':traceback.format_exc()})
    raise
finally:
    if errors is not None:errors.close()
    app.close()
