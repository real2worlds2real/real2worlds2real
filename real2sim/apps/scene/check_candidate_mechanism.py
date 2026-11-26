"""Native rigid mechanism experiment on preserved, source-passing candidates.

This breaks the construction/test dependency without exporting unqualified
assets. It never writes task_check.json or grants whole-scene acceptance.
The plan names exact rounds, explicit solver/collider hypotheses and a local
agent check(ctx). Production authoring and the full scene must be tested later.
"""
import argparse
import copy
import hashlib
import importlib.util
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('session');p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
p.add_argument('--renderer',choices=['rt','pt'],default='rt',help='Initialize the native render backend before native scene creation')
p.add_argument('--diagnostic-unapproved-source', action='store_true',
               help='Explicit construction diagnostic on unapproved rigid source geometry; never source, task or export acceptance')
a=p.parse_args()
from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402
app=SimulationApp({'headless':True,'renderer':'PathTracing' if a.renderer=='pt' else 'RaytracedLighting'})
import sys  # noqa: E402
original_hook=sys.excepthook
def fail(kind,error,tb):
    original_hook(kind,error,tb)
    # An exception must leave explicit failed evidence and shut Kit down before
    # Python unloads its extension modules. post_quit alone needs an event loop
    # that no longer runs here, and could leave a crash with no failure report.
    import traceback
    try:
        folder=globals().get('out')
        if folder is not None and folder.is_dir():
            (folder/'failure.json').write_text(json.dumps({
                'pass':False,'task_approved':False,'error_type':kind.__name__,
                'error':str(error),'traceback':traceback.format_exception(kind,error,tb),
            },indent=2)+'\n')
    finally:
        try:
            monitor=globals().get('errors')
            if monitor is not None:monitor.close()
        finally:
            app.close()
sys.excepthook=fail
import numpy as np  # noqa: E402
import trimesh  # noqa: E402
import omni.usd  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from pxr import UsdPhysics,PhysxSchema  # noqa: E402
from calibrated_scene_reconstruction.scene.session import Session  # noqa: E402
from calibrated_scene_reconstruction.assets.metric_mesh import asset,sdf_settings  # noqa: E402
from calibrated_scene_reconstruction.assets.physical_priors import validate,sample  # noqa: E402
from calibrated_scene_reconstruction.assets.object_pose import rotation  # noqa: E402
from calibrated_scene_reconstruction.sim.mechanism_stage import add_links  # noqa: E402
from calibrated_scene_reconstruction.sim.physics import physics_material,bind_colliders  # noqa: E402
from calibrated_scene_reconstruction.sim.source_moments import author as moments  # noqa: E402
from calibrated_scene_reconstruction.sim.usd import set_pose  # noqa: E402
from calibrated_scene_reconstruction.sim.simulation import apply,apply_rigid_iterations,audit  # noqa: E402
from calibrated_scene_reconstruction.sim.physics_errors import PhysicsErrors  # noqa: E402
from calibrated_scene_reconstruction.scene.support.runtime_inputs import models as support_models  # noqa: E402
from calibrated_scene_reconstruction.sim.support_stage import add_support  # noqa: E402
from calibrated_scene_reconstruction.tasks.rigid.task_checks import Ctx  # noqa: E402
from calibrated_scene_reconstruction.core.paths import ROOT,resolve  # noqa: E402

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')
s=Session(a.session);plan_path=Path(a.plan).resolve();plan=json.loads(plan_path.read_text())
out=(s/a.output).resolve()
if not out.is_relative_to((s/'qa').resolve()):raise ValueError('Candidate experiments stay in session QA')
out.mkdir(parents=True,exist_ok=False)
if not plan.get('basis') or not plan.get('candidates'):raise ValueError('Disclosed candidate experiment required')
check=(s.config_dir/plan['check']).resolve()
if not check.is_relative_to(s.config_dir.resolve()) or not check.is_file():raise ValueError('Agent check must be current config-local source')
profile=plan['simulation'];priors=s.config('agent/physics_priors.json');mechanisms=s.config('agent/mechanisms.json')
source_graph=copy.deepcopy(s.scene_graph)
configuration={'agent/physics_priors.json':priors,'agent/mechanisms.json':mechanisms,'scene_graph.json':source_graph,
               'task.json':s.config('task.json',{})}
additional_inputs={}
for name in plan.get('additional_configs',[]):
    path=(s.config_dir/name).resolve()
    if name in configuration or not path.is_relative_to(s.config_dir.resolve()) or not path.is_file():
        raise ValueError('Additional candidate config must be a distinct current config-local file')
    configuration[name]=json.loads(path.read_text());additional_inputs[str(path)]=sha(path)
for artifact in plan.get('additional_artifacts',[]):
    path=resolve(artifact['path']).resolve()
    if (not any(path.is_relative_to(root.resolve()) for root in [s.dir,s.config_dir])
            or not path.is_file() or sha(path)!=artifact.get('sha256')):
        raise ValueError('Additional candidate artifact must bind an exact current session/config input')
    additional_inputs[str(path)]=sha(path)
write(out/'configuration.json',configuration)
physical=sample(validate(priors,mechanisms),plan.get('case','nominal'))
world=World(stage_units_in_meters=1.,physics_dt=1/profile['physics_hz'],rendering_dt=1/30,backend='numpy')
stage=omni.usd.get_context().get_stage();errors=PhysicsErrors(out/'native_errors.json');errors.guard_steps(world)
apply(stage,profile)
graph=copy.deepcopy(source_graph);graph['dynamic_objects']={}
support=source_graph['task']['support'];material_values=physical['supports'][support]['parameters']
support_mat=physics_material(stage,'/World/Materials/support',material_values['static_friction'],material_values['dynamic_friction'],material_values['restitution'])
support_root='/World/Support/'+support
add_support(stage,support_root,support_models(s)[support]);bind_colliders(stage,[support_root],support_mat)
objects={};records={};inputs={str(plan_path):sha(plan_path),str(check):sha(check),str(Path(__file__).resolve()):sha(__file__),**additional_inputs}
inputs[str(out/'configuration.json')]=sha(out/'configuration.json')
support_fix=source_graph['furniture'][support]['fix']
for path in [support_fix['mesh'],support_fix['visual_glb'],
             *([support_fix['source_solids']['path']] if support_fix.get('source_solids') else []),
             *(entry['path'] for entry in support_fix.get('textures',[]))]:
    path=resolve(path);inputs[str(path)]=sha(path)
for name,proposal in plan['candidates'].items():
    if name not in source_graph['dynamic_objects'] or type(proposal['round']) is not int:raise ValueError('Unknown candidate')
    directory=s/f"objects/{name}/round_{proposal['round']}";f=json.loads((directory/'fit.json').read_text())
    source_geometry_pass = bool(f['ok'] and f['metrics']['ok'])
    if not source_geometry_pass and not a.diagnostic_unapproved_source:
        raise ValueError('This rigid mechanism harness requires passing source geometry; an explicit unapproved-source diagnostic cannot qualify delivery')
    for file,key in [('mesh.npz','mesh_sha256'),('model.py','model_sha256')]:
        if sha(directory/file)!=f[key]:raise ValueError('Candidate snapshot changed')
        inputs[str((directory/file).resolve())]=sha(directory/file)
    inputs[str(directory/'fit.json')]=sha(directory/'fit.json')
    fix={'model':source_graph['dynamic_objects'][name]['fit']['model'],'mesh':str((directory/'mesh.npz').resolve()),
         'params':f['build']['params'],'opening':f['build'].get('opening'),'sdf':sdf_settings(proposal['sdf']),**f['pose']}
    if proposal.get('link_physics'):
        fix['link_physics']=copy.deepcopy(proposal['link_physics'])
    kit=asset(fix)
    if kit.get('joints') or any(part['kind']!='rigid' for part in kit['parts']):raise ValueError('Only unjointed rigid contact candidates in this harness')
    spec=copy.deepcopy(source_graph['dynamic_objects'][name]);spec['fix']=fix
    masses={}
    for part in kit['parts']:
        values=physical['bodies'][name+'/'+part['name']]['parameters'];mesh=trimesh.Trimesh(part['V'],part['F'],process=True)
        if not mesh.is_volume:raise ValueError('Actual closed source solid required for mass')
        masses[part['name']]=float(values['mass'] if 'mass' in values else values['density']*mesh.volume)
    spec['mass']=sum(masses.values())
    spec['_mass_why']='Current sampled material priors integrated over actual closed candidate solids; individual link values and source inertia applied before reset.'
    graph['dynamic_objects'][name]=spec
    root=add_links(stage,'/World/Objects/'+name,kit,fix,spec,support_mat,look_base=s.dir);objects[name]=root
    q=Rotation.from_matrix(rotation(fix)).as_quat()[[3,0,1,2]];set_pose(root,[*fix['xy'],fix['support_z']],q)
    for part in kit['parts']:
        values=physical['bodies'][name+'/'+part['name']]['parameters'];body=stage.GetPrimAtPath(str(root.GetPath())+'/'+part['name'])
        moments(body,part['V'],part['F'],masses[part['name']])
        rb=PhysxSchema.PhysxRigidBodyAPI.Apply(body);rb.CreateLinearDampingAttr(values['linear_damping']);rb.CreateAngularDampingAttr(values['angular_damping'])
        mat=physics_material(stage,str(body.GetPath())+'_material',values['static_friction'],values['dynamic_friction'],values['restitution'],friction_combine='average')
        bind_colliders(stage,[str(body.GetPath())],mat)
    records[name]={'fix':fix,'mass_kg':masses,'source_geometry_pass':source_geometry_pass,
                   'parameters':{k:v for k,v in physical['bodies'].items() if k.startswith(name+'/')}}
apply_rigid_iterations(stage,profile);profile_report=audit(stage,profile)
stage.GetRootLayer().Export(str(out/'initial.usda'))
record={'diagnostic_only':True,'task_approved':False,'candidate_rounds':plan['candidates'],'objects':records,'inputs':inputs,
        'source_geometry_qualified':all(v['source_geometry_pass'] for v in records.values()),
        'diagnostic_unapproved_source_requested':a.diagnostic_unapproved_source,
        'backend':{'name':'isaacsim','version':importlib.metadata.version('isaacsim')},'profile':profile_report,
        'production_profile_validated':False,'basis':plan['basis']}
write(out/'input.json',record)
world.reset()
precision=errors.precision_audit()
if not precision['pass']:
    write(out/'precision_failure.json',precision);errors.close()
    raise ValueError('Native cooking changed requested precision; inspect native_errors')
class CandidateSession:
    scene_graph=graph
    def config(self,name,*args):
        if name not in configuration:raise ValueError('Candidate callback must declare and snapshot additional configuration before simulation: '+name)
        return copy.deepcopy(configuration[name])
    def __getattr__(self,key):return getattr(s,key)
    def __truediv__(self,key):return s/key
ctx=Ctx(CandidateSession(),SimpleNamespace(objects=objects,inputs=SimpleNamespace(simulation=profile)),world,stage,lambda *args:None)
ctx.native_errors=errors;ctx.trial_output=str(out.relative_to(s.dir.resolve()))
module_spec=importlib.util.spec_from_file_location('candidate_check',check);module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
for imported in list(sys.modules.values()):
    file=getattr(imported,'__file__',None)
    if file and Path(file).resolve().is_relative_to(ROOT/'src/calibrated_scene_reconstruction'):
        path=Path(file).resolve()
        if path.is_file():inputs[str(path)]=sha(path)
write(out/'input.json',record)
result=module.check(ctx);errors.check()
changed=[path for path,digest in inputs.items() if not Path(path).is_file() or sha(path)!=digest]
if changed:
    write(out/'changed_inputs.json',{'pass':False,'paths':changed})
    raise ValueError('Candidate experiment inputs changed during simulation: '+', '.join(changed))
write(out/'result.json',{**record,'native_experiment':result,'native_physics':errors.report(),'native_precision':errors.precision_audit(),
                         'scope':'Native selected-candidate experiment. Source appearance, input scene and whole-scene acceptance are separate.'})
print(json.dumps({'candidate_experiment_pass':result['pass'],'task_approved':False,'output':str(out)}),flush=True)
errors.close();app.close()
