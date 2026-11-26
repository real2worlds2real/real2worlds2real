"""Read native retention proof; geometric fold connection cannot certify load."""
import json
from pathlib import Path

from calibrated_scene_reconstruction.scene.acceptance_evidence import mesh_digest,sha


def current(session,obj,link,round_dir,entry):
    root=session.dir.resolve()
    report_path=(root/entry.get('report','')).resolve()
    evidence=entry.get('evidence',{})
    relative=str(report_path.relative_to(root)) if report_path.is_relative_to(root) else ''
    if not evidence or relative not in evidence:
        raise ValueError('Connected-fold support needs hash-bound native retention evidence')
    for name,expected in evidence.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root/'qa') or not path.is_file() or sha(path)!=expected:
            raise ValueError('Native retention evidence changed or missing')
    result=json.loads(report_path.read_text());directory=report_path.parent
    probe=directory/'probe_input.json'
    if sha(probe)!=result.get('probe_input_sha256'):
        raise ValueError('Native retention input changed')
    inputs=json.loads(probe.read_text())
    producer=Path(__file__).resolve().parents[4]/'apps/scene/check_candidate_passive.py'
    if inputs.get('source_sha256')!=sha(producer):
        raise ValueError('Current native retention producer required')
    bound=['probe_input.json','states.npz','candidate_input.json','mechanisms.json',
           'physics_priors.json','retention_input.json','native_physics_errors.json']
    if any(str((directory/name).relative_to(root)) not in evidence for name in bound):
        raise ValueError('Native retention requires raw states and complete bound inputs')
    if result.get('physical_mesh_sha256')!=mesh_digest(Path(round_dir)/'mesh.npz'):
        raise ValueError('Native retention belongs to different physical geometry')
    if (result.get('complete') is not True or result.get('physical_state_pass') is not True
            or result.get('native_physics_errors',{}).get('pass') is not True):
        raise ValueError('Complete physically valid native retention run required')
    for filename,key in [('mechanisms.json','agent/mechanisms.json'),('physics_priors.json','agent/physics_priors.json')]:
        if json.loads((directory/filename).read_text())!=session.config(key,{}):
            raise ValueError('Native retention uses stale representation or material hypotheses')
    simulation={k:v for k,v in session.config('simulation.json',{}).items() if k!='basis'}
    if any(result.get('numerical_profile',{}).get(k)!=v for k,v in simulation.items()):
        raise ValueError('Native retention must use the current deployment simulation profile')
    candidate=json.loads((directory/'candidate_input.json').read_text())
    fit=session.scene_graph['dynamic_objects'][obj].get('fit') or {}
    configured=fit.get('link_physics',{}).get(link,{})
    if any(configured.get(k)!=v for k,v in candidate.get('cloth_physics',{}).items()):
        raise ValueError('Native retention material/contact profile differs from deployment')
    actual=result.get('native_cloth_parameters',{})
    if not actual or any(actual.get(k)!=v for k,v in configured.items()):
        raise ValueError('Actual native cloth settings differ from deployment')
    if any(value!=candidate.get('link_physics',{}).get(name,{})
           for name,value in fit.get('link_physics',{}).items() if name!=link):
        raise ValueError('Native retention rigid-link contact settings differ from deployment')
    if candidate.get('sdf',{})!=fit.get('sdf',{}):
        raise ValueError('Native retention rigid contact profile differs from deployment')
    for group in ['validity_sources','retention_sources']:
        if not inputs.get(group):raise ValueError('Current retention and validity implementations required')
        for name,expected in inputs[group].items():
            path=Path(name)
            if not path.is_file() or sha(path)!=expected:
                raise ValueError('Native retention implementation or requirements changed')
    for name,expected in inputs.get('preparation_files_sha256',{}).items():
        path=directory/name
        if not path.is_file() or sha(path)!=expected:
            raise ValueError('Native release evidence changed')
    measured=result.get('material_retention',{})
    required={'complete_native_run','native_physical_validity','unforced_observation_duration',
              'resolved_observation_sampling','initial_distribution','retained_distribution',
              'bounded_inside_material_loss','no_cloth_or_body_assistance','actual_drive_release',
              'native_error_check','native_initial_support'}
    checks=measured.get('checks',{})
    if (measured.get('pass') is not True or not required<=checks.keys()
            or not all(value is True for value in checks.values())
            or measured.get('lowest_observed_inside_fraction',0)<=0):
        raise ValueError('Released native material retention did not pass')
    declared=session.config('agent/mechanisms.json',{})['objects'][obj]['links'][link]['initial_support']
    if not declared.get('connected_folds'):
        raise ValueError('Connected-fold proof must match the current declared support relation')
    # Final native task checks still validate opening, actual accessibility,
    # contact grasp and extraction. This proof only establishes released hold.
    return sha(report_path)
