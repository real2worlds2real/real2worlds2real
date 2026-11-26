"""Run and preserve task checks across stated physical-prior samples."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from calibrated_scene_reconstruction.scene.objects.mechanisms import evidence_files, fingerprint

CASES = ('nominal', 'low', 'high')


def preserve(s, result, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for mechanism in result.get('mechanisms', {}).values():
        saved = []
        for name in mechanism.get('evidence_files', []):
            source = (s.dir/name).resolve()
            if not source.is_relative_to(s.dir.resolve()):
                raise ValueError('Physics trial evidence must stay in its own session')
            relative = source.relative_to(s.dir.resolve())
            target = source if source.is_relative_to(destination.resolve()) else destination/'evidence'/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target != source:
                shutil.copyfile(source, target)
            saved.append(str(target.resolve().relative_to(s.dir.resolve())))
        mechanism['evidence_files'] = saved
    result['mechanism_evidence_sha256'] = evidence_files(s, result.get('mechanisms', {}))
    if result.get('image'):
        source = (s.dir/result['image']).resolve()
        if not source.is_relative_to(s.dir.resolve()) or not source.is_file():
            raise ValueError('Trial image must be an existing file inside its session')
        target = destination/'images'/source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        result['image'] = str(target.resolve().relative_to(s.dir.resolve()))
        result['trial_image_sha256'] = {result['image']: hashlib.sha256(target.read_bytes()).hexdigest()}
    path = destination/'task_check.json'
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return path


def run(s, executable, app, background, *, fail_fast=True, first='nominal'):
    if first not in CASES:
        raise ValueError('First physical trial must be nominal, low or high')
    order = (first, *(case for case in CASES if case != first))
    (s/'qa/task_check.json').unlink(missing_ok=True)
    root = s/'qa/physics_trials'
    root.mkdir(parents=True, exist_ok=True)
    n = max([int(p.name[6:]) for p in root.glob('round_*') if p.name[6:].isdigit()], default=0)+1
    destination = root/f'round_{n}'
    destination.mkdir()
    from calibrated_scene_reconstruction.scene.objects.physical_priors import declaration
    (destination/'physics_priors.json').write_text(json.dumps(declaration(s, required=True), indent=2)+'\n')
    initial_fingerprint = fingerprint(s)
    # Preserve the actual declarations and check source, not just a composite
    # hash. Later collider/fixture refinements must not erase failed inputs.
    if hasattr(s, 'config_dir'):
        snapshot = destination/'input_snapshot'
        snapshot.mkdir()
        for name in ('scene_graph.json', 'task.json', 'twin.json', 'simulation.json'):
            path = s.config_dir/name
            if path.exists():
                shutil.copy2(path, snapshot/name)
        from calibrated_scene_reconstruction.core.session_code import source_path
        for name in ('task_check.py', 'task_check_inputs.json', 'mechanisms.json', 'physics_priors.json', 'material_references.json','observation_exclusions.json'):
            path = source_path(s.config_dir, name)
            if path.exists():
                shutil.copy2(path, snapshot/name)
        from calibrated_scene_reconstruction.scene.objects.mechanisms import task_check_inputs
        for name, path in task_check_inputs(s).items():
            target = snapshot/'task_check_inputs'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        (snapshot/'fingerprint.json').write_text(json.dumps({'input_fingerprint': initial_fingerprint})+'\n')
    results, files, reports = {}, {}, {}
    for index, case in enumerate(order):
        # The previous child has exited and released its CUDA context. Check
        # again: another session may have started since the parent launcher.
        from calibrated_scene_reconstruction.core.gpu_lease import wait_for_capacity
        wait_for_capacity()
        out = destination/case
        command=[executable, app, str(s.dir), '--background', background,
                 '--physics-case', case, '--trial-output', str(out.relative_to(s.dir))]
        if getattr(s,'config_dir',None) is not None:command+=['--configuration',str(s.config_dir)]
        rc = subprocess.run(command).returncode
        path = out/'task_check.json'
        if rc or not path.exists():
            results[case] = {'pass': False, 'why': f'Physical trial failed (process={rc}, report={path.exists()})'}
        else:
            result = json.loads(path.read_text())
            results[case] = result
            reports[case] = str(path.relative_to(s.dir))
            files[str(path.relative_to(s.dir))] = hashlib.sha256(path.read_bytes()).hexdigest()
            files.update(result.get('mechanism_evidence_sha256', {}))
            files.update(result.get('trial_image_sha256', {}))
        if fail_fast and results[case].get('pass') is not True:
            for remaining in order[index+1:]:
                results[remaining] = {'pass': False, 'why': f'Not run after {case} failed; all cases are still required'}
            break
    current = fingerprint(s)
    passed = current == initial_fingerprint and all(results.get(c, {}).get('pass') is True
                                                   and results[c].get('input_fingerprint') == current for c in CASES)
    result = results.get('nominal', {'pass': False}).copy()
    result.update(pass_=passed, physical_sensitivity={'pass': passed, 'cases': {k: v.get('pass') is True for k, v in results.items()},
                                                     'design': 'nominal and simultaneous low/high prior bounds; not exhaustive joint uncertainty coverage',
                                                     'property_identification_from_task_success': False},
                  sensitivity_evidence_sha256=files)
    result['physical_sensitivity']['trial_reports'] = reports
    result['physical_sensitivity']['trial_order'] = list(order)
    result['pass'] = result.pop('pass_')
    s.write_json('qa/task_check.json', result)
    print(f'[task_check] {"PASS" if passed else "FAIL"}: physical-prior trials {result["physical_sensitivity"]["cases"]}', flush=True)
    return result
