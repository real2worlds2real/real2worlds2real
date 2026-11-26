"""Append-only process-model rounds and feedback into real2sim modeling."""
import copy
import fcntl
import functools
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time
import tempfile

from calibrated_scene_reconstruction.core.paths import ROOT
from calibrated_scene_reconstruction.physics.process.contract import digest, no_relaxation, read, require, validate, write


def exclusive(function):
    @functools.wraps(function)
    def call(recording, *args, **kwargs):
        root = Path(recording) / 'physics_process'
        root.mkdir(parents=True, exist_ok=True)
        with (root / '.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                return function(recording, *args, **kwargs)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
    return call


def rounds(recording):
    return sorted(p for p in (Path(recording).resolve() / 'physics_process').glob('round_[0-9][0-9][0-9][0-9]') if p.is_dir())


def current(recording):
    found = rounds(recording)
    require(bool(found), 'Agent must propose a process plan first')
    return found[-1]


def event(recording, kind, **data):
    root = Path(recording) / 'physics_process'
    root.mkdir(parents=True, exist_ok=True)
    import json
    with (root / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps({'time': time.time(), 'event': kind, **data}, allow_nan=False) + '\n')


@exclusive
def propose(recording, plan, *, input_root=ROOT):
    plan = copy.deepcopy(plan)
    inputs = validate(plan, input_root)
    previous = rounds(recording)
    if previous:
        last = previous[-1]
        old = read(last / 'plan.json')
        revision = plan.get('revision', {})
        require(revision.get('parent_round') == int(last.name[6:]) and revision.get('reason')
                and revision.get('hypothesis') and revision.get('changes'),
                'Revision must explain the previous round, hypothesis and actual changes')
        no_relaxation(old, plan)
        import numpy as np
        with np.load(last / 'observations.npz') as before, np.load(inputs[plan['observations']]['path']) as after:
            indices = old['comparison']['fit_indices'] + old['comparison']['validation_indices']
            require(np.allclose(before['t'][indices], after['t'][indices], rtol=0, atol=1e-9),
                    'Existing observation indices must retain their original timestamps')
        require(plan['model'] != old['model'] or plan['parameters'] != old['parameters']
                or inputs != read(last / 'inputs.json') or plan['channels'] != old['channels']
                or plan['search'] != old['search'] or plan['simulation'] != old['simulation']
                or any(not Path(p).is_file() or digest(p) != h for p,h in read(last/'harness.json').items()),
                'Unchanged failed model: revise the model, observations or parameters')
        if (last / 'geometry_request.json').exists():
            prior = read(last / 'inputs.json')
            changed = any(k not in prior or v['sha256'] != prior[k]['sha256']
                          for k, v in inputs.items() if v['kind'] in {'geometry', 'model_source'})
            require(changed, 'Pending geometry request requires changed, newly reviewed model sources')
    number = int(previous[-1].name[6:]) + 1 if previous else 1
    out = Path(recording).resolve() / 'physics_process' / f'round_{number:04d}'
    require(not out.exists(), 'Round already exists')
    with tempfile.TemporaryDirectory(prefix='.planning_',dir=out.parent) as temporary:
        snapshot_round(Path(temporary),plan,inputs)
        Path(temporary).rename(out)
    event(recording, 'planned', round=number)
    return status(recording)


def snapshot_round(out, plan, inputs):
    write(out / 'plan.json', plan)
    write(out / 'inputs.json', inputs)
    shutil.copy2(inputs[plan['observations']]['path'], out / 'observations.npz')
    for name, item in inputs.items():
        if item['kind'] in {'model_source','model_review'}:
            require(Path(name).name == name, 'Input identifiers must not contain directories')
            directory = out / 'source_inputs'
            directory.mkdir(exist_ok=True)
            shutil.copy2(item['path'], directory / (name + Path(item['path']).suffix))
    from calibrated_scene_reconstruction.physics.process import contract, geometry, resources, search, initial_motion, region_observations, driver_guard, mass_bindings, mask_raster, fit_priority
    from calibrated_scene_reconstruction.physics.replay import data, optim, params
    write(out / 'harness.json', {str(Path(m.__file__).resolve()): digest(m.__file__)
                               for m in (contract, geometry, search, resources, initial_motion, region_observations, driver_guard, mass_bindings, mask_raster, fit_priority, data, optim, params)} |
          {str(Path(__file__).resolve()): digest(__file__)} |
          {str(ROOT/p): digest(ROOT/p) for p in ('apps/physics/process.py','apps/physics/physics_only.kit','tools/run_shared_isaac.sh',
              'tools/run_isaac.sh','src/calibrated_scene_reconstruction/sim/physics.py','src/calibrated_scene_reconstruction/sim/tabletop.py',
              'src/calibrated_scene_reconstruction/sim/robot_articulation.py','src/calibrated_scene_reconstruction/apps/physics/process_native.py',
              'src/calibrated_scene_reconstruction/apps/physics/process_models.py','src/calibrated_scene_reconstruction/apps/physics/process_initial_rest.py',
              'src/calibrated_scene_reconstruction/sim/process_world.py','src/calibrated_scene_reconstruction/core/records.py',
              'src/calibrated_scene_reconstruction/apps/physics/process_scene.py','src/calibrated_scene_reconstruction/physics/process/material_projection.py')})
    snapshot = ['plan.json', 'inputs.json', 'harness.json', 'observations.npz']
    snapshot += [str(p.relative_to(out)) for p in (out/'source_inputs').glob('*')]
    write(out / 'snapshot.json', {p: digest(out / p) for p in snapshot})


def intact(out):
    inputs = read(out / 'inputs.json')
    hashes = {v['path']: v['sha256'] for v in inputs.values()}
    hashes.update(read(out / 'harness.json'))
    hashes.update({str(out / p): h for p, h in read(out / 'snapshot.json').items()})
    if (out / 'result_hashes.json').exists():
        hashes.update({str(out / p): h for p, h in read(out / 'result_hashes.json').items()})
    if (out / 'review_hash.json').exists():
        hashes[str(out / 'review.json')] = read(out / 'review_hash.json')['sha256']
        for p, h in read(out / 'review.json').get('evidence', {}).items():
            path = Path(p)
            hashes[str(path if path.is_absolute() else out / path)] = h
    return all(Path(p).is_file() and digest(p) == h for p, h in hashes.items())


def status(recording):
    previous = rounds(recording)
    if not previous:
        return {'state': 'missing', 'passed': False, 'next_action': 'agent_propose_model_and_parameters'}
    out = previous[-1]
    state, next_action, passed = 'planned', 'run', False
    try:
        unchanged = intact(out)
    except (OSError,ValueError,KeyError,TypeError):
        unchanged = False
    if not unchanged:
        state, next_action = 'stale', 'agent_revise_plan_and_review_changed_inputs'
    elif (out / 'geometry_request.json').exists():
        state, next_action = 'geometry_requested', 'real2sim_rebuild_and_review_then_propose_new_round'
    elif (out / 'observation_request.json').exists():
        state, next_action = 'observations_requested', 'agent_add_observations_or_record_a_justified_prior_only_model'
    elif (out / 'execution_failure.json').exists():
        state, next_action = 'failed', 'inspect_execution_failure_then_revise'
    elif (out / 'review.json').exists():
        passed = read(out / 'review.json')['pass']
        state, next_action = ('passed', 'consume_physics') if passed else ('rejected', 'agent_revise_plan')
    elif (out / 'result.json').exists():
        if not (out / 'result_hashes.json').exists():
            state, next_action = 'interrupted', 'unsealed_native_result_requires_a_new_round'
        else:
            good = read(out / 'result.json').get('pass') is True
            screening=read(out/'plan.json')['simulation'].get('purpose')=='screening'
            state, next_action = (('awaiting_validation','agent_run_finer_native_validation') if screening else
                                 ('awaiting_review', 'agent_review_native_evidence')) if good else ('failed', 'agent_diagnose_and_revise_model_or_parameters')
    elif (out / 'started.json').exists():
        launches = sorted((out / 'launches').glob('[0-9][0-9][0-9][0-9].json'))
        if launches and read(launches[-1]).get('returncode') == 75:
            state, next_action = 'deferred', 'run_when_shared_gpu_has_headroom'
        else:
            with (out.parent/'.lock').open('a') as lock:
                try:
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    fcntl.flock(lock,fcntl.LOCK_UN)
                    state, next_action = 'interrupted', 'inspect_log_then_create_new_round'
                except BlockingIOError:
                    state, next_action = 'running', 'inspect_native_log_and_history_without_changing_sources'
    value = {'state': state, 'passed': passed, 'round': int(out.name[6:]), 'path': str(out), 'next_action': next_action}
    if unchanged and (out / 'result.json').exists():
        value['findings'] = read(out / 'result.json').get('findings', [])
    if (out / 'geometry_request.json').exists():
        value['geometry_request'] = read(out / 'geometry_request.json')
    if (out / 'observation_request.json').exists():
        value['observation_request'] = read(out / 'observation_request.json')
    return value


@exclusive
def run(recording):
    out = current(recording)
    require(status(recording)['state'] in {'planned', 'deferred'}, 'Only an intact, unexecuted or deferred plan can run')
    plan = read(out / 'plan.json')
    if not (out / 'started.json').exists():
        write(out / 'started.json', {'time': time.time()})
    launches = out / 'launches'
    launches.mkdir(exist_ok=True)
    attempt = len(list(launches.glob('*.log'))) + 1
    environment={**os.environ,'SCENE_RECONSTRUCTION_PROCESS_ATTEMPT':str(attempt)}
    resources=plan['simulation'].get('resources')
    if resources:
        environment.update(SCENE_RECONSTRUCTION_ISAAC_EXPECTED_VRAM_MIB=str(resources['expected_vram_mib']),
                           SCENE_RECONSTRUCTION_SHARED_GPU_RESERVE_MIB=str(resources['shared_headroom_mib']))
    command = [str(ROOT / 'tools/run_shared_isaac.sh'), str(ROOT / 'apps/physics/process.py'), str(out.resolve())]
    child = None
    try:
        with (launches / f'{attempt:04d}.log').open('x') as stream:
            child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,env=environment)
            code = child.wait(timeout=plan['search']['timeout_seconds'])
        write(launches / f'{attempt:04d}.json', {'returncode': code})
        if code == 75 and not (out / 'result.json').exists() and not (out / 'trials').exists():
            event(recording, 'gpu_deferred', attempt=attempt)
            return status(recording)
        require(code == 0, f'Native process failed ({code}); inspect {launches / f"{attempt:04d}.log"}')
        require((out / 'result.json').exists(), 'Native worker did not produce a result')
        require(intact(out), 'Inputs or harness changed during native execution')
        seal_result(out)
    except BaseException as exc:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        if not (out / 'result.json').exists():
            write(out / 'result.json', {'pass': False, 'findings': [
                {'id': 'execution', 'kind': 'execution', 'reason': str(exc)}]})
            seal_result(out)
        if not (out / 'execution_failure.json').exists():
            write(out / 'execution_failure.json', {'reason': str(exc), 'pass': False})
        event(recording, 'execution_failed', error=str(exc))
        raise
    event(recording, 'executed', round=out.name, passed=read(out / 'result.json').get('pass') is True)
    return status(recording)


def seal_result(out):
    out = Path(out)
    files = [p for p in out.rglob('*') if p.is_file() and
             p.name not in {'result_hashes.json', 'review.json', 'geometry_request.json'}]
    write(out / 'result_hashes.json', {str(p.relative_to(out)): digest(p) for p in files})


@exclusive
def request_geometry(recording, request):
    out = current(recording)
    require(intact(out), 'Cannot diagnose stale inputs')
    ids = {b['id'] for b in read(out / 'plan.json')['model']['bodies']}
    require(request.get('reason') and request.get('changes') and request.get('bodies')
            and set(request['bodies']) <= ids, 'Specific known bodies, evidence-based reason and changes required')
    require((out / 'result.json').exists(), 'Inspect a native fit before requesting geometry changes')
    write(out / 'geometry_request.json', {**request, 'round': out.name,
          'result_sha256': digest(out / 'result.json'), 'consumer': 'real2sim_modeling_loop'})
    event(recording, 'geometry_requested', bodies=request['bodies'])
    return status(recording)


@exclusive
def request_observations(recording, request):
    out = current(recording)
    require(intact(out) and (out/'result.json').exists(), 'Diagnose an intact executed round first')
    require(request.get('reason') and request.get('needed') and request.get('consequence'),
            'Explain the missing observation, why it is needed and what remains uncertain without it')
    write(out/'observation_request.json', {**request,'result_sha256':digest(out/'result.json')})
    event(recording,'observations_requested',reason=request['reason'])
    return status(recording)


@exclusive
def review(recording, record):
    out = current(recording)
    require(status(recording)['state'] in {'awaiting_review', 'awaiting_validation', 'failed'}, 'A current executed round is required')
    require(type(record.get('pass')) is bool and record.get('reason') and record.get('checks')
            and record.get('result_sha256') == digest(out / 'result.json'), 'Review must bind the current result')
    if record['pass']:
        require(read(out/'plan.json')['simulation'].get('purpose','validation')=='validation',
                'Screening cannot be approved; run the declared finer native validation')
        result = read(out / 'result.json')
        require(result.get('pass') is True, 'Cannot approve a numerically failed process')
        require(result.get('engine', {}).get('name') == 'isaacsim'
                and str(result['engine'].get('version', '')).split('.')[:2] == ['5', '1']
                and result.get('best', {}).get('artifacts'), 'Accepted process requires native Isaac Sim evidence')
        launches = sorted((out / 'launches').glob('[0-9][0-9][0-9][0-9].json'))
        require((out / 'started.json').is_file() and launches and read(launches[-1]).get('returncode') == 0,
                'A successful native harness launch is required; a hand-written result is not execution evidence')
        required = read(out / 'plan.json')['model']['required_checks']
        physical = result['best'].get('physical_checks', {})
        require(set(required) <= set(physical)
                and all(v.get('pass') is True and v.get('reason') for v in physical.values()),
                'Every planned physical plausibility check must pass')
        artifacts = result['best']['artifacts']
        for suffix in ('native_initial.usdc', 'native_states.npz', 'predictions.npz'):
            require(any(p.endswith(suffix) for p in artifacts), 'Missing native artifact: '+suffix)
        for relative, expected in artifacts.items():
            path = (out / relative).resolve()
            require(path.is_relative_to(out.resolve()) and path.is_file() and digest(path) == expected,
                    'Native trial artifact changed or escaped its round')
        from calibrated_scene_reconstruction.physics.process.contract import compare
        import numpy as np
        plan = read(out / 'plan.json')
        inputs = read(out / 'inputs.json')
        observations = dict(np.load(inputs[plan['observations']]['path'], allow_pickle=False))
        predicted = dict(np.load(out / next(p for p in artifacts if p.endswith('predictions.npz')), allow_pickle=False))
        require(compare(plan, observations, predicted) == result['best']['metrics'],
                'Reported process metrics differ from saved predictions')
        require({'model_semantics','physical_plausibility','whole_process','parameter_evidence'} <= set(record['checks'])
                and all(c.get('pass') is True and c.get('reason') for c in record['checks'].values()),
                'Every process plausibility/appearance check must pass with a reason')
        require(record.get('evidence'), 'Review must bind the inspected native evidence')
        for relative, expected in record['evidence'].items():
            file = (out / relative).resolve()
            require(file.is_relative_to(out.resolve()) and file.is_file() and digest(file) == expected,
                    'Reviewed native evidence must belong to this exact round')
    write(out / 'review.json', record)
    write(out / 'review_hash.json', {'sha256': digest(out / 'review.json')})
    event(recording, 'reviewed', passed=record['pass'])
    return status(recording)


def accepted(recording):
    require(status(recording)['passed'], 'Process is not current and accepted')
    out = current(recording)
    result = read(out / 'result.json')
    return {'parameters': result['parameter_evidence'], 'model': read(out / 'plan.json')['model'],
            'parameter_values': result['best']['parameters'],
            'native_artifacts': result['best']['artifacts'], 'round': str(out),
            'native_scene': str(out / next(p for p in result['best']['artifacts'] if p.endswith('native_initial.usdc'))),
            'simulation': read(out / 'plan.json')['simulation'], 'inputs': read(out / 'inputs.json'),
            'limitations': result['validation_scope']}


@exclusive
def attach(recording, session):
    """Publish a read-only result contract; never overwrite a frozen scene asset."""
    from calibrated_scene_reconstruction.core.session import Session
    target = Session(session)
    recording = Path(recording).resolve()
    process = accepted(recording)
    out = current(recording)
    inputs = read(out / 'inputs.json')
    geometry = [Path(v['path']).resolve() for v in inputs.values() if v['kind'] == 'geometry']
    require(any(p.is_relative_to(target.dir.resolve()) for p in geometry),
            'Process must bind geometry from this real2sim scene before attachment')
    files = {v['path']: v['sha256'] for v in inputs.values()}
    files.update({str(out / p): digest(out / p) for p in
                  ('plan.json', 'inputs.json', 'snapshot.json', 'harness.json', 'result.json',
                   'result_hashes.json', 'review.json', 'review_hash.json')})
    files.update(read(out / 'harness.json'))
    for path, h in read(out/'review.json').get('evidence',{}).items():
        files[str((out/path).resolve())]=h
    files.update({str(out / p): h for p, h in read(out / 'result_hashes.json').items()})
    files[str(Path(recording) / 'physics_process/events.jsonl')] = digest(Path(recording) / 'physics_process/events.jsonl')
    marker = target / 'qa/physics_process_required.json'
    if not marker.exists():
        write(marker, {'version': 1, 'recording': str(Path(recording).resolve())})
    temporary = target / 'qa' / ('.process_receipt_' + str(time.time_ns()) + '.json')
    write(temporary, {'version': 1, 'pass': True, 'round': str(out), 'files': files, 'process': process})
    os.replace(temporary, target / 'qa/physics_process.json')
    return {'state': 'attached', 'passed': True, 'path': str(target / 'qa/physics_process.json')}
