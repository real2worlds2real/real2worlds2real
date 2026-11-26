"""Full workflow JSON tools; stage execution and human/model review are separate.

    python -m calibrated_scene_reconstruction.agent.tool schemas
    python -m calibrated_scene_reconstruction.agent.tool status --request request.json --output result_directory
"""
import argparse
import contextlib
import json
import re
import os
import signal
import subprocess
import sys
from functools import cache
from pathlib import Path

from calibrated_scene_reconstruction.core.paths import ROOT, rel, resolve
from calibrated_scene_reconstruction.scene.session import Session
from calibrated_scene_reconstruction.scene.twin.gates import options, status
from calibrated_scene_reconstruction.scene.capture.inputs import validate_session
from calibrated_scene_reconstruction.agent.support_tool import DESCRIPTIONS as SUPPORT_DESCRIPTIONS, execute as support_execute
from calibrated_scene_reconstruction.agent.process_tool import DESCRIPTIONS as PHYSICS_DESCRIPTIONS, execute as physics_execute
from .workflow_schemas import STAGES, object_contracts, schemas


@cache
def descriptions():
    try:
        from .object_tool import DESCRIPTIONS as objects
    except ModuleNotFoundError as exc:
        if exc.name not in {'torch', 'trimesh', 'scipy', 'pxr'}:
            raise
        # MCP also serves schema-inspection clients on hosts without the object execution backend.
        objects = {name: f"Inspect the {name} contract. Running this object tool requires {exc.name}."
                   for name in object_contracts()}
    return {
        **{'stage_' + k.lower(): f'Execute {k}; return current gate state without approving its review.' for k in STAGES},
        **{'object_' + k: v for k, v in objects.items()},
        **{'support_' + k: v for k, v in SUPPORT_DESCRIPTIONS.items()},
        **{'physics_' + k: v for k, v in PHYSICS_DESCRIPTIONS.items()},
        'status': 'Read all current stage and object gates.',
        'record': 'Append an explicit review round without replacing earlier records.',
        'report': 'Build the automatic observation-based check report.',
    }


def __getattr__(name):
    # Preserve the description mapping imported by MCP clients without initializing the execution backends for schemas.
    if name == 'DESCRIPTIONS':
        return descriptions()
    raise AttributeError(name)


def record(session, review, output):
    stage = review['stage']
    if stage == 'S6' and review['pass']:
        require_delivery_inputs(session)
    rounds = []
    for path in (session / 'qa/review').glob(stage + '_r*.json'):
        match = re.fullmatch(re.escape(stage) + r'_r(\d+)\.json', path.name)
        if match:
            rounds.append(int(match.group(1)))
    review = {**review, 'round': review.get('round', max(rounds, default=0) + 1)}
    target = session / f"qa/review/{stage}_r{review['round']}.json"
    if target.exists():
        raise ValueError('Review round already exists; append a new round')
    request = output / 'review_request.json'
    request.write_text(json.dumps(review, indent=2) + '\n')
    with (output / 'record.log').open('w') as log:
        subprocess.run([sys.executable, '-m', 'calibrated_scene_reconstruction.scene.twin.gates', str(session.dir), 'record', str(request)],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    return {'ok': True, 'record': rel(target), 'status': status(session), 'evidence_images': []}


def require_delivery_inputs(session):
    current = status(session)
    unmet = [k for k, v in current['stages'].items() if k != 'S6' and v['applicable'] and not v['passed']]
    unmet += ['OBJ.' + k for k, v in current['objects'].items() if not v['passed']]
    unmet += ['SUP.' + k for k, v in current.get('supports', {}).items() if not v['passed']]
    if not current.get('physics_process', {}).get('passed', True):
        unmet.append('PHYSICS')
    if unmet:
        raise ValueError('Delivery prerequisites failed: ' + ', '.join(unmet))


def execute(name, request, output):
    import jsonschema
    from . import store

    registry = schemas()
    if name not in registry:
        raise ValueError('Unsupported tool: ' + str(name))
    schema = registry[name]
    jsonschema.validate(request, schema['input'])
    session = Session(resolve(request['session']))
    request = {**request, 'session': str(session.dir)}
    if name.startswith(('stage_', 'object_', 'support_')):
        validate_session(session)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    with (contextlib.nullcontext() if name == 'physics_status' else store.session_lock(session)):
        run = store.start(session, name, request)
        try:
            if name.startswith('object_'):
                from .object_tool import DESCRIPTIONS, execute as object_execute

                if name[7:] not in DESCRIPTIONS:
                    raise ValueError('Unsupported tool: ' + name)
                result = object_execute(name[7:], request, out)
            elif name.startswith('support_'):
                result = support_execute(name[8:], request, out)
            elif name.startswith('physics_'):
                result = physics_execute(name[8:], request, out)
            elif name == 'status':
                state = status(session)
                result = {'ok': True, 'status': state, 'evidence_images': []}
            elif name == 'record':
                result = record(session, request['review'], out)
            elif name == 'report':
                with (out / 'report.log').open('w') as log:
                    subprocess.run([sys.executable, '-m', 'calibrated_scene_reconstruction.scene.twin.report', str(session.dir)],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                result = {'ok': True, 'report': rel(session / 'qa/report.json'), 'evidence_images': []}
            else:
                stage = name[6:].upper()
                if stage == 'S6':
                    require_delivery_inputs(session)
                first, last = STAGES[stage]
                opts = options(session)
                if stage == 'S3' and opts and opts['background'] == 'zone':
                    result = {'ok': True, 'applicable': False, 'status': status(session), 'evidence_images': []}
                else:
                    if stage == 'S3' and opts and opts['background'] == 'splat':
                        first, last = 'splat_masks', 'splat_review'
                    elif stage == 'S3' and opts and opts['background'] == 'blender':
                        first, last = 'environment_build', 'environment_build'
                    if session.dir.resolve() != (ROOT / 'sessions' / session.name).resolve():
                        raise ValueError('Shell stage execution requires a repository session directory')
                    command = [str(ROOT / 'pipeline/twin.sh'), session.name, '--from', first, '--to', last,
                               '--object-route', request.get('object_route', 'agent_blender')]
                    if request.get('robot'):
                        command += ['--robots', request['robot']]
                    if request.get('reuse_camera_model'):
                        command += ['--reuse-camera-model']
                    if 'pose_workers' in request:
                        command += ['--pose-workers', str(request['pose_workers'])]
                    with (out / 'stage.log').open('w') as log:
                        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                                 start_new_session=True)
                        try:
                            code = child.wait(timeout=request.get('timeout', 86400))
                        except BaseException:
                            # This process group belongs only to the stage launched above.
                            os.killpg(child.pid, signal.SIGTERM)
                            try:
                                child.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                os.killpg(child.pid, signal.SIGKILL)
                                child.wait(timeout=10)
                            raise
                        if code:
                            raise subprocess.CalledProcessError(code, command)
                    result = {'ok': True, 'stage': stage, 'status': status(session), 'evidence_images': []}
            result['run'] = rel(run)
            jsonschema.validate(result, schema['output'])
            (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
            (run / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
            store.event(run, 'completed', ok=result['ok'])
            return result
        except Exception as exc:
            result = {'ok': False, 'error': str(exc), 'run': rel(run), 'evidence_images': []}
            (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
            (run / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
            store.event(run, 'failed', error=str(exc))
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(prog="calibrated-scene-reconstruction", description=__doc__)
    parser.add_argument('name', choices=['schemas', *schemas()])
    parser.add_argument('--request')
    parser.add_argument('--output')
    args = parser.parse_args(argv)
    if args.name == 'schemas':
        print(json.dumps(schemas(), indent=2))
        return
    if not args.request or not args.output:
        parser.error('--request and --output are required')
    result = execute(args.name, json.loads(Path(args.request).read_text()), args.output)
    print(json.dumps({'ok': result['ok'], 'result': str(Path(args.output) / 'result.json')}))
    if not result['ok']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
