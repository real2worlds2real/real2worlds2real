"""Exclusive tool runs with input fingerprints and append-only events."""
import contextlib
import fcntl
import json
import time
import uuid
from pathlib import Path

from calibrated_scene_reconstruction.assets.model_artifact import digest


def fingerprint(session, *, include_camera_model=False):
    """Current input files, including pose initializers; never read development evals."""
    paths = set(session.config_dir.glob('*.json'))
    paths.update(session.config_dir.glob('agent/**/*.py'))
    for directory in ('capture', 'parsing', 'splat/object_fit'):
        root = session / directory
        paths.update(p for p in root.rglob('*') if p.is_file()
                     and p.suffix in ('.json', '.npz', '.png', '.jpg'))
    for relative in ('splat/align.json', 'splat/object_fit.json'):
        if (session / relative).is_file():
            paths.add(session / relative)
    if include_camera_model:
        for relative in ('splat/data/sparse/full', 'splat/data/images'):
            paths.update(p for p in (session / relative).rglob('*') if p.is_file())
        if (session / 'splat/frames.json').is_file():
            paths.add(session / 'splat/frames.json')
    keyframes = session / 'capture/keyframes.json'
    if keyframes.is_file():
        from calibrated_scene_reconstruction.core.paths import resolve
        meta = json.loads(keyframes.read_text())
        if isinstance(meta.get('video'), str):
            paths.add(resolve(meta['video']))
    splat = session.config('splat.json', {})
    if splat.get('videos'):
        from calibrated_scene_reconstruction.core.paths import resolve
        paths.update(resolve(video) for video in splat['videos'])
    return {str(p): digest(p) for p in sorted(paths) if '__pycache__' not in p.parts}


@contextlib.contextmanager
def session_lock(session):
    root = session / 'agent_runs'
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def start(session, tool, request):
    run_id = uuid.uuid4().hex
    root = session / 'agent_runs' / run_id
    root.mkdir(parents=True)
    with (root / 'input.json').open('x') as stream:
        inputs = fingerprint(session, include_camera_model=request.get('reuse_camera_model', False))
        json.dump({'tool': tool, 'request': request, 'inputs': inputs}, stream, indent=2)
        stream.write('\n')
    event(root, 'started', tool=tool)
    return root


def event(root, kind, **data):
    root = Path(root)
    row = {'time': time.time(), 'run': root.name, 'event': kind, **data}
    for path in (root / 'events.jsonl', root.parent / 'events.jsonl'):
        with path.open('a') as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
