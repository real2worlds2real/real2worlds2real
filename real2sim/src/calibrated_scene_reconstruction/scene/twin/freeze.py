"""Freeze accepted reconstruction files before separate evaluation or background work.

    python -m calibrated_scene_reconstruction.scene.twin.freeze sessions/<name> [--verify]

The manifest is append-forbidden. Verification reads the saved files only and
never rewrites reviews, timestamps, or reconstruction artifacts.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from calibrated_scene_reconstruction.scene.session import Session
from calibrated_scene_reconstruction.scene.twin.gates import status


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def verify(session):
    manifest = session/'qa/independent_freeze.json'
    saved = json.loads(manifest.read_text())
    roots = {'session': session.dir.resolve(), 'config': session.config_dir.resolve()}
    changed = []
    for row in saved['files']:
        root = roots[row['base']]
        path = (root/row['path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Frozen file must remain within its recorded root')
        if not path.is_file() or digest(path) != row['sha256']:
            changed.append({'base': row['base'], 'path': row['path']})
    return {'unchanged': not changed, 'files': len(saved['files']), 'changed': changed}


def freeze(session):
    manifest = session/'qa/independent_freeze.json'
    if manifest.exists():
        raise ValueError('Freeze already exists; verify it or create a derivative session')
    state = status(session)
    if not state['passed']:
        raise ValueError('Freeze requires all applicable gates and object/support reviews to pass')
    roots = {'session': session.dir.resolve(), 'config': session.config_dir.resolve()}
    files = {}

    def collect(base, path):
        paths = sorted(path.rglob('*')) if path.is_dir() else [path]
        for source in paths:
            if source == manifest or '__pycache__' in source.parts or source.suffix in ('.pyc', '.tmp'):
                continue
            if source.is_file():
                relative = str(source.relative_to(roots[base]))
                if not source.resolve().is_relative_to(roots[base]):
                    raise ValueError('Do not freeze links to mutable files outside this session')
                files[base, relative] = {'base': base, 'path': relative, 'sha256': digest(source)}

    collect('config', roots['config'])
    for name in ('capture', 'parsing', 'scene', 'export', 'placements', 'renders', 'qa'):
        collect('session', roots['session']/name)
    graph = session.scene_graph
    for name, spec in graph.get('dynamic_objects', {}).items():
        fix = spec.get('fix') or {}
        if fix.get('model'):
            directory = roots['session']/'objects'/name/f"round_{fix['round']}"
            collect('session', directory)
            result = json.loads((directory/'fit.json').read_text())
            if result.get('fixed_candidate_revalidation'):
                from calibrated_scene_reconstruction.scene.objects.blender.revalidate import verify as verify_lineage
                lineage = verify_lineage(session, name, directory, result)
                # A new round refers to immutable inherited evidence. Freeze
                # those verified dependencies too, not only the new pointer.
                dependencies = [lineage['inherited_freeze'], *lineage['parent_files'], *lineage['look_files']]
                for relative in dependencies:
                    path = (roots['session']/relative).resolve()
                    if not path.is_relative_to(roots['session']) or not path.is_file():
                        raise ValueError('Verified recovery lineage must remain inside this session')
                    collect('session', path)
        for suffix in ('.npz', '.start.json'):
            collect('session', roots['session']/'splat/object_fit'/(name+suffix))
    support = graph.get('task', {}).get('support')
    if support:
        fix = graph.get('furniture', {}).get(support, {}).get('fix') or {}
        if fix.get('model'):
            collect('session', roots['session']/'support'/support/f"round_{fix['round']}")
    for name in ('align.json', 'support_carve.json', 'support_marks.json', 'object_fit.json',
                 'robot_visual_fit.json', 'robot_joint_fit.json', 'robot_fit.json', 'object_refine.json'):
        collect('session', roots['session']/'splat'/name)
    collect('session', roots['session']/'splat/data/sparse/full')
    record = {'version': 1, 'frozen_at': datetime.now(timezone.utc).isoformat(),
              'code': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'basis': 'Current capture, legal calibration and accepted reviews. True task dimensions '
                       'are not reconstruction inputs; any later evaluation must not alter this delivery.',
              'status': state, 'files': list(files.values())}
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    return {'freeze': str(manifest), 'files': len(files), 'sha256': digest(manifest)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    result = (verify if args.verify else freeze)(Session(args.session))
    print(json.dumps(result, indent=2))
    if args.verify and not result['unchanged']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
