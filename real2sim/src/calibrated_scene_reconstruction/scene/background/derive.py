"""Derive an environment experiment from a frozen task without mutating its parent."""
import argparse
import json
from pathlib import Path
import re
import shutil

from calibrated_scene_reconstruction.scene.review.environment import digest
from calibrated_scene_reconstruction.scene.session import Session


def derive(parent, name, background='blender'):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', name) or background not in ('blender', 'splat'):
        raise ValueError('Simple new session name and blender/splat background required')
    child = Session('sessions/'+name)
    if child.dir.exists() or child.config_dir.exists():
        raise ValueError('Derivative destination already exists; never overwrite')
    freeze_path = parent/'qa/independent_freeze.json'
    frozen = json.loads(freeze_path.read_text())
    for row in frozen['files']:
        root = parent.dir if row['base'] == 'session' else parent.config_dir
        source = (root/row['path']).resolve()
        if not source.is_relative_to(root.resolve()) or digest(source) != row['sha256']:
            raise ValueError('Frozen parent input changed: '+str(source))
    child.dir.mkdir(parents=True)
    child.config_dir.mkdir(parents=True)
    copied = {}
    graph = parent.config('scene_graph.json')
    code_roots = {Path('agent/kits')}
    for spec in [*graph.get('dynamic_objects', {}).values(), *graph.get('furniture', {}).values()]:
        for section in ('fit', 'fix'):
            model = (spec.get(section) or {}).get('model')
            if model:
                code_roots.add(Path(model).parent)

    def copy(source, dest):
        if not source.is_file():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        copied[str(source.resolve())] = digest(source)

    for row in frozen['files']:
        if row['base'] == 'config':
            if row['path'].endswith(('.auto.json', '.prev.json')):
                continue
            relative = Path(row['path'])
            if (relative.parts[0] == 'agent' and relative != Path('agent/task_check.py')
                    and not any(relative.is_relative_to(root) for root in code_roots)):
                continue  # Old capture-specific automation can still target its frozen parent.
            copy(parent.config_dir/row['path'], child.config_dir/row['path'])
        elif row['path'].split('/')[0] in ('capture', 'parsing', 'scene', 'objects', 'placements'):
            copy(parent/row['path'], child/row['path'])
        elif row['path'].startswith('splat/'):
            copy(parent/row['path'], child/row['path'])
    # Calibration sparse tracks and registered images are legal capture evidence,
    # not new task fits. Copy ordinary files rather than sharing mutable links.
    for folder in ('splat/data/sparse', 'splat/data/images'):
        root = parent/folder
        for source in root.rglob('*'):
            if source.is_file():
                copy(source, child/folder/source.relative_to(root))
    graph = child.config('scene_graph.json')
    for spec in [*graph.get('dynamic_objects', {}).values(), *graph.get('furniture', {}).values()]:
        fix = spec.get('fix') or {}
        for key in ('mesh', 'visual_glb'):
            if fix.get(key) and fix[key].startswith(str(parent.dir)+'/'):
                fix[key] = str(child.dir)+'/'+fix[key][len(str(parent.dir))+1:]
    graph['_environment_derived_from'] = {'session': parent.name, 'freeze_sha256': digest(freeze_path),
        'scope': 'Background appearance/geometry only. Existing task/support/robot/calibration fixed; not an independent reconstruction.'}
    (child.config_dir/'scene_graph.json').write_text(json.dumps(graph, indent=2)+'\n')
    twin = child.config('twin.json')
    twin.update(background=background, environment_quality_version=1, cousins=0)
    (child.config_dir/'twin.json').write_text(json.dumps(twin, indent=2)+'\n')
    record = {'parent': parent.name, 'parent_freeze_sha256': digest(freeze_path),
              'copy_mode': 'Separate regular files; no links and no inherited environment approval.',
              'input_sha256': copied, 'task_geometry_refitted': False, 'video_sampling_changed': False,
              'actual_size_evaluation_read': False, 'background': background}
    (child/'qa').mkdir(exist_ok=True)
    (child/'qa/derived_inputs.json').write_text(json.dumps(record, indent=2)+'\n')
    for source, expected in copied.items():
        if digest(source) != expected:
            raise ValueError('Parent changed during copy: '+source)
    return child


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parent')
    parser.add_argument('name')
    parser.add_argument('--background', choices=['blender', 'splat'], default='blender')
    args = parser.parse_args()
    result = derive(Session(args.parent), args.name, args.background)
    print(result.dir)


if __name__ == '__main__':
    main()
