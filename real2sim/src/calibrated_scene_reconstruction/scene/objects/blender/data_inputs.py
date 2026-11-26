"""Declared model data travels with immutable Blender source snapshots."""
from pathlib import Path
import shutil

from calibrated_scene_reconstruction.core.session_code import source_path


def files(session, obj, model):
    if not hasattr(session, 'scene_graph'):
        return {}  # Lightweight legacy mesh clients do not declare data assets.
    declared = (session.scene_graph['dynamic_objects'][obj].get('fit') or {}).get('assets', [])
    if not isinstance(declared, list) or any(not isinstance(v, str) for v in declared):
        raise ValueError('Model assets must be a list of config-relative file paths')
    root = Path(model).resolve().parent
    result = {}
    for item in declared:
        path = source_path(session.config_dir, item).resolve()
        if not path.is_file() or not path.is_relative_to(root):
            raise ValueError('Declared model assets must be files under the model directory')
        relative = path.relative_to(root)
        if len(relative.parts) < 2 or relative.parts[0] != 'assets':
            raise ValueError('Put model data under its assets/ subdirectory, separate from round outputs')
        if path in result:
            raise ValueError('Duplicate model asset')
        result[path] = relative
    return result


def snapshot(session, obj, model, output):
    result = files(session, obj, model)
    for path, relative in result.items():
        destination = Path(output)/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    return result
