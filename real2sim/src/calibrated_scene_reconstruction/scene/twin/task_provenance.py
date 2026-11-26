"""Bind ordinary task checks to the actual session, configuration and candidate.

Mechanism checks retain their more detailed contract. This shared producer and
reader also cover rigid tasks without a mechanism declaration, so moving an old
passing JSON or changing its upstream geometry cannot preserve a current pass.
"""
import hashlib
from pathlib import Path

from calibrated_scene_reconstruction.core.paths import ROOT


ENGINES = (
    'apps/scene/task_check.py', 'src/calibrated_scene_reconstruction/scene/twin/task_provenance.py',
    'src/calibrated_scene_reconstruction/scene/twin/runtime_inputs.py', 'src/calibrated_scene_reconstruction/scene/support/runtime_inputs.py',
    'src/calibrated_scene_reconstruction/sim/scene_stage.py', 'src/calibrated_scene_reconstruction/sim/scene_physics.py',
    'src/calibrated_scene_reconstruction/sim/physics.py', 'src/calibrated_scene_reconstruction/sim/physics_errors.py',
    'src/calibrated_scene_reconstruction/sim/object_materials.py', 'src/calibrated_scene_reconstruction/assets/metric_mesh.py',
    'src/calibrated_scene_reconstruction/sim/rigid_fixture.py',
    'src/calibrated_scene_reconstruction/assets/object_pose.py', 'src/calibrated_scene_reconstruction/tasks/rigid/task_checks.py',
    'src/calibrated_scene_reconstruction/tasks/rigid/check_script.py',
)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def snapshot(session):
    """Read current inputs only; QA verdicts and generated outputs are excluded."""
    repository = Path(ROOT).resolve()
    files = {}

    def pin(path):
        path = Path(path).resolve()
        if not path.is_relative_to(repository) or not path.is_file():
            raise ValueError('Task-check input must exist inside its current repository: ' + str(path))
        files[str(path)] = digest(path)

    for path in sorted(session.config_dir.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
            pin(path)
    for required in ('scene_graph.json', 'task.json'):
        pin(session.config_dir / required)
    from calibrated_scene_reconstruction.core.session_code import source_path
    pin(source_path(session.config_dir, 'task_check.py'))
    for group in ('dynamic_objects', 'furniture'):
        for spec in session.scene_graph.get(group, {}).values():
            fix = spec.get('fix') or {}
            for key in ('mesh', 'visual_glb'):
                if fix.get(key):
                    pin(repository / fix[key])
            for key in ('mesh', 'albedo', 'photometric', 'material_finishes'):
                value = (fix.get('look') or {}).get(key)
                if value:
                    pin(session.dir / value)
    for name in ('parsing/world_frame.json', 'splat/align.json', 'splat/robot_visual_fit.json',
                 'splat/robot_joint_fit.json', 'splat/robot_fit.json', 'scene/manifest.json'):
        path = session.dir / name
        if path.is_file():
            pin(path)
    for name in ENGINES:
        pin(repository / name)
    from calibrated_scene_reconstruction.scene.objects.mechanisms import task_check_inputs
    for path in task_check_inputs(session).values():
        pin(path)
    return {'version': 1, 'session': str(session.dir.resolve()),
            'configuration': str(session.config_dir.resolve()), 'input_sha256': files}


def current(session, result):
    """Validate producer-owned inputs and every additional script-supplied hash."""
    recorded = result.get('task_input_provenance')
    if not isinstance(recorded, dict) or recorded != snapshot(session):
        raise ValueError('Task evidence lacks current session/configuration/geometry provenance; rerun task_check')
    if result.get('task_inputs_unchanged') is not True:
        raise ValueError('Task inputs changed during native execution')
    extra = result.get('input_sha256', {})
    if not isinstance(extra, dict):
        raise ValueError('Task script input_sha256 must be a file-hash mapping')
    for name, expected in extra.items():
        path = Path(name)
        path = (path if path.is_absolute() else Path(ROOT) / path).resolve()
        if not path.is_relative_to(Path(ROOT).resolve()) or not path.is_file() or digest(path) != expected:
            raise ValueError('Task script input changed or belongs to another repository: ' + str(name))
    return True
