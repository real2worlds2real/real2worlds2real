"""Opt-in rigid source initialization on the actual modeled support triangles."""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from calibrated_scene_reconstruction.assets.object_pose import rotation
from calibrated_scene_reconstruction.assets.support_placement import TriangleSurface, on_triangles
from calibrated_scene_reconstruction.core.paths import resolve
from .store import digest


def request(session, obj):
    value = session.scene_graph['dynamic_objects'][obj]['fit'].get('support_placement')
    if value is None:
        return None
    if (not isinstance(value, dict) or set(value) != {'mode', 'sample_spacing_m'}
            or value['mode'] != 'actual_triangles'
            or isinstance(value['sample_spacing_m'], bool)
            or not isinstance(value['sample_spacing_m'], (float, int))
            or not np.isfinite(value['sample_spacing_m'])
            or not .0001 <= value['sample_spacing_m'] <= .01):
        raise ValueError('Declare actual_triangles placement and bounded sample_spacing_m')
    return value


def descriptor(session, obj):
    settings = request(session, obj)
    if settings is None:
        return None
    fit = session.scene_graph['dynamic_objects'][obj]['fit']
    seed = fit.get('support_seed')
    if seed:
        from .support_seed import inputs
        provenance = inputs(session, obj)
        root = session/'objects'/seed['object']/f"round_{seed['round']}"
        parent = json.loads((root/'fit.json').read_text())
        mesh = root/'mesh.npz'; pose = parent['pose']
        surface = seed.get('surface', 'uppermost'); name = seed['object']
    else:
        from calibrated_scene_reconstruction.scene.support.blender import verify
        name = session.scene_graph['task']['support']
        verify(session, name)
        fix = session.scene_graph['furniture'][name]['fix']
        mesh = resolve(fix['mesh'])
        if digest(mesh) != fix['mesh_sha256']:
            raise ValueError('Actual support geometry changed')
        pose = {'xy': fix['xy'], 'yaw_deg': fix['yaw_deg'], 'support_z': 0.}
        surface = 'uppermost'; provenance = {str(mesh.resolve()): digest(mesh)}
    return {'version': 1, 'mesh': str(mesh.resolve()), 'mesh_sha256': digest(mesh),
            'pose': pose, 'surface': surface, 'support': name,
            'sample_spacing_m': float(settings['sample_spacing_m']), 'inputs': provenance}


def attach(session, obj, pose):
    actual = descriptor(session, obj)
    if actual is None:
        return pose
    return {**pose, 'support_triangles': actual, 'contact': 'actual_triangle_support_proposal'}


@lru_cache(maxsize=8)
def _surface(path, expected, pose_json):
    if digest(path) != expected:
        raise ValueError('Actual support mesh changed')
    pose = json.loads(pose_json)
    with np.load(path, allow_pickle=False) as mesh:
        vertices = mesh['V']@rotation(pose).T + [*pose['xy'], pose['support_z']]
        return TriangleSurface(vertices, mesh['F'])


def apply(vertices, faces, parts, pose):
    """Recompute only initialization height; never alter triangles or pin bodies."""
    actual = pose['support_triangles']
    if (actual.get('version') != 1 or not parts or any(part.get('kind') != 'rigid' for part in parts)
            or faces is None):
        raise ValueError('Actual triangle placement requires explicit rigid links and actual candidate faces')
    if digest(actual['mesh']) != actual['mesh_sha256']:
        raise ValueError('Actual support mesh changed')
    parent = _surface(actual['mesh'], actual['mesh_sha256'], json.dumps(actual['pose'], sort_keys=True))
    result = on_triangles(vertices, faces, parent, xy=pose['xy'], rotation=rotation(pose),
                          surface=actual['surface'], sample_spacing_m=actual['sample_spacing_m'])
    top = result['support_z']
    return {**pose, 'local_top': top, 'support_z': top + float(pose.get('down_offset', 0.)),
            'support_triangle_measurement': result}


def inputs(session, obj):
    value = descriptor(session, obj)
    if value is None:
        return {}
    return {str(Path(path).resolve() if Path(path).is_absolute() else (session/path).resolve()): expected
            for path, expected in value['inputs'].items()}
