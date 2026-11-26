"""Audited source-proposal poses initialize fresh Blender rounds, never accept them.

Only planar translation and yaw travel through this interface. Support height,
support orientation, geometry, source scores and native checks must be rebuilt
by the existing round harness. Evidence and candidate files are bound by their
actual bytes; every used candidate is copied into the new immutable round.
"""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from calibrated_scene_reconstruction.core.session_code import source_path


def _digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def _local(session, path):
    path = Path(path).resolve()
    if not path.is_file() or not any(path.is_relative_to(Path(root).resolve())
                                     for root in (session.dir, session.config_dir)):
        raise ValueError('Pose seed inputs must be existing current-session or config files')
    return path


def evidence_inputs(session, obj, observations=None):
    """Bind the current metric frame, cached masks and effective source domains."""
    required = ['parsing/posed_frames/cameras.npz', 'splat/align.json',
                f'splat/object_fit/{obj}.npz']
    files = {_local(session, session / name) for name in required}
    world = session / 'parsing/world_frame.json'
    if world.exists():
        files.add(_local(session, world))
    if observations is None:
        from .evidence import load
        observations = load(session, obj, parts=True)
    for relative, expected in observations[0].get('_object_domain_inputs', {}).items():
        path = _local(session, session / relative)
        if _digest(path) != expected:
            raise ValueError('Source domain changed while preparing pose initialization')
        files.add(path)
    return {str(path): _digest(path) for path in sorted(files)}


def _pose(value):
    if not isinstance(value, dict) or set(value) != {'xy', 'yaw_deg'}:
        raise ValueError('Pose initialization carries only xy and yaw_deg; support is rebuilt')
    xy = value['xy']
    if (not isinstance(xy, list) or len(xy) != 2
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in xy)
            or isinstance(value['yaw_deg'], bool)
            or not isinstance(value['yaw_deg'], (int, float))
            or not np.isfinite([*xy, value['yaw_deg']]).all()):
        raise ValueError('Finite planar pose coordinates required')
    return {'xy': list(map(float, xy)), 'yaw_deg': float(value['yaw_deg'])}


def create(session, obj, pose, candidates, *, observations=None):
    """Create a manifest from actual current files; the caller chooses its path.

    ``candidates`` contains the proposal report and geometry or parameter files
    that produced the pose. A candidate's claimed score is never trusted here.
    """
    if obj not in session.scene_graph.get('dynamic_objects', {}):
        raise ValueError('Pose seed needs a current declared object identity')
    paths = [_local(session, path) for path in candidates]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError('Declare distinct actual candidate inputs for pose initialization')
    return {'version': 1, 'object': obj, 'initialization_only': True,
            'pose': _pose(pose), 'evidence_inputs': evidence_inputs(session, obj, observations),
            'candidate_inputs': {str(path): _digest(path) for path in paths}}


def validate(session, obj, manifest, *, observations=None):
    required = {'version', 'object', 'initialization_only', 'pose',
                'evidence_inputs', 'candidate_inputs'}
    if (not isinstance(manifest, dict) or set(manifest) != required
            or type(manifest['version']) is not int or manifest['version'] != 1
            or manifest['object'] != obj or manifest['initialization_only'] is not True):
        raise ValueError('Explicit initialization-only pose manifest and matching object required')
    pose = _pose(manifest['pose'])
    current = evidence_inputs(session, obj, observations)
    if manifest['evidence_inputs'] != current:
        raise ValueError('Pose proposal source evidence or metric frame changed')
    inputs = manifest['candidate_inputs']
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError('Pose proposal requires actual candidate provenance')
    for name, expected in inputs.items():
        path = _local(session, name)
        if str(path) != name or _digest(path) != expected:
            raise ValueError('Pose proposal candidate input changed')
    return pose


def snapshot(session, obj, output, *, observations=None):
    """Copy a configured seed into a newly created round; legacy routes return None.

    The fit request key is ``source_pose_seed`` (a config-relative JSON file).
    Calling this twice for the same output is rejected, preserving the first
    candidate and its provenance even when a later attempt fails.
    """
    request = session.scene_graph['dynamic_objects'][obj].get('fit') or {}
    declared = request.get('source_pose_seed')
    if declared is None:
        return None
    if not isinstance(declared, str) or not declared:
        raise ValueError('source_pose_seed must name a config-relative manifest')
    source = _local(session, source_path(session.config_dir, declared))
    payload = source.read_bytes()
    manifest = json.loads(payload)
    pose = validate(session, obj, manifest, observations=observations)
    directory = Path(output) / 'source_pose_seed'
    directory.mkdir()  # Exclusive: never rewrite another attempt's seed.
    (directory / 'manifest.json').write_bytes(payload)
    copies = {}
    for index, (name, expected) in enumerate(manifest['candidate_inputs'].items()):
        target = directory / f'{index:03d}_{Path(name).name}'
        shutil.copyfile(name, target)
        if _digest(target) != expected:
            raise ValueError('Pose candidate changed while taking its immutable snapshot')
        copies[name] = target.name
    if source.read_bytes() != payload:
        raise ValueError('Pose seed manifest changed while snapshotting')
    record = {'initialization_only': True, 'source': str(source),
              'manifest_sha256': hashlib.sha256(payload).hexdigest(),
              'candidate_snapshots': copies, 'pose': pose}
    (directory / 'snapshot.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def load_snapshot(session, obj, output, *, observations=None):
    """Revalidate and return only an initializer; no metric or approval is returned."""
    directory = Path(output) / 'source_pose_seed'
    if not directory.exists():
        if (session.scene_graph['dynamic_objects'][obj].get('fit') or {}).get('source_pose_seed'):
            raise ValueError('Configured source pose seed was not snapshotted for this round')
        return None
    record = json.loads((directory / 'snapshot.json').read_text())
    manifest_path = directory / 'manifest.json'
    if (_digest(manifest_path) != record['manifest_sha256']
            or _digest(_local(session, record['source'])) != record['manifest_sha256']):
        raise ValueError('Source pose seed or immutable manifest snapshot changed')
    manifest = json.loads(manifest_path.read_text())
    pose = validate(session, obj, manifest, observations=observations)
    if set(record['candidate_snapshots']) != set(manifest['candidate_inputs']):
        raise ValueError('Pose seed candidate snapshot inventory changed')
    for name, relative in record['candidate_snapshots'].items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory.resolve()) or _digest(path) != manifest['candidate_inputs'][name]:
            raise ValueError('Immutable pose candidate snapshot changed')
    return pose
