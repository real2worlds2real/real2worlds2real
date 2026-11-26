"""Reuse existing observation and motion tools without treating old fits as acceptance."""
from pathlib import Path

import numpy as np

from calibrated_scene_reconstruction.core.paths import ROOT
from calibrated_scene_reconstruction.physics.replay.data import load
from .contract import digest, read, require, write


def prepare(recording, output):
    """Translate retained SAM tracks, not old fitted object states, into channels."""
    recording, output = Path(recording).resolve(), Path(output)
    track = dict(np.load(recording / 'track.npz', allow_pickle=False))
    require('centroid' in track, 'Run the existing observation tracker first; do not use simulated tracks as observations')
    t, centroid = np.asarray(track['t'], float), np.asarray(track['centroid'], float).copy()
    valid = np.asarray(track['valid'], bool)
    centroid[~valid] = np.nan
    require(len(t) == len(centroid) and np.isfinite(t).all() and np.all(np.diff(t) > 0), 'Invalid observation clock')
    output.mkdir(parents=True, exist_ok=True)
    require(not (output / 'observations.npz').exists(), 'Observation export already exists')
    np.savez_compressed(output / 'observations.npz', t=t, object_centroid_px=centroid)
    paths = {'video': ('video.mp4', 'video'), 'camera': ('camera.json', 'camera'),
             'robot_metadata': ('meta.json', 'auxiliary')}
    report = {'source': str(recording), 'frames': len(t), 'valid_frames': int(valid.sum()),
              'source_track_sha256': digest(recording / 'track.npz'),
              'scope': 'Observation/trajectory adaptation only; no new calibration or physics acceptance.'}
    if (recording / 'trajectory.npz').exists() or (recording / 'trajectory.csv').exists():
        rec = load(recording)
        name = 'trajectory.npz' if (recording / 'trajectory.npz').exists() else 'trajectory.csv'
        paths['robot_trajectory'] = (name, 'robot_trajectory')
        report['robot'] = {'name': rec.meta['robot'], 'joint_names': rec.joints,
                           'samples': len(rec.t), 'duration': float(rec.t[-1]),
                           'semantics': rec.meta.get('trajectory_semantics', 'unspecified; agent must establish measured vs commanded')}
    inputs = {key: {'path': str(recording / name), 'kind': kind}
              for key, (name, kind) in paths.items() if (recording / name).is_file()}
    inputs['observations'] = {'path': str((output / 'observations.npz').resolve()), 'kind': 'observations'}
    report['historical_fits'] = []
    for p in sorted(recording.glob('replay*/loss_calibrated.json')):
        fit = read(p)
        twin = fit.get('physics', {}).get('twin', {})
        report['historical_fits'].append({'path': str(p), 'loss': fit.get('loss'),
            'geometry_changed': twin.get('depth_scale', 1.) != 1. or twin.get('support_dz', 0.) != 0.,
            'accepted_by_new_process': False})
    write(output / 'inputs.json', inputs)
    write(output / 'legacy_report.json', report)
    return report


def commands(recording, route):
    """Return explicit existing tool invocations; never silently select a template."""
    root = str(Path(recording).resolve())
    require(route in {'replay', 'video', 'soft'}, 'Unknown legacy physics route')
    if route == 'video':
        return [[str(ROOT / 'tools/physics/video_physics.sh'), root]]
    entry = 'replay.py' if route == 'replay' else 'soft.py'
    command = [str(ROOT / 'tools/run_shared_isaac.sh'), str(ROOT / 'apps/physics' / entry), root]
    return [command + ['--calibrate']] if route == 'replay' else [command + ['--render']]
