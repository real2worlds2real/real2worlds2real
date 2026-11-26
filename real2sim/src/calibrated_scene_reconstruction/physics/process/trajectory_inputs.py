"""Measured joint positions for a process driver, without sensor-force inputs.

Preparation preserves source ordering and timestamps. It does not align video
clocks, correct joint zero frames, infer held poses, or approve a native replay.
The original JSON may contain other channels; only the selected measured joint
positions enter the prepared NPZ. Use its receipt as an auxiliary process input.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .contract import digest, number, read, require, write


def normalize(document, specification):
    """Validate named measured positions, convert declared units to SI only.

    ``specification`` selects a measured position field and declares its units,
    clock quality and joint coordinate frame. Mixed revolute/prismatic streams
    use a unit mapping keyed by joint name. No angle wrapping is performed.
    """
    require(specification.get('measurement_kind') == 'measured',
            'This adapter requires actual measured joint positions')
    key = specification.get('position_key', 'measured_rad')
    require(key in {'measured_rad', 'measured_deg', 'measured_positions'},
            'Select a measured joint-position field, not commands or forces')
    definitions = document.get('joints')
    require(isinstance(definitions, list) and definitions,
            'Named joint definitions are required')
    names, groups = [], []
    for joint in definitions:
        require(isinstance(joint, dict), 'Each joint definition must be an object')
        name, group = joint.get('joint'), joint.get('group', '')
        require(isinstance(name, str) and name.strip() == name and bool(name),
                'Joint names must be nonempty and unambiguous')
        require(isinstance(group, str), 'Joint groups must be strings')
        names.append(name)
        groups.append(group)
    require(len(set(names)) == len(names), 'Joint names must be unique')
    t = np.asarray(document.get('t_s'), dtype=np.float64)
    q = np.asarray(document.get(key), dtype=np.float64)
    require(t.ndim == 1 and len(t) >= 2 and np.isfinite(t).all()
            and np.all(np.diff(t) > 0), 'Finite strictly increasing timestamps are required')
    require(q.shape == (len(t), len(names)) and np.isfinite(q).all(),
            'Measured positions must be a finite samples-by-joints array')
    if 'n_frames' in document:
        require(type(document['n_frames']) is int and document['n_frames'] == len(t),
                'Declared frame count differs from measured positions')
    source_units = specification.get('units')
    if isinstance(source_units, str):
        units = [source_units] * len(names)
    else:
        require(isinstance(source_units, dict) and set(source_units) == set(names),
                'Declare units for every joint exactly once')
        units = [source_units[name] for name in names]
    require(all(unit in {'rad', 'deg', 'm'} for unit in units),
            'Joint position units must be rad, deg or m')
    if key in {'measured_rad', 'measured_deg'}:
        require(all(unit == key.removeprefix('measured_') for unit in units),
                'Declared units contradict the selected source field')
    frame = specification.get('coordinate_frame')
    require(isinstance(frame, str) and frame.strip(), 'Declare the joint coordinate frame')
    require(specification.get('frame_conversion') == 'already_in_declared_frame',
            'Frame conversion must be performed and reviewed separately')
    clock = specification.get('clock', {})
    require(clock.get('kind') in {'nominal_frame_rate', 'measured_timestamps'}
            and isinstance(clock.get('reason'), str) and clock['reason'].strip(),
            'Declare clock quality and its evidence basis')
    if clock['kind'] == 'nominal_frame_rate':
        fps, tolerance = clock.get('fps'), clock.get('rounding_tolerance_s')
        require(number(fps) and fps > 0 and number(tolerance) and 0 <= tolerance < .1 / fps,
                'Nominal clock needs positive fps and a bounded rounding tolerance')
        expected = t[0] + np.arange(len(t)) / fps
        require(np.max(np.abs(t - expected)) <= tolerance + 1e-12,
                'Timestamps disagree with the declared nominal frame clock')
        if 'fps' in document:
            require(number(document['fps']) and np.isclose(document['fps'], fps, rtol=0, atol=1e-9),
                    'Source frame rate differs from the clock declaration')
    holds = specification.get('held_groups', {})
    require(isinstance(holds, dict) and set(holds) <= set(groups)
            and all(isinstance(reason, str) and reason.strip() for reason in holds.values()),
            'Held-group declarations need known groups and evidence reasons')
    # A declared hold describes the recording procedure, not an instruction to
    # replace feedback with a fixed pose. Small measured motion stays intact.
    scale = np.array([np.pi / 180 if unit == 'deg' else 1. for unit in units])
    values = {
        't': t.copy(), 'q': q * scale,
        'joint_names': np.asarray(names), 'joint_groups': np.asarray(groups),
        'joint_units': np.asarray(['m' if unit == 'm' else 'rad' for unit in units]),
    }
    metadata = {
        'measurement_kind': 'measured', 'selected_position_key': key,
        'coordinate_frame': frame, 'frame_conversion': 'none',
        'source_units': units, 'clock': dict(clock), 'held_groups': dict(holds),
        'timestamp_operation': 'preserved; no clock shift, stretching or re-zeroing',
        'position_operation': 'declared degree-to-radian conversion only; no wrapping, filtering or held-pose replacement',
        'alignment_status': 'pending', 'native_replay_status': 'not_run',
        'selected_channels': ['t_s', 'joints', key],
        'excluded_channels_policy': 'Commands, force, torque, wrench, tactile, target-object states and all unselected channels are not exported.',
    }
    return values, metadata


def motion_summary(values, metadata):
    """Report variation without interpreting near-constant samples as a lock."""
    t, q = values['t'], values['q']
    rows = []
    for i, name in enumerate(values['joint_names']):
        unit = str(values['joint_units'][i])
        rows.append({'joint': str(name), 'group': str(values['joint_groups'][i]), 'unit': unit,
                     'min': float(q[:, i].min()), 'max': float(q[:, i].max()),
                     'range': float(np.ptp(q[:, i])),
                     'maximum_sample_step': float(np.max(np.abs(np.diff(q[:, i])))),
                     'exactly_constant': bool(np.all(q[:, i] == q[0, i]))})
    return {'samples': len(t), 'joints': len(rows), 'start_s': float(t[0]), 'end_s': float(t[-1]),
            'duration_on_declared_clock_s': float(t[-1] - t[0]), 'clock_kind': metadata['clock']['kind'],
            'time_step_s': {'min': float(np.diff(t).min()), 'max': float(np.diff(t).max())},
            'per_joint': rows,
            'interpretation': 'Variation is observed on the declared clock; constants do not establish rigid attachment or exact hold. Nominal rates are not measured wall-clock velocities.'}


def prepare(source, output, specification):
    """Write a fresh, hash-bound, position-only preparation directory."""
    source, output = Path(source).resolve(), Path(output)
    require(source.is_file(), 'Measured trajectory source does not exist')
    before = digest(source)
    expected = specification.get('source_sha256')
    require(isinstance(expected, str) and before == expected,
            'Source hash must match the reviewed trajectory specification')
    values, metadata = normalize(read(source), specification)
    require(digest(source) == before, 'Trajectory changed during preparation')
    evidence = []
    for item in specification.get('evidence', []):
        path = Path(item['path']).resolve()
        require(path.is_file() and digest(path) == item.get('sha256') and item.get('reason'),
                'Trajectory evidence must exist and match its reviewed hash')
        evidence.append({'path': str(path), 'sha256': item['sha256'], 'reason': item['reason']})
    require(evidence, 'A source-semantics and clock evidence record is required')
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / 'trajectory.npz', **values)
    write(output / 'motion_summary.json', motion_summary(values, metadata))
    write(output / 'specification.json', specification)
    result = {
        'version': 1, 'status': 'prepared_unaligned', **metadata,
        'source': {'path': str(source), 'sha256': before}, 'evidence': evidence,
        'trajectory_sha256': digest(output / 'trajectory.npz'),
        'specification_sha256': digest(output / 'specification.json'),
        'summary_sha256': digest(output / 'motion_summary.json'),
        'producer': {'path': str(Path(__file__).resolve()), 'sha256': digest(__file__)},
        'acceptance': 'Preparation is not camera synchronization, physical calibration or a passing task proof.',
    }
    write(output / 'manifest.json', result)
    return result


def load_prepared(directory):
    """Read position-only arrays after verifying their immutable preparation."""
    directory = Path(directory)
    receipt = read(directory / 'manifest.json')
    require(receipt.get('version') == 1 and receipt.get('measurement_kind') == 'measured',
            'Measured trajectory preparation receipt required')
    require(digest(directory / 'trajectory.npz') == receipt.get('trajectory_sha256')
            and digest(directory / 'specification.json') == receipt.get('specification_sha256')
            and digest(directory / 'motion_summary.json') == receipt.get('summary_sha256'),
            'Prepared trajectory or evidence changed')
    sources = [receipt['source'], *receipt['evidence'], receipt['producer']]
    require(all(Path(item['path']).is_file() and digest(item['path']) == item['sha256'] for item in sources),
            'Trajectory source, evidence or preparation implementation changed')
    with np.load(directory / 'trajectory.npz', allow_pickle=False) as saved:
        require(set(saved.files) == {'t', 'q', 'joint_names', 'joint_groups', 'joint_units'},
                'Prepared trajectory must contain only named joint positions and timestamps')
        values = {key: saved[key].copy() for key in saved.files}
    return values, receipt


def sample_positions(values, times):
    """Interpolate joint coordinates on their own clock, with no extrapolation.

    This helper does not map video time to the trajectory. The caller must use
    a separately reviewed clock mapping. Continuous angles are never unwrapped
    implicitly: an encoded whole revolution remains a whole revolution.
    """
    query = np.asarray(times, float)
    t, q = np.asarray(values['t'], float), np.asarray(values['q'], float)
    require(query.ndim <= 1 and np.isfinite(query).all(), 'Sample times must be finite scalars or vectors')
    require(t.ndim == 1 and len(t) >= 2 and np.isfinite(t).all() and np.all(np.diff(t) > 0)
            and q.ndim == 2 and len(q) == len(t) and np.isfinite(q).all(),
            'Sampling requires a valid finite measured trajectory')
    require(np.all(query >= t[0]) and np.all(query <= t[-1]),
            'No trajectory extrapolation; explicitly model uncovered preparation or rest intervals')
    return np.stack([np.interp(query, t, column) for column in q.T], axis=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('output', help='new preparation directory')
    parser.add_argument('--specification', required=True)
    args = parser.parse_args()
    receipt = prepare(args.source, args.output, read(args.specification))
    print(json.dumps({'status': receipt['status'], 'output': args.output}))


if __name__ == '__main__':
    main()
