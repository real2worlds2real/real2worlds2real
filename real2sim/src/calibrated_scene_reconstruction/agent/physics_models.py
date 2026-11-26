"""Blender model iteration for fixed-camera physics recordings.

    python -m calibrated_scene_reconstruction.agent.physics_models fit RECORDING [--mode video|replay]
    python -m calibrated_scene_reconstruction.agent.physics_models review RECORDING --round N --record FILE

Model source lives in configs/<recording>/agent/objects/object/model.py. Fitting
never approves its own visual evidence; review requires a separate JSON record.
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from calibrated_scene_reconstruction.assets.model_artifact import digest, require_reviewed
from calibrated_scene_reconstruction.core.paths import ROOT
from calibrated_scene_reconstruction.scene.objects.blender.blender_run import Blender
from calibrated_scene_reconstruction.scene.objects.blender.contract import contract, write_json
from calibrated_scene_reconstruction.scene.objects.blender.fit import optimize
from calibrated_scene_reconstruction.scene.objects.blender.evidence import score
from calibrated_scene_reconstruction.physics.video.model_inputs import calibration, observations


def event(root, kind, **data):
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps({'time': time.time(), 'event': kind, **data}) + '\n')


def source(path):
    return ROOT / 'configs' / Path(path).name / 'agent/objects/object/model.py'


def fit(path, mode='video', budget=300):
    if not 10 <= budget <= 2000:
        raise ValueError('Budget must be between 10 and 2000')
    path = Path(path)
    model = source(path)
    if not model.is_file():
        require_reviewed(path, model)
    root = path / 'objects/object'
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'events.jsonl').exists():
        for line in (root / 'events.jsonl').read_text().splitlines():
            previous = json.loads(line)
            if (previous.get('event') == 'reviewed' and previous.get('passed') is False
                    and previous.get('model_sha256') == digest(model)):
                raise ValueError('Rejected model source has not changed; revise model.py before resubmission')
    rounds = [int(p.name[6:]) for p in root.glob('round_*') if p.name[6:].isdigit()]
    number = max(rounds, default=0) + 1
    out = root / f'round_{number}'
    out.mkdir()
    shutil.copyfile(model, out / 'model.py')
    shutil.copyfile(path / 'track.npz', out / 'observation_track.npz')
    obs, ref_pose, context, indices = observations(path, mode)
    write_json(out / 'calibration.json', context)
    paths = ['video.mp4', str((out / 'observation_track.npz').relative_to(path)),
             str((out / 'calibration.json').relative_to(path))]
    paths += ['perception.npz'] if mode == 'video' else ['camera.json', 'model_initialization.json']
    inputs = {p: digest(path / p) for p in paths}
    write_json(out / 'input.json', {'inputs': inputs, 'frames': indices.tolist(), 'mode': mode,
                                  'model_sha256': digest(model), 'budget': budget})
    event(root, 'started', round=number, inputs=inputs, model_sha256=digest(model))
    try:
        with Blender(out / 'model.py', root / 'cache') as builder:
            result = optimize(builder, builder.spec, obs,
                              {'xy': [0., 0.], 'yaw_deg': 0., 'local_top': 0.}, budget)
            artifacts, build = builder.evaluate(result['params'], artifacts=True)
        for name in ('mesh.npz', 'object.glb', 'scene.blend', 'preview_0.png', 'preview_1.png'):
            shutil.copyfile(artifacts / name, out / name)
        with np.load(out / 'mesh.npz') as mesh:
            metrics, predicted = score(mesh['V'], mesh['F'], result['pose'], obs, renders=True)
        if mode == 'video':
            from calibrated_scene_reconstruction.physics.video.scene import read_frames
        else:
            from calibrated_scene_reconstruction.physics.replay.observe import read_video as read_frames
        frames, _ = read_frames(path if mode == 'video' else path / 'video.mp4')
        rows = []
        order = np.argsort(metrics['local_per_frame'])
        selected = list(dict.fromkeys([*order[-4:][::-1], *order[::max(1, len(order) // 4)]]))[:8]
        for k in selected:
            rgb = frames[indices[k]].copy()
            if rgb.shape[:2] != obs[2][k].shape:
                rgb = cv2.resize(rgb, obs[2][k].shape[::-1])
            overlay = rgb.copy()
            for mask, colour in ((obs[2][k], (240, 40, 40)), (predicted[k], (40, 220, 40))):
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, colour, 1)
            rows.append(np.hstack([cv2.resize(im, (384, 256)) for im in (rgb, overlay)]))
        cv2.imwrite(str(out / 'evidence.jpg'), np.vstack(rows)[:, :, ::-1])
        result.update(metrics=metrics, ok=metrics['ok'], round=number, build=build,
                      model_sha256=contract(model)['sha256'], mesh_sha256=digest(out / 'mesh.npz'),
                      input_sha256=digest(out / 'input.json'),
                      reference_pose=ref_pose.tolist(), calibration=context, mode=mode)
        result.pop('mesh', None)
        write_json(out / 'fit.json', result)
        event(root, 'fitted', round=number, ok=result['ok'], model_sha256=result['model_sha256'])
        return result
    except Exception as exc:
        event(root, 'failed', round=number, error=str(exc))
        raise


def review(path, number, record):
    path = Path(path)
    root = path / 'objects/object'
    out = root / f'round_{number}'
    result = json.loads((out / 'fit.json').read_text())
    if digest(source(path)) != result['model_sha256']:
        raise ValueError('Model changed since fitting')
    if digest(out / 'input.json') != result['input_sha256']:
        raise ValueError('Round inputs changed since fitting')
    if calibration(path, result['mode']) != result['calibration']:
        raise ValueError('Camera calibration changed since fitting')
    if (record.get('stage') != 'OBJ.object' or record.get('round') != number
            or record.get('model_sha256') != result['model_sha256']
            or type(record.get('pass')) is not bool or not isinstance(record.get('findings'), list)
            or not record.get('review') or not record.get('checks')):
        raise ValueError('A complete review of this exact source and round is required')
    if record['pass'] and (not result['ok'] or any(f.get('status') != 'resolved' for f in record['findings'])):
        raise ValueError('Passing review requires image thresholds and resolved findings')
    review_path = out / 'review.json'
    with review_path.open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    event(root, 'reviewed', round=number, passed=record['pass'], model_sha256=result['model_sha256'])
    if not record['pass']:
        return record
    ref = result['reference_pose']
    p = result['pose']
    angle = np.radians(ref[3])
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    xy = np.asarray(ref[:2]) + R @ p['xy']
    proof = {'route': 'agent_blender', 'round': number, 'mode': result['mode'],
             'model_sha256': result['model_sha256'], 'mesh_sha256': result['mesh_sha256'],
             'mesh': str((out / 'mesh.npz').relative_to(path)),
             'fit': str((out / 'fit.json').relative_to(path)), 'fit_sha256': digest(out / 'fit.json'),
             'review': str(review_path.relative_to(path)),
             'inputs': json.loads((out / 'input.json').read_text())['inputs'],
             'calibration': result['calibration'],
             'world_pose': [float(xy[0]), float(xy[1]), ref[2] + p['support_z'], ref[3] + p['yaw_deg']],
             'iou_median': result['metrics']['iou_median']}
    write_json(root / 'approved.json', proof)
    require_reviewed(path, source(path))
    return proof


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', choices=('fit', 'review', 'status'))
    parser.add_argument('recording')
    parser.add_argument('--mode', choices=('video', 'replay'), default='video')
    parser.add_argument('--max-evaluations', type=int, default=300)
    parser.add_argument('--round', type=int)
    parser.add_argument('--record')
    args = parser.parse_args()
    if args.action == 'fit':
        result = fit(args.recording, args.mode, args.max_evaluations)
        print(json.dumps({k: result[k] for k in ('ok', 'round', 'model_sha256')}))
        if not result['ok']:
            raise SystemExit(2)
    elif args.action == 'review':
        if args.round is None or not args.record:
            parser.error('review requires --round and --record')
        review(args.recording, args.round, json.loads(Path(args.record).read_text()))
    else:
        proof, _, _ = require_reviewed(args.recording, source(args.recording))
        print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    main()
