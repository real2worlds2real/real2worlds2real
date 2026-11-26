"""Fix source/matched/novel review cameras without changing video selection."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from calibrated_scene_reconstruction.scene.capture.colmap_points import world_points
from calibrated_scene_reconstruction.scene.session import Session


def view_name(stem, offset):
    """Preserve centimetre names while keeping fractional offsets distinct."""
    centimetres = offset*100
    if np.isclose(centimetres, round(centimetres), rtol=0, atol=1e-9):
        return f'{stem}_offset{round(centimetres):02d}'
    return stem+'_offset_m_'+format(float(offset), '.17g').replace('.', 'p')


def novel_view(centre, rotation, distance, offset):
    centre, rotation = np.asarray(centre, float), np.asarray(rotation, float)
    if (centre.shape != (3,) or rotation.shape != (3, 3)
            or not np.isfinite([*centre, *rotation.ravel(), distance, offset]).all()
            or distance <= 0 or offset < 0 or offset > 2
            or not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-5)):
        raise ValueError('Finite rigid camera, positive target distance and offset0..2m required')
    target = centre + distance*rotation[:, 2]
    eye = centre + offset*rotation[:, 0]
    direction = target-eye
    direction /= np.linalg.norm(direction)
    v = np.cross(rotation[:, 2], direction)
    cosine, squared = float(rotation[:, 2]@direction), float(v@v)
    V = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    D = np.eye(3)+V+V@V*((1-cosine)/squared) if squared > 1e-12 else np.eye(3)
    return {'offset_m': float(offset), 'centre': eye.tolist(), 'R_world_cam': (D@rotation).tolist()}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(session, frames=None, count=4, offsets=(.15, .35)):
    if type(count) is not int or not 1 <= count <= 32:
        raise ValueError('Review count must be1..32')
    offsets = list(offsets)
    if (len(offsets) > 8 or len(set(offsets)) != len(offsets)
            or any(not np.isfinite(x) or not 0 < x <= 2 for x in offsets)):
        raise ValueError('Use distinct positive novel offsets up to2m')
    camera = session/'parsing/posed_frames/cameras.npz'
    z = dict(np.load(camera))
    names = list(map(str, z['names']))
    directory = session/'splat/data/images'
    inputs = {str(camera.resolve()): digest(camera)}
    role = 'explicit registered capture views'
    if frames is None:
        available = [n for n in names if (directory/n).is_file()]
        training = session/'splat/train.json'
        if training.exists():
            record = json.loads(training.read_text())
            interval = int(record['settings']['test_every'])
            if interval <= 0:
                raise ValueError('Training held-out interval must be positive')
            train_directory = (session/record.get('data', 'splat/data'))/'images'
            held = set(sorted(p.name for p in train_directory.glob('*.jpg'))[::interval])
            available = [n for n in available if n in held]
            inputs[str(training.resolve())] = digest(training)
            role = 'held-out training views with registered metric cameras'
        if not available:
            raise ValueError('No eligible review views; supply explicit registered frames')
        frames = [available[i] for i in np.unique(np.linspace(0, len(available)-1, min(count, len(available))).astype(int))]
    if (not frames or len(set(frames)) != len(frames)
            or any(Path(n).name != n or n not in names for n in frames)):
        raise ValueError('Use distinct registered image basenames')
    points, _, _ = world_points(session, min_track=3, max_error=2.)
    for relative in ('splat/align.json', 'splat/data/sparse/full/points3D.bin'):
        path = session/relative
        inputs[str(path.resolve())] = digest(path)
    anchors = []
    for name in frames:
        i = names.index(name)
        C, R, K = z['centre'][i], z['R_world_cam'][i], z['K'][i]
        w, h = map(int, z['size'][i])
        if K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0 or min(w, h) <= 0:
            raise ValueError('Invalid registered camera intrinsics or raster')
        pc = (points-C)@R
        depth = pc[:, 2]
        uv = pc[:, :2]/np.maximum(depth[:, None], 1e-9)*[K[0, 0], K[1, 1]]+K[:2, 2]
        visible = ((depth > .2) & (depth < 20) & (uv[:, 0] > .25*w) & (uv[:, 0] < .75*w)
                   & (uv[:, 1] > .25*h) & (uv[:, 1] < .75*h))
        distance = float(np.median(depth[visible])) if visible.sum() > 8 else 2.
        source = directory/name
        inputs[str(source.resolve())] = digest(source)
        anchors.append({'name': name, 'source': str(source.resolve()), 'source_sha256': inputs[str(source.resolve())],
                        'source_size': [w, h], 'K': K.tolist(), 'look_distance_m': distance,
                        'look_distance_basis': 'Median reliable central sparse depth;2m display-only fallback if unavailable.',
                        'views': [novel_view(C, R, distance, offset) for offset in [0., *offsets]]})
    w, h = anchors[0]['source_size']
    width = min(w, 1280)
    return {'purpose': 'Review only; video sampling, calibration and training frames unchanged.',
            'source_role': role, 'width': width, 'height': max(2, round(width*h/w)),
            'novel_offsets_m': offsets, 'truth_for_free_views': False, 'anchors': anchors, 'input_sha256': inputs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session')
    parser.add_argument('--frames', nargs='+')
    parser.add_argument('--count', type=int, default=4)
    parser.add_argument('--offsets', nargs='*', type=float, default=[.15, .35])
    parser.add_argument('--output', default='qa/background_diagnostics/cameras.json')
    args = parser.parse_args()
    session = Session(args.session)
    if (session/'qa/independent_freeze.json').exists():
        raise ValueError('Frozen session: create a background derivative for new review artifacts')
    output = session/args.output
    if output.exists():
        raise ValueError('Review cameras already fixed; reuse them or choose a new output')
    record = build(session, args.frames, args.count, args.offsets)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    print(output, [a['name'] for a in record['anchors']])


if __name__ == '__main__':
    main()
