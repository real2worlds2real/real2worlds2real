"""A provisional world frame from an observed auxiliary scene view.

This supplies orientation/origin for drafting when the primary close view has no
floor. It neither supplies metric scale nor approves S0/S1.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from calibrated_scene_reconstruction.scene.session import Session


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def estimate(points, support):
    """Fit floor and independently observed desk in the same camera coordinates."""
    from calibrated_scene_reconstruction.scene.twin.graph import plane_robust
    points, support = np.asarray(points), np.asarray(support)
    if len(points) < 2000 or len(support) < 1000 or not np.isfinite(points).all():
        raise ValueError('Need sufficient independently observed floor/support points')
    n, c = plane_robust(points)
    if n[1] > 0:
        n = -n
    sn, _ = plane_robust(support)
    angle = float(np.degrees(np.arccos(np.clip(abs(sn@n), 0, 1))))
    height = float(-n@c)
    separation = float(np.median((support-c)@n))
    residual = float(np.percentile(abs((points-c)@n), 95))
    if height <= 0 or angle > 10 or separation <= .05*height or residual > .03*height:
        raise ValueError(f'Floor/support inconsistent: height={height}, separation={separation}, '
                         f'parallel_angle={angle}, residual={residual}')
    x = np.array([1., 0, 0])-n[0]*n
    x /= np.linalg.norm(x)
    R = np.stack([x, np.cross(n, x), n])
    o = -height*n
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, -R@o
    return {'R_wc': R.tolist(), 'o': o.tolist(), 'T_world_cam_cv': T.tolist(),
            'camera_height_m': height,
            'diagnostics': {'floor_points': len(points), 'support_points': len(support),
                            'parallel_angle_deg': angle, 'floor_p95_residual': residual,
                            'support_separation': separation},
            'scope': 'Provisional source geometry units; metric scale and pose still require S1'}


def support_instance(instances, masks, selected=None):
    """Use one observed task support, never a union of different room tables."""
    from calibrated_scene_reconstruction.scene.twin.graph import SUPPORT
    candidates = [i for i, item in enumerate(instances)
                  if SUPPORT.search(item['label']) and np.asarray(masks[i]).any()]
    if selected is not None:
        if type(selected) is not int or selected not in candidates:
            raise ValueError('Selected support must be a current nonempty table/desk observation')
        return selected
    if len(candidates) != 1:
        raise ValueError('Select --support-id from current primary observations; '
                         f'found support instances {candidates}')
    return candidates[0]


def prepare(s, view, box, prompt='gray carpet floor', support_id=None):
    from calibrated_scene_reconstruction.scene.capture.perceive import run_sam3, overlay
    from calibrated_scene_reconstruction.core.gpu import device
    from calibrated_scene_reconstruction.scene.capture.multiview import Views
    from calibrated_scene_reconstruction.scene.capture.plane_observations import select_floor
    paths = [s/'capture/views'/f'{view}.png', s/'parsing/multiview/da3.npz',
             s/'parsing/geometry.npz', s/'parsing/multiview.json']
    before = {str(p.relative_to(s.dir)): sha(p) for p in paths}
    image = Image.open(paths[0]).convert('RGB')
    root = s/'qa/draft_floor'
    number = max([int(p.name[6:]) for p in root.glob('round_*') if p.name[6:].isdigit()], default=0)+1
    out = root/f'round_{number}'
    out.mkdir(parents=True)
    proposals = run_sam3(image, ['wooden tabletop'], device(), threshold=.3,
                         regions=[{'prompt': prompt, 'label': 'floor', 'box': box, 'threshold': .3}])
    candidates = [{'mask': p['mask'], 'prompt': prompt, 'confidence': p['score']}
                  for p in proposals if p['label'] == 'floor']
    supports = [p['mask'] for p in proposals if p['label'] == 'wooden tabletop']
    mask, decisions = select_floor(candidates, supports)
    overlay(np.asarray(image), [{'mask': mask, 'box': [0, 0, image.width, image.height],
                                'label': 'auxiliary floor', 'score': 1.}]).save(out/'review.png')
    Image.fromarray(mask.astype(np.uint8)*255).save(out/'mask.png')
    geo = dict(np.load(paths[2]))
    views = Views(s, dict(np.load(paths[1])), geo)
    index = views.names.index(view)
    X = views.lift(index, mask)
    instances = json.loads((s/'parsing/instances.json').read_text())
    masks = s.masks()
    support_id = support_instance(instances, masks, support_id)
    sm = masks[support_id] & geo['mask']
    support = geo['points'][sm][::10]
    frame = estimate(X, support)
    np.savez_compressed(out/'points.npz', floor=X, support=support)
    write(out/'candidate.json', frame)
    for p in paths:
        if sha(p) != before[str(p.relative_to(s.dir))]:
            raise ValueError('Source changed during plane preparation')
    for name in ['mask.png', 'review.png', 'points.npz', 'candidate.json']:
        before[str((out/name).relative_to(s.dir))] = sha(out/name)
    for name in ['instances.json', 'masks.npz']:
        before['parsing/'+name] = sha(s/'parsing'/name)
    write(out/'manifest.json', {'view': view, 'prompt': prompt, 'box': box,
                               'support_id': support_id,
                               'support_label': instances[support_id]['label'],
                               'source_files': before, 'candidate_decisions': decisions})
    write(out/'review.json', {'manifest_sha256': sha(out/'manifest.json'), 'pass': False, 'basis': ''})
    return out


def verify(s, out):
    out = Path(out).resolve()
    if not out.is_relative_to((s/'qa/draft_floor').resolve()):
        raise ValueError('Floor evidence must be inside this capture')
    manifest = json.loads((out/'manifest.json').read_text())
    review = json.loads((out/'review.json').read_text())
    if (review.get('pass') is not True or not isinstance(review.get('basis'), str)
            or not review['basis'].strip() or review.get('manifest_sha256') != sha(out/'manifest.json')):
        raise ValueError('Auxiliary floor needs an actual source-mask review')
    for name, expected in manifest['source_files'].items():
        path = (s.dir/name).resolve()
        if not path.is_relative_to(s.dir.resolve()) or sha(path) != expected:
            raise ValueError('Auxiliary floor source changed: '+name)
    points = np.load(out/'points.npz')
    candidate = json.loads((out/'candidate.json').read_text())
    if candidate != estimate(points['floor'], points['support']):
        raise ValueError('Auxiliary frame differs from observed floor/support geometry')
    return candidate


def activate(s, out):
    candidate = verify(s, out)
    target = s/'parsing/world_frame.json'
    if target.exists():
        raise ValueError('Preserve existing world frame before changing the observation')
    out = Path(out)
    candidate['observed_auxiliary_floor'] = str(out.relative_to(s.dir))
    write(target, candidate)


def verify_active(s):
    frame = json.loads((s/'parsing/world_frame.json').read_text())
    relative = frame.pop('observed_auxiliary_floor', None)
    if not relative or verify(s, s/relative) != frame:
        raise ValueError('Missing or changed observed auxiliary-floor frame')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session')
    parser.add_argument('--view')
    parser.add_argument('--box', type=int, nargs=4)
    parser.add_argument('--support-id', type=int,
                        help='Reviewed primary table/desk instance; required if ambiguous')
    parser.add_argument('--activate', type=Path)
    args = parser.parse_args()
    s = Session(args.session)
    if args.activate:
        activate(s, args.activate)
    elif args.view and args.box:
        print(prepare(s, args.view, args.box, support_id=args.support_id))
    else:
        parser.error('Use --view/--box to prepare, inspect evidence, then --activate a reviewed round')


if __name__ == '__main__':
    main()
