"""Reconcile complete object identity from independently reviewed source parts.

An optional query may include nearby objects. Reviewing its parts must not
leave that rejected identity in the whole-object union. Every acquired view,
original query, source window and negative background remains available. This
module sees source labels and RGB only; no mesh or rendered error is accepted.
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def merge(names, parts):
    """Union every declared reviewed part, with no view/part selection."""
    if not names or len(set(names)) != len(names) or not parts:
        raise ValueError('Unique complete source views and reviewed parts required')
    result = None
    for part in parts:
        if part['names'] != names:
            raise ValueError('Every part must retain every source frame in order')
        masks = np.asarray(part['masks'])
        if (masks.ndim != 3 or len(masks) != len(names) or masks.dtype != bool
                or min(masks.shape[1:]) < 1):
            raise ValueError('Native boolean source-part images required')
        if result is None:
            result = masks.copy()
        else:
            if result.shape != masks.shape:
                raise ValueError('Source part resolutions differ')
            result |= masks
    if any(not mask.any() for mask in result):
        raise ValueError('No source view may disappear during object composition')
    return result


def review_frames(names, review):
    if review.get('basis') != 'source_image_only' or set(review.get('frames', {})) != set(names):
        raise ValueError('Whole identity review must cover every original RGB view')
    for name in names:
        row = review['frames'][name]
        if (row.get('status') != 'verified' or row.get('complete_visible_parts') is not True
                or not isinstance(row.get('basis'), str) or not row['basis'].strip()):
            raise ValueError('Verify complete visible object identity independently in every source frame')


def _part(session, obj, path, names):
    """Recheck an existing source review; it grants no whole-object review."""
    from .part_label_review import components
    root = (session/'qa/source_parts'/obj).resolve()
    path = (session.dir/path).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Part review must belong to this current source object')
    manifest_path, review_path, labels_path = [path/name for name in ['manifest.json', 'review.json', 'labels.npz']]
    manifest, review = [json.loads(p.read_text()) for p in [manifest_path, review_path]]
    expected = {'raw_labels_sha256': session/'splat/object_fit'/f'{obj}.components.acquired.npz',
                'acquisition_sha256': session/'splat/object_fit'/f'{obj}.acquired.npz',
                'whole_observations_sha256': session/'splat/object_fit'/f'{obj}.npz',
                'cameras_sha256': session/'parsing/posed_frames/cameras.npz',
                'proposed_labels_sha256': labels_path}
    if (manifest['object'] != obj or manifest['connectivity'] != 8
            or review['manifest_sha256'] != sha(manifest_path)
            or any(manifest.get(key) != sha(file) for key, file in expected.items())):
        raise ValueError('Part review source or identity fingerprint changed')
    inputs = {str(file.relative_to(session.dir.resolve())): sha(file)
              for file in [manifest_path, review_path, labels_path]}
    inputs.update({str(file.resolve().relative_to(session.dir.resolve())): sha(file)
                   for file in expected.values()})
    for relative, checksum in {**manifest.get('parent_inputs', {}),
                               **manifest.get('source_domain_inputs', {})}.items():
        source = (session.dir/relative).resolve()
        if not source.is_relative_to(session.dir.resolve()) or not source.is_file() or sha(source) != checksum:
            raise ValueError('Part source provenance changed')
        inputs[str(source.relative_to(session.dir.resolve()))] = checksum
    if manifest.get('source_annotations_sha256'):
        annotation = path/'source_annotations.json'
        if sha(annotation) != manifest['source_annotations_sha256']:
            raise ValueError('Part source annotations changed')
        inputs[str(annotation.relative_to(session.dir.resolve()))] = sha(annotation)
    with np.load(labels_path) as data:
        if list(map(str, data['names'])) != names:
            raise ValueError('Part review must preserve source ordering')
        masks = data['masks']
    if ([row['frame'] for row in manifest['rows']] != names or set(review['frames']) != set(names)):
        raise ValueError('Part identity review misses or changes source frames')
    selected = np.zeros_like(masks)
    for k, row in enumerate(manifest['rows']):
        rgb = session/'splat/data/images'/row['frame']
        if sha(rgb) != row['rgb_sha256']:
            raise ValueError('Reviewed source RGB changed')
        inputs[str(rgb.relative_to(session.dir))] = row['rgb_sha256']
        labels, items = components(masks[k]);decision = review['frames'][row['frame']]
        keep = decision.get('keep_components')
        if (items != row['components'] or not isinstance(keep, list)
                or any(type(i) is not int for i in keep) or len(set(keep)) != len(keep)
                or not set(keep) <= {item['id'] for item in items}
                or decision.get('status') not in {'verified', 'unobserved'}
                or (decision['status'] == 'verified') != bool(keep)
                or not str(decision.get('basis', '')).strip()):
            raise ValueError('Each source part needs resolved independent identity')
        selected[k] = np.isin(labels, keep)
    return {'names': names, 'masks': selected}, inputs


def prepare(session, obj, part_reviews):
    """Create a pending whole-RGB review, retaining the original query archive."""
    if (not isinstance(part_reviews, list) or not part_reviews
            or len(set(part_reviews)) != len(part_reviews)):
        raise ValueError('Distinct independently reviewed source parts required')
    observation = session/'splat/object_fit'/f'{obj}.npz'
    cameras = session/'parsing/posed_frames/cameras.npz'
    with np.load(cameras) as data:
        all_names = list(map(str, data['names']))
    with np.load(observation, allow_pickle=True) as data:
        names = [all_names[i] for i in data['frames']]; windows = data['windows']
    parts, inputs = [], {str(file.relative_to(session.dir)): sha(file) for file in [observation, cameras]}
    for path in part_reviews:
        part, provenance = _part(session, obj, path, names)
        parts.append(part);inputs.update(provenance)
    masks = merge(names, parts)
    root = session/'qa/source_identity'/obj;root.mkdir(parents=True, exist_ok=True)
    number = max((int(p.name[6:]) for p in root.glob('round_*') if p.name[6:].isdigit()), default=0)+1
    out = root/f'round_{number}';out.mkdir()
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    inputs[str((out/'source.py').relative_to(session.dir))] = sha(out/'source.py')
    np.savez_compressed(out/'labels.npz', names=names, masks=masks)
    inputs[str((out/'labels.npz').relative_to(session.dir))] = sha(out/'labels.npz')
    rows = []
    for name, mask, window in zip(names, masks, windows):
        image = cv2.imread(str(session/'splat/data/images'/name));overlay = image.copy()
        contours, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 220, 0), 2)
        x, y, w, h = np.rint(window[:4]).astype(int)
        tiles = [cv2.resize(view[y:y+h, x:x+w], (384, 384)) for view in [image, overlay]]
        for tile in tiles:cv2.putText(tile, name, (5, 22), 0, .6, (0, 255, 255), 1)
        rows.append(np.hstack(tiles))
    for start in range(0, len(rows), 6):cv2.imwrite(str(out/f'source_only_{start//6}.jpg'), np.vstack(rows[start:start+6]))
    manifest = {'version': 1, 'object': obj, 'basis': 'source_image_only', 'names': names,
                'part_reviews': part_reviews, 'inputs': inputs, 'accepted': False,
                'scope': 'All declared source parts, original view order/windows and background; no geometry input.'}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    review = {'manifest_sha256': sha(out/'manifest.json'), 'basis': 'source_image_only',
              'frames': {name: {'status': 'pending', 'complete_visible_parts': False, 'basis': ''} for name in names}}
    (out/'review.json').write_text(json.dumps(review, indent=2)+'\n')
    return out


def load(session, obj, cameras, frames, masks, windows):
    relative = session.scene_graph['dynamic_objects'][obj]['fit'].get('source_identity_review')
    if relative is None:return masks, {}
    out = (session.dir/relative).resolve()
    if not out.is_relative_to((session/'qa/source_identity'/obj).resolve()):
        raise ValueError('Whole source identity review escaped object scope')
    manifest, review = [json.loads((out/name).read_text()) for name in ['manifest.json', 'review.json']]
    names = [str(cameras['names'][i]) for i in frames]
    if (manifest.get('version') != 1 or manifest['object'] != obj or manifest['basis'] != 'source_image_only'
            or manifest['names'] != names or review['manifest_sha256'] != sha(out/'manifest.json')):
        raise ValueError('Whole source identity inputs or view ordering changed')
    review_frames(names, review)
    required = {f'splat/object_fit/{obj}.npz', 'parsing/posed_frames/cameras.npz',
                str((out/'labels.npz').relative_to(session.dir.resolve())),
                *(f'splat/data/images/{name}' for name in names)}
    if not required <= set(manifest['inputs']):raise ValueError('Whole source identity lacks original acquisition provenance')
    for relative, checksum in manifest['inputs'].items():
        path = (session.dir/relative).resolve()
        if not path.is_relative_to(session.dir.resolve()) or not path.is_file() or sha(path) != checksum:
            raise ValueError('Whole source identity provenance changed')
    result = []
    with np.load(out/'labels.npz') as data:
        native = data['masks']
        if (list(map(str, data['names'])) != names or native.ndim != 3
                or len(native) != len(names) or len(masks) != len(names)
                or len(windows) != len(names) or native.dtype != bool):
            raise ValueError('Whole source labels changed view identity or complete coverage')
        for full, mask, window in zip(native, masks, windows):
            x, y, w, h = np.rint(window[:4]).astype(int);crop = full[y:y+h, x:x+w]
            if full.dtype != bool or crop.shape != (h, w) or not crop.any():
                raise ValueError('Whole source composition cannot remove a source view')
            result.append(cv2.resize(crop.astype('uint8'), mask.shape[::-1], interpolation=cv2.INTER_NEAREST)>0)
    inputs = {**manifest['inputs'], **{str((out/name).relative_to(session.dir.resolve())): sha(out/name)
                                    for name in ['manifest.json', 'review.json']}}
    return result, inputs
