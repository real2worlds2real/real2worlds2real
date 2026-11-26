"""Explicit source-RGB polygon corrections, followed by component review.

This tool never reads candidate geometry, scores or renders. It preserves all
source frames and queries; newly annotated masks remain unapproved proposals.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .blender.store import digest


def annotate(names, masks, records, rgb_hashes):
    original = np.asarray(masks)
    if (original.dtype != np.bool_ or original.ndim != 3 or len(original) != len(names)
            or len(set(names)) != len(names) or not records or not set(records) <= set(names)):
        raise ValueError('Unique complete source frames and native boolean masks required')
    result = original.copy()
    h, w = original.shape[1:]
    for name, record in records.items():
        if (set(record) != {'rgb_sha256', 'polygons'} or record['rgb_sha256'] != rgb_hashes.get(name)
                or not isinstance(record['polygons'], list) or not record['polygons']):
            raise ValueError('Explicit polygons must bind unchanged source RGB')
        index = names.index(name)
        for request in record['polygons']:
            if (set(request) != {'xy', 'label', 'basis'} or type(request['label']) is not int
                    or request['label'] not in (0, 1) or not isinstance(request['basis'], str)
                    or not request['basis'].strip()):
                raise ValueError('Each polygon needs source identity, binary label and basis')
            xy = np.asarray(request['xy'], float)
            if (xy.ndim != 2 or xy.shape[1:] != (2,) or not 3 <= len(xy) <= 2048
                    or not np.isfinite(xy).all() or (xy < 0).any()
                    or (xy[:, 0] >= w).any() or (xy[:, 1] >= h).any()):
                raise ValueError('Finite source-pixel polygons inside the original image required')
            polygon = np.rint(xy).astype('int32')
            if abs(cv2.contourArea(polygon)) < 1:
                raise ValueError('Source polygon must enclose a resolved image region')
            region = np.zeros((h, w), 'uint8')
            cv2.fillPoly(region, [polygon], 1)
            result[index, region.astype(bool)] = bool(request['label'])
        if original[index].any() and not result[index].any():
            raise ValueError('A source annotation cannot silently remove a whole observed view')
    return result


def prepare(session, obj, proposal, annotations):
    from calibrated_scene_reconstruction.core.session_code import source_path
    from .part_label_review import prepare as component_review
    root = (session / 'qa/source_parts' / obj).resolve()
    parent = (session.dir / proposal).resolve()
    if not parent.is_relative_to(root):
        raise ValueError('Keep the source proposal in this object review scope')
    source = source_path(session.config_dir, annotations)
    instructions = json.loads(source.read_text())
    if (set(instructions) != {'version', 'object', 'frames'} or instructions['version'] != 1
            or instructions['object'] != obj):
        raise ValueError('Matching object and version-1 source annotations required')
    manifest = json.loads((parent / 'manifest.json').read_text())
    expected = {
        'raw_labels_sha256': digest(session / 'splat/object_fit' / f'{obj}.components.acquired.npz'),
        'acquisition_sha256': digest(session / 'splat/object_fit' / f'{obj}.acquired.npz'),
        'whole_observations_sha256': digest(session / 'splat/object_fit' / f'{obj}.npz'),
        'cameras_sha256': digest(session / 'parsing/posed_frames/cameras.npz')}
    if (manifest['object'] != obj or any(manifest.get(k) != v for k, v in expected.items())
            or manifest['proposed_labels_sha256'] != digest(parent / 'labels.npz')):
        raise ValueError('Source proposal acquisition or cameras changed')
    hashes = {**manifest.get('parent_inputs', {}), **manifest.get('source_domain_inputs', {})}
    for name, sha in hashes.items():
        path = (session.dir / name).resolve()
        if not path.is_relative_to(session.dir.resolve()) or digest(path) != sha:
            raise ValueError('Source proposal provenance changed')
    with np.load(parent / 'labels.npz') as data:
        names, masks = list(map(str, data['names'])), data['masks']
    if any(Path(name).name != name for name in names):
        raise ValueError('Source frame names must be local image basenames')
    if not isinstance(instructions['frames'], dict) or not set(instructions['frames']) <= set(names):
        raise ValueError('Annotate only existing source frames')
    rgb_hashes = {name: digest(session / 'splat/data/images' / name) for name in instructions['frames']}
    selected = annotate(names, masks, instructions['frames'], rgb_hashes)
    number = max((int(p.name[6:]) for p in root.glob('round_*') if p.name[6:].isdigit()), default=0) + 1
    archive = root / f'round_{number}'
    archive.mkdir()
    (archive / 'source_annotations.json').write_bytes(source.read_bytes())
    (archive / 'source.py').write_bytes(Path(__file__).read_bytes())
    for path in [parent / 'manifest.json', parent / 'labels.npz', archive / 'source_annotations.json', archive / 'source.py']:
        hashes[str(path.relative_to(session.dir.resolve()))] = digest(path)
    for name, sha in rgb_hashes.items():
        hashes[str((session / 'splat/data/images' / name).resolve().relative_to(session.dir.resolve()))] = sha
    np.savez_compressed(archive / 'labels.npz', names=names, masks=selected)
    metadata = dict(manifest, proposed_labels_sha256=digest(archive / 'labels.npz'),
                    alternative_prompt='Explicit source-RGB identity polygons; review required',
                    source_annotations_sha256=digest(archive / 'source_annotations.json'), parent_inputs=hashes)
    (archive / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    output = component_review(session, obj, manifest['prompt'], label_proposal=str(archive.relative_to(session.dir.resolve())))
    path = output / 'manifest.json'
    metadata = json.loads(path.read_text()); metadata['parent_inputs'].update(hashes)
    path.write_text(json.dumps(metadata, indent=2) + '\n')
    review_path = output / 'review.json'; review = json.loads(review_path.read_text())
    review['manifest_sha256'] = digest(path)
    review_path.write_text(json.dumps(review, indent=2) + '\n')
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('session'); p.add_argument('object'); p.add_argument('--proposal', required=True)
    p.add_argument('--annotations', required=True)
    args = p.parse_args()
    from calibrated_scene_reconstruction.scene.session import Session
    print(prepare(Session(args.session), args.object, args.proposal, args.annotations))
