"""Reviewed source-instance corrections for support segmentation.

Rejected instances become unknown in both whole-support and part evidence.
No candidate mesh, fitted dimensions or render residual selects pixels.
"""
import hashlib
import json

import cv2
import numpy as np

from calibrated_scene_reconstruction.core.paths import rel, resolve
from calibrated_scene_reconstruction.core.session_code import source_path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(session, manifest, associated):
    name = session.scene_graph['task']['support']
    request = session.scene_graph['furniture'][name].get('fit', {})
    reference = request.get('part_identity_review')
    if not reference:
        return {}, {}
    if manifest is None:
        raise ValueError('Support part identity review needs archived source instances')
    path = source_path(session.config_dir, reference)
    review = json.loads(path.read_text())
    if (review.get('version') != 1 or review.get('support') != name
            or review.get('observations_sha256') != digest(manifest)
            or not isinstance(review.get('exclusions'), list)):
        raise ValueError('Support part identity review changed or belongs to another observation revision')
    ignored, inputs, seen = {}, {rel(path): digest(path)}, set()
    for item in review['exclusions']:
        frame = item.get('frame')
        if frame not in associated or not isinstance(item.get('basis'), str) or not item['basis'].strip():
            raise ValueError('Source identity correction needs a known frame and review basis')
        source = session/'splat/data/images'/frame
        records = [r for g in associated[frame].get('attached_parts', []) for r in g['selected']]
        matches = [r for r in records if r.get('path') == item.get('mask')]
        if len(matches) != 1 or item['mask'] in seen:
            raise ValueError('Identity correction must name one distinct archived source instance')
        seen.add(item['mask'])
        mask_path = resolve(item['mask'])
        if not mask_path.resolve().is_relative_to(session.dir.resolve()):
            raise ValueError('Support source masks must remain inside this capture')
        if (digest(source) != item.get('source_sha256')
                or digest(mask_path) != item.get('mask_sha256')
                or digest(mask_path) != matches[0]['sha256']):
            raise ValueError('Reviewed support source RGB or mask changed')
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(str(source))
        if mask is None or image is None or mask.shape != image.shape[:2] or not (mask < 128).any():
            raise ValueError('Source identity correction needs a nonempty native-resolution mask')
        ignored.setdefault(frame, np.zeros(mask.shape, bool))
        ignored[frame] |= mask < 128
        inputs.update({rel(source): digest(source), rel(mask_path): digest(mask_path)})
    return ignored, inputs
