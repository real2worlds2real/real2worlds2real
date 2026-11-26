"""Rasterize reviewed source-image regions without consulting a fitted model.

For calibration exclusions a region can conservatively cover an unestimated
part; it is unknown evidence, never a positive CAD silhouette target.
"""
import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np


def rasterize(image, row):
    raw = Path(image).read_bytes()
    if hashlib.sha256(raw).hexdigest() != row['sha256']:
        raise ValueError('Source image changed since annotation: ' + str(image))
    rgb = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if rgb is None:
        raise ValueError('Unreadable annotation image')
    height, width = rgb.shape[:2]
    if not row.get('reason', '').strip():
        raise ValueError('A source-only reason is required for every annotation')
    reference = np.asarray(row['coordinate_size'], float)
    if reference.shape != (2,) or not np.isfinite(reference).all() or np.any(reference <= 0):
        raise ValueError('Finite positive annotation coordinate size required')
    mask = np.full((height, width), 255, np.uint8)
    for polygon in row['polygons']:
        points = np.asarray(polygon, float)
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or
                not np.isfinite(points).all() or np.any(points < 0) or np.any(points > reference)):
            raise ValueError('Annotation polygon must lie inside its source image')
        pixels = np.rint(points * [width, height] / reference).astype(np.int32)
        cv2.fillPoly(mask, [pixels], 0)
    return mask


def write(session, annotation, suffix):
    if not re.fullmatch(r'_[A-Za-z0-9_]+', suffix):
        raise ValueError('Local observation suffix required')
    path = Path(annotation)
    specification = json.loads(path.read_text())
    if specification.get('basis') != 'source_image_only':
        raise ValueError('Annotations must be based on source images, not candidate fits')
    images = session / 'splat/data/images'
    prepared = []
    for name, row in specification['frames'].items():
        if Path(name).name != name or not name.endswith('.jpg'):
            raise ValueError('Local capture JPG name required')
        mask = rasterize(images / name, row)
        output = images / (Path(name).stem + suffix + '.png')
        if output.exists():
            old = cv2.imread(str(output), 0)
            if old is None or not np.array_equal(old, mask):
                raise ValueError('Preserve existing observations; choose a new suffix')
        prepared.append((output, mask))
    for output, mask in prepared:
        if not output.exists() and not cv2.imwrite(str(output), mask):
            raise OSError('Could not write annotation mask')
    session.write_json('splat/masks' + suffix + '.json', {
        'basis': 'source_image_only', 'annotation_path': str(path),
        'annotation_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'images': {name: row['sha256'] for name, row in specification['frames'].items()},
        'use': specification.get('use'),
    })
    return len(prepared)
