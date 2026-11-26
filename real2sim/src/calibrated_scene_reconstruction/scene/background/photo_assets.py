"""Convert declared source photographs to renderer-calibrated linear emission.

These assets preserve captured lighting. They are not reflectance estimates and
must be replaced when a cousin changes the corresponding lighting or layout.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import tifffile

from calibrated_scene_reconstruction.scene.background.display_response import invert
from calibrated_scene_reconstruction.scene.review.environment import digest


def convert(names, assets, response, response_path, output):
    if not isinstance(names, list) or any(not isinstance(n, str) for n in names) or len(set(names)) != len(names):
        raise ValueError('captured_photos must be distinct logical asset names')
    if names and response is None:
        raise ValueError('Captured photographs require a validated display response')
    result = {}
    for name in names:
        if name not in assets:
            raise ValueError('Unknown captured photograph asset: '+name)
        source = Path(assets[name])
        with Image.open(source) as image:
            if image.mode not in ('RGB', 'RGBA') or image.format not in ('PNG', 'JPEG', 'WEBP'):
                raise ValueError('Captured photograph must be an sRGB 8-bit RGB image: '+name)
            rgb = np.asarray(image)
            if rgb.shape[-1] == 4 and np.any(rgb[..., 3] != 255):
                raise ValueError('Captured photo transparency needs a separate geometry/coverage proof: '+name)
            radiance = invert(rgb[..., :3]/255., response)
        # Some renderer texture caches alias distinct files with the same
        # basename. Names identify the actual float pixels, across build rounds.
        pixels_sha=hashlib.sha256(radiance.tobytes(order='C')).hexdigest()
        target = Path(output)/name/(name+'_'+pixels_sha[:16]+'_radiance.tiff')
        target.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(target, radiance, photometric='rgb')
        proof = {'source':str(source.resolve()), 'source_sha256':digest(source),
                 'radiance':str(target.resolve()), 'radiance_sha256':digest(target),
                 'response':str(Path(response_path).resolve()), 'response_sha256':digest(response_path),
                 'scope':__doc__.strip()}
        target.with_suffix('.json').write_text(json.dumps(proof, indent=2)+'\n')
        result[name] = proof
        assets[name] = str(target.resolve())
    return result
