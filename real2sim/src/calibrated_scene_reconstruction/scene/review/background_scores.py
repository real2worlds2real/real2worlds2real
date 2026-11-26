"""Captured-view background detail diagnostics; never score novel views as paired."""
import argparse
import json
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from calibrated_scene_reconstruction.core.paths import resolve
from calibrated_scene_reconstruction.scene.capture.valid_pixels import ValidPixels, resize_valid
from calibrated_scene_reconstruction.scene.review.cameras import digest
from calibrated_scene_reconstruction.scene.session import Session


def score(reference, rendered, domain):
    reference, rendered = np.asarray(reference), np.asarray(rendered)
    domain = np.asarray(domain, bool)
    if (reference.shape != rendered.shape or reference.ndim != 3 or reference.shape[2] != 3
            or domain.shape != reference.shape[:2] or min(reference.shape[:2]) < 3):
        raise ValueError('Same RGB rasters and a matching observed-domain mask required')
    if not np.isfinite(reference).all() or not np.isfinite(rendered).all():
        raise ValueError('RGB diagnostics require finite image data')
    count = int(domain.sum())
    if not count:
        return {'pixels': 0, 'psnr_db': None, 'mean_abs_rgb': None, 'gradient_energy_ratio': None}
    a, b = reference.astype(float)/255., rendered.astype(float)/255.
    error = (a-b)[domain]
    inner = cv2.erode(domain.astype('uint8'), np.ones((3, 3), 'uint8')) > 0
    energies = []
    for rgb in (a, b):
        grey = (rgb@np.array([.2126, .7152, .0722])).astype('float32')
        dx = cv2.Sobel(grey, cv2.CV_32F, 1, 0)
        dy = cv2.Sobel(grey, cv2.CV_32F, 0, 1)
        energies.append(float((dx*dx+dy*dy)[inner].mean()) if inner.any() else 0.)
    return {'pixels': count, 'psnr_db': float(-10*np.log10(max(float((error*error).mean()), 1e-12))),
            'mean_abs_rgb': float(np.abs(error).mean()),
            'gradient_energy_ratio': energies[1]/energies[0] if energies[0] > 1e-10 else None}


def camera_manifest(review):
    bound = {str(resolve(p).resolve()): p for p in review['input_sha256']}
    explicit = review.get('camera_manifest')
    if explicit is not None:
        path = str(resolve(explicit).resolve())
        if path not in bound:
            raise ValueError('Camera manifest must be bound by the render input hashes')
        return json.loads(Path(path).read_text())
    # Older renders recorded hashes but no explicit role. Discover by the camera
    # contract, never by a capture-specific filename or prefix.
    candidates = []
    for path in bound:
        if Path(path).suffix == '.json':
            value = json.loads(Path(path).read_text())
            if isinstance(value, dict) and all(k in value for k in ('anchors', 'width', 'height', 'truth_for_free_views')):
                candidates.append(value)
    if len(candidates) != 1:
        raise ValueError('Review must bind exactly one camera manifest')
    return candidates[0]


def run(session, review_directory, output, mask_suffix='_sam'):
    if not re.fullmatch(r'_[A-Za-z0-9_]+', mask_suffix):
        raise ValueError('Use a local independently observed background-mask suffix')
    output, review_directory = Path(output), Path(review_directory)
    if output.exists():
        raise ValueError('Preserve previous diagnostics; use a new output')
    path = review_directory/'review.json'
    review = json.loads(path.read_text())
    inputs = {**review['input_sha256'], str(path.resolve()): digest(path)}
    cameras = camera_manifest(review)
    if cameras.get('truth_for_free_views') is not False:
        raise ValueError('Novel views must be explicitly unpaired')
    for source, expected in inputs.items():
        if digest(resolve(source)) != expected:
            raise ValueError('Review input changed: '+source)
    output.mkdir(parents=True)
    validity = ValidPixels(session)
    rows = []
    for anchor, result in zip(cameras['anchors'], review['views'], strict=True):
        indices = [i for i, view in enumerate(anchor['views']) if view['offset_m'] == 0]
        if len(indices) != 1:
            raise ValueError('Expected exactly one matched captured camera')
        source = Path(anchor['source'])
        if Path(result['source']).resolve() != source.resolve():
            raise ValueError('Camera/source review ordering differs')
        render = Path(result['renders'][indices[0]])
        prediction = np.array(Image.open(render).convert('RGB'))
        raw = np.array(Image.open(source).convert('RGB'))
        if [raw.shape[1], raw.shape[0]] != anchor['source_size']:
            raise ValueError('Source raster dimensions differ from calibrated review manifest')
        mask_path = session/'splat/data/images'/(source.stem+mask_suffix+'.png')
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != raw.shape[:2]:
            raise ValueError('Independently observed background mask missing or wrong raster: '+str(mask_path))
        valid = validity.get(source.name, raw.shape[:2])
        domain = resize_valid((mask > 128) & valid, (prediction.shape[1], prediction.shape[0]))
        reference = np.array(Image.fromarray(raw).resize((prediction.shape[1], prediction.shape[0]), Image.Resampling.BICUBIC))
        metrics = score(reference, prediction, domain)
        inputs[str(mask_path.resolve())] = digest(mask_path)
        inputs[str(render.resolve())] = digest(render)
        rows.append({'frame': source.name, 'offset_m': 0., **metrics})
        error = np.clip(np.abs(reference.astype(float)-prediction).mean(2)*3., 0, 255).astype('uint8')
        error = cv2.applyColorMap(error, cv2.COLORMAP_INFERNO)[:, :, ::-1]
        error[~domain] = 24
        panel = Image.new('RGB', (1920, 390), (24, 24, 24))
        draw = ImageDraw.Draw(panel)
        for i, (im, label) in enumerate(zip([reference, prediction, error], ['Original '+source.name, 'Matched exported scene', 'RGB error in observed background only'])):
            panel.paste(ImageOps.pad(Image.fromarray(im), (640, 360)), (640*i, 30))
            draw.text((640*i+8, 8), label, fill='white')
        panel.save(output/(source.stem+'.jpg'), quality=95)
    inputs.update({str(resolve(p)): h for p, h in validity.inputs.items()})
    for source, expected in inputs.items():
        if digest(resolve(source)) != expected:
            raise ValueError('Diagnostic input changed: '+source)
    supported = [r for r in rows if r['pixels']]
    record = {'scope': 'Clamped sRGB PNGs, matched captured views only. Independent background labels intersect geometric rectification validity. No novel-view ground truth or automatic photorealism acceptance.',
              'gradient_interpretation': 'Detail-energy diagnostic, not a quality gate; noise, blur, lighting and registration affect it.',
              'frames': rows, 'mean_frame_psnr_db': float(np.mean([r['psnr_db'] for r in supported])) if supported else None,
              'input_sha256': inputs, 'acceptance_automated': False}
    (output/'scores.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session')
    parser.add_argument('review_directory')
    parser.add_argument('--output', required=True)
    parser.add_argument('--mask-suffix', default='_sam')
    args = parser.parse_args()
    record = run(Session(args.session), args.review_directory, args.output, args.mask_suffix)
    print('Captured frames', len(record['frames']), 'mean PSNR', record['mean_frame_psnr_db'], 'review still required')


if __name__ == '__main__':
    main()
