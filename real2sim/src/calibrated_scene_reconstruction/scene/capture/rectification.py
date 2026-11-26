"""Bind selected capture pixels to the camera used by sparse reconstruction.

Self-calibration follows frame selection. Re-extract those exact source frames
and rectify them before perception; never undistort already resized/rectified
pixels a second time. Preserve changed pixels and stale SAM cache as evidence.
"""
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def camera_identity(camera):
    k = np.asarray(camera['K'], float); d = np.asarray(camera['dist'], float)
    size = camera['size']
    if (k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0
            or d.shape not in [(4,), (5,), (8,), (12,), (14,)] or not np.isfinite(d).all()
            or len(size) != 2 or any(type(n) is not int or n <= 0 for n in size)):
        raise ValueError('Finite camera intrinsics/distortion and source dimensions required')
    return {'K': k.tolist(), 'dist': d.tolist(), 'size': size}


def rectify_frame(frame, camera, output_size):
    """Same full-resolution undistortion followed by resize as capture.py."""
    camera = camera_identity(camera)
    if [frame.shape[1], frame.shape[0]] != camera['size']:
        raise ValueError('Source video and camera dimensions differ')
    rgb = frame[:, :, ::-1]
    if any(camera['dist']):
        rgb = cv2.undistort(rgb, np.asarray(camera['K']), np.asarray(camera['dist']))
    image = Image.fromarray(rgb)
    if image.size != tuple(output_size):
        image = image.resize(tuple(output_size), Image.Resampling.LANCZOS)
    return image


def ensure(session):
    camera = session.config('camera.json', {})
    selection = session/'capture/keyframes.json'
    if not camera or not selection.is_file():
        return False
    if (session/'qa/independent_freeze.json').exists():
        raise ValueError('Frozen capture cannot be re-rectified')
    from calibrated_scene_reconstruction.core.media import VideoCapture, color_profile
    from calibrated_scene_reconstruction.core.paths import resolve
    camera = camera_identity(camera)
    selected = json.loads(selection.read_text()); source = resolve(selected['video'])
    profile = color_profile(source)
    if session.read_json('capture/color.json') != profile:
        raise ValueError('Capture color conversion changed; use a fresh source session')
    paths = {int(selected['photo']): [session/'capture/rgb.png']}
    for i in selected['views']:
        paths.setdefault(int(i), []).append(session/f'capture/views/f{i:04d}.png')
    expected = {f'f{i:04d}.png' for i in selected['views']}
    if {p.name for p in (session/'capture/views').glob('*.png')} != expected:
        raise ValueError('Mixed-camera supplementary photos need their own rectification, not primary video intrinsics')
    basis = {'camera': camera, 'source_sha256': digest(source), 'selection_sha256': digest(selection),
             'color_profile': profile}
    pointer = session/'capture/rectification.json'
    if pointer.exists():
        previous = json.loads(pointer.read_text())
        if previous.get('basis') == basis and all(
                (session/path).is_file() and digest(session/path) == sha
                for path, sha in previous['images'].items()):
            return False
    root = session/'capture/rectification'; root.mkdir(exist_ok=True)
    number = max([int(p.name[6:]) for p in root.glob('round_*') if p.name[6:].isdigit()], default=0)+1
    out = root/f'round_{number}'; out.mkdir()
    (out/'basis.json').write_text(json.dumps(basis, indent=2)+'\n')
    sizes = {p: Image.open(p).size for values in paths.values() for p in values}
    changed = []; images = {}; pending = {}
    with VideoCapture(source) as reader:
        for i in range(max(paths)+1):
            ok, pixels = reader.read()
            if not ok:
                raise ValueError(f'Source ended before selected frame {i}')
            for p in paths.get(i, []):
                relative = str(p.relative_to(session.dir)); target = out/'new'/p.relative_to(session/'capture')
                target.parent.mkdir(parents=True, exist_ok=True)
                rectify_frame(pixels, camera, sizes[p]).save(target)
                pending[p] = target; images[relative] = digest(target)
                if digest(p) != images[relative]:
                    before = out/'before'/p.relative_to(session/'capture')
                    before.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, before)
                    changed.append(relative)
    # Publish only after every requested frame decoded successfully.
    for p, target in pending.items():
        shutil.copy2(target, p)
    cache = session/'parsing/multiview/sam3_cache.npz'
    if changed and cache.exists():
        shutil.move(cache, out/'previous_sam3_cache.npz')
    report = {'basis': basis, 'images': images, 'changed': changed,
              'requires_fresh_perception_and_multiview': bool(changed), 'accepted': False,
              'scope': 'Pixel preparation only; no camera, object or task acceptance.'}
    (out/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    pointer.write_text(json.dumps(report, indent=2)+'\n')
    print(f'[rectification] {len(images)} selected images, {len(changed)} changed; old pixels retained', flush=True)
    return bool(changed)


def main():
    import argparse
    from calibrated_scene_reconstruction.scene.session import Session
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('session')
    ensure(Session(parser.parse_args().session))


if __name__ == '__main__':
    main()
