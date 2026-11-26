"""Source-bound exposure evidence for primary capture review; no geometry policy."""
import hashlib
import json
import math

import cv2

from calibrated_scene_reconstruction.core.media import color_profile, exposure_metrics, extract_frames


def detail_brackets(session, source, frames, output, *, exposure_evs=(0., 2.),
                    camera_path=None, crops=None):
    """Inspect dark or bright functional details from the original video.

    This is additional source evidence, not a replacement for canonical RGB,
    a new camera, a segmentation label or a geometry/task approval. Exposure
    shifts happen in linear HDR before tone mapping. Positive brackets can
    reveal dark boundaries; they cannot recover information absent in source.
    """
    from pathlib import Path
    import numpy as np

    frames = list(frames); exposure_evs = list(exposure_evs)
    if (not frames or any(type(i) is not int or i < 0 for i in frames)
            or len(set(frames)) != len(frames)):
        raise ValueError('Distinct nonnegative source frame indices required')
    # The installed FFmpeg exposure filter accepts +/-3 per stage. Keep this
    # diagnostic inside that range; do not silently clip or return partial RGB.
    if (not exposure_evs or any(not isinstance(ev, (int, float))
                               or not math.isfinite(ev) or abs(ev) > 3 for ev in exposure_evs)
            or len(set(exposure_evs)) != len(exposure_evs)):
        raise ValueError('Distinct exposure brackets within -3..3 EV required')
    source = Path(source).resolve()
    output = (session/output).resolve()
    if not output.is_relative_to((session/'qa').resolve()):
        raise ValueError('Detail exposure evidence belongs under session QA')
    camera = None; camera_digest = None
    if camera_path is not None:
        camera_path = Path(camera_path).resolve()
        camera = json.loads(camera_path.read_text()); camera_digest = _sha(camera_path)
        k, dist = np.asarray(camera['K'], float), np.asarray(camera['dist'], float)
        if k.shape != (3, 3) or not np.isfinite(k).all() or not np.isfinite(dist).all():
            raise ValueError('Finite camera calibration required for rectified detail crops')
    crops = crops or {}
    if set(crops)-set(frames):
        raise ValueError('Detail crops must refer to selected source frames')
    profiles = {str(ev): color_profile(source, exposure_ev=ev) for ev in exposure_evs}
    before = _sha(source)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for ev in exposure_evs:
        decoded = extract_frames(source, frames, exposure_ev=ev)
        for index in frames:
            pixels = decoded[index]
            if camera is not None:
                if tuple(camera['size']) != (pixels.shape[1], pixels.shape[0]):
                    raise ValueError('Calibration dimensions differ from source frame')
                pixels = cv2.undistort(pixels, k, dist)
            box = crops.get(index, [0, 0, pixels.shape[1], pixels.shape[0]])
            if (len(box) != 4 or any(type(x) is not int for x in box)
                    or not 0 <= box[0] < box[2] <= pixels.shape[1]
                    or not 0 <= box[1] < box[3] <= pixels.shape[0]):
                raise ValueError('Detail crop must lie inside the decoded image')
            image = pixels[box[1]:box[3], box[0]:box[2]]
            name = f'frame_{index:06d}_ev{ev:+g}.png'
            if not cv2.imwrite(str(output/name), image):
                raise ValueError('Could not write detail source pixels')
            rows.append({'frame': index, 'exposure_ev': ev, 'crop_xyxy': list(box),
                         'image': name, 'sha256': _sha(output/name),
                         'exposure': exposure_metrics(image)})
    if _sha(source) != before or (camera_path is not None and _sha(camera_path) != camera_digest):
        raise ValueError('Source or calibration changed during bracket extraction')
    _write(output/'manifest.json', {'version': 1, 'source': str(source),
           'source_sha256': before, 'profiles': profiles,
           'camera': None if camera_path is None else {'path': str(camera_path), 'sha256': camera_digest},
           'rows': rows, 'canonical_cache_changed': False, 'task_pass': False,
           'scope': 'Additional exposure evidence only; inspect boundary identity before changing geometry'})
    return output


def same_profile(expected, recorded):
    """Verify existing pixels without requiring the consumer's decoder libraries.

    Producer versions remain recorded and hashed. Read-only consumers (including
    Isaac) do not decode these images. Actual cache reuse during new decoding
    still compares the complete producer profile in the capture entrypoints.
    """
    return (isinstance(recorded, dict)
            and {k: v for k, v in expected.items() if k != 'versions'}
            == {k: v for k, v in recorded.items() if k != 'versions'})


def _sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def prepare(session, source, frames):
    """Archive canonical source frames and linear HDR brackets, then require review."""
    from pathlib import Path
    source = Path(source).resolve()
    profile = color_profile(source)
    hdr = profile['mode'] == 'hdr_base_to_srgb'
    selected = {i: pixels for i, pixels in frames.items()
                if hdr or exposure_metrics(pixels)['white_at_ceiling_fraction'] > .01}
    directory = session/'capture/exposure'
    directory.mkdir(parents=True, exist_ok=True)
    number = max([int(p.name[6:]) for p in directory.glob('round_*') if p.name[6:].isdigit()], default=0)+1
    out = directory/f'round_{number}'
    out.mkdir()
    alternate = extract_frames(source, list(selected), exposure_ev=-2.) if hdr and selected else {}
    before = _sha(source)
    images, rows = {}, []
    for i, pixels in selected.items():
        name = f'frame_{i:06d}.png'
        cv2.imwrite(str(out/name), pixels)
        images[name] = _sha(out/name)
        other = None
        if i in alternate:
            other = f'frame_{i:06d}_ev_minus2.png'
            cv2.imwrite(str(out/other), alternate[i])
            images[other] = _sha(out/other)
        rows.append({'frame': i, 'canonical': name, 'alternate': other, 'exposure': exposure_metrics(pixels)})
    manifest = {'schema': 1, 'source': str(source), 'source_sha256': before,
                'profile': profile, 'rows': rows, 'images': images,
                'scope': 'Review evidence only; exposure alternates are the same camera/frame'}
    if _sha(source) != before:
        raise ValueError('Video changed while preparing exposure evidence')
    _write(out/'manifest.json', manifest)
    _write(out/'review.json', {'manifest_sha256': _sha(out/'manifest.json'),
                             'frames': {str(row['frame']): {'status': 'pending', 'basis': ''} for row in rows},
                             'instruction': 'Inspect source, canonical and alternate for task-part boundaries. '
                             'If source detail is clipped or still ambiguous, choose another view or abstain. '
                             'Do not infer connected parts merely from bright merged regions.'})
    _write(directory/'current.json', {'directory': str(out.relative_to(session.dir)),
                                     'manifest_sha256': _sha(out/'manifest.json')})
    return out


def current(session):
    """Reject stale, unreviewed or unresolved exposure evidence at S0/S6."""
    pointer = session/'capture/exposure/current.json'
    if not pointer.exists():
        color_path = session/'capture/color.json'
        if color_path.exists() and json.loads(color_path.read_text()).get('mode') == 'hdr_base_to_srgb':
            raise ValueError('HDR capture needs exposure evidence and boundary review')
        return None  # Preserved legacy captures do not acquire fabricated approval.
    info = json.loads(pointer.read_text())
    out = (session.dir/info['directory']).resolve()
    if not out.is_relative_to((session/'capture/exposure').resolve()):
        raise ValueError('Exposure evidence must remain within this capture')
    manifest_path = out/'manifest.json'
    if _sha(manifest_path) != info['manifest_sha256']:
        raise ValueError('Exposure manifest changed')
    manifest = json.loads(manifest_path.read_text())
    from pathlib import Path
    source = Path(manifest['source'])
    if _sha(source) != manifest['source_sha256'] or not same_profile(color_profile(source), manifest['profile']):
        raise ValueError('Exposure source or color conversion changed; prepare a new capture')
    for name, expected in manifest['images'].items():
        image = (out/name).resolve()
        if not image.is_relative_to(out) or _sha(image) != expected:
            raise ValueError('Exposure review pixels changed')
    review_path = out/'review.json'
    review = json.loads(review_path.read_text())
    frames = review.get('frames', {})
    if (review.get('manifest_sha256') != info['manifest_sha256']
            or set(frames) != {str(row['frame']) for row in manifest['rows']}
            or any(not isinstance(row, dict) or row.get('status') != 'resolved'
                   or not isinstance(row.get('basis'), str) or not row['basis'].strip() for row in frames.values())):
        raise ValueError('Resolve exposure and task-boundary review before S0')
    return _sha(review_path)
