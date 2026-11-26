"""Agent-reviewed media uses, separated from static scene camera observations.

The plan chooses evidence roles; filenames never choose scale or object identity.
Prepared closeups and demonstrations are modeling evidence, not registered static
cameras. A new plan invalidates dependent object and task checks.
"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from calibrated_scene_reconstruction.core.paths import ROOT
from calibrated_scene_reconstruction.core.media import VideoCapture

ROLES = {'scene', 'geometry', 'appearance', 'motion', 'metric_calibration'}
IMAGES = {'.jpg', '.jpeg', '.png', '.webp', '.tif', '.tiff'}
VIDEOS = {'.mp4', '.mov', '.m4v', '.avi', '.mkv', '.webm'}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _path(item, root):
    path = Path(item['path'])
    if path.is_absolute():
        raise ValueError('Media paths must be relative to the workspace data directory')
    path = (root/path).resolve()
    if not path.is_relative_to((root/'data').resolve()) or not path.is_file():
        raise ValueError('Import media into workspace data/ before preparing evidence')
    return path


def validate(plan, root=None):
    """Validate provenance and role boundaries without decoding any media."""
    from .inputs import reject_measured_sizes
    reject_measured_sizes(plan, 'inputs.json')
    root = Path(ROOT if root is None else root).resolve()
    if not isinstance(plan, dict) or plan.get('version') != 1 or not _text(plan.get('agent_review')):
        raise ValueError('inputs.json requires version1 and an agent review')
    items = plan.get('assets')
    if not isinstance(items, list) or not items:
        raise ValueError('At least one reviewed input is required')
    identifiers, hashes, scene_ids, scale_ids = set(), set(), set(), set()
    for item in items:
        identifier = item.get('id', '')
        if not isinstance(identifier, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', identifier) or identifier in identifiers:
            raise ValueError('Media identifiers must be unique safe names')
        identifiers.add(identifier)
        path = _path(item, root)
        kind = item.get('kind')
        if kind not in {'image', 'video'} or path.suffix.lower() not in (IMAGES if kind == 'image' else VIDEOS):
            raise ValueError('Media kind and file extension disagree')
        sha = digest(path)
        if sha != item.get('sha256') or sha in hashes:
            raise ValueError('Media changed, or the same media was declared twice')
        hashes.add(sha)
        uses = item.get('uses')
        if not isinstance(uses, list) or not uses:
            raise ValueError('Each input requires explicit uses')
        intervals = []
        for use in uses:
            role = use.get('role')
            if role not in ROLES or not all(_text(use.get(k)) for k in ('target', 'state', 'basis', 'camera_group')):
                raise ValueError('Each use requires a role, target, state, basis and camera group')
            if use.get('identity') not in {'confirmed', 'unconfirmed'}:
                raise ValueError('Object identity must be explicitly reviewed')
            if use['identity'] == 'unconfirmed' and role != 'appearance':
                raise ValueError('Unconfirmed identity cannot constrain geometry, motion or scale')
            if 'state_relation' in use and use['state_relation'] not in {'same', 'different', 'unknown'}:
                raise ValueError('State relation must compare this use to the primary initial state')
            parts = []
            for field in ('invariant_parts', 'state_dependent_parts'):
                value = use.get(field, [])
                if (not isinstance(value, list) or any(not _text(name) for name in value)
                        or len(set(value)) != len(value)):
                    raise ValueError('Part state scopes must be distinct named lists')
                parts.append(set(value))
            if parts[0] & parts[1]:
                raise ValueError('A part cannot be both invariant and state-dependent in the same use')
            if kind == 'video':
                interval = use.get('interval_s')
                samples = use.get('sample_times_s')
                if (not isinstance(interval, list) or len(interval) != 2
                        or any(type(x) not in (int, float) or not math.isfinite(x) for x in interval)
                        or not 0 <= interval[0] < interval[1]):
                    raise ValueError('Video use requires a finite increasing time interval')
                if (not isinstance(samples, list) or not 1 <= len(samples) <= 120
                        or any(type(x) not in (int, float) or not math.isfinite(x)
                               or not interval[0] <= x < interval[1] for x in samples)
                        or sorted(set(samples)) != samples or (role == 'motion' and len(samples) < 3)):
                    raise ValueError('Choose bounded ordered samples within each interval; motion needs at least3')
                intervals.append((role, interval))
            elif 'interval_s' in use or 'sample_times_s' in use or role == 'motion':
                raise ValueError('A still image has no motion sequence or video interval')
            if role == 'scene':
                scene_ids.add(identifier)
            if role == 'metric_calibration':
                if use.get('method') not in {'board', 'robot_joints', 'robot_visual', 'robot_parts'}:
                    raise ValueError('Metric calibration requires board or robot evidence, never object dimensions')
                scale_ids.add(identifier)
        for role, span in intervals:
            if role == 'scene' and any(other == 'motion' and max(span[0], span2[0]) < min(span[1], span2[1])
                                       for other, span2 in intervals):
                raise ValueError('Moving-object intervals cannot enter static scene reconstruction')
    if plan.get('primary_scene') not in scene_ids:
        raise ValueError('Select a reviewed primary scene input')
    scale = plan.get('metric_scale', {})
    if (scale.get('source') not in scale_ids
            or scale.get('method') not in {'board', 'robot_joints', 'robot_visual', 'robot_parts'}
            or not _text(scale.get('basis'))):
        raise ValueError('Declare the source and method for metric scale')
    matching = [use for item in items if item['id'] == scale['source'] for use in item['uses']
                if use['role'] == 'metric_calibration' and use.get('method') == scale['method']]
    if not matching:
        raise ValueError('Metric scale method differs from the reviewed input use')
    return plan


def load(session):
    if not hasattr(session, 'config_dir'):
        return None
    path = session.config_dir/'inputs.json'
    return validate(json.loads(path.read_text())) if path.exists() else None


def scene_intervals(session, path, *, primary=False):
    """Fail closed when an undeclared/closeup/motion input enters static SfM.

    None preserves legacy whole-file behavior. An empty list represents an
    approved scene still image. Video ranges are half-open, in seconds.
    """
    plan = load(session)
    if plan is None:
        return None
    path = (ROOT/Path(path)).resolve()
    for item in plan['assets']:
        if _path(item, ROOT) == path:
            if primary and item['id'] != plan['primary_scene']:
                raise ValueError('Capture input differs from the reviewed primary scene')
            uses = [use for use in item['uses'] if use['role'] == 'scene']
            if uses:
                if len({use['camera_group'] for use in uses}) != 1:
                    raise ValueError('A static scene stream needs one camera group; independently calibrate separate lens/zoom segments')
                return [use['interval_s'] for use in uses] if item['kind'] == 'video' else []
    raise ValueError('Static scene input has no reviewed scene use; keep closeups and motion separate')


def check_scene_view_camera(session, primary, view):
    """Legacy --views applies primary rectification; do not mix another lens."""
    plan = load(session)
    if plan is None:
        return
    groups = []
    for path in (primary, view):
        source = (ROOT/Path(path)).resolve()
        groups.append({use['camera_group'] for item in plan['assets'] if _path(item, ROOT) == source
                       for use in item['uses'] if use['role'] == 'scene'})
    if len(groups[0]) != 1 or groups[0] != groups[1]:
        raise ValueError('--views uses primary camera rectification; different camera groups need independent registration')


def in_intervals(seconds, intervals):
    return intervals is None or any(lo <= seconds < hi for lo, hi in intervals)


def assert_static_frames(session, names, videos, fps):
    """Reject retained frames outside reviewed static intervals on cache reuse."""
    if load(session) is None:
        return
    intervals = [scene_intervals(session, path, primary=i == 0) for i, path in enumerate(videos)]
    for name in names:
        match = re.fullmatch(r'(\d+)_(\d+)\.jpg', name)
        if match is None:
            raise ValueError('Retained static camera has an unknown media/frame identity: '+name)
        video, frame = map(int, match.groups())
        if (not 0 <= video < len(videos) or not math.isfinite(fps[video]) or fps[video] <= 0
                or not in_intervals(frame/fps[video], intervals[video])):
            raise ValueError('Retained camera lies outside reviewed static scene inputs: '+name)


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def prepare(session, *, root=None):
    """Extract only agent-selected frames into role-separated, append-only QA.

    Additional camera groups retain raw intrinsics. No primary camera calibration,
    metric pose or static-scene membership is invented for them.
    """
    import cv2
    from calibrated_scene_reconstruction.core.media import extract_frames, exposure_metrics, frame_timestamp
    root = Path(ROOT if root is None else root).resolve()
    plan = validate(json.loads((session.config_dir/'inputs.json').read_text()), root)
    directory = session/'capture/additional_inputs'
    directory.mkdir(parents=True, exist_ok=True)
    number = max([int(p.name[6:]) for p in directory.glob('round_*') if p.name[6:].isdigit()], default=0)+1
    out = directory/f'round_{number}'
    out.mkdir()
    _write(out/'plan.json', plan)
    rows, boards, profiles, exposure_frames = [], {}, {}, {}
    for item in plan['assets']:
        source = _path(item, root)
        reader = VideoCapture(str(source)) if item['kind'] == 'video' else None
        if reader is not None:
            fps = reader.get(cv2.CAP_PROP_FPS)
            count = reader.get(cv2.CAP_PROP_FRAME_COUNT)
            if not reader.isOpened() or not math.isfinite(fps) or fps <= 0 or count <= 0:
                raise ValueError('Cannot read reviewed video: '+item['id'])
            profiles[item['id']] = reader.profile
            selected = sorted({round(t*fps) for use in item['uses'] for t in use.get('sample_times_s', [])})
            pixels_by_frame = extract_frames(source, selected)
            hdr = reader.profile['mode'] == 'hdr_base_to_srgb'
            dark_by_frame = extract_frames(source, selected, exposure_ev=-2.) if hdr else {}
        try:
            for i, use in enumerate(item['uses']):
                if reader is not None and use['interval_s'][1] > count/fps+1/fps:
                    raise ValueError('Reviewed interval exceeds the actual video duration')
                images = []
                for j, seconds in enumerate(use.get('sample_times_s', [None])):
                    if reader is None:
                        image = ImageOps.exif_transpose(Image.open(source)).convert('RGB')
                        frame = None
                        decoded_time = None
                    else:
                        frame = round(seconds*fps)
                        pixels = pixels_by_frame[frame]
                        image = Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))
                        decoded_time = frame_timestamp(source, frame)
                    target = out/f'{item["id"]}_{i}_{j:03d}.png'
                    image.save(target)
                    quality = exposure_metrics(np.asarray(image)[:, :, ::-1])
                    alternate = None
                    if reader is not None and hdr:
                        alternate = target.with_name(target.stem+'_ev_minus2.png')
                        Image.fromarray(dark_by_frame[frame][:, :, ::-1]).save(alternate)
                        boards[str(alternate.relative_to(session.dir))] = digest(alternate)
                    if (reader is not None and hdr) or quality['white_at_ceiling_fraction'] > .01:
                        exposure_frames[str(target.relative_to(session.dir))] = {'status': 'pending', 'basis': ''}
                    images.append((target, image.copy(), seconds))
                    rows.append({'asset': item['id'], 'use': i, 'role': use['role'], 'target': use['target'],
                                 'state': use['state'], 'identity': use['identity'], 'camera_group': use['camera_group'],
                                 'seconds': seconds, 'frame': frame, 'image': str(target.relative_to(session.dir)),
                                 'decoded_time_s': decoded_time, 'upright_size': list(image.size),
                                 'exposure': quality,
                                 'exposure_alternate': str(alternate.relative_to(session.dir)) if alternate else None,
                                 'sha256': digest(target), 'registered_camera': False,
                                 'metric_scale_established': False})
                width, height, columns = 384, 240, 3
                board = Image.new('RGB', (columns*width, math.ceil(len(images)/columns)*height), '#242424')
                draw = ImageDraw.Draw(board)
                for j, (_, image, seconds) in enumerate(images):
                    image.thumbnail((width, height-38))
                    x, y = j % columns*width, j//columns*height
                    board.paste(image, (x+(width-image.width)//2, y))
                    draw.text((x+6, y+height-34), f'{item["id"]} / {use["role"]} / {seconds}s', fill='white')
                board_path = out/f'{item["id"]}_{i}_review.jpg'
                board.save(board_path)
                boards[str(board_path.relative_to(session.dir))] = digest(board_path)
        finally:
            if reader is not None:
                reader.release()
    # Detect source changes during decoding before publishing a current pointer.
    validate(plan, root)
    manifest = {'version': 1, 'plan_sha256': _hash(plan), 'rows': rows, 'review_boards': boards,
                'color_profiles': profiles,
                'scope': 'Agent evidence only; closeups/motion have no registered static cameras or independent metric scale'}
    _write(out/'manifest.json', manifest)
    if exposure_frames:
        _write(out/'exposure_review.json', {
            'manifest_sha256': digest(out/'manifest.json'), 'frames': exposure_frames,
            'instruction': 'Inspect original video, canonical frame and exposure alternate. Resolve task-part boundaries; '
                           'record what is visible, what remains hidden and why. Darkening cannot recover clipped source. '
                           'Mark resolved only when remaining ambiguity does not affect the declared use; otherwise select '
                           'another view or abstain. Brackets share one source frame, not independent cameras.'})
    _write(directory/'current.json', {'directory': str(out.relative_to(session.dir)),
                                     'manifest_sha256': digest(out/'manifest.json')})
    return out


def evidence(session, *, require_review=True):
    """Fresh source and prepared-image provenance for object/task round inputs."""
    if not hasattr(session, 'config_dir') or not (session.config_dir/'inputs.json').exists():
        return None
    plan = load(session)
    pointer = session/'capture/additional_inputs/current.json'
    if not pointer.exists():
        raise ValueError('Prepare and review the declared additional inputs before fitting')
    current = json.loads(pointer.read_text())
    out = (session.dir/current['directory']).resolve()
    if not out.is_relative_to((session/'capture/additional_inputs').resolve()):
        raise ValueError('Additional-input manifest must remain inside this session')
    path = out/'manifest.json'
    manifest = json.loads(path.read_text())
    if digest(path) != current['manifest_sha256'] or manifest['plan_sha256'] != _hash(plan):
        raise ValueError('Additional-input plan or prepared evidence changed; prepare again')
    from calibrated_scene_reconstruction.core.media import color_profile
    from calibrated_scene_reconstruction.scene.capture.exposure import same_profile
    for item in plan['assets']:
        if item['kind'] == 'video':
            profile = color_profile(_path(item, ROOT))
            previous = manifest.get('color_profiles', {}).get(item['id'])
            if not same_profile(profile, previous) and (previous is not None or profile['mode'] != 'opencv_sdr'):
                raise ValueError('Video color management changed; prepare and review new evidence')
    for row in manifest['rows']:
        image = (session.dir/row['image']).resolve()
        if not image.is_relative_to(out) or digest(image) != row['sha256']:
            raise ValueError('Prepared input image changed; prepare again')
    if not manifest.get('review_boards'):
        raise ValueError('Review boards have no source hashes; prepare again')
    for name, expected in manifest['review_boards'].items():
        board = (session.dir/name).resolve()
        if not board.is_relative_to(out) or digest(board) != expected:
            raise ValueError('Prepared review board changed; prepare again')
    required = {row['image'] for row in manifest['rows']
                if (manifest.get('color_profiles', {}).get(row['asset'], {}).get('mode') == 'hdr_base_to_srgb'
                    or row.get('exposure', {}).get('white_at_ceiling_fraction', 0) > .01)}
    exposure_sha, unresolved = None, []
    if required:
        review_path = out/'exposure_review.json'
        review = json.loads(review_path.read_text()) if review_path.exists() else {}
        reviewed = review.get('frames', {})
        if review.get('manifest_sha256') != digest(path) or set(reviewed) != required:
            raise ValueError('Exposure review does not bind the prepared frames')
        unresolved = [name for name, v in reviewed.items()
                      if not isinstance(v, dict) or v.get('status') != 'resolved' or not _text(v.get('basis'))]
        if unresolved and require_review:
            raise ValueError('Resolve exposure and task-boundary review before fitting')
        exposure_sha = digest(review_path)
    return {'plan_sha256': _hash(plan), 'manifest_sha256': digest(path),
            'directory': str(out.relative_to(session.dir.resolve())), 'metric_scale': plan['metric_scale'],
            'engine_sha256': digest(Path(__file__)),
            'exposure_review_sha256': exposure_sha,
            'unresolved_exposure_frames': unresolved,
            'assets': {item['id']: item['sha256'] for item in plan['assets']}}


def modeling_inputs(session, target=None):
    """Expose role/state-aware media to the same agent that writes the model."""
    provenance = evidence(session, require_review=False)
    if provenance is None:
        return {'ok': True, 'supplemental_evidence': None, 'uses': [],
                'evidence_images': ['capture/rgb.png'] if (session/'capture/rgb.png').exists() else []}
    out = session/provenance['directory']
    plan = json.loads((out/'plan.json').read_text())
    manifest = json.loads((out/'manifest.json').read_text())
    uses, images = [], []
    for asset in plan['assets']:
        for index, use in enumerate(asset['uses']):
            matches = target is None or use['target'] == target or use['target'].startswith(target+'/')
            if not matches and use['role'] not in {'scene', 'metric_calibration'}:
                continue
            board = out/f'{asset["id"]}_{index}_review.jpg'
            images.append(str(board.relative_to(session.dir)))
            images.extend(row['exposure_alternate'] for row in manifest['rows']
                          if row['asset'] == asset['id'] and row['use'] == index and row.get('exposure_alternate'))
            uses.append({'asset': asset['id'], **use,
                         'observations': [row for row in manifest['rows'] if row['asset'] == asset['id'] and row['use'] == index]})
    return {'ok': True, 'supplemental_evidence': provenance, 'agent_review': plan['agent_review'],
            'exposure_review': str((out/'exposure_review.json').relative_to(session.dir))
                               if (out/'exposure_review.json').exists() else None,
            'uses': uses, 'evidence_images': images,
            'scope': 'Agent modeling evidence. Estimate each camera and part state before any metric image loss; motion is not a static scene. Declared scale source still requires its normal calibration gate.'}


def main():
    from calibrated_scene_reconstruction.scene.session import Session
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session')
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    session = Session(args.session)
    if args.prepare:
        print(prepare(session))
    else:
        print(json.dumps(evidence(session), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
