"""Persistent video sources and time-addressable visual evidence for the agent."""
from __future__ import annotations

import bisect
import hashlib
import json
import math
import re
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from .artifacts import atomic_json, sha256_file
from .blender import FinalRenderResult


def retain_input_videos(root, inputs):
    """Retain the original stream, rather than only temporary overview frames."""
    root = Path(root)
    for index, record in enumerate(inputs):
        if record['kind'] != 'video':
            continue
        source = Path(record['source'])
        target = root/'inputs'/f'video_{index:03d}{source.suffix.lower()}'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha256_file(target) != record['sha256']:
            raise ValueError('The input video changed while retaining it')
        record.update(video_id=f'input-{index:03d}', retained_path=str(target.relative_to(root)))
    return inputs


def register_video(store, source, *, video_id, role):
    """Host registration for a movie to review; the agent cannot open arbitrary paths."""
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', video_id):
        raise ValueError('Invalid video identifier')
    source = Path(source).resolve()
    digest = sha256_file(source)
    target = store.root/'video-sources'/f'{digest}{source.suffix.lower()}'
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file():
        shutil.copy2(source, target)
    if sha256_file(target) != digest:
        raise ValueError('Registered video copy is corrupt')
    catalog_path = store.root/'video_catalog.json'
    catalog = json.loads(catalog_path.read_text()) if catalog_path.is_file() else {}
    catalog[video_id] = {'video_id': video_id, 'role': role,
                         'path': str(target.relative_to(store.root)), 'sha256': digest}
    atomic_json(catalog_path, catalog)
    return catalog[video_id]


def video_sources(store):
    records = {}
    manifest = store.root/'input_manifest.json'
    if manifest.is_file():
        for index, record in enumerate(json.loads(manifest.read_text()).get('inputs', [])):
            if record.get('kind') == 'video' and record.get('retained_path'):
                video_id = record.get('video_id', f'input-{index:03d}')
                records[video_id] = {'video_id': video_id, 'role': 'reference',
                                     'path': record['retained_path'], 'sha256': record['sha256']}
    catalog = store.root/'video_catalog.json'
    if catalog.is_file():
        records.update(json.loads(catalog.read_text()))
    for path in sorted(store.attempts_dir.glob('attempt-*/motion-review-*/candidate_motion.mp4')):
        video_id = f'candidate-{path.parent.parent.name}-{path.parent.name}'
        records[video_id] = {'video_id': video_id, 'role': 'candidate_simulation',
                             'path': str(path.relative_to(store.root)), 'sha256': sha256_file(path)}
    for record in records.values():
        path = (store.root/record['path']).resolve()
        if not path.is_relative_to(store.root.resolve()) or not path.is_file():
            raise ValueError('A registered video is missing or outside the retained run')
        if sha256_file(path) != record['sha256']:
            raise ValueError('A registered video changed; re-register it before reviewing')
    return records


def _index(store, record):
    import av

    output = store.root/'video-review'/record['sha256']
    output.mkdir(parents=True, exist_ok=True)
    index_path = output/'source_index.json'
    if index_path.is_file():
        return json.loads(index_path.read_text()), output
    rows = []
    with av.open(str(store.root/record['path'])) as container:
        stream = container.streams.video[0]
        fallback_fps = float(stream.average_rate) if stream.average_rate else None
        first_time = None
        for ordinal, frame in enumerate(container.decode(stream)):
            if frame.time is not None:
                raw_time = float(frame.time)
                first_time = raw_time if first_time is None else first_time
                timestamp, method = raw_time-first_time, 'decoder_presentation_timestamp'
            elif fallback_fps:
                timestamp, method = ordinal/fallback_fps, 'nominal_fps_fallback'
            else:
                raise ValueError('Video lacks both presentation timestamps and usable frame rate')
            if rows and timestamp < rows[-1]['timestamp_seconds']:
                raise ValueError('Decoder returned nonmonotonic presentation timestamps')
            rows.append({'frame_index': ordinal, 'timestamp_seconds': timestamp,
                         'timestamp_source': method, 'pts': frame.pts})
        if not rows:
            raise ValueError('Video contains no decodable frames')
        tail = (rows[-1]['timestamp_seconds']-rows[-2]['timestamp_seconds']) if len(rows) > 1 else 0
        tail = tail if tail > 0 else 1/(fallback_fps or 24)
        index = {'video_id': record['video_id'], 'source_sha256': record['sha256'],
                 'width': stream.width, 'height': stream.height, 'nominal_fps': fallback_fps,
                 'duration_seconds': rows[-1]['timestamp_seconds']+tail,
                 'total_frames': len(rows), 'frames': rows,
                 'audio_reviewed': False}
    if sha256_file(store.root/record['path']) != record['sha256']:
        raise ValueError('Video changed during indexing')
    atomic_json(index_path, index)
    return index, output


def list_video_sources(store):
    import av

    result = []
    for record in video_sources(store).values():
        with av.open(str(store.root/record['path'])) as container:
            stream = container.streams.video[0]
            duration = float(stream.duration*stream.time_base) if stream.duration is not None else None
            result.append({**record, 'width': stream.width, 'height': stream.height,
                           'declared_frames': stream.frames, 'duration_seconds': duration})
    return {'videos': result, 'input_transport': 'Decoded ordered images with presentation timestamps; no native MP4 model input',
            'next_action': 'Use inspect_video for an overview, a dense time interval or exact original frame indices; crop normalized image coordinates for detail.'}


def select_frames(index, *, mode, start_seconds, end_seconds, sample_fps, frame_indices, max_frames):
    if type(max_frames) is not int or not 1 <= max_frames <= 120:
        raise ValueError('max_frames must lie between 1 and 120')
    times = [row['timestamp_seconds'] for row in index['frames']]
    if mode == 'frames':
        if not frame_indices or any(type(i) is not int or not 0 <= i < len(times) for i in frame_indices):
            raise ValueError('Supply valid original frame indices')
        selected = sorted(set(frame_indices))
        candidates = selected
    else:
        start = 0.0 if start_seconds is None else start_seconds
        end = index['duration_seconds'] if end_seconds is None else end_seconds
        if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end <= index['duration_seconds']+1e-6:
            raise ValueError('Choose a valid [start_seconds, end_seconds) interval')
        candidates = list(range(bisect.bisect_left(times, start), bisect.bisect_left(times, end)))
        if not candidates:
            raise ValueError('No frames lie inside the requested interval')
        if mode == 'overview':
            count = min(max_frames, len(candidates))
            low, high = times[candidates[0]], times[candidates[-1]]
            selected = sorted({min(candidates[-1], bisect.bisect_left(times, low+i*(high-low)/max(count-1, 1)))
                               for i in range(count)})
        elif mode == 'interval':
            if sample_fps is None:
                selected = candidates
            else:
                if not math.isfinite(sample_fps) or sample_fps <= 0:
                    raise ValueError('sample_fps must be positive')
                count = math.ceil((end-start)*sample_fps)
                if count > 100000:
                    raise ValueError('Requested sampling density is too large')
                selected = sorted({i for step in range(count)
                                   if (i := bisect.bisect_left(times, start+step/sample_fps)) in candidates})
        else:
            raise ValueError('Choose overview, interval or frames')
    if len(selected) > max_frames:
        next_time = times[selected[max_frames]]
        raise ValueError(f'This request selects {len(selected)} frames, above max_frames={max_frames}. '
                         f'No frames were silently dropped. Split the interval near {next_time:.6f}s or explicitly choose sample_fps.')
    if not selected:
        raise ValueError('No frames match the selected sampling times')
    return selected, selected == candidates


def inspect_video(store, *, video_id, mode, start_seconds, end_seconds, sample_fps, frame_indices, crop, max_frames, detail=None):
    import av

    records = video_sources(store)
    if video_id not in records:
        raise ValueError('Unknown video_id; call list_video_sources')
    record = records[video_id]
    index, cache = _index(store, record)
    selected, complete = select_frames(index, mode=mode, start_seconds=start_seconds,
                                      end_seconds=end_seconds, sample_fps=sample_fps,
                                      frame_indices=frame_indices, max_frames=max_frames)
    if detail not in {None, 'overview', 'full'}:
        raise ValueError('detail must be overview, full or null (automatic)')
    full = detail == 'full' or (detail is None and mode in {'frames', 'interval'} and len(selected) <= 24)
    if full and len(selected) > 24:
        raise ValueError('Full-resolution viewing supports up to 24 frames per call. Split dense intervals or exact-frame lists; no selected frames were dropped.')
    if crop is not None and (len(crop) != 4 or any(not math.isfinite(x) or not 0 <= x <= 1 for x in crop)
                             or crop[0] >= crop[2] or crop[1] >= crop[3]):
        raise ValueError('crop is [left, top, right, bottom] in normalized displayed image coordinates')
    key = hashlib.sha256(json.dumps({'frames': selected, 'crop': crop}, sort_keys=True).encode()).hexdigest()[:16]
    output = cache/key
    output.mkdir(parents=True, exist_ok=True)
    images, image_evidence = [], []
    remaining = set(selected)
    decoded = {}
    with av.open(str(store.root/record['path'])) as container:
        for ordinal, frame in enumerate(container.decode(video=0)):
            if ordinal in remaining:
                image = frame.to_image().convert('RGB')
                rotation = getattr(frame, 'rotation', 0)
                if rotation:
                    image = image.rotate(rotation, expand=True)
                if crop is not None:
                    width, height = image.size
                    image = image.crop((math.floor(crop[0]*width), math.floor(crop[1]*height),
                                        math.ceil(crop[2]*width), math.ceil(crop[3]*height)))
                path = output/f'frame_{ordinal:08d}.jpg'
                image.save(path, quality=96)
                decoded[ordinal] = path
                remaining.remove(ordinal)
            if not remaining:
                break
    if remaining:
        raise ValueError('Video decoding did not recover all requested frames')
    for offset in range(0, len(selected), 12):
        sheet = Image.new('RGB', (1536, 936), '#eeeeee')
        draw = ImageDraw.Draw(sheet)
        for local, ordinal in enumerate(selected[offset:offset+12]):
            with Image.open(decoded[ordinal]) as image:
                thumb = ImageOps.contain(image, (384, 288))
                x, y = local % 4*384, local // 4*312
                sheet.paste(thumb, (x+(384-thumb.width)//2, y+(288-thumb.height)//2))
            timestamp = index['frames'][ordinal]['timestamp_seconds']
            draw.text((x+5, y+290), f'original frame {ordinal} | {timestamp:.6f}s', fill='black')
        path = output/f'sheet_{offset//12:03d}.jpg'
        sheet.save(path, quality=94)
        images.append(path)
        image_evidence.append({'path': str(path), 'caption': f'Video {video_id}: chronological overview of original frames {selected[offset:offset+12]}. Thumbnails; request full detail/crops to judge small features.'})
    if full or (crop is not None and len(selected) <= 24):
        for ordinal in selected:
            path = decoded[ordinal]
            timestamp = index['frames'][ordinal]['timestamp_seconds']
            with Image.open(path) as image:
                dimensions = list(image.size)
            images.append(path)
            image_evidence.append({'path': str(path), 'caption': f'Video {video_id}, original frame {ordinal}, presentation time {timestamp:.6f}s, decoded image {dimensions[0]}x{dimensions[1]}, normalized crop {crop}. Full retained image follows.'})
    summary = {'ok': True, 'video_id': video_id, 'source_sha256': record['sha256'],
               'mode': mode, 'source_total_frames': index['total_frames'], 'source_duration_seconds': index['duration_seconds'],
               'complete_requested_interval': complete, 'complete_video_coverage': selected == list(range(index['total_frames'])),
               'sampling': 'all frames in the requested interval' if mode == 'interval' and sample_fps is None else 'explicit frame selection or sampled overview',
               'crop': crop, 'audio_reviewed': False,
               'full_resolution_frames_returned': selected if full or (crop is not None and len(selected) <= 24) else [],
               'image_evidence': image_evidence,
               'frames': [{**index['frames'][i], 'image': str(decoded[i].relative_to(store.root))} for i in selected],
               'directory': str(output.relative_to(store.root)),
               'next_action': 'Use exact frame indices or smaller dense intervals to resolve motion/occlusion/contact. Overview coverage alone does not certify no transient defects. Distinguish observations from hidden geometry or unmeasured scale assumptions.'}
    atomic_json(output/'review.json', summary)
    store.append_event({'type': 'video_inspected', 'video_id': video_id, 'mode': mode, 'frames': selected,
                        'complete_video_coverage': summary['complete_video_coverage']})
    return FinalRenderResult(video_id, True, summary, images)


def promote_video_frames(store, *, video_id, frame_indices, reason):
    """Make selected uncropped views addressable by existing geometry/camera tools."""
    if not reason.strip() or not frame_indices or len(set(frame_indices)) > 24:
        raise ValueError('Explain why these 1..24 frames are useful as reference views')
    result = inspect_video(store, video_id=video_id, mode='frames', start_seconds=None,
                           end_seconds=None, sample_fps=None, frame_indices=frame_indices,
                           crop=None, max_frames=24, detail='full')
    run_path = store.root/'run.json'
    run = json.loads(run_path.read_text())
    references = run['references']
    promoted = []
    for frame in result.summary['frames']:
        provenance = {'video_id': video_id, 'source_sha256': result.summary['source_sha256'],
                      'frame_index': frame['frame_index'], 'timestamp_seconds': frame['timestamp_seconds']}
        existing = next((i for i, reference in enumerate(references) if reference.get('video_frame') == provenance), None)
        if existing is None:
            existing = len(references)
            source = store.root/frame['image']
            destination = store.root/'references'/f'video_{existing:04d}_{frame["frame_index"]:08d}.jpg'
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            references.append({'source_name': destination.name, 'path': str(destination.relative_to(store.root)),
                               'sha256': sha256_file(destination), 'video_frame': provenance})
        promoted.append({**provenance, 'reference_index': existing, 'path': references[existing]['path']})
    atomic_json(run_path, run)
    store.append_event({'type': 'video_frames_promoted', 'video_id': video_id, 'frames': promoted, 'reason': reason})
    return {'ok': True, 'references': promoted,
            'next_action': 'These reference indices can be selected by geometry/camera tools. Use different viewpoints with a consistent object configuration for multiview geometry; do not combine articulated/deforming states as if they were one rigid scene. Scale and physical parameters remain unmeasured.'}
