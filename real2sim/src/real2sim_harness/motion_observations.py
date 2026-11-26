"""Import video motion evidence from existing trackers, masks or measured points.

No tracker/model is selected or launched implicitly. Presentation times and
pixel-domain checks keep external perception interchangeable without promoting
its estimates to metric or physical ground truth.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .artifacts import atomic_json, sha256_file


def video_timeline(path):
    import av

    rows, size, first = [], None, None
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if frame.time is None or not np.isfinite(float(frame.time)):
                raise ValueError("Motion measurement requires decoder presentation timestamps")
            timestamp = float(frame.time)
            first = timestamp if first is None else first
            timestamp -= first
            rotation = int(getattr(frame, "rotation", 0)) % 360
            if rotation not in {0, 90, 180, 270}:
                raise ValueError("Unsupported display rotation; explicitly normalize video first")
            displayed = [frame.height, frame.width] if rotation in {90, 270} else [frame.width, frame.height]
            if (size is not None and displayed != size) or (rows and timestamp <= rows[-1]["video_time_s"]):
                raise ValueError("Motion source must have stable dimensions and strictly increasing PTS")
            size = displayed
            rows.append({"frame_index": index, "video_time_s": timestamp, "pts": frame.pts,
                         "time_base": str(frame.time_base), "rotation": rotation})
    if len(rows) < 2:
        raise ValueError("Need at least two decoded video frames")
    return {"schema_version": 1, "image_size": size, "frames": rows,
            "timestamp_source": "decoder presentation timestamps", "first_pts_time_s": first}


def _text(row, key):
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Supply {key} evidence")
    return value


def _archive_samples(track, base):
    with np.load(base / track["archive"], allow_pickle=False) as data:
        # Compatible with calibrated_scene_reconstruction replay track.npz. Do not silently treat its planar
        # xy estimates as image points or trust nominal-FPS time as exact video PTS.
        points = np.asarray(data[track.get("value_key", "centroid")], float)
        valid = np.asarray(data[track.get("valid_key", "valid")])
        if points.ndim != 2 or points.shape[1] != 2 or valid.shape != (len(points),):
            raise ValueError("Expected Nx2 image points and N validity flags")
        if not np.isin(valid, [False, True]).all():
            raise ValueError("Archive validity must contain booleans or 0/1")
        scores = np.asarray(data["score"], float) if "score" in data else np.ones(len(points))
        if scores.shape != (len(points),) or not np.isfinite(scores).all():
            raise ValueError("Invalid archive scores")
        indices = track["frame_indices"]
        if len(indices) != len(points):
            raise ValueError("Map each archive sample to an explicit original video frame")
        return [{"frame_index": frame, "point_px": point.tolist(),
                 "sample_state": "observed" if bool(ok) and np.isfinite(point).all() else "missing",
                 "score": float(score)}
                for frame, point, ok, score in zip(indices, points, valid, scores, strict=True)]


def prepare(request, base, output):
    base, output = Path(base), Path(output)
    video = base / request["video"]
    video_hash = sha256_file(video)
    if request.get("video_sha256") != video_hash:
        raise ValueError("video_sha256 must match the video used for these measurements")
    timeline = video_timeline(video)
    frame_rows, size = timeline["frames"], timeline["image_size"]
    clock = request.get("clock", {})
    scale = float(clock.get("seconds_per_video_second", 1.0))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("seconds_per_video_second must be positive")
    rate_source = clock.get("rate_source")
    if "seconds_per_video_second" in clock and not rate_source:
        raise ValueError("Explicit playback-rate correction needs rate_source")
    camera = request["camera_motion"]
    if camera.get("kind") not in {"fixed", "registered", "moving_unknown"}:
        raise ValueError("Declare fixed, registered or moving_unknown camera motion")
    _text(camera, "source")
    tracks, quality, points_by_frame, seen = [], [], {}, set()
    for spec in request["tracks"]:
        for key in ("id", "source", "frame", "observable"):
            _text(spec, key)
        if spec.get("units", "px") != "px":
            raise ValueError("This importer accepts image pixels, not planar/metric trajectory estimates")
        if spec["id"] in seen:
            raise ValueError("Duplicate track id")
        seen.add(spec["id"])
        if spec.get("image_size") != size:
            raise ValueError("Measurements must use displayed full video pixels; explicitly transform resized/cropped evidence first")
        residual_scale = float(spec["residual_scale"])
        if not np.isfinite(residual_scale) or residual_scale <= 0:
            raise ValueError("residual_scale must be positive pixels")
        if ("archive" in spec) == ("samples" in spec):
            raise ValueError("Supply samples or a tracker archive, not both")
        samples = _archive_samples(spec, base) if "archive" in spec else spec["samples"]
        if len(samples) < 2:
            raise ValueError("Need at least two timed samples per track")
        indices = [s["frame_index"] for s in samples]
        if (any(type(i) is not int or not 0 <= i < len(frame_rows) for i in indices)
                or indices != sorted(set(indices))):
            raise ValueError("Frame indices must be distinct, increasing, and in the decoded video")
        values, states, diagnostic = [], [], []
        for sample in samples:
            state = sample.get("sample_state", "observed")
            if state not in {"observed", "missing", "interpolated", "predicted"}:
                raise ValueError("Invalid sample_state")
            reasons, point = [], None
            score = float(sample.get("score", 1.0))
            if not np.isfinite(score):
                raise ValueError("Sample score must be finite")
            if sample.get("occluded", False):
                reasons.append("declared_occlusion")
            if score < float(spec.get("min_score", 0.0)):
                reasons.append("below_selected_score")
            if "mask" in sample:
                if "point_px" in sample or spec["observable"] != "visible_mask_centroid":
                    raise ValueError("Mask samples measure visible_mask_centroid only")
                with Image.open(base / sample["mask"]) as image:
                    if list(image.size) != size:
                        raise ValueError("Mask must match displayed full video dimensions")
                    mask = np.asarray(image.convert("L")) > 0
                y, x = np.nonzero(mask)
                if len(x):
                    point = [float(x.mean()), float(y.mean())]
                    if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
                        reasons.append("border_truncation")
                else:
                    reasons.append("empty_mask")
            elif "point_px" in sample:
                array = np.asarray(sample["point_px"], float)
                if array.shape != (2,):
                    raise ValueError("point_px must contain two coordinates")
                if np.isfinite(array).all():
                    point = array.tolist()
                    if not (0 <= point[0] < size[0] and 0 <= point[1] < size[1]):
                        reasons.append("out_of_image")
            if point is None:
                reasons.append("nonfinite_or_absent_point")
            if point is None or reasons:
                state = "missing"
            values.append(point if state != "missing" else [None, None])
            states.append(state)
            diagnostic.append({"frame_index": sample["frame_index"], "score": score,
                               "sample_state": state, "reasons": reasons,
                               "raw_point_px": point})
            if point is not None:
                points_by_frame.setdefault(sample["frame_index"], []).append((spec["id"], point, state))
        tracks.append({"id": spec["id"], "units": "px", "frame": spec["frame"],
                       "image_size": size,
                       "observable": spec["observable"], "source": spec["source"],
                       "residual_scale": residual_scale,
                       "time_s": [frame_rows[i]["video_time_s"] * scale for i in indices],
                       "values": values, "sample_state": states,
                       "frame_indices": indices})
        quality.append({"id": spec["id"], "samples": diagnostic,
                        "observed_samples": states.count("observed"), "total_samples": len(states)})
    if not tracks:
        raise ValueError("Supply at least one track")
    warnings = []
    if not rate_source:
        warnings.append("Playback rate is unverified; video seconds are not certified physical seconds")
    if camera["kind"] == "moving_unknown":
        warnings.append("Camera motion is unresolved; image displacement is not object world motion")
    if any(t["observable"] == "visible_mask_centroid" for t in tracks):
        warnings.append("Visible silhouette centroid can change with rotation or occlusion; compare the same rendered observable")
    events = {}
    for name, index in request.get("event_frames", {}).items():
        if index is not None and (type(index) is not int or not 0 <= index < len(frame_rows)):
            raise ValueError("Event frame outside decoded video")
        events[name] = None if index is None else frame_rows[index]["video_time_s"] * scale
    observations = {"schema_version": 1, "clock_source": "decoder PTS; " + str(rate_source or "playback rate unknown"),
                    "playback_rate_evidence_supplied": bool(rate_source), "camera_motion": camera,
                    "source_video_sha256": video_hash, "tracks": tracks, "events": events}
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "observations.json", observations)
    atomic_json(output / "timeline.json", timeline)
    atomic_json(output / "quality.json", quality)
    previews = _previews(video, points_by_frame, output)
    if sha256_file(video) != video_hash:
        raise ValueError("Video changed while measuring")
    report = {"ok": True, "operation": "motion_observations", "acceptance": False,
              "observations": "observations.json", "timeline": "timeline.json", "quality": "quality.json",
              "source_video_sha256": video_hash, "playback_rate_evidence_supplied": bool(rate_source),
              "tracks": [{k: row[k] for k in ("id", "observed_samples", "total_samples")} for row in quality],
              "warnings": warnings, "preview_images": previews}
    atomic_json(output / "report.json", report)
    return report


def _previews(video, points, output):
    import av

    indices = sorted(points)
    selected = {indices[i] for i in (0, len(indices)//2, len(indices)-1)} if indices else set()
    previews = []
    if not selected:
        return previews
    with av.open(str(video)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index not in selected:
                continue
            image = frame.to_image().convert("RGB")
            rotation = getattr(frame, "rotation", 0)
            if rotation:
                image = image.rotate(rotation, expand=True)
            draw = ImageDraw.Draw(image)
            for label, (x, y), state in points[index]:
                color = "lime" if state == "observed" else "red"
                draw.ellipse((x-4, y-4, x+4, y+4), outline=color, width=2)
                draw.text((x+6, y+4), label, fill=color)
            name = f"measurement-{index:06d}.png"
            image.save(output / name)
            previews.append(name)
            if len(previews) == len(selected):
                break
    return previews
