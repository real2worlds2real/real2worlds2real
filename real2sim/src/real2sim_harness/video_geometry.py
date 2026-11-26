"""Optional calibrated multi-stream SfM, independent of scene/object recipes."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .artifacts import sha256_file
from .posed_observations import file_path, write_report


def match_pairs(groups, *, window=10, cross_stride=10, loop_stride=10, max_pairs=50000):
    if (any(type(v) is not int for v in (window, cross_stride, loop_stride, max_pairs))
            or not 1 <= window <= 100 or not 1 <= cross_stride <= 1000
            or not 0 <= loop_stride <= 1000 or not 1 <= max_pairs <= 200000):
        raise ValueError("Invalid sequential/cross-stream/loop pair budget")
    pairs = set()
    def add(a, b):
        if a != b:
            pairs.add(tuple(sorted((a, b))))
        if len(pairs) > max_pairs:
            raise ValueError("Matching pair budget exceeded; select fewer frames or increase strides")
    for group in groups:
        for i, a in enumerate(group):
            for b in group[i + 1:i + window + 1]:
                add(a, b)
        if loop_stride:
            selected = group[::loop_stride]
            for i, a in enumerate(selected):
                for b in selected[i + 1:]:
                    add(a, b)
    for i, group in enumerate(groups):
        for other in groups[i + 1:]:
            for a in group[::cross_stride]:
                for b in other[::cross_stride]:
                    add(a, b)
    return sorted(pairs)


def select_frames(stream, base):
    """Sharp representative per time bucket; preserve original order and timestamps."""
    if ("video" in stream) == ("frames" in stream):
        raise ValueError("Each stream needs either a video or ordered frame records")
    limit = stream.get("max_frames", 250)
    if type(limit) is not int or not 2 <= limit <= 1000:
        raise ValueError("max_frames must be 2..1000 per stream")
    if "frames" in stream:
        rows = stream["frames"]
        if not 2 <= len(rows) <= limit:
            raise ValueError("Supply 2..max_frames ordered frames")
        previous = -np.inf
        for index, row in enumerate(rows):
            time = float(row.get("time_seconds", index))
            if not np.isfinite(time) or time <= previous:
                raise ValueError("Stream timestamps must be finite and strictly increasing")
            previous = time
            path = file_path(base, row["image"])
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError("Cannot read stream frame")
            yield image, {"time_seconds": time, "source_frame": index, "source_sha256": sha256_file(path)}
        return
    path = file_path(base, stream["video"])
    interval = float(stream.get("interval_seconds", .5))
    start = float(stream.get("start_seconds", 0))
    end = float(stream.get("end_seconds", 1800))
    if not np.isfinite([interval, start, end]).all() or not .02 <= interval <= 60 or not 0 <= start < end:
        raise ValueError("Use a finite video interval and ordered nonnegative time bounds")
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError("Video has no readable finite frame rate")
    digest, best, bucket, emitted, index = sha256_file(path), None, None, 0, 0
    previous_time = -np.inf
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            has_pts = np.isfinite(timestamp) and timestamp >= 0 and timestamp > previous_time
            time = float(timestamp) if has_pts else max(index / fps, previous_time + 1 / fps)
            if time > end:
                break
            previous_time = time
            current = int((time - start) / interval)
            index += 1
            if time < start:
                continue
            if bucket is not None and current != bucket:
                yield best[1], best[2]
                emitted += 1
                if emitted >= limit:
                    return
                best = None
            bucket = current
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            score = float(cv2.Laplacian(gray, cv2.CV_32F).var())
            if best is None or score > best[0]:
                best = (score, image, {"time_seconds": time, "source_frame": index - 1,
                                       "timestamp_source": "decoder_pts" if has_pts else "nominal_fps_fallback",
                                       "source_sha256": digest, "sharpness": score})
        if best is not None and emitted < limit:
            yield best[1], best[2]
    finally:
        cap.release()


def _model(reconstructions):
    if not reconstructions:
        raise ValueError("COLMAP could not register the selected views")
    return max(reconstructions.values(), key=lambda rec: (rec.num_reg_images(), rec.num_points3D()))


def _self_calibrate(images, output, threads):
    import pycolmap
    output.mkdir()
    database = output / "database.db"
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "OPENCV"
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.num_threads = threads
    pycolmap.extract_features(database, images, camera_mode=pycolmap.CameraMode.SINGLE,
                             reader_options=reader, extraction_options=extraction, device=pycolmap.Device.cpu)
    sequence = pycolmap.SequentialPairingOptions()
    sequence.overlap = 10
    matching = pycolmap.FeatureMatchingOptions()
    matching.num_threads = threads
    pycolmap.match_sequential(database, pairing_options=sequence, matching_options=matching,
                             device=pycolmap.Device.cpu)
    mapper = pycolmap.IncrementalPipelineOptions()
    mapper.num_threads, mapper.min_model_size = threads, 2
    rec = _model(pycolmap.incremental_mapping(database, images, output / "sparse", options=mapper))
    camera = next(iter(rec.cameras.values()))
    fx, fy, cx, cy, *dist = list(map(float, camera.params))
    result = {"image_size": [camera.width, camera.height],
            "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "distortion": dist},
            "source": "COLMAP per-stream OPENCV self-calibration; estimated, not measured"}
    from .artifacts import atomic_json
    atomic_json(output / "camera_estimate.json", {"camera": result,
        "reprojection_px": float(rec.compute_mean_reprojection_error()),
        "registered_images": rec.num_reg_images()})
    return result


def recover(request, base, output):
    import pycolmap

    from .calibration import camera_spec, intrinsic_matrix
    from .multiview import _camera_record, _image_record
    from .posed_observations import load_views
    load_views({**request, "views": []}, base, min_views=0)  # Common frame/schema validation.
    streams = request.get("streams", [])
    threads = request.get("num_threads", 4)
    if not 1 <= len(streams) <= 16 or type(threads) is not int or not 1 <= threads <= 32:
        raise ValueError("Use 1..16 streams and 1..32 threads")
    output = Path(output)
    raw, images = output / "raw_frames", output / "images"
    images.mkdir(parents=True, exist_ok=False)
    groups, cameras, retained, hashes, stream_summaries = [], [], [], {}, []
    for i, stream in enumerate(streams):
        folder = raw / f"stream-{i:02d}"
        folder.mkdir(parents=True)
        frames, shape = [], None
        for j, (image, evidence) in enumerate(select_frames(stream, base)):
            if shape is None:
                shape = image.shape[:2]
            if image.shape[:2] != shape or min(shape) < 8 or max(shape) > 4096:
                raise ValueError("A stream must keep one8..4096pixel displayed resolution")
            path = folder / f"{j:05d}.png"
            cv2.imwrite(str(path), image)
            frames.append((path, evidence))
        if len(frames) < 2:
            raise ValueError("Each stream needs at least two selected frames")
        supplied = stream.get("camera")
        camera = camera_spec(supplied) if supplied else _self_calibrate(folder, output / f"selfcal-{i:02d}", threads)
        if camera["image_size"] != [shape[1], shape[0]]:
            raise ValueError("Stream displayed resolution must match its calibration")
        k = intrinsic_matrix(camera)
        distortion = np.asarray(camera["intrinsics"].get("distortion", []), float)
        group = []
        for j, (path, evidence) in enumerate(frames):
            image = cv2.imread(str(path))
            name = f"s{i:02d}_{j:05d}.png"
            fixed = cv2.undistort(image, k, distortion) if distortion.size and distortion.any() else image
            cv2.imwrite(str(images / name), fixed)
            group.append(name)
            retained.append({"name": name, "stream": i, **evidence})
            hashes[name] = sha256_file(images / name)
        groups.append(group)
        cameras.append(camera)
        stream_summaries.append({"stream": i, "selected_count": len(frames),
            "first_time_seconds": frames[0][1]["time_seconds"],
            "last_time_seconds": frames[-1][1]["time_seconds"],
            "selection_budget_filled": len(frames) >= stream.get("max_frames", 250),
            "calibration": "supplied" if supplied else "estimated"})
    pairs = match_pairs(groups, **request.get("matching", {}))
    pair_file = output / "pairs.txt"
    pair_file.write_text("".join(f"{a} {b}\n" for a, b in pairs))
    database = output / "database.db"
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.num_threads = threads
    for group, camera in zip(groups, cameras, strict=True):
        k = intrinsic_matrix(camera)
        reader = pycolmap.ImageReaderOptions()
        reader.camera_model = "PINHOLE"
        reader.camera_params = ",".join(map(str, [k[0, 0], k[1, 1], k[0, 2], k[1, 2]]))
        pycolmap.extract_features(database, images, image_names=group, camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=reader, extraction_options=extraction, device=pycolmap.Device.cpu)
    pairing = pycolmap.ImportedPairingOptions()
    pairing.match_list_path = pair_file
    matching = pycolmap.FeatureMatchingOptions()
    matching.num_threads = threads
    pycolmap.match_image_pairs(database, pairing_options=pairing, matching_options=matching, device=pycolmap.Device.cpu)
    mapper = pycolmap.IncrementalPipelineOptions()
    mapper.num_threads, mapper.min_model_size = threads, 2
    mapper.ba_refine_focal_length = mapper.ba_refine_principal_point = mapper.ba_refine_extra_params = False
    rec = _model(pycolmap.incremental_mapping(database, images, output / "sparse", options=mapper))
    (output / "sparse/selected").mkdir()
    rec.write(output / "sparse/selected")
    rec.export_PLY(output / "sparse.ply")
    return write_report(output, {"schema_version": 1, "ok": True, "operation": "video_cameras",
        "backend": {"name": "pycolmap", "version": pycolmap.__version__}, "frame": request["frame"],
        "input_hashes": hashes, "stream_calibrations": cameras, "stream_summaries": stream_summaries, "selected_frames": retained,
        "pair_count": len(pairs), "registered_images": rec.num_reg_images(),
        "points3d": rec.num_points3D(), "mean_reprojection_error_px": rec.compute_mean_reprojection_error(),
        "cameras": [_camera_record(cam) for _, cam in sorted(rec.cameras.items())],
        "images": [_image_record(im) for _, im in sorted(rec.images.items())],
        "sparse_model": "sparse/selected", "points": "sparse.ply", "preview_images": [],
        "limits": ["Static geometry and adequate overlap required; moving participants can corrupt poses.",
                   "Self-calibration remains an estimate. Absolute scale requires independent registration.",
                   "A registered subset does not establish coverage of every input stream."]})
