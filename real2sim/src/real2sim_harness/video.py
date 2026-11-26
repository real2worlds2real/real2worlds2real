from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifacts import atomic_json, sha256_file

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def prepare_reference_inputs(
    sources: list[Path], output: Path, *, video_frames: int = 12
) -> tuple[list[Path], list[dict[str, Any]]]:
    """Normalize image/video inputs while retaining frame-to-original provenance."""
    references = []
    inputs = []
    for input_index, raw_source in enumerate(sources):
        source = raw_source.resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        record = {
            "source": str(source),
            "source_name": source.name,
            "sha256": sha256_file(source),
            "reference_indices": [],
        }
        if source.suffix.lower() in VIDEO_SUFFIXES:
            if video_frames == 0:
                record.update(kind="video", video=None)
                inputs.append(record)
                continue
            frame_dir = output / f"input_{input_index:03d}"
            manifest = extract_keyframes(source, frame_dir, count=video_frames)
            record.update(kind="video", video=manifest)
            for frame in manifest["frames"]:
                frame["reference_index"] = len(references)
                record["reference_indices"].append(len(references))
                references.append(frame_dir / frame["path"])
        else:
            record.update(kind="image")
            record["reference_indices"].append(len(references))
            references.append(source)
        inputs.append(record)
    return references, inputs


def extract_keyframes(video: Path, output: Path, *, count: int = 12) -> dict[str, Any]:
    """Initial overview only; exact timestamps come from decoded presentation times."""
    import av

    video = video.resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    if not 2 <= count <= 120:
        raise ValueError("Frame count must be between 2 and 120")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    times = []
    first_time = None
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else None
        for ordinal, frame in enumerate(container.decode(stream)):
            if frame.time is not None:
                raw_time = float(frame.time)
                first_time = raw_time if first_time is None else first_time
                timestamp = raw_time-first_time
            elif fps:
                timestamp = ordinal/fps
            else:
                raise ValueError("Video lacks presentation timestamps and a usable frame rate")
            times.append(timestamp)
    if not times:
        raise ValueError("Video has no decoded frames")
    total, retained = len(times), min(count, len(times))
    indices = sorted({round(i*(total-1)/max(retained-1, 1)) for i in range(retained)})
    selected = set(indices)
    frames = []
    with av.open(str(video)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in selected:
                name = f"frame_{len(frames):03d}_{index:08d}.jpg"
                image = frame.to_image().convert("RGB")
                rotation = getattr(frame, "rotation", 0)
                if rotation:
                    image = image.rotate(rotation, expand=True)
                image.save(output/name, quality=95)
                frames.append({"path": name, "frame_index": index,
                               "timestamp_seconds": round(times[index], 6)})
            if len(frames) == retained:
                break
    if len(frames) != retained:
        raise RuntimeError("Could not decode all selected video frames")
    tail = times[-1]-times[-2] if len(times) > 1 else 0
    tail = tail if tail > 0 else 1/(fps or 24)
    manifest = {"schema_version": 2, "source": str(video), "fps": fps,
                "total_frames": total, "duration_seconds": times[-1]+tail,
                "frames": frames, "overview_only": retained != total,
                "timestamp_source": "decoder presentation times when available; nominal-fps fallback otherwise"}
    atomic_json(output/"manifest.json", manifest)
    return manifest
