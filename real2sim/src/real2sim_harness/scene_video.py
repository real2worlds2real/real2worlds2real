"""CPU encoding and decoding of retained simulator video evidence."""
from __future__ import annotations

import hashlib
import math
from fractions import Fraction
from pathlib import Path

from PIL import Image

from .artifacts import sha256_file


def evidence_path(folder, value):
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError("Use output-relative media paths")
    path = (Path(folder) / value).resolve()
    if not path.is_relative_to(Path(folder).resolve()) or not path.is_file():
        raise ValueError("Media is missing or outside its output directory")
    return path


def _decode_movie(path):
    """Decode the complete stream, so a nonempty corrupt file cannot pass review."""
    import av

    count, previous, first, last = 0, None, None, None
    with av.open(str(path)) as source:
        if not source.streams.video:
            raise ValueError("Movie has no video stream")
        stream = source.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else None
        for frame in source.decode(stream):
            timestamp = float(frame.time) if frame.time is not None else count / fps if fps else None
            if timestamp is None or not math.isfinite(timestamp) or (previous is not None and timestamp <= previous):
                raise ValueError("Movie requires strictly increasing frame timestamps")
            if frame.width != stream.width or frame.height != stream.height:
                raise ValueError("Movie changes frame dimensions")
            first = timestamp if first is None else first
            previous, last = timestamp, timestamp
            count += 1
        if count < 2:
            raise ValueError("Return at least two decoded frames for a continuous video")
        duration = last - first + (1 / fps if fps else (last - first) / (count - 1))
        return {"frames": count, "width": stream.width, "height": stream.height,
                "fps": fps, "duration_seconds": duration, "sha256": sha256_file(Path(path))}


def inspect_movie(path):
    import av

    try:
        return _decode_movie(path)
    except av.error.FFmpegError as exc:
        raise ValueError("Movie cannot be decoded: " + str(exc)) from exc


def encode_sequence(folder, sequence, *, filename="scene_simulation.mp4"):
    """Encode actual ordered simulation captures, never interpolate sparse snapshots."""
    import av

    if not isinstance(sequence, dict):
        raise ValueError("Frame sequence must be an object")
    fps, names = sequence.get("fps"), sequence.get("frames")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or not 1 <= fps <= 120:
        raise ValueError("Frame sequence FPS must be from 1 to 120")
    if (not isinstance(names, list) or not 2 <= len(names) <= 18000
            or not all(isinstance(v, str) for v in names) or len(set(names)) != len(names)):
        raise ValueError("Return a bounded ordered list of distinct captured frame paths")
    if sequence.get("time_basis") != "simulation_seconds":
        raise ValueError("Declare simulation_seconds for the capture time basis")
    files = [evidence_path(folder, name) for name in names]
    if len(set(files)) != len(files):
        raise ValueError("Captured frames must refer to distinct files")
    output = Path(folder) / filename
    digest, size = hashlib.sha256(), None
    try:
        with av.open(str(output), "w") as movie:
            stream = movie.add_stream("libx264", rate=Fraction(str(fps)))
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "18", "preset": "medium"}
            for index, path in enumerate(files):
                digest.update(sha256_file(path).encode())
                with Image.open(path) as source:
                    rgb = source.convert("RGB")
                    if size is None:
                        size = rgb.size
                        if min(size) < 2 or max(size) > 4096 or any(v % 2 for v in size):
                            raise ValueError("Use even frame dimensions from 2 to 4096 pixels")
                        stream.width, stream.height = size
                    elif rgb.size != size:
                        raise ValueError("Captured frame sizes differ")
                    frame = av.VideoFrame.from_image(rgb)
                frame.pts = index
                frame.time_base = Fraction(1, 1) / Fraction(str(fps))
                for packet in stream.encode(frame):
                    movie.mux(packet)
            for packet in stream.encode():
                movie.mux(packet)
        metadata = inspect_movie(output)
        if metadata["frames"] != len(files) or abs(metadata["duration_seconds"] - len(files) / fps) > 1 / fps:
            raise ValueError("Encoded movie does not match its captured sequence")
    except av.error.FFmpegError as exc:
        output.unlink(missing_ok=True)
        raise ValueError("Video encoder failed: " + str(exc)) from exc
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return {"path": filename, "source_frames_sha256": digest.hexdigest(),
            "time_basis": "simulation_seconds", **metadata}
