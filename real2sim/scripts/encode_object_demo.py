"""Encode Blender demo frames to MP4 and a compact looping GIF on a CPU node."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from real2sim_harness.video_change import frame_change, has_visible_change  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--name", default="asset_demo")
    args = parser.parse_args()
    output = args.output.resolve()
    report_path = output / "animation_report.json"
    report = json.loads(report_path.read_text())
    count = report["frames"]
    frames = sorted((output / "frames").glob("frame_*.png"))
    if (
        report.get("preview_only")
        or count != round(report["duration_seconds"] * report["fps"])
        or len(frames) != count
    ):
        raise ValueError("Expected a complete Blender sequence matching its motion report")
    if not args.name or Path(args.name).name != args.name:
        raise ValueError("Output name must be a filename stem")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    mp4 = output / f"{args.name}.mp4"
    gif = output / f"{args.name}.gif"
    common = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    subprocess.run(
        [
            *common,
            "-framerate",
            str(report["fps"]),
            "-i",
            str(output / "frames" / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(mp4),
        ],
        check=True,
    )
    subprocess.run(
        [
            *common,
            "-i",
            str(mp4),
            "-filter_complex",
            "fps=12,scale=640:-1:flags=lanczos,split[a][b];"
            "[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=sierra2_4a",
            "-loop",
            "0",
            str(gif),
        ],
        check=True,
    )
    capture = cv2.VideoCapture(str(mp4))
    try:
        if not capture.isOpened():
            raise RuntimeError("MP4 could not be decoded")
        decoded = 0
        first_frame = None
        last_frame = None
        largest_change = 0.0
        largest_region = 0
        largest_changed_pixels = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if first_frame is None:
                first_frame = frame.copy()
            change = frame_change(first_frame, frame)
            largest_change = max(largest_change, change["mean_change"])
            largest_region = max(largest_region, change["largest_changed_region_px"])
            largest_changed_pixels = max(largest_changed_pixels, change["changed_pixels"])
            last_frame = frame
            decoded += 1
        if decoded != count:
            raise RuntimeError(f"MP4 decoded {decoded}/{count} frames")
        if not has_visible_change(largest_region):
            raise RuntimeError("Video is visually static despite having multiple frames")
        video = {
            "frames": decoded,
            "fps": capture.get(cv2.CAP_PROP_FPS),
            "width": capture.get(cv2.CAP_PROP_FRAME_WIDTH),
            "height": capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
            "bytes": mp4.stat().st_size,
            "max_mean_pixel_change_from_start": round(largest_change, 4),
            "largest_changed_region_px": largest_region,
            "max_significantly_changed_pixels": largest_changed_pixels,
            "change_check_scope": "Visible image change, including localized thin objects; not a physics or motion-quality certificate.",
            "loop_seam_mean_pixel_change": round(
                float(cv2.absdiff(last_frame, first_frame).mean()), 4
            ),
        }
    finally:
        capture.release()
    with Image.open(gif) as animation:
        duration_ms = 0
        for index in range(animation.n_frames):
            animation.seek(index)
            duration_ms += animation.info.get("duration", 0)
        gif_report = {
            "frames": animation.n_frames,
            "duration_ms": duration_ms,
            "loop": animation.info.get("loop"),
            "bytes": gif.stat().st_size,
        }
        if abs(duration_ms - report["duration_seconds"] * 1000) > 100:
            raise RuntimeError("GIF duration differs from the rendered animation")
        if gif_report["loop"] != 0:
            raise RuntimeError("GIF is not configured to loop indefinitely")
    result = {"ok": True, "mp4": video, "gif": gif_report}
    (output / "encoding_report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
