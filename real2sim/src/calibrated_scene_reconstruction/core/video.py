"""Video primitives shared by rigid recording, deformable tasks and presentation."""
import subprocess
from pathlib import Path

import numpy as np


def open_rgb_pipe(path, width, height, fps, output_args, *, executable="ffmpeg"):
    """Start raw RGB encoding with the caller's codec arguments unchanged."""
    return subprocess.Popen([executable, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", *output_args, str(path)],
                            stdin=subprocess.PIPE)


class VideoWriter:
    """Frames (H, W, 3) uint8 encoded as they come: piped to ffmpeg (libx264, yuv420p) by a background thread, so a
    recording stores video instead of an image per frame and camera (~0.7 GB per episode as PNGs)."""

    def __init__(self, path, width, height, fps=30, crf=16, *, executable="ffmpeg"):
        import queue
        import threading

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.proc = open_rgb_pipe(path, width, height, fps,
                                  ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
                                   "-pix_fmt", "yuv420p", "-threads", "2"], executable=executable)
        self.q = queue.Queue(maxsize=64)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while (im := self.q.get()) is not None:
            self.proc.stdin.write(im)

    def write(self, im):
        self.q.put(np.ascontiguousarray(im, np.uint8).tobytes())

    def close(self):
        self.q.put(None)
        self.thread.join()
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg failed to encode a recording")


def encode_video(frames_dir, out_path, fps=30):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(Path(frames_dir) / "%05d.jpg"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19", str(out_path)], check=True)
