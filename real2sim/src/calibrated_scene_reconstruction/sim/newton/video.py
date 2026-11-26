"""Recordings of the Newton scenes' cameras (calibrated_scene_reconstruction.sim.newton.recording): frames piped to ffmpeg as they come."""
import shutil
from pathlib import Path

import numpy as np


from calibrated_scene_reconstruction.core.video import VideoWriter

def find_ffmpeg():
    """ffmpeg on PATH (the machine's, with the Isaac python on the machine), else the static binary of an
    imageio-ffmpeg wheel under envs/ (the Isaac container has no ffmpeg of its own; the repo is mounted at its own
    path, so a wheel's binary is reachable), else None."""
    if (exe := shutil.which("ffmpeg")):
        return exe
    root = Path(__file__).resolve().parents[3]
    for exe in sorted(root.glob("envs/*/lib/python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*")):
        if exe.is_file() and exe.stat().st_size > 1_000_000:
            return str(exe)
    return None


class Recorder(VideoWriter):
    """Frames (H, W, 3) uint8 piped to ffmpeg (libx264, yuv420p) as they come (calibrated_scene_reconstruction.sim.runtime.VideoWriter, with
    the ffmpeg find_ffmpeg finds); without one, JPEG frames in <path's directory>/frames to encode afterwards."""

    def __init__(self, path, width, height, fps=30, crf=16):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path, self.n, self.proc = path, 0, None
        exe = find_ffmpeg()
        if exe is None:
            print(f"[warn] no ffmpeg: writing JPEG frames to {path.parent / 'frames'}", flush=True)
            (path.parent / "frames").mkdir(exist_ok=True)
            return
        super().__init__(path, width, height, fps, crf, executable=exe)

    def write(self, im):
        im = np.ascontiguousarray(im, np.uint8)
        if self.proc is None:
            from PIL import Image

            Image.fromarray(im).save(self.path.parent / "frames" / f"{self.n:05d}.jpg", quality=92)
        else:
            self.q.put(im.tobytes())
        self.n += 1

    def close(self):
        if self.proc is None:
            return
        super().close()
