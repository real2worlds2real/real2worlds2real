"""A dataset's JPEG frame cache (calibrated_scene_reconstruction.policy.diffusion.frame_cache_jpeg: every frame of a camera resized and
JPEG-encoded into <dataset>/cache/<camera>_<H>x<W>_q<quality>.jpg with its .idx.npy offsets), built ahead of use:
tools/export_dp_official.py copies its frames, the LeRobot trainer reads it (frame_cache_quality). CPU only.

    PYTHONPATH=src envs/policy/bin/python tools/build_frame_cache.py <dataset> [--camera head] [--size 240 384] \
        [--quality 90]
"""
import argparse
import json
import time
from pathlib import Path

from calibrated_scene_reconstruction.policy.diffusion import frame_cache_jpeg

if __name__ == "__main__":  # the pool's spawned workers import this file
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="an calibrated_scene_reconstruction.datagen.export.lerobot output (LeRobot v2.1)")
    ap.add_argument("--camera", default="head")
    ap.add_argument("--size", nargs=2, type=int, default=[240, 384], metavar=("H", "W"))
    ap.add_argument("--quality", type=int, default=90)
    a = ap.parse_args()
    root = Path(a.dataset)
    t = time.time()
    frame_cache_jpeg(root, json.loads((root / "meta" / "info.json").read_text()), a.camera, tuple(a.size), a.quality)
    print(f"{time.time() - t:.0f} s", flush=True)
