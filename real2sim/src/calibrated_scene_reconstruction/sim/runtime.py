"""Helpers for Isaac entry points (apps/*.py): annotators, variants, video encoding."""
import json
import subprocess as subprocess  # noqa: F401 -- compatibility for callers patching the encoder
import time
from pathlib import Path

import numpy as np


from calibrated_scene_reconstruction.core.video import encode_video as encode_video  # noqa: F401 -- legacy public import
from calibrated_scene_reconstruction.core.video import VideoWriter as VideoWriter  # noqa: F401 -- legacy public import

def rgb_annotator(camera_path, width, height, semantic=False):
    import omni.replicator.core as rep

    rp = rep.create.render_product(str(camera_path), (width, height))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach(rp)
    if not semantic:
        return rgb
    seg = rep.AnnotatorRegistry.get_annotator("semantic_segmentation", init_params={"colorize": False})
    seg.attach(rp)
    return rgb, seg


def grab(annotator, render=None, tries=30):
    """The annotator's RGB frame (H, W, 3) uint8; with `render` (e.g. world.render), a frame whose render product is
    not ready (replicator raises TypeError on its missing overscan, or hands back an empty array; seen on busy shared
    GPUs, where 5 renders were not always enough) rendered again, up to `tries` times. Not ready after them: TypeError."""
    for k in range(tries):
        try:
            a = np.asarray(annotator.get_data())
            if a.ndim == 3 and a.size:
                return a[..., :3].copy()
        except TypeError:
            pass
        if render is None or k == tries - 1:
            raise TypeError("the render product gave no frame")
        render()


def warm(annotators, render, timeout=120.0):
    """Render (no physics step) until every annotator gives a frame, at most `timeout` s, then the seconds it took: a
    search that ran for minutes without a Kit frame (planning, physics validation) can leave the render products
    without data for a while. Still none: TypeError."""
    t0 = time.time()
    while True:
        try:
            for a in annotators:
                grab(a)
            return time.time() - t0
        except TypeError:
            if time.time() - t0 > timeout:
                raise
            render()


def load_variant(path):
    return json.loads(Path(path).read_text()) if path else {}
