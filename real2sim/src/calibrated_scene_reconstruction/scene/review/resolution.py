"""The render size of an app that renders from the photo camera: the photo's aspect at `long_side` pixels.

Isaac's RTX camera takes its horizontal field of view from the horizontal aperture and its vertical one from the
render's aspect (square pixels: the vertical aperture does not count), so a photo camera rendered at another aspect
is squashed or stretched vertically against the photo. No Isaac imports (outside calibrated_scene_reconstruction.sim): the apps
call this before SimulationApp."""
from pathlib import Path

from calibrated_scene_reconstruction.core.resolution import scaled_even_size


def photo_resolution(session, long_side=960):
    from PIL import Image

    w, h = Image.open(Path(session) / "capture" / "rgb.png").size
    s = long_side / max(w, h)
    return scaled_even_size(w, h, s)
