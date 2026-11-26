"""Random looks for the recorded Newton rope scenes (calibrated_scene_reconstruction.sim.newton.recording, apps/newton_rope_letter.py
--look random): domain randomization of what the cameras see, drawn once per episode -- the table's, the floor's and
the background's colours, the light's direction and shadows, each camera's mount (moved a few cm, turned and zoomed a
few degrees) and the image's exposure and white balance; colour / contrasting draw the rope's and the dashes'.
"""
from __future__ import annotations

import colorsys

import numpy as np

# neutrals: white, light grey, grey, charcoal, black, navy, beige, khaki, brown, olive
NEUTRALS = [(0.92, 0.92, 0.9), (0.78, 0.78, 0.78), (0.55, 0.56, 0.58), (0.28, 0.28, 0.3), (0.1, 0.1, 0.11),
            (0.12, 0.15, 0.3), (0.82, 0.74, 0.6), (0.62, 0.56, 0.4), (0.38, 0.27, 0.18), (0.36, 0.38, 0.24)]


def color(rng, neutral_p=0.4):
    """A colour (rgb in [0, 1]): a neutral, or any hue at a moderate to full saturation."""
    if rng.random() < neutral_p:
        return np.clip(np.asarray(NEUTRALS[rng.integers(len(NEUTRALS))]) + rng.normal(0.0, 0.03, 3), 0.0, 1.0)
    return np.asarray(colorsys.hsv_to_rgb(rng.random(), rng.uniform(0.25, 0.9), rng.uniform(0.3, 0.95)))


def contrasting(rng, c, min_diff=0.25):
    """A colour at least `min_diff` from c (the mean absolute difference), so one shows against the other."""
    for _ in range(50):
        d = color(rng)
        if np.abs(d - c).mean() >= min_diff:
            return d
    return 1.0 - c


def table_color(rng):
    """A table top: wood, a neutral (white, grey, black) or a coloured laminate."""
    r = rng.random()
    if r < 0.45:
        return np.asarray(colorsys.hsv_to_rgb(rng.uniform(0.05, 0.11), rng.uniform(0.3, 0.65), rng.uniform(0.3, 0.8)))
    if r < 0.8:
        return np.full(3, rng.uniform(0.15, 0.9)) + rng.normal(0.0, 0.02, 3)
    return np.asarray(colorsys.hsv_to_rgb(rng.random(), rng.uniform(0.15, 0.5), rng.uniform(0.3, 0.85)))


def scene_look(rng, cameras, unit=1.0):
    """The scene's random look. cameras: {name: (eye, target, vertical fov deg)} (in the scene's units, `unit` of
    them a centimetre), returned moved: the eye up to ~4 cm, the target ~3 cm, the fov +-3 deg."""
    el, az = np.radians(rng.uniform(35.0, 80.0)), rng.uniform(0.0, 2 * np.pi)
    light = -np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])  # pointing down at the table

    def moved(c):
        return (np.asarray(c[0], float) + unit * np.clip(rng.normal(0.0, 2.0, 3), -4.0, 4.0),
                np.asarray(c[1], float) + unit * np.clip(rng.normal(0.0, 1.5, 3), -3.0, 3.0),
                float(c[2] + rng.uniform(-3.0, 3.0)))

    cams = {k: moved(c) for k, c in cameras.items()}
    look = {"table": np.clip(table_color(rng), 0.0, 1.0),
            "floor": np.clip(np.full(3, rng.uniform(0.2, 0.8)) + rng.normal(0.0, 0.05, 3), 0.0, 1.0),
            "background": np.clip(np.full(3, rng.uniform(0.35, 0.95)) + rng.normal(0.0, 0.05, 3), 0.0, 1.0),
            "light_direction": light, "shadows": bool(rng.random() < 0.8),
            "gain": np.clip(rng.uniform(0.8, 1.2) * rng.normal(1.0, 0.04, 3), 0.6, 1.4)}  # exposure, white balance
    look["cameras"] = cams
    return look


# Public legacy spellings.
colour = color
table_colour = table_color
