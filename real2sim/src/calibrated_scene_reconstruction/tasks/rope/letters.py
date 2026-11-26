"""Capital letters a rope can lie in: each one stroke, without going over itself (apps/newton_rope_letter.py's goals).

A letter is a polyline in its own box (u to the right, v up, the height 1); arcs are sampled finely. `goal_points`
places it on the table at the rope's length (scaled so the stroke is as long as the rope) and samples it at the
rope's segment midpoints by arc length, so segment i of the rope belongs at point i (or, the rope the other way round,
at point n - 1 - i). `dashes` cuts the stroke into the dashed line drawn on the table as the visual goal.
"""
from __future__ import annotations

import numpy as np


def _arc(cx, cy, rx, ry, a0, a1, n=24):
    a = np.radians(np.linspace(a0, a1, n))
    return np.stack([cx + rx * np.cos(a), cy + ry * np.sin(a)], 1)


def _join(*parts):
    out = [np.asarray(parts[0], float)]
    for p in parts[1:]:
        p = np.asarray(p, float)
        out.append(p[1:] if np.allclose(p[0], out[-1][-1]) else p)
    return np.concatenate(out)


LETTERS = {
    "C": _arc(0.4, 0.5, 0.4, 0.5, 50.0, 310.0, 40),
    "J": _join([[0.6, 1.0], [0.6, 0.3]], _arc(0.35, 0.3, 0.25, 0.3, 0.0, -180.0)),
    "L": np.array([[0.0, 1.0], [0.0, 0.0], [0.6, 0.0]]),
    "M": np.array([[0.0, 0.0], [0.0, 1.0], [0.4, 0.4], [0.8, 1.0], [0.8, 0.0]]),
    "N": np.array([[0.0, 0.0], [0.0, 1.0], [0.7, 0.0], [0.7, 1.0]]),
    "S": _join(_arc(0.35, 0.75, 0.3, 0.25, 20.0, 270.0, 30), _arc(0.35, 0.25, 0.3, 0.25, 90.0, -160.0, 30)),
    "U": _join([[0.0, 1.0], [0.0, 0.35]], _arc(0.35, 0.35, 0.35, 0.35, 180.0, 360.0), [[0.7, 0.35], [0.7, 1.0]]),
    "V": np.array([[0.0, 1.0], [0.35, 0.0], [0.7, 1.0]]),
    "W": np.array([[0.0, 1.0], [0.25, 0.0], [0.5, 0.7], [0.75, 0.0], [1.0, 1.0]]),  # wide: its inner corners rounded too
    "Z": np.array([[0.0, 1.0], [0.7, 1.0], [0.0, 0.0], [0.7, 0.0]]),
}


def length(poly):
    return float(np.linalg.norm(np.diff(poly, axis=0), axis=1).sum())


def at_arclength(poly, s):
    """Points at arc lengths s along the polyline."""
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))])
    return np.stack([np.interp(s, cum, poly[:, 0]), np.interp(s, cum, poly[:, 1])], 1)


def round_corners(poly, radius, step=None):
    """The polyline with its corners rounded over about `radius`: resampled finely by arc length and smoothed (a
    Gaussian of that width; the ends extended straight, so they stay put)."""
    L = length(poly)
    step = step or radius / 8
    p = at_arclength(poly, np.linspace(0.0, L, max(2, int(np.ceil(L / step))) + 1))
    sigma = radius / step
    k = np.arange(-int(3 * sigma), int(3 * sigma) + 1)
    w = np.exp(-0.5 * (k / sigma) ** 2)
    w /= w.sum()
    n = len(k) // 2
    q = np.pad(p, ((n, n), (0, 0)), mode="reflect", reflect_type="odd")
    return np.stack([np.convolve(q[:, i], w, mode="valid") for i in range(2)], 1)


def rope_length_for(letter, height, corner=0.03):
    """The rope (the stroke's length) the letter takes at about this height, its corners rounded over `corner`."""
    return length(round_corners(LETTERS[letter] * height, corner)) if corner > 0 else length(LETTERS[letter] * height)


def place(letter, rope_length, centre, up_angle, corner=0.03):
    """The letter's stroke on the table (2-D, the rope's units): scaled to the rope's length, its corners rounded
    over `corner` (a rope a centimetre thick does not lie in a sharp corner), its box centred on `centre`, its up
    direction at up_angle (rad, from +x)."""
    p = LETTERS[letter]
    p = p * (rope_length / length(p))
    if corner > 0:
        p = round_corners(p, corner)
        p = p * (rope_length / length(p))
    p = p - (p.min(0) + p.max(0)) / 2
    up = np.array([np.cos(up_angle), np.sin(up_angle)])
    right = np.array([up[1], -up[0]])  # a quarter turn clockwise from up: read from behind the letter's foot
    return np.asarray(centre, float) + p[:, :1] * right + p[:, 1:] * up


def goal_points(stroke, n_seg):
    """The stroke sampled at n_seg segment midpoints by arc length."""
    L = length(stroke)
    return at_arclength(stroke, (np.arange(n_seg) + 0.5) * L / n_seg)


def dashes(stroke, dash, gap):
    """The stroke cut into dashes: a list of polylines, each `dash` long, `gap` apart (the last one shortened)."""
    L = length(stroke)
    out, s = [], 0.0
    while s < L:
        e = min(s + dash, L)
        ss = np.linspace(s, e, max(2, int(np.ceil((e - s) / (dash / 6))) + 1))
        out.append(at_arclength(stroke, ss))
        s = e + gap
    return out


from calibrated_scene_reconstruction.assets.polyline import distance_to as distance_to  # noqa: E402,F401 -- public helper
