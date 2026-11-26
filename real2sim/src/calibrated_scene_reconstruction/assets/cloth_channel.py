"""Isometric bowed-sheet fixtures for testing friction in a narrow channel."""
import numpy as np
from scipy.optimize import brentq

from calibrated_scene_reconstruction.assets.cloth_rest import triangulate_grid


def bowed(width, length, span, spacing):
    """Return a flat intrinsic chart bowed across X, extruded along Z.

    The discrete edge lengths, rather than a sampled smooth arc, preserve the
    membrane metric exactly. A flat natural bending reference can then develop
    contact pressure against two real surfaces; this supplies no attachment.
    """
    if (not np.isfinite([width, length, span, spacing]).all()
            or not 0 < span < width or min(length, spacing) <= 0):
        raise ValueError('Positive sheet dimensions and a span below material width required')
    columns = max(3, int(np.ceil(width / spacing)) + 1)
    rows = max(3, int(np.ceil(length / spacing)) + 1)
    u = np.linspace(0., width, columns)
    v = np.linspace(0., length, rows)
    directions = np.linspace(-1., 1., columns - 1)
    def angles(phi):
        return directions * phi
    def projected(phi):
        return float(np.sum(np.diff(u) * np.cos(angles(phi))))
    # Keep every transverse segment moving forward, so the initial fixture
    # cannot manufacture a self-fold or intersect its own channel edge.
    if projected(np.pi / 2) >= span:
        raise ValueError('Span too narrow for a monotone single bow; use a resolved fold')
    phi = brentq(lambda value: projected(value) - span, 0., np.pi / 2)
    x = np.r_[0., np.cumsum(np.diff(u) * np.cos(angles(phi)))]
    y = np.r_[0., np.cumsum(np.diff(u) * np.sin(angles(phi)))]
    x -= span / 2
    uu, vv = np.meshgrid(u, v)
    points = np.c_[np.tile(x, rows), np.tile(y, rows), vv.ravel()]
    rest = np.c_[uu.ravel(), vv.ravel(), np.zeros(uu.size)]
    faces, chart, report = triangulate_grid(points, rest, rows, columns)
    return points, faces, chart, report
