"""Smooth radial shape field for structured rigid-mesh proposals.

An envelope constrains an existing detailed surface; it never replaces it.
Source appearance, finite thickness and physical contacts still need validation.
"""
import numpy as np


def warp(vertices, centre, scale_grid, *, weights=None):
    vertices = np.asarray(vertices, float); centre = np.asarray(centre, float)
    scale_grid = np.asarray(scale_grid, float)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or centre.shape != (3,)
            or scale_grid.ndim != 2 or min(scale_grid.shape) < 4
            or not np.isfinite(vertices).all() or not np.isfinite(centre).all()
            or not np.isfinite(scale_grid).all() or (scale_grid <= 0).any()):
        raise ValueError('Finite vertices, centre and positive regular radial field required')
    offset = vertices - centre; radius = np.linalg.norm(offset, axis=1)
    azimuth = np.mod(np.arctan2(offset[:, 1], offset[:, 0]), 2 * np.pi)
    polar = np.arccos(np.clip(offset[:, 2] / np.maximum(radius, 1e-12), -1, 1))
    u = azimuth / (2 * np.pi) * scale_grid.shape[1]
    v = polar / np.pi * (scale_grid.shape[0] - 1)
    a = np.floor(u).astype(int) % scale_grid.shape[1]
    b = np.minimum(np.floor(v).astype(int), scale_grid.shape[0] - 2)
    du, dv = u - np.floor(u), v - b
    scale = ((1-dv) * ((1-du) * scale_grid[b, a] + du * scale_grid[b, (a+1) % scale_grid.shape[1]])
             + dv * ((1-du) * scale_grid[b+1, a] + du * scale_grid[b+1, (a+1) % scale_grid.shape[1]]))
    if weights is not None:
        weights = np.asarray(weights, float)
        if weights.shape != radius.shape or not np.isfinite(weights).all() or ((weights < 0) | (weights > 1)).any():
            raise ValueError('One finite blend weight in0..1 per vertex required')
        scale = 1 + weights * (scale - 1)
    return centre + offset * scale[:, None]
