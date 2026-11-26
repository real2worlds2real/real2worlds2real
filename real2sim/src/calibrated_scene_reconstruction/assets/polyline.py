"""Distances to rendered polyline asset segments."""
import numpy as np


def distance_to(points, lines):
    """Each point's (N, 2) least distance to the polylines' segments."""
    a = np.concatenate([ln[:-1] for ln in lines])
    b = np.concatenate([ln[1:] for ln in lines])
    d = np.full(len(points), np.inf)
    ab = b - a
    ab2 = np.maximum((ab ** 2).sum(1), 1e-12)
    for i in range(0, len(a), 64):  # chunks of segments: (N, 64) at a time
        aa, vv, ww = a[i:i + 64], ab[i:i + 64], ab2[i:i + 64]
        t = np.clip(((points[:, None] - aa[None]) * vv[None]).sum(2) / ww[None], 0.0, 1.0)
        q = aa[None] + t[..., None] * vv[None]
        d = np.minimum(d, np.linalg.norm(points[:, None] - q, axis=2).min(1))
    return d
