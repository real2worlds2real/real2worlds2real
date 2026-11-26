"""Parametric templates for thin structures single-view generators cannot recover (e.g. faucets)."""
import numpy as np


def tube_mesh(path_pts, radius, segs=18):
    """Sweep a circle along a polyline with parallel-transport frames."""
    P = np.asarray(path_pts, float)
    T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    n = np.cross(T[0], [0, 1, 0] if abs(T[0][1]) < 0.9 else [1, 0, 0])
    n /= np.linalg.norm(n)
    ang = np.linspace(0, 2 * np.pi, segs, endpoint=False)
    rings = []
    for i in range(len(P)):
        if i > 0:
            n = n - (n @ T[i]) * T[i]
            n /= np.linalg.norm(n)
        b = np.cross(T[i], n)
        rings.append(P[i] + radius * (np.cos(ang)[:, None] * n + np.sin(ang)[:, None] * b))
    F = []
    for i in range(len(P) - 1):
        for j in range(segs):
            a, b2 = i * segs + j, i * segs + (j + 1) % segs
            F += [[a, a + segs, b2], [b2, a + segs, b2 + segs]]
    return np.concatenate(rings), np.array(F)


def cylinder(center, radius, height, segs=24):
    center = np.asarray(center, float)
    V, F = tube_mesh([center, center + [0, 0, height]], radius, segs)
    V = np.vstack([V, center + [0, 0, height], center])
    ti, bi = len(V) - 2, len(V) - 1
    caps = [[segs + j, segs + (j + 1) % segs, ti] for j in range(segs)] + [[(j + 1) % segs, j, bi] for j in range(segs)]
    return V, np.vstack([F, caps])


def merge(parts):
    Vs, Fs, off = [], [], 0
    for V, F in parts:
        Vs.append(V)
        Fs.append(F + off)
        off += len(V)
    return np.vstack(Vs), np.vstack(Fs)


def gooseneck_faucet(base, spout_xy, height):
    """Bridge-style gooseneck faucet: base, vertical riser, 180-degree arc towards the spout, two lever handles."""
    bx, by, bz = base
    r_arc = max(0.06, (bx - spout_xy[0]) / 2)
    top = height - r_arc
    path = [[bx, by, bz + 0.03], [bx, by, top]]
    path += [[bx - r_arc + r_arc * np.cos(a), by, top + r_arc * np.sin(a)] for a in np.linspace(0, np.pi, 24)[1:]]
    path.append([bx - 2 * r_arc, by, top - 0.05])
    parts = [tube_mesh(path, 0.012), cylinder([bx, by, bz], 0.026, 0.035)]
    for dy in (-0.11, 0.11):
        parts.append(cylinder([bx, by + dy, bz], 0.018, 0.07))
        parts.append(tube_mesh([[bx, by + dy, bz + 0.07], [bx - 0.06, by + dy, bz + 0.10]], 0.006))
    return merge(parts)


TEMPLATES = {"gooseneck_faucet": (gooseneck_faucet, {"color": (0.80, 0.58, 0.30), "roughness": 0.28, "metallic": 1.0})}
