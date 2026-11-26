"""Silhouettes of triangle meshes by filling their projected triangles (CPU, OpenCV).

cv2.fillPoly called once with every triangle fills by the even-odd rule, so wherever two triangles overlap (always:
front and back faces) the silhouette has a hole. Here every triangle is filled on its own; `simplify` keeps that
affordable for a reconstructed mesh of 10^5 faces.
"""
import numpy as np


def fill(tri_uv, shape):
    """Union (H, W) bool of triangles given as pixel coordinates (M, 3, 2); triangles with a NaN are skipped."""
    import cv2

    canvas = np.zeros(shape, np.uint8)
    tri_uv = np.asarray(tri_uv, float)
    ok = np.all(np.isfinite(tri_uv), (1, 2))
    for t in np.round(tri_uv[ok]).astype(np.int32):
        cv2.fillConvexPoly(canvas, t, 1)
    return canvas.astype(bool)


def simplify(V, F, max_faces=2500):
    """A coarser mesh with the same outline: vertices merged on a grid that is coarsened until at most max_faces
    faces remain. For silhouettes and colliders, not for rendering. Returns (V, F)."""
    V, F = np.asarray(V, float), np.asarray(F)
    if len(F) <= max_faces:
        return V, F
    cell = float(np.ptp(V, 0).max()) / 64
    while True:
        key = np.floor((V - V.min(0)) / cell).astype(np.int64)
        _u, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.ravel()
        n = inv.max() + 1
        Vn = np.stack([np.bincount(inv, V[:, k], n) for k in range(3)], 1) / np.bincount(inv, minlength=n)[:, None]
        Fn = inv[F]
        Fn = Fn[(Fn[:, 0] != Fn[:, 1]) & (Fn[:, 1] != Fn[:, 2]) & (Fn[:, 0] != Fn[:, 2])]
        Fn = np.unique(np.sort(Fn, 1), axis=0)
        if len(Fn) <= max_faces:
            return Vn, Fn
        cell *= 1.3
