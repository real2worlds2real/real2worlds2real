"""Primitive asset triangles, bottom at z=0."""
import numpy as np

def primitive_mesh(shape, dims, n=32):
    """Vertices (bottom face at z = 0, centred in xy) and triangles of an upright cylinder {radius, height} or box
    {size_x, size_y, height}."""
    h = dims["height"]
    if shape == "cylinder":
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        ring = np.stack([dims["radius"] * np.cos(a), dims["radius"] * np.sin(a)], 1)
        V = np.concatenate([np.c_[ring, np.zeros(n)], np.c_[ring, np.full(n, h)], [[0, 0, 0], [0, 0, h]]])
        F = []
        for i in range(n):
            j = (i + 1) % n
            F += [[i, j, n + j], [i, n + j, n + i], [2 * n, j, i], [2 * n + 1, n + i, n + j]]
        return V, np.array(F)
    if shape == "box":
        x, y = dims["size_x"] / 2, dims["size_y"] / 2
        V = np.array([[sx * x, sy * y, z] for z in (0, h) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        F = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                      [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
        return V, F
    raise ValueError(f"unknown shape {shape}")
