"""Convex geometry used to inspect a robot motion sweep."""
import numpy as np
from scipy.spatial import ConvexHull


class Hull:
    """Convex hull with a bounding-box prefilter; its facet planes are computed on first use (a Delaunay
    triangulation per sampled link pose took 25 GB over a pick-and-place sweep's ~20000 hulls)."""

    def __init__(self, P):
        self.P, self.lo, self.hi, self._eq = P, P.min(0), P.max(0), None

    def overlaps(self, lo, hi):
        return bool(np.all(self.lo <= hi) and np.all(lo <= self.hi))

    def contains(self, Q, eps=1e-9):
        m = np.all((Q >= self.lo) & (Q <= self.hi), axis=1)
        if m.any():
            if self._eq is None:  # joggled: link meshes have coplanar / duplicate points
                self._eq = ConvexHull(self.P, qhull_options="QJ").equations
            Qm = Q[m]
            m[m] = np.all(Qm @ self._eq[:, :3].T + self._eq[:, 3] <= eps, axis=1)
        return m

    def shifted(self, d):
        """The hull translated by d (its facet planes moved with it, not recomputed)."""
        h = Hull(self.P + d)
        if self._eq is not None:
            h._eq = self._eq.copy()
            h._eq[:, 3] -= self._eq[:, :3] @ d
        return h
