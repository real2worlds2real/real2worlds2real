"""World-space USD surface sampling and collision-support geometry."""
import numpy as np
from pxr import Usd, UsdGeom

from .usd import set_pose


def solid_supports(boxes, tol=0.005):
    """Colliders only: a box carrying a thin slab (counter top on its cabinet) is extended up through the slab, so
    objects on the counter rest on a solid block; a 4 cm slab alone lets the solver push a resting object through.
    Only a slab counts, something covering at least half the box: a rail or a frame's band standing on a table
 raised the whole table's collider to the band's top, and the cups set on the table started inside it
    and were pushed through to the floor."""
    out = {k: (list(lo), list(hi)) for k, (lo, hi) in boxes.items()}
    for a, (alo, ahi) in boxes.items():
        area = max((ahi[0] - alo[0]) * (ahi[1] - alo[1]), 1e-9)
        for b, (blo, bhi) in boxes.items():
            ox = min(ahi[0], bhi[0]) - max(alo[0], blo[0])
            oy = min(ahi[1], bhi[1]) - max(alo[1], blo[1])
            if a != b and ox > 0 and oy > 0 and ox * oy >= 0.5 * area and abs(ahi[2] - blo[2]) < tol:
                out[a][1][2] = max(out[a][1][2], bhi[2])
    return out


def surface_points(prim, n=60000, seed=0):
    """World points spread over the faces of every mesh under a prim, by area."""
    cache, tris = UsdGeom.XformCache(), []
    for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if not p.IsA(UsdGeom.Mesh):
            continue
        m = UsdGeom.Mesh(p)
        V, counts, idx = m.GetPointsAttr().Get(), m.GetFaceVertexCountsAttr().Get(), m.GetFaceVertexIndicesAttr().Get()
        if not V or not counts:
            continue
        M = np.asarray(cache.GetLocalToWorldTransform(p))
        V = (np.c_[np.asarray(V, float), np.ones(len(V))] @ M)[:, :3]
        idx, at = np.asarray(idx), np.r_[0, np.cumsum(counts)[:-1]]
        for k in range(1, int(max(counts)) - 1):  # each polygon as a fan of triangles
            f = np.asarray(counts) > k + 1
            tris.append(V[np.stack([idx[at[f]], idx[at[f] + k], idx[at[f] + k + 1]], 1)])
    if not tris:
        return np.zeros((0, 3))
    T = np.concatenate(tris)
    area = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(T), n, p=area / area.sum()) if area.sum() > 0 else rng.choice(len(T), n)
    a, b = rng.random((2, n, 1))
    flip = (a + b) > 1
    a, b = np.where(flip, 1 - a, a), np.where(flip, 1 - b, b)
    T = T[pick]
    return T[:, 0] + a * (T[:, 1] - T[:, 0]) + b * (T[:, 2] - T[:, 0])


def top_slabs(prim, cell=0.05, t=0.03):
    """What a room object offers to stand on, from what is seen of it: per cell of its footprint a thin slab at the
    top of its points there (a desk's top where its lamp is not; the snap picks the highest one under an item's
    bottom). Sampled over its faces by area: a library desk's top is two triangles, its vertices only at the corners
."""
    P = surface_points(prim)
    if not len(P):
        return []
    lo = P[:, :2].min(0)
    ij = np.floor((P[:, :2] - lo) / cell).astype(int)
    top = {}
    for (i, j), z in zip(map(tuple, ij), P[:, 2]):
        top[(i, j)] = max(top.get((i, j), -np.inf), float(z))
    return [(np.r_[lo + [i * cell, j * cell], z - t], np.r_[lo + [(i + 1) * cell, (j + 1) * cell], z])
            for (i, j), z in top.items()]


def fit_footprint(prim, box, yaw, lim=(0.5, 2.5)):
    """A library stand-in stretched across to the piece it stands for (its measured box, which its box collider
    already is), its height kept: scaled by height alone a model keeps its own proportions. Per axis within `lim`; a piece turned off the room's axes (its box is then the
    turned footprint's bounds) is left as it is."""
    from .clearance import world_points

    if abs((np.degrees(yaw) + 45) % 90 - 45) > 5:
        return
    P = world_points(prim, 20000)
    lo, hi = P.min(0), P.max(0)
    k = np.clip((box[1][:2] - box[0][:2]) / np.maximum(hi[:2] - lo[:2], 1e-6), *lim)
    t = np.asarray(prim.GetAttribute("xformOp:translate").Get(), float)
    c = (lo[:2] + hi[:2]) / 2
    xy = (box[0][:2] + box[1][:2]) / 2 - k * (c - t[:2])  # the box's centre stays where it is
    set_pose(prim, [xy[0], xy[1], t[2]], scale=(float(k[0]), float(k[1]), 1.0))
