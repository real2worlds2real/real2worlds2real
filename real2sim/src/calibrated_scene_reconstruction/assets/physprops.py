"""Semantic mass and centre of mass for simulated objects: a reconstruction or a library model is a surface, the
simulator fills it with uniform density, and real objects are not uniform -- a table lamp's weight is in its base
(uniformly filled, the shade puts the centre of mass up high and it tips over at a small angle), a potted plant's
in the soil, a mug's and a vase's in their thick bottoms.

    mass_props(label, V) -> {"class", "mass", "com"}   (com in the frame of V)

Per class: an effective density for the closed hull's volume (kg / m^3, clipped to the class's mass range) and the
height of the centre of mass as a fraction of the object's height, over its base axis (the lowest 30 % of the points:
a mug's body, not its handle). The first class whose pattern matches the label wins; unknown labels keep uniform
density at 300 kg / m^3 (hollow household objects).

    rest_pose(V, com) -> (R, tip angle)   rolls(label)

Round produce rolls on a rigid hull where the real thing stops: it is laid on the flat spot of its hull that tips
latest (a library lemon modelled standing on its tip lies down on its side) and given angular damping in place of the
rolling resistance of its soft contact (sim.scene, sim.assets; distractors, not the task target).
"""
import re

import numpy as np
import trimesh

CLASSES = [  # (pattern, density of the hull volume, (min kg, max kg), centre-of-mass height / object height)
    (r"lamp|lantern", 350, (0.4, 4.0), 0.2),
    (r"plant|planter|flower ?pot", 500, (0.2, 6.0), 0.25),
    (r"vase|jar|jug|pitcher|kettle|carafe|bottle|teapot", 250, (0.15, 2.5), 0.35),
    (r"cup|mug|tumbler|glass", 450, (0.08, 0.6), 0.35),
    (r"bowl|plate|dish|tray|\bpan\b|\bpot\b", 350, (0.1, 2.0), 0.3),
    (r"apple|orange|lemon|lime|pear|peach|tomato|potato|onion|fruit", 850, (0.05, 0.4), 0.5),
    (r"banana", 600, (0.08, 0.3), 0.5),
    (r"bread|loaf|croissant|bun", 250, (0.05, 0.8), 0.45),
    (r"book|notebook", 650, (0.1, 2.0), 0.5),
    (r"box|carton|\btin\b|\bcan\b", 300, (0.05, 2.0), 0.5),
    (r"basket", 120, (0.1, 1.5), 0.35),
    (r"frame|picture|photo", 500, (0.1, 2.0), 0.5),
]
DEFAULT = ("", 300, (0.05, 3.0), 0.5)


def semantic_class(label):
    lab = (label or "").lower().replace("_", " ")
    return next((c for c in CLASSES if re.search(c[0], lab)), DEFAULT)


def mass_from_box(label, size):
    """Mass (kg) of an object of this label from its bounding box (scene-graph drafting, before a mesh exists): a round
    household object fills about two thirds of its box."""
    _, density, (lo, hi), _ = semantic_class(label)
    return float(np.clip(density * 0.65 * float(np.prod(np.maximum(size, 0.02))), lo, hi))


def mass_props(label, V):
    """Mass (kg) and centre of mass (in the frame of the vertices V) of an object of this label."""
    V = np.asarray(V, float)
    pattern, density, (lo, hi), frac = semantic_class(label)
    flat = np.linalg.matrix_rank(V - V.mean(0), tol=1e-6) < 3  # a single plane: no hull volume
    vol = float(np.prod(np.ptp(V, axis=0))) if flat else float(trimesh.convex.convex_hull(V).volume)
    zmin, h = V[:, 2].min(), np.ptp(V[:, 2])
    base = V[V[:, 2] < zmin + 0.3 * h]
    c = np.median(base[:, :2], axis=0) if len(base) >= 10 else (V.min(0) + V.max(0))[:2] / 2
    return {"class": pattern or "default", "mass": float(np.clip(density * vol, lo, hi)),
            "com": [float(c[0]), float(c[1]), float(zmin + frac * h)]}


ROLLING = r"apple|orange|lemon|lime|pear|peach|plum|tomato|potato|onion|fruit|banana|\bball\b"  # produce, balls
ROLLING_DAMPING = 4.0  # 1/s angular damping: the rolling resistance of a soft round body (a few cm of roll at most)
SUPPORT_TOL = 0.001  # a resting body touches the support over the hull points this close to it: a soft fruit's
#                       flat spot under its own weight, a scan's base ring
TIP_MIN_DEG = 3.0  # an authored pose this close to tipping over is not a resting pose
STEADIER = 1.5  # a flat spot this many times steadier than the authored pose wins
ROUND_DEG = 12.0  # resting poses that tip at a smaller angle roll: produce, balls, a banana on its back


def rolls(label):
    """Round produce and balls: on a rigid hull PhysX lets them roll on, where a real one's flat spot and its soft
    contact (rolling resistance) stop it within millimetres."""
    return bool(re.search(ROLLING, (label or "").lower().replace("_", " ")))


def _rot_to(n, target=(0.0, 0.0, -1.0)):
    """Rotation taking unit vector n onto target (Rodrigues; a half turn about x when they are opposite)."""
    n, t = np.asarray(n, float), np.asarray(target, float)
    v, c = np.cross(n, t), float(n @ t)
    if c < -1 + 1e-9:
        return np.diag([1.0, -1.0, -1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def tip_angle(V, com):
    """Degrees the body standing on its lowest points (V, z up) tilts before its centre of mass leaves the support
    polygon (the hull points within SUPPORT_TOL of the bottom); <= 0: it is not resting."""
    V = np.asarray(V, float)
    sup = V[V[:, 2] < V[:, 2].min() + SUPPORT_TOL][:, :2]
    h = max(float(com[2] - V[:, 2].min()), 1e-6)
    c = np.asarray(com[:2], float)
    if len(np.unique(np.round(sup, 5), axis=0)) < 3 or np.linalg.matrix_rank(sup - sup.mean(0), tol=1e-5) < 2:
        return float(np.degrees(np.arctan2(-np.min(np.linalg.norm(sup - c, axis=1)), h)))
    from scipy.spatial import ConvexHull

    hull = ConvexHull(sup)  # equations: n . x + d <= 0 inside, |n| = 1
    margin = float(-np.max(hull.equations[:, :2] @ c + hull.equations[:, 2]))
    return float(np.degrees(np.arctan2(margin, h)))


def level_rotation(V, max_deg=25.0, min_deg=0.5):
    """Rotation that sets a reconstructed body (V, z up) down on its own base: the largest flat facet of its convex
    hull facing within `max_deg` of straight down onto the support. A photo reconstruction's frame can be tilted by
    its pose error. Identity below `min_deg` or without such a facet (a body resting on a point)."""
    V = np.asarray(V, float)
    if len(V) > 6000:
        V = V[np.random.default_rng(0).choice(len(V), 6000, replace=False)]
    hull = trimesh.convex.convex_hull(V)
    best = None
    for n, a in zip(hull.facets_normal, hull.facets_area):
        ang = float(np.degrees(np.arccos(np.clip(-n[2], -1.0, 1.0))))
        if ang <= max_deg and (best is None or a > best[1]):
            best = (n, a, ang)
    if best is None or best[2] < min_deg:
        return np.eye(3)
    return _rot_to(best[0])


def rest_pose(V, com, candidates=64):
    """How a body rests on a support: (R, tip angle in degrees). R = identity while the authored pose (V, z up) rests
    (tips at >= TIP_MIN_DEG) about as well as any other; otherwise the rotation onto the flat spot of its convex hull
    that tips latest -- a library lemon modelled standing on its tip lies down on its side, as a real one would."""
    V, com = np.asarray(V, float), np.asarray(com, float)
    if len(V) > 3000:
        V = V[np.random.default_rng(0).choice(len(V), 3000, replace=False)]
    tip0 = tip_angle(V, com)
    hull = trimesh.convex.convex_hull(V)
    normals = list(hull.facets_normal) + list(hull.face_normals)
    areas = list(hull.facets_area) + list(hull.area_faces)
    best = (tip0, np.eye(3))
    for k in np.argsort(areas)[::-1][:candidates]:
        R = _rot_to(normals[k])
        t = tip_angle(hull.vertices @ R.T, R @ com)
        if t > best[0] + 1e-6:
            best = (t, R)
    if tip0 >= TIP_MIN_DEG and best[0] < STEADIER * tip0:  # the authored pose rests about as well: keep it
        return np.eye(3), tip0
    return best[1], best[0]
