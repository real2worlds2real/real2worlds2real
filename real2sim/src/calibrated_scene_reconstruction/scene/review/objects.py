"""Automatic checks of every item a built scene holds, for the agent's object review (apps/scene/review_objects.py):

  support   rays down from the item's lowest points to the first collider that is not the item: the smallest gap
            is how far it floats (> FLOAT: nothing under it), a gap below -SINK it is sunk; an item hung on a wall
            (within WALL of a wall, a frame / clock / mirror / shelf / sconce) or from the ceiling (a lamp within
            CEIL of it) needs no support
  balance   rays down from points over the item's whole footprint, from just above its base: the share of its
            footprint over the surface it stands on (< OVER: it overhangs) and whether its centre (of its points)
            lies within the supported part (else it would tip); the support rays start from its lowest points only,
            which a box overhanging a desk's edge keeps on the desk
  collider  a room object stands on and holds things by one box collider (its model's own colliders off): the box
            beyond its visible footprint is a ledge nobody sees: the visible share of the box's footprint (< SEEN: flagged);
            for a thing that holds things (a desk, a cabinet) the share of the box's top its visible top fills
            (a room object built since with its meshes as its collider passes: what is seen is what collides)
  upright   a legged piece (sofa, chair, table) with legs in its top band and none in its bottom one is upside down
  size      height and footprint within the range of its class (SIZES; a book row is measured per book, not as a row)
  material  a glass or metal class built as an opaque diffuse reconstruction (a photo texture on a SAM 3D shell)
A plant in a basket or fruit in a bowl intersects it and stands on what is inside it: `contains` pairs are no overlap
and the held one no floating.
Pure functions over measurements; the app does the ray casts and renders."""
import re

import numpy as np

FLOAT, SINK, WALL, CEIL, OVER, SEEN = 0.012, 0.02, 0.04, 0.15, 0.85, 0.85
HUNG = r"frame|picture|clock|mirror|shelf|sconce|poster|painting|tv\b|television|curtain|blind"
HANGING = r"lamp|pendant|chandelier|light"
# (pattern, height range m, largest footprint side range m)
SIZES = [
    (r"book", (0.12, 0.42), (0.01, 1.2)),
    (r"cup|mug|tumbler", (0.05, 0.17), (0.04, 0.16)),
    (r"glass", (0.05, 0.25), (0.04, 0.12)),
    (r"bowl", (0.03, 0.2), (0.08, 0.45)),
    (r"plate", (0.01, 0.06), (0.12, 0.35)),
    (r"tray", (0.01, 0.3), (0.2, 0.7)),
    (r"vase|jar|pitcher|jug|carafe", (0.08, 0.8), (0.05, 0.45)),
    (r"bottle", (0.08, 0.45), (0.03, 0.15)),
    (r"plant", (0.08, 2.2), (0.05, 1.2)),
    (r"basket", (0.06, 0.6), (0.15, 0.7)),
    (r"frame|picture", (0.08, 1.2), (0.01, 1.5)),
    (r"stool", (0.4, 0.95), (0.25, 0.6)),
    (r"chair", (0.7, 1.15), (0.35, 0.75)),
    (r"bench", (0.35, 0.6), (0.3, 2.2)),
    (r"table", (0.45, 1.15), (0.4, 2.5)),
    (r"lamp", (0.15, 1.9), (0.08, 0.7)),
    (r"apple|orange|lemon|lime|pear|peach|plum|fruit", (0.04, 0.13), (0.04, 0.13)),
]
CONTAINER = r"basket|planter|pot\b|bowl|vase|tray|crate|box"
HELD = r"plant|flower|herb|fruit|apple|orange|lemon|lime|pear|peach|plum|banana|bread|loaf|bun|egg|vegetable"
GLASSY = r"glass|tumbler|carafe|bottle"
METALLIC = r"kettle|pot\b|tray|faucet|tap\b|pan\b|cooktop|steel|metal"


def size_check(label, ext):
    lab = (label or "").lower().replace("_", " ")
    rule = next((r for r in SIZES if re.search(r[0], lab)), None)
    if rule is None:
        return {"ok": True, "note": "no size range for this class"}
    (h0, h1), (f0, f1) = rule[1], rule[2]
    h, f = float(ext[2]), float(max(ext[0], ext[1]))
    ok = h0 * 0.85 <= h <= h1 * 1.15 and f0 * 0.85 <= f <= f1 * 1.15
    return {"ok": ok, "height_m": round(h, 3), "footprint_m": round(f, 3), "range": [list(rule[1]), list(rule[2])]}


def support_check(label, gaps, wall_dist, ceil_dist):
    """gaps: the per-ray gap (m) between an item's lowest points and the first other collider under them (None: no
    hit); wall_dist: distance of its bounding box to the nearest wall; ceil_dist: of its top to the ceiling."""
    lab = (label or "").lower().replace("_", " ")
    g = np.array([v for v in gaps if v is not None], float)
    out = {"min_gap_m": round(float(g.min()), 4) if len(g) else None,
           "median_gap_m": round(float(np.median(g)), 4) if len(g) else None,
           "hit_share": round(len(g) / max(len(gaps), 1), 2), "wall_m": round(float(wall_dist), 3),
           "ceiling_m": round(float(ceil_dist), 3)}
    if re.search(HANGING, lab) and ceil_dist < CEIL:
        return {**out, "ok": True, "mount": "ceiling"}
    if re.search(HUNG, lab) and wall_dist < WALL:
        return {**out, "ok": True, "mount": "wall"}
    if not len(g):
        return {**out, "ok": False, "problem": "nothing under it"}
    if g.min() > FLOAT:
        return {**out, "ok": False, "problem": f"floating {100 * g.min():.1f} cm"}
    if np.median(g) < -SINK:
        return {**out, "ok": False, "problem": f"sunk {-100 * np.median(g):.1f} cm"}
    return {**out, "ok": True}


def hull(xy):
    """Convex hull (counter-clockwise) of 2D points, monotone chain."""
    P = sorted(map(tuple, np.round(np.asarray(xy, float), 5)))
    if len(P) < 3:
        return np.array(P)

    def half(pts):
        h = []
        for p in pts:
            while len(h) >= 2 and np.cross(np.subtract(h[-1], h[-2]), np.subtract(p, h[-2])) <= 0:
                h.pop()
            h.append(p)
        return h
    return np.array(half(P)[:-1] + half(P[::-1])[:-1])


def inside_hull(H, q, tol=0.0):
    """Whether q lies within the convex polygon H (counter-clockwise), `tol` metres of slack."""
    if len(H) < 3:
        return False
    for a, b in zip(H, np.roll(H, -1, 0)):
        e = b - a
        if np.cross(e, np.asarray(q) - a) < -tol * np.linalg.norm(e):
            return False
    return True


def balance_check(drops, com_xy, mount=None):
    """drops: (x, y, drop) per ray from over the item's footprint -- how far below its base the first surface under
    that point lies (None: nothing); com_xy: its centre. On the floor or hung, nothing to check."""
    if mount or not drops:
        return {"ok": True}
    d = np.array([[x, y, np.inf if v is None else v] for x, y, v in drops], float)
    on = d[:, 2] < 0.03
    share = float(on.mean())
    out = {"over_support": round(share, 2)}
    H = hull(d[on, :2]) if on.sum() >= 3 else []
    if len(H) >= 3 and not inside_hull(H, com_xy, 0.01):  # a degenerate hull (one cell, one row): the share decides
        return {**out, "ok": False, "problem": f"its centre off its support ({100 * share:.0f} % of it over it): it tips"}
    if share < OVER:
        return {**out, "ok": False, "problem": f"overhangs: {100 * (1 - share):.0f} % of its footprint over nothing"}
    return {**out, "ok": True}


LEGGED = r"sofa|couch|chair|table|desk|bench|stool|bed"
HOLDS = r"table|desk|counter|cabinet|shelf|shelving|dresser|nightstand|bench|workbench|box|crate"


def _bands(P, cell=0.03):
    """Occupancy (2D) of the item's points in its bottom and top 12 % of height, over its footprint's grid."""
    lo, hi = P.min(0), P.max(0)
    h = hi[2] - lo[2]
    n = np.maximum(np.ceil((hi[:2] - lo[:2]) / cell).astype(int), 1)
    out = []
    for m in (P[:, 2] <= lo[2] + 0.12 * h, P[:, 2] >= hi[2] - 0.12 * h):
        g = np.zeros(n[::-1], np.uint8)
        ij = np.clip(((P[m, :2] - lo[:2]) / cell).astype(int), 0, n - 1)
        g[ij[:, 1], ij[:, 0]] = 1
        out.append(g)
    return out


def _legs(g):
    """Whether an occupancy band looks like legs: 3 or more small pieces spread over the footprint."""
    from scipy import ndimage

    lab, k = ndimage.label(g)
    if k < 3:
        return False
    sizes = ndimage.sum(g, lab, range(1, k + 1))
    small = [i + 1 for i, s_ in enumerate(sizes) if s_ <= 0.04 * g.size]
    if len(small) < 3 or g.mean() > 0.2:
        return False
    cs = np.array(ndimage.center_of_mass(g, lab, small))
    span = np.ptp(cs, 0) / np.maximum(np.array(g.shape) - 1, 1)
    return bool((span > 0.5).all())


def upright_check(label, P):
    """A legged piece (sofa, chair, table) with legs at its top and none at its bottom lies upside down."""
    if not re.search(LEGGED, label or "", re.I) or len(P) < 500:
        return {"ok": True}
    bottom, top = _bands(P)
    if _legs(top) and not _legs(bottom):
        return {"ok": False, "problem": "upside down: legs at its top"}
    return {"ok": True}


def collider_check(visible_lo, visible_hi, box_lo, box_hi, label=None, P=None, cell=0.03):
    """The share of a box collider's footprint the item's visible points cover: for a thing that holds things on its
    top (a desk, a cabinet) the share of the box's top its visible points fill within 5 cm under it, else the overlap of their bounds."""
    if P is not None and re.search(HOLDS, label or "", re.I):
        lo, hi = np.asarray(box_lo, float), np.asarray(box_hi, float)
        n = np.maximum(np.ceil((hi[:2] - lo[:2]) / cell).astype(int), 1)
        m = (P[:, 2] >= hi[2] - 0.05) & (P[:, 2] <= hi[2] + 0.02) & np.all(P[:, :2] >= lo[:2], 1) & np.all(
            P[:, :2] <= hi[:2], 1)
        g = np.zeros(n[::-1], np.uint8)
        ij = np.clip(((P[m, :2] - lo[:2]) / cell).astype(int), 0, n - 1)
        g[ij[:, 1], ij[:, 0]] = 1
        import cv2

        g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))  # the sampling's gaps
        share = float(g.mean())
        if share < SEEN:
            return {"ok": False, "visible_share": round(share, 2),
                    "problem": f"its collider's top is {100 * share:.0f} % covered by what is seen"}
        return {"ok": True, "visible_share": round(share, 2)}
    lo, hi = np.maximum(visible_lo[:2], box_lo[:2]), np.minimum(visible_hi[:2], box_hi[:2])
    share = float(np.prod(np.clip(hi - lo, 0, None)) / max(np.prod(np.asarray(box_hi[:2]) - box_lo[:2]), 1e-9))
    if share < SEEN:
        return {"ok": False, "visible_share": round(share, 2),
                "problem": f"its collider reaches past what is seen ({100 * share:.0f} % of it visible)"}
    return {"ok": True, "visible_share": round(share, 2)}


def material_check(label, kind):
    """kind: how the item is built -- 'recon' (SAM 3D shell with a photo texture), 'library', 'kit', 'template'."""
    lab = (label or "").lower().replace("_", " ")
    if kind != "recon":
        return {"ok": True}
    if re.search(GLASSY, lab):
        return {"ok": False, "problem": "glass rebuilt as an opaque shell"}
    if re.search(METALLIC, lab):
        return {"ok": False, "problem": "metal rebuilt as a diffuse shell"}
    return {"ok": True}


def contains(label_a, label_b):
    """One of the two holds the other (a plant in a basket, apples in a bowl)."""
    a, b = ((x or "").lower().replace("_", " ") for x in (label_a, label_b))
    return bool((re.search(CONTAINER, a) and re.search(HELD, b)) or (re.search(CONTAINER, b) and re.search(HELD, a)))
