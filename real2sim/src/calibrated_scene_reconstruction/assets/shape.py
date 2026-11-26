"""Orientation semantics of furniture and props from their geometry (z-up points: a library model in its own frame,
as recon.furniture.load_zup returns it, or a placed item's world points).

profile(V) -> {height, seat_h, back, long, stand}
  back   direction (xy unit vector) from the footprint centre towards a backrest, None when there is none: the top
         band of a chair-like stool sits on one side of the footprint, a backless stool's top band (the seat) is
         centred on it;
  seat_h seat height: the highest level whose depth along `back` still covers most of the footprint (the backrest
         above it is thin); the top for backless pieces;
  long   footprint long-axis angle (rad, mod pi) when the footprint is elongated, else None;
  stand  direction towards a standing prop's support leg (a picture frame's easel), None when absent: the lower
         band sits on one side along the thin footprint axis; the prop's front faces the opposite way;
  handle direction of a handle (a mug's), None when absent: more than 1 % of the points lie well outside the
         body's radius (1.35 x the median distance from the body axis), all in one direction (mean unit
         direction > 0.9).
Placement rules: a stool's backrest points away from the counter it stands at; an elongated swap keeps the twin
instance's long axis; a front-facing prop (picture frame) shows its front to the room / photo camera.
"""
import numpy as np

SEAT_MIN, KNEE_ROOM = 0.55, 0.12  # a stool's seat: at least this high, and this far below the counter slab's underside
BACK_OFFSET = 0.2  # top-band centroid offset (fraction of the footprint depth) that marks a backrest
ELONGATED = 1.3


def angle(v):
    return float(np.arctan2(v[1], v[0]))


def unit(a):
    return np.array([np.cos(a), np.sin(a)])


def wrap(a, period=2 * np.pi):
    """Angle in [-period/2, period/2)."""
    return float((a + period / 2) % period - period / 2)


def footprint_axes(V):
    """Principal axes of the xy footprint: (long-axis angle, long / short extent ratio)."""
    P = V[:, :2] - V[:, :2].mean(0)
    w, U = np.linalg.eigh(np.cov(P.T))
    a = angle(U[:, 1])
    ext = [np.ptp(P @ U[:, k]) for k in (1, 0)]
    return a, ext[0] / max(ext[1], 1e-6)


def profile(V):
    V = np.asarray(V, float)
    lo, hi = V.min(0), V.max(0)
    h = hi[2] - lo[2]
    c = (lo[:2] + hi[:2]) / 2
    out = {"height": float(h), "seat_h": float(h), "back": None, "long": None, "stand": None, "handle": None}
    axis = np.median(V[:, :2], 0)
    r = np.linalg.norm(V[:, :2] - axis, axis=1)
    far = r > 1.35 * np.median(r)
    if far.sum() > 0.01 * len(V):
        off = V[far, :2].mean(0) - axis
        d = (V[far, :2] - axis) / r[far, None]
        if np.linalg.norm(off) > 0.25 * np.ptp(V[:, :2], 0).max() and np.linalg.norm(d.mean(0)) > 0.9:
            out["handle"] = (off / np.linalg.norm(off)).tolist()
    top = V[V[:, 2] > lo[2] + 0.8 * h]
    off = top[:, :2].mean(0) - c
    depth = np.ptp(V[:, :2] @ (off / (np.linalg.norm(off) + 1e-9)))
    if np.linalg.norm(off) > BACK_OFFSET * depth:
        b = off / np.linalg.norm(off)
        out["back"] = b.tolist()
        proj = V[:, :2] @ b
        full = np.ptp(proj)
        for z in np.arange(hi[2], lo[2] + 0.3 * h, -0.01):  # from the top down: first level as deep as the seat
            band = (V[:, 2] <= z) & (V[:, 2] > z - 0.03)
            if band.sum() > 10 and np.ptp(proj[band]) > 0.6 * full:
                out["seat_h"] = float(z - lo[2])
                break
    a, ratio = footprint_axes(V)
    if ratio > ELONGATED:
        out["long"] = wrap(a, np.pi)
        thin = unit(a + np.pi / 2)
        low = V[V[:, 2] < lo[2] + 0.15 * h]
        s = float((low[:, :2].mean(0) - c) @ thin)
        if abs(s) > 0.15 * np.ptp(V[:, :2] @ thin):
            out["stand"] = (np.sign(s) * thin).tolist()
    return out


def toward_box(xy, lo, hi):
    """Unit xy direction from a point to a box footprint; under it (a stool under a counter overhang): the inward
    normal of the nearest edge."""
    xy = np.asarray(xy[:2], float)
    p = np.clip(xy, lo[:2], hi[:2])
    if np.linalg.norm(p - xy) > 1e-6:
        return (p - xy) / np.linalg.norm(p - xy)
    d = [xy[0] - lo[0], hi[0] - xy[0], xy[1] - lo[1], hi[1] - xy[1]]
    return np.array([(1, 0), (-1, 0), (0, 1), (0, -1)][int(np.argmin(d))], float)


def nearest_counter(xy, boxes, top=(0.7, 1.15)):
    """(name, lo, hi) of the nearest box whose top is at counter / bar height."""
    best = None
    for name, (lo, hi) in boxes.items():
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        if not top[0] <= hi[2] <= top[1]:
            continue
        d = np.linalg.norm(np.clip(xy[:2], lo[:2], hi[:2]) - np.asarray(xy[:2]))
        if best is None or d < best[0]:
            best = (d, name, lo, hi)
    return best[1:] if best else None


def seat_range(boxes, name, lo, hi):
    """Seat heights that suit the support next to a seat: from 0.40 m below its top (a dining chair at a 0.75 m table,
    a bar stool at a 1.05 m counter) up to its slab underside less KNEE_ROOM; never below 0.35 m."""
    return max(0.35, float(hi[2]) - 0.40), underside(boxes, name, lo, hi) - KNEE_ROOM


def underside(boxes, name, lo, hi):
    """Lowest bottom among the boxes at a counter's top level over its footprint: the slab underside (knee room)."""
    zs = [b_lo[2] for b_lo, b_hi in (map(np.asarray, v) for v in boxes.values())
          if abs(b_hi[2] - hi[2]) < 0.02 and b_lo[2] > 0.3
          and b_lo[0] < hi[0] and lo[0] < b_hi[0] and b_lo[1] < hi[1] and lo[1] < b_hi[1]]
    return float(min(zs)) if zs else float(hi[2])
