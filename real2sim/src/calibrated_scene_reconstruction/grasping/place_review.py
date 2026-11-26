"""Automatic checks of sampled place targets (apps/review_receptacles.py): the measurements are taken in Isaac (rays,
a physics settle, renders), the verdicts here. Pure numpy.

    footprint_points(rd) -> world xy points covering a placed receptacle's footprint (support rays)
    geometry_checks(rec, rd, support_box, r_obj, target, cfg) -> {check: {"ok": bool, ...}}
    support_check(hits, top, support) / settle_check(...) / visible_check(...) -> {"ok": bool, ...}

A receptacle passes when every check does; the agent reads the failures next to the renders and decides per kind
(scene graph task.place: weights, keep-out areas, a `_review` note).
"""
from __future__ import annotations

import numpy as np

from ..assets import receptacles as rcp
from .place import PlaceCfg

SUPPORT_TOL = 0.003     # m, the support top under the footprint vs the receptacle's bottom
SETTLE_DZ = 0.005       # m, the settled object's base vs the place surface
SETTLE_TILT = 10.0      # deg
SETTLE_SLIDE = 0.02     # m
SIZE_AREA = 0.2         # footprint area / support top area
SIZE_SIDE = 0.6         # footprint's longest side / support top's shorter side
TEXTURE_TOL = 0.25      # UV scale vs the texture's real-world size
VISIBLE_FRAC = 0.001    # changed pixels of the robot view (a coaster across the table: ~0.2 %)


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _hull(P):
    """Convex hull (counter-clockwise) of 2-D points, monotone chain."""
    P = np.unique(np.round(np.asarray(P, float), 6), axis=0)
    if len(P) < 3:
        return P

    def half(pts):
        h = []
        for p in pts:
            while len(h) >= 2 and _cross(h[-1] - h[-2], p - h[-2]) <= 0:
                h.pop()
            h.append(p)
        return h[:-1]

    return np.array(half(P) + half(P[::-1]))


def outline(rec):
    """Footprint outline in the receptacle frame: the hull of its parts, or the `half` rectangle of a scanned model."""
    if rec.parts:
        return _hull(np.concatenate([np.asarray(p["V"])[:, :2] for p in rec.parts]))
    hx, hy = rec.half
    return np.array([[hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy]])


def footprint_points(rec, rd, inset=0.002, n_edge=3):
    """World xy points on a placed receptacle's footprint: its centre, its outline shrunk by `inset` and points along
    each outline edge."""
    O = outline(rec)
    c = O.mean(0)
    O = c + (O - c) * (1 - inset / max(np.linalg.norm(O - c, axis=1).max(), 1e-6))
    pts = [c]
    for i in range(len(O)):
        a, b = O[i], O[(i + 1) % len(O)]
        pts += [a + (b - a) * t for t in np.arange(n_edge) / n_edge]
    return np.array([rcp.to_world(rd, p) for p in pts])


def support_check(hits, top, support):
    """Rays down through the footprint (receptacle absent): [(hit z or None, collider path or None)]. Each must hit the
    support at its top: a lower surface (off the edge) leaves the receptacle floating, a higher one sinks it into
    something, another collider means it overlaps that."""
    gaps = [None if z is None else float(top - z) for z, _ in hits]
    other = sorted({p for z, p in hits if p is not None and support not in p})
    missed = sum(z is None for z, _ in hits)
    worst = max((abs(g) for g in gaps if g is not None), default=None)
    ok = missed == 0 and not other and worst is not None and worst <= SUPPORT_TOL
    return {"ok": bool(ok), "max_gap_m": worst, "missed": missed, "other": other, "rays": len(hits)}


def _up(q):
    """World direction of a body's +z for an orientation quaternion (wxyz)."""
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


def settle_check(p_goal, p_end, q_end, rd, q_rest=(1, 0, 0, 0)):
    """The target object dropped onto the place surface (base at `p_goal`, its resting height on the bare support
    carried over, plus a few mm) after a physics settle: resting on it, upright (its tilt measured from its resting
    orientation `q_rest` on the bare support), where it was put, inside the accept region."""
    p_goal, p_end = np.asarray(p_goal, float), np.asarray(p_end, float)
    dz, slide = float(p_end[2] - p_goal[2]), float(np.linalg.norm(p_end[:2] - p_goal[:2]))
    tilt = float(np.degrees(np.arccos(np.clip(_up(q_end) @ _up(q_rest), -1.0, 1.0))))
    inside = rcp.inside(rd, p_end[:2])
    ok = abs(dz) <= SETTLE_DZ and tilt <= SETTLE_TILT and slide <= SETTLE_SLIDE and inside
    return {"ok": bool(ok), "dz_m": dz, "tilt_deg": tilt, "slide_m": slide, "inside": bool(inside)}


def visible_check(img_without, img_with):
    """Fraction of the robot view's pixels the receptacle changes."""
    d = np.abs(np.asarray(img_with, float) - np.asarray(img_without, float)).sum(-1)
    frac = float((d > 30).mean())
    return {"ok": frac >= VISIBLE_FRAC, "changed_frac": frac}


def uv_scale(part):
    """Metres per UV unit of a part's faces seen from above (median over the faces with an xy area of
    1 / sqrt|det d(uv)/d(xy)|; pieces may be offset or turned, as bamboo slats), None without such faces."""
    V, UV, F = np.asarray(part["V"], float), np.asarray(part["UV"], float), np.asarray(part["F"])
    e1, e2 = V[F[:, 1], :2] - V[F[:, 0], :2], V[F[:, 2], :2] - V[F[:, 0], :2]
    u1, u2 = UV[F[:, 1]] - UV[F[:, 0]], UV[F[:, 2]] - UV[F[:, 0]]
    a_xy, a_uv = _cross(e1.T, e2.T), _cross(u1.T, u2.T)
    ok = np.abs(a_xy) > 1e-8
    if not ok.any():
        return None
    return float(np.median(np.sqrt(np.abs(a_xy[ok]) / np.maximum(np.abs(a_uv[ok]), 1e-18))))


def texture_check(rec):
    """Every textured part shows its texture at the real-world size: a Poly Haven texture's `dimensions`
    (assets.receptacles.TEX_SIZE), a generated set within its range (PROC_TILE). Scanned models keep their own maps."""
    out = []
    for p in rec.parts:
        m = p["material"]
        if "tex" in m:
            want = (rcp.TEX_SIZE[m["tex"]],) * 2
        elif "proc" in m:
            want = rcp.PROC_TILE[m["proc"]]
        else:
            continue
        got = uv_scale(p)
        ok = got is not None and want[0] * (1 - TEXTURE_TOL) <= got <= want[1] * (1 + TEXTURE_TOL)
        out.append({"part": p["name"], "look": rcp.look(m), "m_per_image": got, "real_m": list(want), "ok": bool(ok)})
    return {"ok": all(o["ok"] for o in out), "parts": out}


def geometry_checks(rec, rd, support_box, r_obj, target, obstacles=(), cfg=None):
    """The checks that need no simulator: inside the support top, clear of the obstacles, holds the object, sized
    for the table, textures at real scale, the task text names it."""
    from .place import _rect_clear
    c = cfg or PlaceCfg()
    lo, hi = np.asarray(support_box[0], float)[:2], np.asarray(support_box[1], float)[:2]
    W = np.array([rcp.to_world(rd, p) for p in outline(rec)])
    margin = float(min((W - lo).min(), (hi - W).min()))
    corners = rcp.footprint(rd)
    blocked = [i for i, b in enumerate(obstacles) if not _rect_clear(corners, b, c.obstacle_margin)]
    a = rd["accept"]
    inset = 0.5 * r_obj
    if rec.type == "container":
        room = float(rec.info.get("r_flat", a[1] + inset))
        holds = room >= r_obj + rcp.FINGER_ROOM - 1e-6
    else:
        room = float((a[1] if a[0] == "circle" else min(a[1], a[2])) + inset)
        holds = room >= 0.9 * r_obj - 1e-6
    hx, hy = rd["half"]
    side = 2 * max(hx, hy)
    area = 4 * hx * hy
    s_area = float(np.prod(hi - lo))
    text = rcp.task_text(target, rd)
    return {
        "inside": {"ok": margin >= c.edge_margin - 1e-6, "edge_margin_m": margin},
        "clear": {"ok": not blocked, "blocked_by": blocked},
        "holds": {"ok": bool(holds), "place_radius_m": room, "r_obj": float(r_obj)},
        "size": {"ok": area <= SIZE_AREA * s_area and side <= SIZE_SIDE * float(min(hi - lo)),
                 "footprint_m": [2 * hx, 2 * hy], "area_frac": float(area / s_area),
                 "side_frac": float(side / min(hi - lo))},
        "texture": texture_check(rec),
        "text": {"ok": rcp.KINDS[rec.kind]["label"] in text, "task": text},
    }
