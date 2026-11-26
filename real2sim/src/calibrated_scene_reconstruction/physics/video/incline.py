"""An object on a support that is not level (meta.json "motion": "incline", "level": prompt of a surface that is).

    envs/recon/bin/python -m calibrated_scene_reconstruction.physics.video.fit real/<name>     (fit.py hands "incline" recordings to run() here)

Gravity pulls the object down the slope with g sin(theta), kinetic friction holds it back with mu g cos(theta):

    it slides, with acceleration a      kinetic friction = tan(theta) - a / (g cos(theta)); static friction < tan(theta)
    it stays where it was put           static friction >= tan(theta); nothing else is measured

theta is the angle between the support's plane and a level surface in the same picture: perception fits both in
MoGe-2's points of the same frames (perception.npz level_n), track.py writes the direction of gravity in the
twin's frame (twin.json "gravity", "tilt_deg"). The track itself lies in the support's plane as on a table: the
scene frame is the support's, whichever way it lies. The mass does not enter here either.

A support that tilts while the video runs (meta.json "support_moves": true; perception then fits its plane on every
frame, twin.json "tilt_deg_t"): the object rides it until the tilt reaches the angle of repose, tan of which is the
static friction; from there it slides, by the same law with the tilt of every frame.
"""
from calibrated_scene_reconstruction.physics.common.schema import read_document
import json
from pathlib import Path

import numpy as np

from .fit import PIXELS, PRIOR, spikes
from .scene import G, ray_plane, unusable

TILT_STD_DEG = 1.5

PATH_TILT_STD_DEG = 2.0  # deg: the tilt of a path laid through the object's own places in MoGe-2's points
MOVED = 0.006       # m: an object that went less far than this did not move


def support_hull(per, twin):
    """The support's outline in the plane (N, 2) m, convex: where its mask's border rays meet z = 0."""
    import cv2
    from scipy.spatial import ConvexHull

    cs, _ = cv2.findContours(per["support"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    uv = np.concatenate([c.reshape(-1, 2) for c in cs]).astype(float)
    X = ray_plane(twin["K"], twin["T_world_cam"], uv)[:, :2]
    X = X[np.all(np.isfinite(X), 1)]
    if len(X) < 4:
        return None
    return X[ConvexHull(X).vertices]


def inside(hull, xy, margin=0.0):
    """Which points (N, 2) lie in the convex polygon `hull` (counter-clockwise), `margin` m from its border."""
    if hull is None:
        return np.ones(len(xy), bool)
    a, b = hull, np.roll(hull, -1, 0)
    e = b - a
    nrm = np.stack([-e[:, 1], e[:, 0]], 1) / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)
    d = np.einsum("nkd,kd->nk", xy[:, None, :] - a[None], nrm)  # distance inside of every edge
    return np.all(d >= margin, 1)


def fit_ramp(t, xy):
    """Constant acceleration along a line through positions xy (N, 2) at times t (N,): p = (x0, y0, heading rad,
    v0, a), a signed along the heading. Returns {p, cov, rms, distance}."""
    from scipy.optimize import least_squares

    t = np.asarray(t, float) - t[0]
    xy = np.asarray(xy, float)
    d = xy[-1] - xy[0]
    D = max(np.linalg.norm(d), 1e-4)

    def pos(p):
        s = p[3] * t + 0.5 * p[4] * t ** 2
        return np.stack([p[0] + s * np.cos(p[2]), p[1] + s * np.sin(p[2])], 1)

    best = None
    for v0 in (0.0, D / t[-1]):  # released from rest, or already under way
        p0 = [xy[0, 0], xy[0, 1], np.arctan2(d[1], d[0]), v0, 2 * (D - v0 * t[-1]) / t[-1] ** 2]
        r = least_squares(lambda p: (pos(p) - xy).ravel(), p0, loss="soft_l1", f_scale=0.005,
                          x_scale=[0.05, 0.05, 0.1, 0.5, 2.0])
        if best is None or r.cost < best.cost:
            best = r
    p = best.x
    res = pos(p) - xy
    s2 = float((res ** 2).sum() / max(res.size - 5, 1))
    return {"p": p, "cov": s2 * np.linalg.pinv(best.jac.T @ best.jac), "rms": float(np.sqrt((res ** 2).sum(1).mean())),
            "distance": float(abs(p[3] * t[-1] + 0.5 * p[4] * t[-1] ** 2)), "positions": pos(p)}


def run_along_path(per, track, twin, out, pr):
    """An incline seen along itself (a board from its side: it shows as a line, and MoGe-2 fits no plane through
    a line). The object cannot be stood on a plane the camera lies in, but it need not be: a body that slides
    straight down a slope moves on a line, and that line is seen. It is laid through the places MoGe-2 saw the
    object at in the frames it looked at (perception object_xyz, at the twin's scale); its angle to the level
    surface is the tilt, and each frame's place along it is where the line comes closest to the ray through the
    mask's centroid. From there on it is the fit of any incline: constant acceleration a along the path,
    friction tan(tilt) - a / (g cos(tilt)). Fills and returns `out`."""
    from calibrated_scene_reconstruction.physics.common.observations import mask_stats

    def refuse(why):
        out["note"] = ("the support is seen along itself and is no plane in the picture; " + why
                       + ": nothing is measured")
        return out

    if "level_n" not in per.files:
        return refuse("no level surface was found either (meta.json 'level'), so which way is down is not known")
    more = "object_at_xyz" in per.files and len(per["object_at_xyz"])
    if not more and ("object_xyz" not in per.files or len(per["object_xyz"]) == 0):
        return refuse("perception placed the object nowhere in depth")
    d = -np.median(np.asarray(per["level_n"], float), 0)
    d /= np.linalg.norm(d)
    s0 = float(twin.get("scale", {}).get("factor", 1.0))
    X_all = np.asarray(per["object_at_xyz"] if more else per["object_xyz"], float) * s0
    at_all = np.asarray(per["object_at_frames"] if more else per["plane_frames"], int)
    ok = np.all(np.isfinite(X_all), 1)
    X_all, at_all = X_all[ok], at_all[ok]
    size = float(np.nanmedian(per["object_size"])) * s0 if np.isfinite(per["object_size"]).any() else 0.0
    K, t = np.asarray(per["K"], float), np.asarray(per["t"], float)
    uv = np.array([mask_stats(m)[0] for m in per["object"]])
    seen = np.all(np.isfinite(uv), 1) & (per["score"] > 0.05) & ~track["border"]
    r = np.c_[(uv[:, 0] - K[0, 2]) / K[0, 0], (uv[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(uv))]
    r /= np.linalg.norm(r, axis=1, keepdims=True)
    rel_scale = float(twin.get("scale", {}).get("rel_std", 0.09))
    free = seen & (~track["contact"] | left_alone(per["object"], per["manipulator"], track["contact"]))
    tilt, lines = None, []
    for i0, i1 in runs(free):
        i0 = letting_go(per["object"], per["manipulator"], i0)
        k = np.arange(i0, i1 + 1)
        k = k[seen[k]]
        if len(k) < 5:
            continue
        # a slide is a straight line in the picture, as far as it is one: where the object leaves the board (it
        # drops to the floor, it tumbles) its track bends, and what comes after is no part of the slide
        k = k[:straight_prefix(uv[k], max(3.0, 0.03 * float(np.linalg.norm(uv[k[-1]] - uv[k[0]]))))]
        if len(k) < 5:
            continue
        # the line of this slide: through the places MoGe-2 saw the object at while it went along it
        inside = (at_all >= k[0]) & (at_all <= k[-1])
        X, at = X_all[inside], at_all[inside]
        if len(X) < 2:
            lines.append(f"the object was placed in depth in {len(X)} of the frames of its slide")
            continue
        c = X.mean(0)
        u = np.linalg.svd(X - c)[2][0]
        span = float(np.ptp((X - c) @ u))
        off = float(np.sqrt((((X - c) - np.outer((X - c) @ u, u)) ** 2).sum(1).mean()))
        if span < max(size, 0.05 * np.linalg.norm(c)) or off > 0.15 * span:  # it goes its own length at the least
            lines.append(f"the places the object was seen at ({len(X)}, {span:.2f} m apart, {off:.2f} m off one "
                         "line) do not make the line of a slide")
            continue
        if u @ d < 0:
            u = -u
        tilt = float(np.arcsin(np.clip(u @ d, -1, 1)))
        # the place along the line closest to each frame's ray (from the camera, the origin)
        ru = r[k] @ u
        den = np.where(1 - ru ** 2 > 1e-6, 1 - ru ** 2, np.nan)
        s_ = ((r[k] @ c) * ru - c @ u) / den
        lam = (r[k] @ c) + s_ * ru
        gap = np.linalg.norm(c + np.outer(s_, u) - lam[:, None] * r[k], axis=1)  # how far the ray passes the line
        r_px = float(np.linalg.norm(c) / K[0, 0])                               # metres a pixel is, at the path
        near = np.isfinite(s_) & (gap < max(0.5 * size, 5 * r_px))
        k, s_ = k[near], s_[near]
        if len(k) < 5:
            continue
        reach = max(MOVED, 3 * PIXELS * r_px)
        away = np.abs(s_ - np.median(s_[:3])) > reach
        if not away.any():
            continue  # it stays: but a tilt read from a path needs a path
        a0 = int(np.nonzero(~away)[0][-1]) if (~away).any() else 0
        ks, ss = k[a0:], s_[a0:]
        # and it ends where the object gets no further (it has come to the board's foot and lies there): frames
        # of it at rest are no part of a slide down a slope
        went = (ss - ss[0]) * np.sign(ss[-1] - ss[0])
        end = int(np.argmax(went >= went[-1] - reach)) + 1
        ks, ss = ks[:end], ss[:end]
        if len(ks) < 5:
            continue
        f = fit_ramp(t[ks], np.c_[ss, np.zeros(len(ks))])
        a = float(f["p"][4] * np.cos(f["p"][2]))  # fit_ramp's heading is 0 or pi on a line
        sa = float(np.sqrt(max(f["cov"][4, 4], 0.0)))
        mu = float(np.tan(tilt) - a / (G * np.cos(tilt)))
        d_th = (1 - a / G * np.sin(tilt)) / np.cos(tilt) ** 2
        se = {"fit": sa / (G * np.cos(tilt)), "scale": abs(a) * rel_scale / (G * np.cos(tilt)),
              "tilt": abs(d_th) * np.radians(PATH_TILT_STD_DEG)}
        why = ("the model does not explain the track" if not f["rms"] < max(0.006, 0.05 * f["distance"])
               else "a pixel is too much of the path" if PIXELS * r_px > 0.15 * f["distance"]
               else "too short to time" if len(ks) < 6
               else "it does not accelerate down the slope" if a <= 0 else None)
        out["free_phases"].append({
            "first_frame": int(ks[0]), "last_frame": int(ks[-1]), "frames": int(len(ks)), "t0": float(t[ks[0]]),
            "release_speed": float(f["p"][3] * np.cos(f["p"][2])), "acceleration": a, "distance": f["distance"],
            "seconds": float(t[ks[-1]] - t[ks[0]]), "rms": f["rms"], "tilt_deg": float(np.degrees(tilt)),
            "tilt_std_deg": PATH_TILT_STD_DEG,
            "tilt_at_start_deg": float(np.degrees(tilt)), "coefficient": mu,
            "stderr": float(np.sqrt(sum(v ** 2 for v in se.values()))), "stderr_parts": se, "metres_per_pixel": r_px,
            "ok": why is None, "rejected": why, "path_camera": {"point": c.tolist(), "direction": u.tolist()},
            "path_start": float(f["p"][0]),
            "path_seen_at": at.tolist(), "path_span": span, "path_scatter": off})
    if not out["free_phases"] and lines:
        return refuse(lines[0])
    out["stays"] = []
    out["tilt_from"] = "the object's own path (the support is seen along itself)"
    good = [p for p in out["free_phases"] if p["ok"]]
    if not good:
        return refuse("the object never slides free along a line" if not out["free_phases"] else
                      "no free slide measures an acceleration (" + "; ".join(sorted({p["rejected"] for p in
                                                                                   out["free_phases"]})) + ")")
    w = np.array([p["distance"] for p in good])
    cs = np.array([p["coefficient"] for p in good])
    mu = float((cs * w).sum() / w.sum())
    spread = float(np.sqrt((w * (cs - mu) ** 2).sum() / w.sum())) if len(cs) > 1 else 0.0
    se = float(np.hypot(np.sqrt((w ** 2 * np.array([p["stderr"] for p in good]) ** 2).sum()) / w.sum(),
                        spread / np.sqrt(len(cs))))
    out["free_phase"] = max(good, key=lambda p: p["distance"])
    tilt = np.radians(out["free_phase"]["tilt_deg"])
    how = (f"tilt of the object's path against a level surface, the path laid through where MoGe-2 saw the object "
           f"(+- {PATH_TILT_STD_DEG:.0f} deg)")
    out["parameters"]["support.kinetic_friction"] = kinetic(mu, se, len(good), spread, how)
    # a bound must hold: the tilt is known to PATH_TILT_STD_DEG, and the bound is tan of the tilt and twice that
    seen = float(np.tan(tilt))
    ub = float(np.tan(min(tilt + 2 * np.radians(PATH_TILT_STD_DEG), np.radians(89.0))))
    mu_s = float(min(mu * pr["static_ratio"], 0.98 * seen))
    out["parameters"]["support.static_friction"] = {
        "value": mu_s, "upper_bound": ub, "seen": seen, "source": "bounded",
        "how": f"it slid when let go at {np.degrees(tilt):.1f} deg: below tan of that and twice the tilt's "
               "uncertainty; the value is kinetic x the prior static / kinetic ratio, held under tan of the tilt"}
    out["physics"]["support"] = {"static_friction": max(mu_s, mu, 0.0), "dynamic_friction": max(mu, 0.0)}
    return out


def kinetic(mu, se, phases, spread, how):
    """The parameter a slide down a slope gives: tan(tilt) - a / (g cos(tilt)) with its error, and where that is
    no more than its own error (a steep slope: the slide is all but a fall along it, and the tilt's error is
    most of the friction) only what it is under."""
    if mu > se:
        return {"value": mu, "stderr": se, "source": "measured", "phases": phases, "spread": spread,
                "how": "tan(tilt) - acceleration of the free slide / (g cos(tilt)); " + how}
    return {"value": None, "upper_bound": float(max(mu, 0.0) + 2 * se), "seen": mu, "stderr": se, "source": "bounded",
            "phases": phases, "spread": spread,
            "how": f"tan(tilt) - acceleration of the free slide / (g cos(tilt)) is {mu:.3f} +- {se:.3f}, no more "
                   "than its own error: the slide is not told from one without friction, and the friction is "
                   "under that and twice the error; " + how}


def straight_prefix(P, tol):
    """How many of the points P (n, 2), from the first, lie on one straight line within tol (their units)."""
    e = len(P)
    while e > 3:
        c = P[:e].mean(0)
        v = np.linalg.svd(P[:e] - c)[2][0]
        if np.abs((P[:e] - c) @ np.array([-v[1], v[0]])).max() <= tol:
            break
        e -= max(1, e // 20)
    return e


def cumint2(f, t):
    """The double integral of f over t from t[0] (trapezoid)."""
    v = np.r_[0.0, np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(t))]
    return np.r_[0.0, np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))]


def fit_tilting(t, xy, tilt_deg):
    """A slide on a support whose tilt changes: along its line the object obeys s'' = g (sin th - mu cos th), so
    s(t) = s0 + v0 t + A(t) - mu B(t) with A, B the double integrals of g sin th and g cos th: linear in
    (s0, v0, mu). Returns fit_ramp's dict with "mu" and "mu_std" and the mean acceleration as p[4]."""
    t = np.asarray(t, float) - t[0]
    xy = np.asarray(xy, float)
    d = xy[-1] - xy[0]
    u = d / max(np.linalg.norm(d), 1e-9)
    s_ = (xy - xy[0]) @ u
    th = np.radians(np.asarray(tilt_deg, float))
    A, B = cumint2(G * np.sin(th), t), cumint2(G * np.cos(th), t)
    M = np.stack([np.ones_like(t), t, -B], 1)
    x = np.linalg.lstsq(M, s_ - A, rcond=None)[0]
    model = M @ x + A
    res = np.c_[model - s_, (xy - xy[0]) @ np.array([-u[1], u[0]])]
    s2 = float((res ** 2).sum() / max(res.size - 5, 1))
    cov = s2 * np.linalg.pinv(M.T @ M)
    a = float(np.mean(G * (np.sin(th) - x[2] * np.cos(th))))
    return {"p": np.array([xy[0, 0] + x[0] * u[0], xy[0, 1] + x[0] * u[1], np.arctan2(u[1], u[0]), x[1], a]),
            "cov": np.diag([0, 0, 0, cov[1, 1], 0.0]), "rms": float(np.sqrt((res ** 2).sum(1).mean())),
            "distance": float(abs(model[-1] - model[0])), "positions": xy[0] + np.outer(model, u),
            "mu": float(x[2]), "mu_std": float(np.sqrt(max(cov[2, 2], 0.0)))}


def slip_from_rest(t, s_obs, tilt_deg, t_lo, t_hi, steps=60):
    """When an object that rode a tilting support began to slide, and its kinetic friction: it is at rest on the
    support until t_on and then obeys s'' = g (sin th - mu cos th) from rest. For every t_on between t_lo and
    t_hi the one mu that fits the displacements s_obs (N,) at times t best is found (linear), and the t_on with
    the smallest residual kept. A threshold on the displacement sees the slip only once it has gone that far;
    this puts it where the motion says it began. Returns {t_on, mu, rms, model (N,)} or None."""
    t, s_obs, th = np.asarray(t, float), np.asarray(s_obs, float), np.radians(np.asarray(tilt_deg, float))
    best = None
    for t_on in np.linspace(t_lo, t_hi, steps):
        tt = np.linspace(t_on, t[-1], 200)
        thh = np.interp(tt, t, th)
        A, B = cumint2(G * np.sin(thh), tt), cumint2(G * np.cos(thh), tt)
        a, b = np.interp(t, tt, A, left=0.0), np.interp(t, tt, B, left=0.0)
        if not (b ** 2).sum() > 0:
            continue
        mu = float((b * (a - s_obs)).sum() / (b ** 2).sum())
        model = np.maximum(a - mu * b, 0.0)
        cost = float(((model - s_obs) ** 2).sum())
        if best is None or cost < best[0]:
            best = (cost, t_on, mu, model)
    if best is None:
        return None
    return {"t_on": float(best[1]), "mu": best[2], "rms": float(np.sqrt(best[0] / len(t))), "model": best[3]}


def board_frames(per, twin):
    """The frame a long support that tilts carries, on every image: (camera-to-support 4x4 (F, 4, 4), NaN where the
    support is not seen; gravity's direction in that frame (F, 3)). The origin is the middle of the support, x runs
    along its length, y is the level direction across it (the hinge), z its normal towards the camera.

    The support's two ends in the image are two rays; its direction in space is the one track.outline_lean reads
    the tilt from (in the plane through the camera and the mask's long axis, at right angles to the hinge); and
    only one distance puts a support of its length (twin.json support_half_length) between the two rays. So what is
    drawn in this frame lies on the support as the image shows it, wherever MoGe-2 put its plane. Gravity leans in
    it by twin.json's tilt_deg_t, the tilt the closed form uses."""
    from .track import hinge_of

    K = np.asarray(per["K"], float)
    Ki = np.linalg.inv(K)
    up = np.nanmedian(per["level_n_t"], 0)
    up /= np.linalg.norm(up)
    n_t = np.asarray(per["plane_n_t"], float)
    hinge = hinge_of(up, n_t[np.all(np.isfinite(n_t), 1)], per["plane_axis_t"])
    F, half = len(per["support_t"]), float(twin["support_half_length"])
    tilt = np.radians(np.asarray(twin["tilt_deg_t"], float))
    T, g, prev = np.full((F, 4, 4), np.nan), np.full((F, 3), np.nan), None
    if hinge is None:
        return T, g
    for i, s in enumerate(per["support_t"]):
        ys, xs = np.nonzero(s)
        if len(xs) < 50:
            continue
        P = np.stack([xs, ys], 1).astype(float)
        c = P.mean(0)
        a = np.linalg.eigh(np.cov((P - c).T))[1][:, -1]
        if (prev is not None and a @ prev < 0) or (prev is None and a[0] < 0):
            a = -a
        prev = a
        lo, hi = np.percentile((P - c) @ a, [2, 98])
        r_lo, r_hi = Ki @ np.r_[c + lo * a, 1.0], Ki @ np.r_[c + hi * a, 1.0]
        d = np.cross(hinge, np.cross(r_lo, r_hi))
        d /= max(np.linalg.norm(d), 1e-12)
        lam = np.linalg.lstsq(np.stack([-r_lo, r_hi], 1), 2 * half * d, rcond=None)[0]
        if lam.sum() < 0:  # the other way along the line: from the low end of the mask's axis to its high end
            d, lam = -d, -lam
        if not np.all(lam > 0):
            continue
        mid = 0.5 * (lam[0] * r_lo + lam[1] * r_hi)
        y = np.cross(up, d)  # level, and across the support: a hinge a little off must not roll the support
        y /= max(np.linalg.norm(y), 1e-12)
        z = np.cross(d, y)
        R = np.stack([d, y, z])  # rows: the support's axes in camera coordinates
        T[i] = np.eye(4)
        T[i, :3, :3], T[i, :3, 3] = R, -R @ mid
        down = -np.sign(d @ up) if abs(d @ up) > 1e-6 else 0.0  # which way along x is downhill
        g[i] = np.sin(tilt[i]) * down, 0.0, -np.cos(tilt[i])
    return T, g


def board_edges(mask):
    """The two long edges of a long support's mask, along the middle three fifths of its length: two (N, 2) pixel
    arrays (one point per step along its axis on either side), or None."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        return None
    P = np.stack([xs, ys], 1).astype(float)
    c = P.mean(0)
    a = np.linalg.eigh(np.cov((P - c).T))[1][:, -1]
    n = np.array([-a[1], a[0]])
    along, across = (P - c) @ a, (P - c) @ n
    lo, hi = np.percentile(along, [2, 98])
    one, two = [], []
    for s0 in np.linspace(lo + 0.2 * (hi - lo), hi - 0.2 * (hi - lo), 25):
        k = np.abs(along - s0) < 1.0
        if k.sum() >= 3:
            one.append(c + s0 * a + across[k].min() * n), two.append(c + s0 * a + across[k].max() * n)
    return (np.array(one), np.array(two)) if len(one) >= 5 else None


def board_look(frames, support_t, covers, K, T_t, idx, px_per_m=1200.0):
    """What a long support looks like, for drawing it in its twin: {"lo", "hi": the corners (x, y) of its top face
    in board_frames' frame, "thick": how thick it is (m), "texture": (h, w, 3) uint8, the top face's picture with
    its top row at y = hi}. From the frames of `idx` it is seen from highest above its plane. Its length is its
    mask's outline cut with the plane. Across, the mask shows the top face and the side towards the camera: its far
    edge is the top face's, its near edge the underside's, so the width and the thickness are the pair with which
    the frames, each seeing the side at its own angle, agree on the width. The picture is the median of what those
    frames show of the top face, less what `covers` (F, H, W) hides of it."""
    import cv2

    from calibrated_scene_reconstruction.physics.common.observations import project

    K = np.asarray(K, float)
    idx = np.array([i for i in idx if np.isfinite(T_t[i, 0, 0]) and support_t[i].sum() > 50])
    if len(idx) == 0:
        return None
    good = idx[np.argsort(T_t[idx, 2, 3])[-max(3, min(12, len(idx) // 3)):]]
    ends, far, near = [], [], []
    for i in good:
        cs, _ = cv2.findContours(support_t[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        X = ray_plane(K, T_t[i], np.concatenate(cs).reshape(-1, 2).astype(float))[:, 0]
        e = board_edges(support_t[i])
        if e is None or np.isfinite(X).sum() < 8:
            continue
        ends.append(np.nanpercentile(X, [2, 98]))
        y = [np.nanmedian(ray_plane(K, T_t[i], q)[:, 1]) for q in e]
        j = int(np.argmax(np.abs(np.array(y) - T_t[i, 1, 3])))  # the edge further from the camera across the board
        far.append((i, e[j])), near.append((i, e[1 - j]))
    if len(ends) < 2:
        return None
    x0, x1 = np.median(ends, 0)

    def widths(th):  # the top face's two edges across, per frame, with the near edge of the mask on the underside
        yf = np.array([np.nanmedian(ray_plane(K, T_t[i], q)[:, 1]) for i, q in far])
        yn = np.array([np.nanmedian(ray_plane(K, T_t[i], q, z=-th)[:, 1]) for i, q in near])
        return yf, yn

    grid = np.linspace(0.0, 0.04, 21)
    spread = [np.std(np.abs(np.subtract(*widths(th)))) for th in grid]
    thick = float(grid[int(np.argmin(spread))])
    if not 0.002 <= thick <= 0.036:  # the frames see its side at too much the same angle to say: a centimetre
        thick = 0.01
    yf, yn = widths(thick)
    lo, hi = np.array([x0, min(np.median(yf), np.median(yn))]), np.array([x1, max(np.median(yf), np.median(yn))])
    w, h = (max(8, int(round(v * px_per_m))) for v in hi - lo)
    gx, gy = np.meshgrid(lo[0] + (np.arange(w) + 0.5) / w * (hi[0] - lo[0]),
                         hi[1] - (np.arange(h) + 0.5) / h * (hi[1] - lo[1]))
    P = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], 1)
    stack = []
    for i in good:
        uv = project(K, T_t[i], P)[0].astype(np.float32).reshape(h, w, 2)
        pic = cv2.remap(frames[i], uv[..., 0], uv[..., 1], cv2.INTER_LINEAR).astype(np.float32)
        hid = cv2.dilate(covers[i].astype(np.uint8), np.ones((9, 9), np.uint8)) | (~support_t[i]).astype(np.uint8)
        pic[cv2.remap(hid, uv[..., 0], uv[..., 1], cv2.INTER_NEAREST, borderValue=1) > 0] = np.nan
        stack.append(pic)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        tex = np.nanmedian(np.stack(stack), 0)
    gap = ~np.isfinite(tex[..., 0])
    tex = np.nan_to_num(tex, nan=float(np.nanmedian(tex)) if np.isfinite(tex).any() else 128.0).astype(np.uint8)
    if gap.any() and not gap.all():
        tex = cv2.inpaint(tex, gap.astype(np.uint8), 5, cv2.INPAINT_TELEA)
    return {"lo": lo, "hi": hi, "thick": thick, "texture": tex}


def place_on_board(K, T_t, uv, frames, z):
    """Where an object that rides a tilting support stands on it, in board_frames' frame (x, y): the one place
    at height z whose image through every frame's T_t is the object's centroid uv in the frames `frames`. A ray
    cut with a support seen from its edge runs far along it; a place drawn onto every image does not."""
    from scipy.optimize import least_squares

    from calibrated_scene_reconstruction.physics.common.observations import project

    k = np.array([i for i in frames if np.isfinite(T_t[i, 0, 0]) and np.all(np.isfinite(uv[i]))])
    if len(k) < 3:
        return None
    best = k[np.argsort(T_t[k, 2, 3])[-max(3, len(k) // 4):]]  # seen from highest above its plane
    x0 = np.nanmedian(np.stack([ray_plane(K, T_t[i], uv[i], z=z)[:2] for i in best]), 0)
    if not np.all(np.isfinite(x0)):
        x0 = np.zeros(2)
    r = least_squares(lambda v: np.concatenate([project(K, T_t[i], np.r_[v, z][None])[0][0] - uv[i] for i in k]), x0,
                      loss="soft_l1", f_scale=3.0)
    return r.x


def letting_go(obj, manip, first, most=8):
    """The frame a free run may start at when a hand lets the object go rather than pushes it: back from `first`
    (the first frame the masks no longer count as touching) over the frames in which the hand's mask is already
    clear of the object's and falling behind (the gap between them grows from frame to frame)."""
    import cv2

    def gap(k):
        if not obj[k].any() or not manip[k].any():
            return np.inf if obj[k].any() else np.nan
        return float(cv2.distanceTransform((~manip[k]).astype(np.uint8), cv2.DIST_L2, 3)[obj[k]].min())

    k, g1 = first, gap(first)
    while k > 0 and first - k < most:
        g0 = gap(k - 1)
        if not (np.isfinite(g0) and 1.0 < g0 < g1):
            break
        k, g1 = k - 1, g0
    return k


def left_alone(obj, manip, contact, reach=40, moved=1.5):
    """Frames (F,) in which the masks touch and yet nothing acts on the object: it moves (its mask's centroid by
    more than `moved` px since the frame before) while the manipulator's mask around it (within `reach` px of the
    object's box) is the same as the frame before. A hand that stands still beside a board, or an arm the object
    passes in the image, does not push what slides away from it."""
    import cv2

    from calibrated_scene_reconstruction.physics.common.observations import mask_stats

    out = np.zeros(len(obj), bool)
    H, W = obj[0].shape
    for i in np.nonzero(contact)[0]:
        if i == 0 or not obj[i].any() or not obj[i - 1].any():
            continue
        if np.linalg.norm(mask_stats(obj[i])[0] - mask_stats(obj[i - 1])[0]) < moved:
            continue
        both = obj[i] | obj[i - 1]
        ys, xs = np.nonzero(both)
        y0, y1, x0, x1 = max(ys.min() - reach, 0), min(ys.max() + reach, H), max(xs.min() - reach, 0), min(xs.max() + reach, W)
        hid = cv2.dilate(both[y0:y1, x0:x1].astype(np.uint8), np.ones((7, 7), np.uint8)) > 0  # what the object covers
        m0, m1 = manip[i - 1][y0:y1, x0:x1] & ~hid, manip[i][y0:y1, x0:x1] & ~hid
        out[i] = (m0 ^ m1).sum() <= max(12, 0.03 * (m0 | m1).sum())
    return out


def runs(free, least=5, gap=2):
    """[(first, last)] of the runs of True in `free`, holes of up to `gap` frames bridged."""
    free = free.copy()
    idx = np.nonzero(free)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= gap + 1:
            free[a:b] = True
    out, i, F = [], 0, len(free)
    while i < F:
        if free[i]:
            j = i
            while j + 1 < F and free[j + 1]:
                j += 1
            if j - i + 1 >= least:
                out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def run(path, prior=None):
    """physics dict of an incline recording (module doc)."""
    path = Path(path)
    per, track = np.load(path / "perception.npz"), np.load(path / "track.npz")
    twin = read_document(path / "twin.json")
    pr = dict(PRIOR, **(prior or {}))
    out = {"motion": "incline", "free_phase": None, "free_phases": [], "parameters": {
        "object.mass": {"value": pr["mass"], "range": pr["mass_range"],
                        "source": "prior (meta.json)" if prior and "mass" in prior else "default",
                        "how": "not visible in a video of free motion"},
        "object.restitution": {"value": pr["restitution"], "source": "default", "how": "no impact fitted"}},
        "physics": {"object": {"mass": pr["mass"], "restitution": pr["restitution"]}}}
    if unusable(per) is not None and "tilt_deg_t" not in twin:  # the support is no plane in the picture
        return run_along_path(per, track, twin, out, pr)
    if "gravity" not in twin:
        out["note"] = ("no level surface was found beside the support (meta.json 'level'): its tilt is unknown, "
                       "nothing is measured")
        return out
    t, xy = track["t"], track["xy"]
    valid = track["valid"] & ~track["border"] & ~spikes(xy, track["valid"])
    moving = "tilt_deg_t" in twin
    if "on_support" in track.files:  # a support that moves: track.py tested every frame against that frame's outline
        on = track["on_support"] & valid
    else:
        hull = support_hull(per, twin)
        on = np.zeros(len(t), bool)
        on[valid] = inside(hull, xy[valid])
    tilt_t = np.asarray(twin["tilt_deg_t"], float) if moving else np.full(len(t), twin["tilt_deg"])
    rel_scale = float(twin.get("scale", {}).get("rel_std", 0.09))
    # the tilt's uncertainty: a board's outline is within 1.5 deg, two planes of one MoGe-2 point map within 3
    tilt_std = TILT_STD_DEG * (2.0 if twin.get("tilt_from") == "planes" else 1.0)
    h_cam, f_px = float(twin["camera_height"]), float(np.asarray(twin["K"])[0][0])
    stays = []
    # a support that tilts: the object rides it (it keeps its place along the support in the image, track "along")
    # until it slips; the fixed-support loop below then sees only the slide, from the frame the ride ends
    ride_end, slip_at, ride_of = {}, {}, {}
    if moving and "along" in track.files:
        q = track["along"]
        for i0, i1 in runs(np.isfinite(q) & ~track["contact"] & ~track["border"], least=8):
            k = np.arange(i0, i1 + 1)
            k = k[np.isfinite(q[k])]
            third = max(4, len(k) // 3)
            hold = float(np.median(q[k[:third]]))
            reach = max(0.04, 5 * 1.48 * float(np.median(np.abs(q[k[:third]] - hold))))
            away = np.abs(q[k] - hold) > reach
            if away.any() and away[-1]:
                a0 = int(np.nonzero(~away)[0][-1])
                ride_end[int(k[a0])] = (int(i0), int(i1))
                ride_of[int(k[a0])] = int(i0)
                if a0 >= 3:
                    stays.append({"first_frame": int(i0), "last_frame": int(k[a0]),
                                  "seconds": float(t[k[a0]] - t[i0]), "tilt_deg": float(np.nanmedian(tilt_t[k[:a0 + 1]])),
                                  "tilt_deg_max": float(tilt_t[k[a0]]), "rode": True})
            else:
                stays.append({"first_frame": int(i0), "last_frame": int(i1), "seconds": float(t[i1] - t[i0]),
                              "tilt_deg": float(np.nanmedian(tilt_t[k])), "tilt_deg_max": float(np.nanmax(tilt_t[k]))})
    spans = ([(a, b) for a, (_i, b) in ride_end.items()] if moving and "along" in track.files
             else runs(valid & on & (~track["contact"] | left_alone(per["object"], per["manipulator"], track["contact"]))))
    beside = set()
    if not moving:
        # a run the manipulator's mask is beside to its end (the object passes an arm in the image): its last
        # stretch of motion is a candidate, held to more below (it must be a clean slide from rest down the slope)
        for i0, i1 in runs(valid & on & track["contact"], least=8):
            if any(a <= i1 + 3 and i0 <= b for a, b in spans):
                continue
            k = np.arange(i0, i1 + 1)
            k = k[valid[k]]
            step = np.linalg.norm(np.diff(xy[k], axis=0), axis=1)
            j = len(step) - 1
            while j > 0 and (step[j] > 0.002 or step[j - 1] > 0.002):
                j -= 1
            if len(k) - j >= 8:
                spans = spans + [(int(k[max(j - 2, 0)]), int(i1))]
                beside.add(int(k[max(j - 2, 0)]))
    for i0, i1 in spans:
        strict = int(i0) in beside
        if not strict:
            i0 = letting_go(per["object"], per["manipulator"], i0)
        k = np.arange(i0, i1 + 1)
        k = k[valid[k]]
        if len(k) < 5:
            continue
        # where it starts to move: the last frame within reach of where the run began (it may never leave it).
        # The reach is MOVED, or more where the positions of the run's first third scatter more (the support's own
        # frame, read off its mask on every image, is not as still as a table)
        ref = np.median(xy[k[:max(3, len(k) // 3)]], 0) if moving else np.median(xy[k[:3]], 0)
        dist = np.linalg.norm(xy[k] - ref, axis=1)
        reach = max(MOVED, 4 * 1.48 * float(np.median(dist[:max(3, len(k) // 3)]))) if moving else MOVED
        away = dist > reach
        if moving:  # r_px: metres a pixel is on the plane where the object is, seen from the twin's camera
            r_px = float(h_cam / f_px * (1 + (twin["object"]["pose"][0] ** 2 + twin["object"]["pose"][1] ** 2) / h_cam ** 2))
        else:
            r_px = float(np.median((np.linalg.norm(xy[k], axis=1) ** 2 + h_cam ** 2) / (h_cam * f_px)))
        if moving and "along" in track.files:
            k = k[on[k]]  # the slide, where the plane frame holds it (the ride ended at i0)
            if len(k) < 5:
                continue
            away = np.r_[False, np.ones(len(k) - 1, bool)]
        elif not away.any() or np.linalg.norm(xy[k[-1]] - ref) < max(reach, 3 * PIXELS * r_px):
            stays.append({"first_frame": int(i0), "last_frame": int(i1), "seconds": float(t[k[-1]] - t[k[0]]),
                          "tilt_deg": float(np.nanmedian(tilt_t[k])), "tilt_deg_max": float(np.nanmax(tilt_t[k]))})
            continue
        a0 = int(np.nonzero(~away)[0][-1]) if (~away).any() else 0  # from its last frame within reach on
        ks = k[a0:]
        if len(ks) < 5:
            continue

        def fit(ks):
            return fit_tilting(t[ks], xy[ks], tilt_t[ks]) if moving else fit_ramp(t[ks], xy[ks])

        f = fit(ks)
        # the slide may have begun while the hand's mask was still near (it lets go and stays): the frames before,
        # back to where the object stood, belong to it when the same free slide explains them as well
        kb = np.arange(max(ks[0] - 12, 0), ks[0])
        kb = kb[valid[kb] & on[kb]]
        if len(kb) >= 2 and not moving:
            rest = np.linalg.norm(xy[kb] - np.median(xy[kb[:2]], 0), axis=1) < MOVED
            first = int(np.nonzero(rest)[0][-1]) if rest.any() else 0
            ke = np.r_[kb[first:], ks]
            fe = fit(ke)
            if len(ke) > len(ks) and fe["rms"] < max(1.5 * f["rms"], 2 * PIXELS * r_px):
                ks, f = ke, fe
        while len(ks) > 6:  # the last frames, over the support's edge, are a fall and not a slide
            err = np.linalg.norm(f["positions"] - xy[ks], axis=1)
            if err[-1] < max(3 * f["rms"], 2 * PIXELS * r_px):
                break
            ks = ks[:-1]
            f = fit(ks)
        th = np.radians(float(np.nanmedian(tilt_t[ks])))
        th0 = np.radians(float(tilt_t[ks[0]]))
        a, sa = float(f["p"][4]), float(np.sqrt(max(f["cov"][4, 4], 0.0)))
        mu = f["mu"] if moving else float(np.tan(th) - a / (G * np.cos(th)))
        if moving and "along" in track.files and "support_half_length" in twin:
            # from rest: the frames before the slide was noticed belong to the fit, and say when it began. The
            # displacement is the object's place along the support in the image (clean where the plane is not),
            # in metres by the support's length
            q, i_ride = track["along"], ride_of.get(int(i0), int(i0))
            kw = np.arange(max(int(i0) - 14, i_ride), ks[-1] + 1)
            kw = kw[np.isfinite(q[kw])]
            hold = float(np.median(q[kw[:5]]))
            sr = slip_from_rest(t[kw], np.abs(q[kw] - hold) * twin["support_half_length"], tilt_t[kw],
                                t[kw[0]], t[min(int(i0) + 2, len(t) - 1)])
            if sr is not None and sr["rms"] < 0.012 and 0.0 < sr["mu"] < 2.0:
                mu, th0 = sr["mu"], np.radians(float(np.interp(sr["t_on"], t, tilt_t)))
                slip_at[int(i0)] = float(np.degrees(th0))
                f = dict(f, rms=max(sr["rms"], 1e-4))
        d_th = (1 - a / G * np.sin(th)) / np.cos(th) ** 2
        se = {"fit": f["mu_std"] if moving else sa / (G * np.cos(th)),
              "scale": abs(a) * rel_scale / (G * np.cos(th)), "tilt": abs(d_th) * np.radians(tilt_std)}
        g_xy = np.asarray(twin["gravity"], float)[:2]
        off = float(np.degrees(np.arccos(np.clip(np.array([np.cos(f["p"][2]), np.sin(f["p"][2])])
                                                 @ (g_xy / max(np.linalg.norm(g_xy), 1e-9)), -1, 1))))
        seen = float(t[ks[-1]] - t[ks[0]])
        why = ("the model does not explain the track" if not f["rms"] < max(0.006, 0.05 * f["distance"])
               else "grazing view: a pixel is too much of the path" if PIXELS * r_px > 0.15 * f["distance"]
               else "too short to time" if len(ks) < 6
               else "it does not accelerate down the slope" if a <= 0 and not moving
               else "it moves across the slope, not down it" if off > 35 and not moving
               else "the manipulator is beside it and it is not a clean slide from rest down the slope"
               if strict and not (len(ks) >= 8 and off < 20 and abs(f["p"][3]) < 0.15
                                  and f["rms"] < max(0.003, 0.02 * f["distance"]))
               else None)
        out["free_phases"].append({
            "first_frame": int(ks[0]), "last_frame": int(ks[-1]), "frames": int(len(ks)), "t0": float(t[ks[0]]),
            "start_xy": [float(f["p"][0]), float(f["p"][1])], "heading_deg": float(np.degrees(f["p"][2])),
            "release_speed": float(f["p"][3]), "acceleration": a, "distance": f["distance"], "seconds": seen,
            "rms": f["rms"], "tilt_deg": float(np.degrees(th)), "tilt_std_deg": tilt_std,
            "tilt_at_start_deg": float(np.degrees(th0)),
            "heading_off_downhill_deg": off, "coefficient": mu, "stderr": float(np.sqrt(sum(v ** 2 for v in se.values()))),
            "stderr_parts": se, "metres_per_pixel": r_px, "ok": why is None, "rejected": why,
            "manipulator_beside": bool(strict)})
    out["stays"] = stays
    good = [p for p in out["free_phases"] if p["ok"]]
    tan_stay = max((np.tan(np.radians(s["tilt_deg_max"])) for s in stays), default=None)
    how_tilt = f"tilt of the support against a level surface in MoGe-2's points (+- {tilt_std:.1f} deg)"
    if good:
        w = np.array([p["distance"] for p in good])
        c = np.array([p["coefficient"] for p in good])
        mu = float((c * w).sum() / w.sum())
        spread = float(np.sqrt((w * (c - mu) ** 2).sum() / w.sum())) if len(c) > 1 else 0.0
        se = float(np.hypot(np.sqrt((w ** 2 * np.array([p["stderr"] for p in good]) ** 2).sum()) / w.sum(),
                            spread / np.sqrt(len(c))))
        out["free_phase"] = max(good, key=lambda p: p["distance"])
        out["parameters"]["support.kinetic_friction"] = kinetic(mu, se, len(good), spread, how_tilt)
        # it slid when let go on this slope: static friction is below tan(tilt); on a support that tilts it is
        # tan of the tilt at which the slide began
        th0 = min(p["tilt_at_start_deg"] for p in good)
        if moving and ride_end:  # where the motion says the slip began, else where it was noticed
            th0 = float(np.median(list(slip_at.values()) or [tilt_t[a] for a in ride_end]))
        if moving:
            out["parameters"]["support.static_friction"] = {
                "value": float(np.tan(np.radians(th0))), "source": "measured",
                "stderr": float(np.radians(tilt_std) / np.cos(np.radians(th0)) ** 2),
                "how": f"tan of the tilt ({th0:.1f} deg) at which the object began to slide; " + how_tilt}
            mu_s = float(np.tan(np.radians(th0)))
        else:
            # a bound must hold: the tilt is known to tilt_std, and the bound is tan of the tilt and twice that
            seen = float(np.tan(np.radians(th0)))
            ub = float(np.tan(np.radians(min(th0 + 2 * tilt_std, 89.0))))
            mu_s = float(min(mu * pr["static_ratio"], 0.98 * seen)) if tan_stay is None else float(max(tan_stay, mu))
            out["parameters"]["support.static_friction"] = {
                "value": mu_s, "upper_bound": ub, "seen": seen, "source": "bounded",
                "how": f"it slid when let go at {th0:.1f} deg: below tan of that and twice the tilt's uncertainty; "
                       "the value is kinetic x the prior static / kinetic ratio, held under tan of the tilt; "
                       + how_tilt}
        out["physics"]["support"] = {"static_friction": max(mu_s, mu, 0.0), "dynamic_friction": max(mu, 0.0)}
    elif ride_end:  # it rode the support up to an angle and slid from there, but the slide itself was not timed
        # (so when it began is only where it was noticed to have left its place, which is after it began: the
        # tilt then is the most its static friction can be, and no more is said)
        th0 = float(np.median([tilt_t[a] for a in ride_end]))
        out["parameters"]["support.static_friction"] = {
            "value": None, "upper_bound": float(np.tan(np.radians(th0 + 2 * tilt_std))), "source": "bounded",
            "how": f"under tan of the tilt ({th0:.1f} deg, and twice its uncertainty) at which the object was seen to "
                   "have left its place: the slide itself was not timed, so when it began is not known; " + how_tilt}
        out["physics"]["support"] = {"static_friction": float(np.tan(np.radians(th0)))}
        why = "; ".join(sorted({p["rejected"] for p in out["free_phases"]})) or "too few frames of it on the support"
        out["note"] = (f"the object rides the support and has left its place by {th0:.1f} deg: static friction under "
                       f"{np.tan(np.radians(th0 + 2 * tilt_std)):.3f}; the slide itself measures nothing ({why})")
    elif stays:
        # a bound must hold: the tilt is known to tilt_std, and the bound is tan of the tilt less twice that
        seen = float(np.degrees(np.arctan(tan_stay)))
        lb = float(np.tan(np.radians(max(seen - 2 * tilt_std, 0.0))))
        out["parameters"]["support.static_friction"] = {
            "lower_bound": lb, "needed": float(tan_stay), "source": "bounded",
            "how": f"the object stays at rest on a support tilted by {seen:.1f} deg: at least tan of that, less twice "
                   f"the tilt's uncertainty; " + how_tilt}
        out["note"] = (f"the object stays where it was put on a support tilted by {seen:.1f} deg: static friction >= "
                       f"{lb:.3f}, nothing else is measured")
    else:
        out["note"] = ("the object never slides free on the support: nothing is measured" if not out["free_phases"]
                       else "no free slide measures an acceleration (" + "; ".join(sorted({p["rejected"] for p in
                            out["free_phases"]})) + "): nothing is measured")
    return out


def describe(path, phys):
    if phys["free_phase"] is None:
        return f"{path}: {phys['note']}"
    m, fp = phys["parameters"]["support.kinetic_friction"], phys["free_phase"]
    s = phys["parameters"]["support.static_friction"]
    st = (f"static {s['value']:.3f} +- {s['stderr']:.3f}" if s["source"] == "measured"
          else f"static < {s['upper_bound']:.3f}")
    kin = (f"kinetic_friction {m['value']:.3f} +- {m['stderr']:.3f}" if m.get("value") is not None
           else f"kinetic_friction < {m['upper_bound']:.3f}")
    return (f"{path}: {kin} ({st}) [{fp['first_frame']}-"
            f"{fp['last_frame']}: tilt {fp['tilt_deg']:.1f} deg, a {fp['acceleration']:.2f} m/s^2 over "
            f"{fp['distance']:.2f} m, rms {1000 * fp['rms']:.1f} mm]")
