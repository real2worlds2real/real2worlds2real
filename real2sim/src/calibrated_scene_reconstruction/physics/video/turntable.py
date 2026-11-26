"""An object on a level support that turns about an axis (meta.json "motion": "turntable").

    envs/recon/bin/python -m calibrated_scene_reconstruction.physics.video.fit real/<name>     (fit.py hands "turntable" recordings to run() here)

While the object rides the support it moves on a circle about the axis, and what holds it there is static friction:
at the distance r from the axis, with the angular velocity omega and the angular acceleration alpha, it needs

    a = r sqrt(omega^4 + alpha^2)        (towards the axis and along the circle)

so static friction >= a / g for as long as it rides, and = a / g at the moment its distance from the axis begins to
change. Once it slides, whatever the support does under it, friction pulls it with mu_k g: the size of its
acceleration is the kinetic friction, and only its direction depends on the support's motion. The mass does not
enter. Only the object is tracked.

What a ride is taken to prove is what it holds for half a turn: r omega^2 with the angular velocity of half
turns (a circle seen at a slant is an ellipse, and the angle about its middle runs unevenly within a half turn
and exactly over one). A support that turns one way and then the other (a reversal) asks for more than that for
a moment, r alpha, which a video resolves poorly: an object whose circle is wider after a reversal has slipped in
it, and its static friction lies between what it rode and what the reversal asked.

The support's plane and axis come from its outline where it is a disc (track.round_support), and from the
object's own circle where it goes round far enough (plane_of_a_ride): a camera that looks along the support
leaves a depth map's plane a few degrees off, which is a quarter of the path's depth.
"""
from calibrated_scene_reconstruction.physics.common.schema import read_document
import json
from pathlib import Path

import numpy as np

from .fit import PIXELS, PRIOR, spikes
from .incline import runs, support_hull
from .scene import G


def polygon_middle(hull):
    x, y = hull[:, 0], hull[:, 1]
    cr = x * np.roll(y, -1) - np.roll(x, -1) * y
    return np.array([((x + np.roll(x, -1)) * cr).sum(), ((y + np.roll(y, -1)) * cr).sum()]) / (3 * cr.sum())


def circles_about(X, parts):
    """One middle for several circles, least squares: X (N, 2), parts: index arrays, a circle each (an object
    that slipped outwards between two rides goes round the same axis further out). -> (centre (2,), radii,
    rms of the distances from the circles over their radii)."""
    A = np.zeros((len(X), 2 + len(parts)))
    A[:, :2] = 2 * X
    for j, k in enumerate(parts):
        A[k, 2 + j] = 1.0
    use = np.concatenate(parts)
    sol = np.linalg.lstsq(A[use], (X[use] ** 2).sum(1), rcond=None)[0]
    c = sol[:2]
    rad = np.sqrt(np.maximum(sol[2:] + c @ c, 1e-12))
    miss = np.concatenate([(np.linalg.norm(X[k] - c, axis=1) - rad[j]) / rad[j] for j, k in enumerate(parts)])
    return c, rad, miss


def plane_of_a_ride(uv, K, T, hc, least_turn=300.0, parts=None):
    """The support's plane from the object that rides it: while it rides, the object goes round a circle, and
    its image is an ellipse that is a circle's from one direction only (of two: the nearer to the plane T is
    from, MoGe-2's). uv (N, 2): the object's pixels, which must show at least `least_turn` degrees of the turn; hc: the
    height above the plane of the point they show; parts: index arrays into uv of rides that may differ in
    radius (circles_about; default one). Returns (T', centre (2,), radius of the first, rms / radius) or None
    when the path does not go round far enough to say. A camera that looks along the support (a turntable seen
    from the side) leaves the depth map's plane a few degrees off, which here is a quarter of the path's depth."""
    from scipy.optimize import least_squares

    from .scene import camera_pose, ray_plane

    T = np.asarray(T, float)
    parts = [np.arange(len(uv))] if parts is None else parts
    n0, h = T[2, :3] / np.linalg.norm(T[2, :3]), float(T[2, 3])
    X0 = ray_plane(K, T, uv, z=hc)[:, :2]
    if not np.all(np.isfinite(X0)):
        return None
    c0 = circles_about(X0, parts)[0]
    seen = np.zeros(36, bool)  # which tenths of degrees of the turn the path shows (frames may be missing)
    seen[(np.degrees(np.arctan2(X0[:, 1] - c0[1], X0[:, 0] - c0[0])) % 360 // 10).astype(int)] = True
    if 10 * seen.sum() < least_turn:
        return None
    mid = np.linalg.inv(K) @ np.r_[uv.mean(0), 1.0]
    mid /= np.linalg.norm(mid)
    depth = (h - hc) / max(-(n0 @ mid), 1e-6)  # along the middle ray to the path's own plane
    e1 = np.cross(n0, [1.0, 0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n0, e1)

    def pose(q):
        nn = n0 + q[0] * e1 + q[1] * e2
        nn /= np.linalg.norm(nn)
        return camera_pose(nn, float(depth * max(-(nn @ mid), 1e-6)) + hc)

    def res(q):
        X = ray_plane(K, pose(q), uv, z=hc)[:, :2]
        if not np.all(np.isfinite(X)):
            return np.full(sum(len(k) for k in parts), 1.0)
        return circles_about(X, parts)[2]

    best = least_squares(res, [0.0, 0.0], bounds=([-0.35, -0.35], [0.35, 0.35]), loss="soft_l1", f_scale=0.02,
                         x_scale=[0.05, 0.05])
    T1 = pose(best.x)
    c, rad, miss = circles_about(ray_plane(K, T1, uv, z=hc)[:, :2], parts)
    return T1, c, float(rad[0]), float(np.sqrt(np.mean(miss ** 2)))


def turning(t, xy, centre, window=9):
    """(r, omega, alpha) per sample of a path about `centre`: distance (m), angular velocity (rad/s) and angular
    acceleration (rad/s^2), each from a quadratic through `window` samples around it."""
    d = xy - centre
    r, phi = np.linalg.norm(d, axis=1), np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    n, h = len(t), window // 2
    rs, om, al = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        a, b = max(0, i - h), min(n, i + h + 1)
        if b - a < 5:
            continue
        tt = t[a:b] - t[i]
        c = np.polyfit(tt, phi[a:b], 2)
        om[i], al[i] = c[1], 2 * c[0]
        rs[i] = np.polyval(np.polyfit(tt, r[a:b], 1), 0.0)
    return rs, om, al


def circle_through(xy):
    """(middle (2,), radius) of the circle through points xy (n, 2) by least squares, or None for under 8 of them
    or points that are no arc."""
    xy = np.asarray(xy, float)
    xy = xy[np.all(np.isfinite(xy), 1)]
    if len(xy) < 8:
        return None
    sol = np.linalg.lstsq(np.c_[2 * xy, np.ones(len(xy))], (xy ** 2).sum(1), rcond=None)[0]
    r2 = sol[2] + sol[0] ** 2 + sol[1] ** 2
    return (sol[:2], float(np.sqrt(r2))) if r2 > 0 else None


def half_turns(t, phi, turn=np.pi):
    """The angular velocity over every `turn` (half a turn) of an unwrapped angle phi(t): (N,) rad/s, nan where
    less than that is left. Half a turn is what a slanted view of a circle does not distort; a whole one (2 pi)
    is what an axis that is not quite the circle's middle does not either."""
    out = np.full(len(t), np.nan)
    for i in range(len(t)):
        d = np.abs(phi[i:] - phi[i])
        j = np.nonzero(d >= turn)[0]
        if len(j):
            j = i + j[0]
            f = (turn - abs(phi[j - 1] - phi[i])) / max(abs(phi[j] - phi[i]) - abs(phi[j - 1] - phi[i]), 1e-9)
            out[i] = turn / max(t[j - 1] + f * (t[j] - t[j - 1]) - t[i], 1e-9)
    return out


def one_way(t, om, least=10, least_deg=180.0):
    """[(first, last)]: the stretches over which the angular velocity om (N,) keeps its sense and turns at least
    `least_deg` degrees."""
    top = np.nanpercentile(np.abs(om), 90) if np.isfinite(om).any() else 0.0
    sense = np.where(np.isfinite(om) & (np.abs(om) > max(0.2, 0.25 * top)), np.sign(om), 0.0)
    out, i, n = [], 0, len(om)
    while i < n:
        if sense[i] == 0:
            i += 1
            continue
        j = i
        while j + 1 < n and sense[j + 1] == sense[i]:
            j += 1
        if j - i + 1 >= least and np.degrees(abs(float(np.sum(0.5 * (om[i + 1:j + 1] + om[i:j]) * np.diff(t[i:j + 1]))))) >= least_deg:
            out.append((i, j))
        i = j + 1
    return out


def pulled(q, t, x0, v0=None, sub=8):
    """The path (N, 2) of a body that friction of constant size pulls along a direction that turns smoothly:
    at rest at x0 until q[0], then |a| = q[1] along the angle q[2] + q[3] s + q[4] s^2 (s: seconds since q[0]).
    v0: its velocity at t[0] when it is not at rest then (q[0] is then t[0])."""
    ts, A, p0, p1, p2 = q[:5]
    tt = np.linspace(t[0], t[-1], (len(t) - 1) * sub + 1)
    dt = tt[1] - tt[0]
    s = np.clip(tt - ts, 0, None)
    psi = p0 + p1 * s + p2 * s * s
    a = A * np.stack([np.cos(psi), np.sin(psi)], 1) * (tt >= ts)[:, None]
    v = np.cumsum(a, 0) * dt + (0.0 if v0 is None else np.asarray(v0, float))
    x = np.asarray(x0, float) + np.cumsum(v, 0) * dt
    return np.stack([np.interp(t, tt, x[:, 0]), np.interp(t, tt, x[:, 1])], 1)


def fit_pulled(t, X, at_rest=True, sigma=None):
    """The size of the acceleration of a slide X (N, 2) at times t, as `pulled` explains it: (|a| m/s^2, its
    standard error, rms m across and along the view, the fitted path). at_rest: X begins with the body at rest
    (its first frames give x0); else its velocity at t[0] is fitted too. sigma (2,): what a pixel is worth across
    the view and along it, m (a camera that looks along the support sees the second several times worse: the
    camera stands over the origin)."""
    from scipy.optimize import least_squares

    X = np.asarray(X, float)
    x0 = np.median(X[:4], 0) if at_rest else X[0]
    d = X[-1] - x0
    end = np.arctan2(d[1], d[0])
    v_init = (X[min(3, len(X) - 1)] - X[0]) / max(t[min(3, len(X) - 1)] - t[0], 1e-6)
    along = X.mean(0) / max(np.linalg.norm(X.mean(0)), 1e-9)
    axes = np.stack([[-along[1], along[0]], along])  # across the view, along it
    sg = np.array([0.004, 0.004]) if sigma is None else np.asarray(sigma, float)

    def miss(q):
        if at_rest:
            return (pulled(q, t, x0) - X) @ axes.T
        return (pulled(np.r_[t[0], q[:4]], t, x0 + q[6:8], q[4:6]) - X) @ axes.T

    def res(q):
        return (miss(q) / sg).ravel()

    best = None
    for ts in ((t[0], t[len(t) // 4]) if at_rest else (t[0],)):
        for p0 in end + np.array([-1.5, -0.7, 0.0, 0.7, 1.5]):
            for p1 in (-3.0, 0.0, 3.0):
                q0 = [ts, 2.5, p0, p1, 0.0] if at_rest else [2.5, p0, p1, 0.0, v_init[0], v_init[1], 0.0, 0.0]
                r = least_squares(res, q0, loss="soft_l1", f_scale=2.0,
                                  x_scale=[0.05, 1, 0.5, 2, 5] if at_rest else [1, 0.5, 2, 5, 0.1, 0.1, 0.003, 0.003])
                if best is None or r.cost < best.cost:
                    best = r
    k = 1 if at_rest else 0
    m = miss(best.x)
    chi = max(1.0, float(np.mean(res(best.x) ** 2)))
    try:
        cov = np.linalg.inv(best.jac.T @ best.jac) * chi
        se = float(np.sqrt(max(cov[k, k], 0.0)))
    except np.linalg.LinAlgError:
        se = float("nan")
    return abs(float(best.x[k])), se, np.sqrt(np.mean(m ** 2, 0)), X + m @ axes


def support_turn(t, path, axis, sigma=0.02):
    """How the support turns, from the path of an object that rides it: a smoothing spline (callable, rad at
    seconds; .derivative() its angular velocity) through the object's own angle about the axis, over the frames
    it is seen in (path rows that are not NaN). sigma: rad, what one angle is good for. Where the object slips
    (a reversal) its angle is still the support's to within the slip. None under 8 frames."""
    from scipy.interpolate import UnivariateSpline

    path = np.asarray(path, float)
    k = np.nonzero(np.all(np.isfinite(path), 1))[0]
    if len(k) < 8:
        return None
    phi = np.unwrap(np.arctan2(path[k, 1] - axis[1], path[k, 0] - axis[0]))
    return UnivariateSpline(np.asarray(t, float)[k], phi, k=3, s=len(k) * sigma ** 2)


def run(path, prior=None):
    """physics dict of a turntable recording (module doc)."""
    from calibrated_scene_reconstruction.physics.common.observations import mask_stats
    from .scene import ray_plane
    from .track import over_support

    path = Path(path)
    per, track = np.load(path / "perception.npz"), np.load(path / "track.npz")
    twin = read_document(path / "twin.json")
    pr = dict(PRIOR, **(prior or {}))
    out = {"motion": "turntable", "free_phase": None, "rides": [], "reversals": [], "parameters": {
        "object.mass": {"value": pr["mass"], "range": pr["mass_range"],
                        "source": "prior (meta.json)" if prior and "mass" in prior else "default",
                        "how": "not visible in a video of free motion"}},
        "physics": {"object": {"mass": pr["mass"], "restitution": pr["restitution"]}}}
    circle = twin.get("support_circle")  # track.round_support: the rim's circle, in the plane it makes a circle in
    if circle is not None:
        centre = np.array(circle["centre"], float)
    else:
        hull = support_hull(per, twin)
        if hull is None:
            out["note"] = "the support's outline is not seen: its axis is unknown, nothing is measured"
            return out
        centre = polygon_middle(hull)
    t, xy = track["t"], track["xy"].copy()
    valid = track["valid"] & ~track["border"] & ~spikes(xy, track["valid"])
    K = np.asarray(twin["K"], float)
    hc = float(twin["object"].get("centroid_height", twin["object"]["dims"]["height"] / 2))
    uv = np.array([mask_stats(m)[0] for m in per["object"]])
    free = valid & ~track["contact"] & np.all(np.isfinite(uv), 1) & (per["score"] > 0.05)  # not a filled-in frame
    on = over_support(per["object"], per["support"])
    if circle is not None:  # its middle over the disc
        on &= np.linalg.norm(track["xy"] - np.array(circle["centre"], float), axis=1) < 1.1 * circle["radius"]
    ride = None  # the plane from the ride itself, when the object goes round far enough (plane_of_a_ride)
    tries = [np.nonzero(free & on)[0]]  # every frame it is seen on the support, or one run of them, or its start
    for i0, i1 in runs(free, least=10):
        k = np.arange(i0, i1 + 1)
        k = k[free[k]]
        tries += [k[:end] for end in (len(k), 2 * len(k) // 3, len(k) // 2, len(k) // 3) if end >= 10]
    for k in tries:
        got = plane_of_a_ride(uv[k], K, twin["T_world_cam"], hc) if len(k) >= 10 else None
        if got is not None and got[3] < 0.04:
            # again with a circle for every stretch it turns one way: it may ride further out after a reversal
            X = ray_plane(K, got[0], uv[k], z=hc)[:, :2]
            parts = [np.arange(a, b + 1) for a, b in one_way(t[k], turning(t[k], X, got[1])[1])]
            more = plane_of_a_ride(uv[k], K, twin["T_world_cam"], hc, parts=parts) if len(parts) > 1 else None
            ride = more if more is not None and more[3] < got[3] else got
            break
    out["support_plane"] = {"from": "MoGe-2" if circle is None else "the support's outline (a disc)"}
    if ride is not None:
        T1, centre = ride[0], ride[1]
        xy[free] = ray_plane(K, T1, uv[free], z=hc)[:, :2]
        out["support_plane"] = {"from": "the object's ride (a circle)", "radius": float(ride[2]), "rms_rel": float(ride[3]),
                                "T_world_cam": np.asarray(T1, float).tolist(),
                                "turned_deg": float(np.degrees(np.arccos(np.clip(
                                    T1[2, :3] @ np.asarray(twin["T_world_cam"], float)[2, :3], -1, 1))))}
        twin = dict(twin, camera_height=float(T1[2, 3]))
    rel_scale = float(twin.get("scale", {}).get("rel_std", 0.09))
    h_cam, f_px = float(twin["camera_height"]), float(np.asarray(twin["K"])[0][0])
    if ride is None and circle is None and h_cam < 0.35 * float(np.linalg.norm(centre)):
        out["note"] = ("the camera looks along the support (it stands %.2f m above it, %.2f m away), the support's "
                       "outline is not a disc's and the object does not go round far enough to fix the plane from "
                       "its ride: nothing is measured" % (h_cam, np.linalg.norm(centre)))
        return out
    rode, slipped, slides = [], [], []
    for i0, i1 in runs(free & on, least=8, gap=5):
        k = np.arange(i0, i1 + 1)
        k = k[free[k]]
        if len(k) < 8:
            continue
        raw = np.linalg.norm(xy[k] - centre, axis=1)
        r_px = float(np.median((np.linalg.norm(xy[k], axis=1) ** 2 + h_cam ** 2) / (h_cam * f_px)))  # m per pixel
        r, om, al = turning(t[k], xy[k], centre)
        phi = np.unwrap(np.arctan2(xy[k, 1] - centre[1], xy[k, 0] - centre[0]))
        ways = one_way(t[k], om)
        level = None  # (frames within k, radius) the object last kept its distance over
        for a, b in ways:
            whole = abs(phi[b] - phi[a]) >= 2.2 * np.pi



            own = None if whole else circle_through(xy[k[a:b + 1]])
            if own is not None and np.linalg.norm(own[0] - centre) < 0.3 * own[1]:
                ph = np.unwrap(np.arctan2(xy[k[a:b + 1], 1] - own[0][1], xy[k[a:b + 1], 0] - own[0][0]))
                rw = np.linalg.norm(xy[k[a:b + 1]] - own[0], axis=1)
            else:
                ph, rw = phi[a:b + 1], raw[a:b + 1]
            w = half_turns(t[k[a:b + 1]], ph, 2 * np.pi if whole else np.pi)
            if not np.isfinite(w).any():
                continue
            rr = float(np.median(rw))
            sd = 1.48 * float(np.median(np.abs(rw - rr)))
            if sd > 0.1 * rr:  # not a circle about this axis: the plane or the axis is off
                continue



            w_ = float(np.nanmedian(w))
            one = {"first_frame": int(k[a]), "last_frame": int(k[b]), "seconds": float(t[k[b]] - t[k[a]]), "radius": rr,
                   "radius_sd": sd, "omega": w_ * float(np.sign(np.nanmedian(om[a:b + 1]))),
                   "omega_top": float(np.nanmax(w)), "needed": rr * w_ ** 2 / G, "metres_per_pixel": r_px}
            if (out["rides"] and 0 < t[one["first_frame"]] - t[out["rides"][-1]["last_frame"]] < 1.0
                    and out["rides"][-1]["omega"] * one["omega"] < 0):
                was = out["rides"][-1]  # a reversal: is the circle wider after it, and what it asked for as far as
                e0 = int(np.searchsorted(k, was["last_frame"]))  # the frame rate resolves it (5 frames)
                al5 = turning(t[k[max(e0 - 4, 0):a + 5]], xy[k[max(e0 - 4, 0):a + 5]], centre, window=5)[2]
                step = rr - was["radius"]
                tol = max(0.003, 3 * float(np.hypot(sd, was["radius_sd"])) / np.sqrt(min(b - a + 1, 20)))
                out["reversals"].append({"frame": int(k[a]), "radius_step": float(step), "slips": bool(abs(step) > tol),
                                         "needed_about": float(0.5 * (was["radius"] + rr) * np.nanmax(np.abs(al5)) / G)
                                         if np.isfinite(al5).any() else None})
            out["rides"].append(one)
            level = (np.arange(a, b + 1), rr)
        far = float(np.median(np.linalg.norm(xy[k], axis=1)))
        view = np.median(xy[k], 0) / max(far, 1e-9)
        axes = np.stack([[-view[1], view[0]], view])  # across the view and along it, and what a pixel is worth there
        sigma = PIXELS * np.array([np.hypot(far, h_cam) / f_px, (far ** 2 + h_cam ** 2) / (h_cam * f_px)])
        if level is None and not ways:  # it never went round: where it rested at the start
            still = np.linalg.norm((xy[k] - np.median(xy[k[:6]], 0)) @ axes.T / sigma, axis=1) < 3.0
            n0 = int(np.argmin(still)) if not still.all() else len(k)
            if n0 >= 6:
                level = (np.arange(n0), float(np.median(raw[:n0])))
        if level is None:
            continue
        idx, r0 = level
        reach = max(0.006, 3 * PIXELS * r_px, 5 * 1.48 * float(np.median(np.abs(raw[idx] - r0))))
        if ways:  # it leaves its circle
            away = np.abs(raw - r0) > reach
        else:  # it leaves where it rested
            away = np.linalg.norm((xy[k] - np.median(xy[k[idx]], 0)) @ axes.T / sigma, axis=1) > 3.0
        away[:idx[0]] = False
        if not (away.any() and away[-1]):
            continue
        a0 = int(np.nonzero(~away)[0][-1])  # the last frame at its distance: from here it slides
        if ways and a0 >= idx[0] + 5 and np.isfinite(om[a0]):  # out of a ride: static friction is what it needed then
            w = half_turns(t[k[idx[0]:a0 + 1]], phi[idx[0]:a0 + 1])
            w_now = float(np.nanmax(np.abs(w[-max(1, len(w) // 4):]))) if np.isfinite(w).any() else abs(float(om[a0]))
            w_now = max(w_now, abs(float(np.nanmedian(om[max(a0 - 6, idx[0]):a0 + 1]))))
            trend = np.polyfit(t[k[max(a0 - 14, idx[0]):a0 + 1]], phi[max(a0 - 14, idx[0]):a0 + 1], 2)[0] * 2
            slipped.append({"frame": int(k[a0]), "radius": r0, "omega": w_now, "alpha": float(trend),
                            "needed": r0 * float(np.sqrt(w_now ** 4 + trend ** 2)) / G})
        s = np.arange(max(a0 - 8, 0), len(k)) if not ways else np.arange(a0, len(k))
        moved = float(np.linalg.norm(xy[k[s[-1]]] - xy[k[a0]]))
        # a slide leaves the circle: a body that only rides an arc about the axis (a support that starts gently,
        # a video cut before half a turn) keeps its distance from it, and what holds it there is no friction
        # measured. A slider cuts across: three frames or more stand off the circle by three times what a pixel
        # is worth there
        rad = (xy[k[s]] - centre) / np.maximum(raw[s], 1e-9)[:, None]
        off = np.linalg.norm(((raw[s] - r0)[:, None] * rad) @ axes.T / sigma, axis=1)
        if len(s) >= 8 and moved > 0.03 and ((off > 3.0).sum() >= 3 if not ways
                                             else abs(raw[-1] - r0) > max(0.03, 0.3 * r0)):
            A, se, rms, fitted = fit_pulled(t[k[s]], xy[k[s]], at_rest=not ways, sigma=sigma)
            # a slide that a constant pull does not explain to a twelfth of its length, or whose acceleration the
            # fit knows to no better than a quarter, measures none: too few frames of it, or not a free slide
            if float(np.hypot(*rms)) > 0.08 * moved or se > 0.25 * A:
                out.setdefault("slides_rejected", []).append(
                    {"first_frame": int(k[s[0]]), "last_frame": int(k[s[-1]]), "frames": int(len(s)), "distance": moved,
                     "acceleration": A, "acceleration_se": se, "rms": float(np.hypot(*rms)),
                     "why": "a constant friction does not explain the path, or too few frames show it"})
                continue
            slides.append({"first_frame": int(k[s[0]]), "last_frame": int(k[s[-1]]), "frames": int(len(s)),
                           "seconds": float(t[k[s[-1]]] - t[k[s[0]]]), "distance": moved, "acceleration": A,
                           "acceleration_se": se, "rms": float(np.hypot(*rms)), "rms_across": float(rms[0]),
                           "rms_along": float(rms[1]), "friction": A / G, "from_rest": not ways,
                           "fitted": fitted.tolist(), "frames_used": k[s].tolist()})
    out["slides"] = slides
    out["axis"] = [float(centre[0]), float(centre[1])]
    out["path"] = [[float(a), float(b)] if free[i] and on[i] and np.isfinite(a + b) else None for i, (a, b) in enumerate(xy)]
    if not out["rides"] and not slides:
        out["note"] = ("the object is seen to slide on the support, but " + out["slides_rejected"][0]["why"]
                       + ": nothing is measured" if out.get("slides_rejected") else
                       "the object is never seen riding the support or sliding on it on its own: nothing is measured")
        return out
    how = ("what holds the object on its circle about the support's axis, r omega^2 / g over half a turn, from its "
           "own path; scales with the scene")
    rode = [r_["needed"] for r_ in out["rides"]]
    gave = [v for v in out["reversals"] if v["slips"]]
    notes = []
    if slipped:
        v = np.array([s_["needed"] for s_ in slipped])
        mu = float(np.median(v))
        out["parameters"]["support.static_friction"] = {
            "value": mu, "stderr": float(np.hypot(mu * rel_scale, np.std(v) / np.sqrt(len(v)) if len(v) > 1 else 0.15 * mu)),
            "source": "measured", "rides": len(slipped),
            "how": "r sqrt(omega^4 + alpha^2) / g when the object's distance from the axis begins to change"}
        out["physics"]["support"] = {"static_friction": mu}
        out["slip"] = max(slipped, key=lambda s_: s_["needed"])
    elif rode:
        # a bound must hold: it is in proportion to the scene's scale, and is lowered by twice what is not known
        # of that (`needed` is what the twin, at its own scale, has to give the object)
        need = max(rode)
        lo = need * max(1 - 2 * rel_scale, 0.0)
        out["parameters"]["support.static_friction"] = {
            "lower_bound": lo, "needed": need, "source": "bounded",
            "how": how + f"; it rides the support; lowered by twice the scale's uncertainty ({100 * rel_scale:.0f} %)"}
        notes.append(f"the object rides the support (up to {max(abs(r_['omega']) for r_ in out['rides']):.2f} rad/s "
                     f"at {out['rides'][0]['radius']:.3f} m from the axis): static friction >= {lo:.3f}")
        if gave:  # not an upper bound: the reversal is over in a frame or two
            about = [v["needed_about"] for v in gave if v["needed_about"]]
            out["parameters"]["support.static_friction"]["slips_in_reversals"] = len(gave)
            notes.append(f"its circle is {1000 * float(np.mean([v['radius_step'] for v in gave])):.0f} mm wider after "
                         f"{len(gave)} of {len(out['reversals'])} reversals of the support (they ask for more, as far "
                         f"as the frame rate shows about {max(about):.2f}: not a bound)" if about else "it slips in reversals")
    if slides:
        v = np.array([s_["friction"] for s_ in slides])
        wgt = np.array([s_["frames"] for s_ in slides], float)
        mu = float((v * wgt).sum() / wgt.sum())
        fit_se = float(np.sqrt(np.sum((wgt * np.array([s_["acceleration_se"] / G for s_ in slides])) ** 2)) / wgt.sum())
        out["parameters"]["support.friction"] = {
            "value": mu, "stderr": float(np.sqrt((mu * rel_scale) ** 2 + fit_se ** 2 + (0.1 * mu) ** 2)),
            "source": "measured", "slides": len(slides),
            "how": "the size of the object's acceleration while it slides on the support, over g (friction pulls a "
                   "sliding body with mu g whatever the support does under it); scales with the scene"}
        out["physics"].setdefault("support", {})["friction"] = mu
        out["free_phase"] = {k_: v_ for k_, v_ in max(slides, key=lambda s_: s_["frames"]).items()
                             if k_ not in ("fitted", "frames_used")}
    elif slipped:
        out["free_phase"] = out["slip"]
    if notes:
        out["note"] = "; ".join(notes)
    return out


def describe(path, phys):
    pm, bits = phys["parameters"], []
    st, kf = pm.get("support.static_friction"), pm.get("support.friction")
    if kf:
        fp = phys["free_phase"]
        bits.append(f"kinetic_friction {kf['value']:.3f} +- {kf['stderr']:.3f} (it slides {fp['distance']:.2f} m in "
                    f"{fp['seconds']:.2f} s at {fp['acceleration']:.2f} m/s^2, rms {fp['rms'] * 1000:.1f} mm)")
    if st and st["source"] == "measured":
        sl = phys["slip"]
        bits.append(f"static_friction {st['value']:.3f} +- {st['stderr']:.3f} (it slips at {sl['radius']:.3f} m from "
                    f"the axis: omega {sl['omega']:.2f} rad/s, alpha {sl['alpha']:.1f} rad/s^2)")
    elif st:
        bits.append(phys["note"])
    if not bits:
        bits.append(phys.get("note", "nothing is measured"))
    return f"{path}: " + "; ".join(bits) + f" [plane: {phys.get('support_plane', {}).get('from', '-')}]"
