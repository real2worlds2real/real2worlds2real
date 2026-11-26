"""An object in the air: a dropped or tossed thing that bounces (meta.json "motion": "drop").

A flight is the one motion in a video whose acceleration is known: g. That makes it a ruler and a clock at once.
Between two bounces the object leaves the support and returns to it, so a flight of duration T rises g T^2 / 8,
in metres, whatever the camera: the scene's scale follows from how high that looks. And the ratio of two successive
flight times is the restitution of the pair object / support, with no scale and no camera in it at all.

The model, in the twin's frame (the support is z = 0) at the twin's present scale, with gravity g' free:

    nodes    k = 1..n   the bounces: time tau_k and place (x_k, y_k), the object's centre at height r there
    arc k               from node k to k + 1: straight in xy, z = r + g' (t - tau_k)(tau_{k+1} - t) / 2
    the fall            before node 1, back to the release: node 1 + V (t - tau_1) - g' z (t - tau_1)^2 / 2

fitted to the mask's centroid in every frame (pixels, robust). g' is what gravity looks like if the twin's scale
were right: the scale is off by the factor G / g'. r is the radius of the disc of the mask's area where the object
touches down.

    python -m calibrated_scene_reconstruction.physics.video.fit real/<name>     (fit.py hands "drop" recordings to run() here)

Writes physics.json {"motion": "drop", "free_phase": {first_frame, last_frame, radius, start {t, position,
velocity}, nodes, rms_px, gravity_apparent}, "bounces": [{t, flight_before, flight_after, speed_in, restitution}],
"parameters": {"object.restitution", "scene.scale", priors}, "physics"} and, once per recording, rescales twin.json
and track.npz by the gravity's scale fused with what was known (twin.json scale.gravity records it).

A flight whose touchdown the video does not show (it begins after one bounce and ends before the next) still has
gravity's curvature in it, but nothing stands it on the support: it is placed where MoGe-2 sees the object in the
frames it looked at (perception object_xyz) and fitted as one arc (fit_arc). That gives the scale, and no
restitution.

Gravity's scale and the scene's own (MoGe-2, the object's size) should agree. Where they are far apart the clock is
the likelier culprit (footage slowed k times shows gravity k^2 weaker), nothing is rescaled, and physics.json says
which slow_motion would reconcile them: whether the footage is slowed is for whoever knows the video to say.

Not modelled: air drag (a ping-pong ball at 3 m/s loses a few percent of g to it), spin, and a restitution that
depends on the speed of the impact; the bounces are listed one by one for that reason.
"""
from calibrated_scene_reconstruction.physics.common.schema import read_document
import json
from pathlib import Path

import numpy as np

from calibrated_scene_reconstruction.physics.common.observations import mask_stats, project
from .scene import G, ray_plane

ALONG_DEG = 30.0  # a camera that looks along gravity within this sees no arc of a fall
GRAVITY_MODEL_STD = 0.04  # relative: what drag, the centroid of a blurred mask and frame timing leave in g'


def down_coordinate(K, T, uv):
    """The pixel track's coordinate along the image of gravity (px, growing downwards in the scene)."""
    m = np.nanmedian(uv, 0)
    X = ray_plane(K, T, m, z=0.0)
    if not np.all(np.isfinite(X)):
        return uv[:, 1].copy()
    d = project(K, T, X)[0] - project(K, T, X + [0, 0, 0.05])[0]
    return uv @ (d / max(np.linalg.norm(d), 1e-9))


def bounces(q, free, min_px=3.0, min_gap=3):
    """Frames at which the object bounces: in free flight the down coordinate q is convex (it accelerates
    downwards), so every local maximum of q that stands out by min_px is a touch of the support. min_px is a
    tenth of the object's size in the picture as run() calls it: under that a hop is the mask's own flicker."""
    from scipy.signal import find_peaks

    out = []
    i, F = 0, len(q)
    while i < F:
        if not free[i]:
            i += 1
            continue
        j = i
        while j + 1 < F and free[j + 1]:
            j += 1
        if j - i >= 2 * min_gap:
            pk, _ = find_peaks(q[i:j + 1], prominence=min_px, distance=min_gap)
            out.append((i, j, [int(i + p) for p in pk]))
        i = j + 1
    return out


def held(uv, i, j, fast=6.0, slow=2.0, least=3):
    """The last frame of the hold a free run begins with: an object that hangs in a gripper no prompt found is not
    falling yet, and one that creeps out of it is not either. The hold ends before the first two steps in a row
    faster than `fast` px a frame, if the object moved slower than `slow` px a frame on most frames until then.
    i when the run starts in motion."""
    k = np.arange(i, j + 1)
    k = k[np.all(np.isfinite(uv[k]), 1)]
    if len(k) < least + 3:
        return i
    step = np.linalg.norm(np.diff(uv[k], axis=0), axis=1) / np.diff(k)
    go = np.nonzero((step[:-1] > fast) & (step[1:] > fast))[0]
    if len(go) == 0 or go[0] < least or np.median(step[:go[0]]) > slow:
        return i
    return int(k[go[0]])


def positions(p, t, n, r, fall):
    """Centre positions (N, 3) at times t and which frames the model covers (N,). p = [g', (tau, x, y) per node,
    then (vx, vy, vz) of the fall into node 1 when `fall`]."""
    g = p[0]
    nodes = p[1:1 + 3 * n].reshape(n, 3)
    P, used = np.full((len(t), 3), np.nan), np.zeros(len(t), bool)
    if fall:
        V = p[1 + 3 * n:4 + 3 * n]
        m = t < nodes[0, 0]
        dt = t[m] - nodes[0, 0]
        P[m] = np.c_[nodes[0, 1] + V[0] * dt, nodes[0, 2] + V[1] * dt, r + V[2] * dt - 0.5 * g * dt ** 2]
        used |= m
    for k in range(n - 1):
        t0, t1 = nodes[k, 0], nodes[k + 1, 0]
        m = (t >= t0) & (t <= t1)
        s = (t[m] - t0) / max(t1 - t0, 1e-6)
        P[m] = np.c_[nodes[k, 1] + s * (nodes[k + 1, 1] - nodes[k, 1]), nodes[k, 2] + s * (nodes[k + 1, 2] - nodes[k, 2]),
                     r + 0.5 * g * (t[m] - t0) * (t1 - t[m])]
        used |= m
    return P, used


def fill_gaps(uv, max_gap=6):
    """uv (F, 2) with runs of up to max_gap unseen frames filled by interpolation (a ball is a blur where it
    bounces and the segmenter loses it for a few frames), and which frames are then covered (F,)."""
    seen = np.all(np.isfinite(uv), 1)
    out, cov = uv.copy(), seen.copy()
    idx = np.nonzero(seen)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap + 1:
            w = (np.arange(a + 1, b) - a) / (b - a)
            out[a + 1:b] = uv[a] + w[:, None] * (uv[b] - uv[a])
            cov[a + 1:b] = True
    return out, cov


def fit_flight(t, uv, K, T, r, peaks, first, fall=True, uv_fill=None, slack=None, rest=False):
    """The model through the pixel track uv (N, 2) of one free run (frames first.. of the video; peaks are bounce
    frames in the same numbering). Returns a dict or None when there is no flight to fit."""
    from scipy.optimize import least_squares

    n = len(peaks)
    idx = np.array(peaks) - first
    ok = np.all(np.isfinite(uv), 1)
    fall = bool(fall and idx[0] >= 4)
    if n < 2 and not fall:
        return None
    nodes0 = []
    uvf = uv if uv_fill is None else uv_fill
    for i in idx:
        X = ray_plane(K, T, uvf[i], z=r)
        if not np.all(np.isfinite(X)):
            return None
        nodes0 += [t[i], X[0], X[1]]
    p0 = [G] + nodes0
    if fall:
        # a start for the fall: the release seen as if it happened at the support (the fit lifts it)
        X0 = ray_plane(K, T, uv[np.nonzero(ok)[0][0]], z=r)
        dt = t[idx[0]] - t[np.nonzero(ok)[0][0]]
        vxy = (np.array(nodes0[1:3]) - X0[:2]) / max(dt, 1e-3) if np.all(np.isfinite(X0)) else np.zeros(2)
        p0 += [0.3 * vxy[0], 0.3 * vxy[1], -G * dt]
    p0 = np.array(p0, float)
    dtf = float(np.median(np.diff(t)))
    lo, hi = np.full(len(p0), -np.inf), np.full(len(p0), np.inf)
    lo[0], hi[0] = 0.02, 80.0  # footage slowed twenty times, or sped up three, still has its own g'
    for k in range(n):  # a bounce stays within a frame and a half of where the track turns (more where it was unseen)
        w = (1.5 + (0 if slack is None else slack[k])) * dtf
        lo[1 + 3 * k], hi[1 + 3 * k] = p0[1 + 3 * k] - w, p0[1 + 3 * k] + w
    if fall:
        hi[-1] = 0.0

    # one camera sees a fall's speed along its line of sight only as the ball growing: with gravity free that
    # trades against the scale. So the fall is held to a small speed along the ray (0.3 m/s; a ball thrown at the
    # camera is not this model's), and one let go from rest (`rest`: it hung still before) to none either way
    # (0.1 m/s), each weighted like a pixel on every frame of the fall
    u = np.array(nodes0[1:3], float) / max(np.linalg.norm(nodes0[1:3]), 1e-6)
    w_fall = np.sqrt(max(int((ok & (t < t[idx[0]])).sum()), 1)) if fall else 0.0

    def res_px(p):
        P, used = positions(p, t, n, r, fall)
        m = used & ok
        e = np.zeros((len(t), 2))
        e[m] = np.nan_to_num(project(K, T, P[m])[0] - uv[m], nan=50.0)
        return e.ravel()

    def res(p):
        e = res_px(p)
        if not fall:
            return e
        V = p[-3:-1]
        return np.r_[e, w_fall * (V @ u) / (0.1 if rest else 0.3),
                     w_fall * (V @ np.array([-u[1], u[0]])) / 0.1 if rest else 0.0]

    best = None
    for g0 in (G, 0.5 * G, 2 * G, 0.1 * G):  # the last: footage slowed three times shows a tenth of gravity
        p = p0.copy()
        p[0] = g0
        if fall:
            p[-1] = -g0 * dt
        s = least_squares(res, np.clip(p, lo + 1e-9, hi - 1e-9), bounds=(lo, hi), loss="soft_l1", f_scale=2.0,
                          x_scale=np.r_[1.0, np.tile([0.02, 0.05, 0.05], n), [0.3, 0.3, 0.3] if fall else []])
        if best is None or s.cost < best.cost:
            best = s
    p = best.x
    P, used = positions(p, t, n, r, fall)
    m = used & ok
    e = res_px(p).reshape(-1, 2)[m]
    dof = max(e.size - len(p), 1)
    cov = float((e ** 2).sum() / dof) * np.linalg.pinv(best.jac.T @ best.jac)
    # the typical residual: single frames go wrong where the ball is a streak (half of it segmented, at a bounce)
    typical = float(1.2 * np.sqrt(np.median((e ** 2).sum(1))))
    return {"p": p, "n": n, "fall": fall, "used": m, "P": P, "rms_px": float(np.sqrt((e ** 2).sum(1).mean())),
            "typical_px": typical,
            "g": float(p[0]), "g_std": float(np.sqrt(max(cov[0, 0], 0.0))), "frames": int(m.sum())}


def fit_arc(t, uv, K, T, seen_at, seen_xyz):
    """One flight with no touchdown in view: centre P(t) = P0 + V (t - t0) - g' z (t - t0)^2 / 2 through the pixel
    track uv (N, 2), held to where MoGe-2 saw the object (world points seen_xyz (M, 3) at indices seen_at into t,
    each to 3 % of its distance from the camera) and, as a fall, to a small speed along the line of sight. Returns
    a dict like fit_flight's, or None for a track that shows no curvature (gravity bends it by under eight times
    its own residual)."""
    from scipy.optimize import least_squares

    ok = np.all(np.isfinite(uv), 1)
    seen_xyz = np.asarray(seen_xyz, float).reshape(-1, 3)
    if ok.sum() < 8 or len(seen_at) == 0:
        return None
    t0 = float(t[np.nonzero(ok)[0][0]])
    tt = t - t0
    cam = T[:3, 3]
    dist = np.linalg.norm(seen_xyz - cam, axis=1)
    ray = (seen_xyz.mean(0) - cam) / np.linalg.norm(seen_xyz.mean(0) - cam)

    def place(p):
        return p[1:4] + np.outer(tt, p[4:7]) - 0.5 * p[0] * np.outer(tt ** 2, [0.0, 0.0, 1.0])

    def res(p):
        P = place(p)
        e = np.zeros((len(t), 2))
        e[ok] = np.nan_to_num(project(K, T, P[ok])[0] - uv[ok], nan=50.0)
        w = np.sqrt(ok.sum() / max(len(seen_at), 1))
        return np.r_[e.ravel(), (w * (P[seen_at] - seen_xyz) / (0.03 * dist[:, None])).ravel(),
                     np.sqrt(ok.sum()) * (p[4:7] @ ray) / 0.3]

    best = None
    for g0 in (G, 0.3 * G, 0.1 * G, 3 * G):
        p0 = np.r_[g0, seen_xyz[0], np.zeros(3)]  # at rest where it was first placed: the fit moves it
        lo, hi = np.full(7, -np.inf), np.full(7, np.inf)
        lo[0], hi[0] = 0.02, 80.0
        r_ = least_squares(res, p0, bounds=(lo, hi), loss="soft_l1", f_scale=2.0,
                           x_scale=[1.0, 0.05, 0.05, 0.05, 0.3, 0.3, 0.3])
        if best is None or r_.cost < best.cost:
            best = r_
    p = best.x
    P = place(p)
    e = project(K, T, P[ok])[0] - uv[ok]
    n_px = 2 * int(ok.sum())
    cov = float((best.fun[:n_px] ** 2).sum() / max(n_px - 7, 1)) * np.linalg.pinv(best.jac.T @ best.jac)
    # the curvature the track shows, in pixels: gravity over the flight against the residual
    bend_px = float(np.linalg.norm(project(K, T, P[ok][[0, -1]].mean(0)[None])[0][0]
                                   - project(K, T, P[ok][len(P[ok]) // 2][None])[0][0]))
    typical = float(1.2 * np.sqrt(np.median((e ** 2).sum(1))))
    if bend_px < 8 * max(typical, 0.3):
        return None
    return {"p": p, "n": 0, "fall": True, "arc": True, "used": ok, "P": P,
            "rms_px": float(np.sqrt((e ** 2).sum(1).mean())), "typical_px": typical, "g": float(p[0]),
            "g_std": float(np.sqrt(max(cov[0, 0], 0.0))), "frames": int(ok.sum()), "t0": t0, "bend_px": bend_px}


def restitutions(f, dt):
    """[{t, flight_before, flight_after, speed_in, restitution, stderr}] per bounce of a fit_flight result that has
    a flight on both sides of it. speed_in is in true metres per second: a flight time and g give it without scale."""
    p, n = f["p"], f["n"]
    tau = p[1:1 + 3 * n:3]
    T = np.diff(tau)
    out = []
    if f["fall"] and n >= 2:
        vin = abs(p[-1])
        out.append({"t": float(tau[0]), "flight_before": None, "flight_after": float(T[0]),
                    "speed_in": float(vin * G / f["g"]),
                    "restitution": float(f["g"] * T[0] / 2 / max(vin, 1e-6)), "stderr": float(dt / 4 / T[0] + 0.05)})
    for k in range(1, n - 1):
        if min(T[k], T[k - 1]) < 4 * dt:  # a flight of three frames is not timed
            continue


        last = 2.0 if k == n - 2 else 1.0
        out.append({"t": float(tau[k]), "flight_before": float(T[k - 1]), "flight_after": float(T[k]),
                    "speed_in": float(G * T[k - 1] / 2), "restitution": float(T[k] / T[k - 1]),
                    "stderr": float(T[k] / T[k - 1] * dt / 4 * np.hypot(last / T[k], 1 / T[k - 1])
                                    + (0.05 if last > 1 and n > 3 else 0.0))})
    return out


def radius_at(K, T, uv, area):
    """The radius (m) of a ball that touches the support and shows as a disc of `area` pixels about uv: the disc's
    radius at the distance of the ball's middle, which is its own radius above the support (taken on the support,
    a ball seen from low down is put behind where it is: a ball of 22 cm by a quarter of its size)."""
    r = 0.0
    for _ in range(4):
        r = float(np.sqrt(area / np.pi) * np.linalg.norm(ray_plane(K, T, uv, z=r) - T[:3, 3]) / K[0, 0])
    return r


def run(path, prior=None):
    """physics dict of a drop recording; rescales twin.json and track.npz by gravity's scale (module doc)."""
    from .fit import PRIOR
    from .track import MONOCULAR_SCALE_STD, _rescale

    path = Path(path)
    per, track = np.load(path / "perception.npz"), dict(np.load(path / "track.npz"))
    twin = read_document(path / "twin.json")
    pr = dict(PRIOR, **(prior or {}))
    K, T, t = np.asarray(twin["K"], float), np.asarray(twin["T_world_cam"], float), per["t"]
    obj = per["object"]
    uv = np.array([mask_stats(m)[0] for m in obj])
    area = np.array([m.sum() for m in obj], float)
    uvf, covered = fill_gaps(uv)
    seen = np.all(np.isfinite(uv), 1)
    free = covered & ~track["contact"] & ~track["border"]
    out = {"motion": "drop", "free_phase": None, "bounces": [], "parameters": {
        "object.mass": {"value": pr["mass"], "range": pr["mass_range"], "source": "prior (meta.json)"
                        if prior and "mass" in prior else "default", "how": "not visible in a video of free motion"}},
        "physics": {"object": {"mass": pr["mass"], "restitution": pr["restitution"]}}}
    # a fall towards the camera or away from it is no arc in the picture: it shows as a change of size, and what
    # the picture has of gravity is its part across the line of sight, under half of it within 30 degrees
    sight = np.linalg.inv(K) @ np.r_[np.nanmedian(uv[seen], 0), 1.0] if seen.any() else np.array([0.0, 0.0, 1.0])
    along = float(np.degrees(np.arccos(np.clip(abs(sight / np.linalg.norm(sight) @ T[:3, :3].T[:, 2]), 0, 1))))
    if along < ALONG_DEG:
        out["note"] = (f"the camera looks along gravity (within {along:.0f} degrees of it at the object): a fall "
                       "is then a change of size and no arc in the picture, nothing is measured")
        return out, twin, track
    q = down_coordinate(K, T, uvf)
    fits = []
    px = obj.shape[2] / 960.0  # the pixel thresholds below are of a picture 960 wide
    tol = 2.5 * max(1.0, px)
    runs = bounces(np.nan_to_num(q, nan=np.nanmin(q)), free, min_px=3.0 * px)
    for i, j, pk in runs:
        i_run = i
        i0 = held(uvf, i, j, fast=6.0 * px, slow=2.0 * px)  # released from rest: the fall starts where the hold ends
        if i0 > i:
            i, pk = i0, [b for b in pk if b > i0 + 2]
            out["released_from_rest"] = int(i0)
        if not pk:
            continue
        gaps = np.diff(pk)  # the last hops, a few frames each, run into rolling: the flights end before them
        if (gaps < 4).any():
            pk = pk[:int(np.argmax(gaps < 4)) + 1]
        # the radius: the disc of the mask's area, at the distance of the place it touches down
        near = [int(np.nonzero(seen)[0][np.argmin(np.abs(np.nonzero(seen)[0] - b))]) for b in pk]
        r = radius_at(K, T, uvf[pk[0]], np.median(area[near]))
        # where it bounces a ball is a streak and its mask too large: its size is read where it lies still after
        # the last bounce, when the video shows that
        kk = np.arange(pk[-1] + 1, min(j, len(uv) - 2) + 1)
        kk = kk[seen[kk] & seen[kk - 1] & seen[kk + 1]]
        kk = kk[np.linalg.norm(uv[kk + 1] - uv[kk - 1], axis=1) < 1.5 * px] if len(kk) else kk
        if len(kk) >= 3:
            rr = [radius_at(K, T, uv[q_], area[q_]) for q_ in kk]
            if np.isfinite(np.median(rr)):
                r = float(np.median(rr))
        f = None
        while pk:  # a last "bounce" that is none (someone walks past, the ball is caught) spoils the fit: drop it
            k = np.arange(i, pk[-1] + 1)
            slack = [int((~seen[max(b - 3, 0):b + 4]).sum()) for b in pk]
            f = fit_flight(t[k], uv[k], K, T, r, pk, i, uv_fill=uvf[k], slack=slack, rest=i0 > i_run)
            if f is None:
                break
            last = f["used"] & (t[k] >= f["p"][1 + 3 * (f["n"] - 2)]) if f["n"] >= 2 else f["used"]
            err = np.linalg.norm(project(K, T, f["P"][last])[0] - uv[k][last], axis=1)
            if f["typical_px"] < tol and len(err) >= 4 and np.median(err) < tol:  # the last flight too is one
                break
            pk = pk[:-1]
        if not pk:
            continue
        if f is not None and f["fall"] and f["n"] >= 3:
            # the flights between bounces are pinned at both ends, the fall only at its end: where the two give
            # gravity differently (a release seen in three frames, a first bounce unseen) the fall is left out
            k2 = np.arange(pk[0], pk[-1] + 1)
            f2 = fit_flight(t[k2], uv[k2], K, T, r, pk, pk[0], fall=False, uv_fill=uvf[k2], slack=slack)
            if f2 is not None and f2["typical_px"] < tol and abs(np.log(f["g"] / f2["g"])) > 0.10:
                f, k, i = f2, k2, pk[0]
        if f is not None and f["typical_px"] < tol and f["frames"] >= 6:
            fits.append(dict(f, first=int(i), last=int(pk[-1]), r=r, k=k))
    sc = twin.get("scale", {"factor": 1.0, "rel_std": MONOCULAR_SCALE_STD, "source": "MoGe-2 metric depth"})
    if not fits and "object_xyz" in per.files and len(per["object_xyz"]):
        # no touchdown is in view: the longest free run as one arc, placed where MoGe-2 saw the object
        Tm = np.asarray(twin["T_world_cam"], float)
        s0 = float(sc.get("factor", 1.0))  # MoGe-2's points at the twin's present scale, as its camera is
        at = {int(a): Tm[:3, :3] @ (np.asarray(x, float) * s0) + Tm[:3, 3]
              for a, x in zip(per["plane_frames"], per["object_xyz"]) if np.all(np.isfinite(x))}
        for i, j, _pk in sorted(runs, key=lambda r_: r_[0] - r_[1]):
            i = held(uvf, i, j, fast=6.0 * px, slow=2.0 * px)
            k = np.arange(i, j + 1)
            inside = [a for a in at if i <= a <= j and seen[a]]
            if len(k) < 8 or not inside:
                continue
            f = fit_arc(t[k], uv[k], K, T, np.array(inside) - i, np.array([at[a] for a in inside]))
            # an arc is held at no bounce: what gravity bends it by is the measure of its misfit (a light ball's
            # drag is a percent of that)
            if f is not None and f["typical_px"] < max(tol, 0.03 * f["bend_px"]):
                area_k = area[k][f["used"]]
                dist_k = np.linalg.norm(f["P"][f["used"]] - T[:3, 3], axis=1)
                fits.append(dict(f, first=int(i), last=int(j), k=k,
                                 r=float(np.median(np.sqrt(area_k / np.pi) * dist_k / K[0, 0]))))
                break
    if not fits:
        out["note"] = ("no flight is seen: neither between a release and a bounce or two bounces, nor one long enough "
                       "to show gravity's curve where the object can be placed: nothing is measured")
        return out, twin, track
    f = max(fits, key=lambda x: x["frames"])
    dt = float(np.median(np.diff(t)))
    # gravity's scale, fused with what the twin's scale was known to (once: a second run finds the twin rescaled)
    g_all = np.array([x["g"] for x in fits])
    w_all = np.array([x["frames"] for x in fits], float)
    g_app = float((g_all * w_all).sum() / w_all.sum())
    g_rel = float(np.hypot(np.hypot(f["g_std"] / f["g"], GRAVITY_MODEL_STD),
                           np.std(g_all) / g_app if len(fits) > 1 else 0.0))
    if "gravity" not in sc:
        # gravity and the scale known before should agree; if they are far apart the clock is the likelier
        # culprit (slow motion by a factor k makes gravity look k^2 weaker), and nothing is rescaled by a clock
        # that may be wrong
        conflict = bool(abs(np.log(G / g_app)) > 2.5 * np.hypot(g_rel, sc["rel_std"]) + 0.1)
        w_m, w_g = 1 / sc["rel_std"] ** 2, 1 / g_rel ** 2
        fac = 1.0 if conflict else float(np.exp(w_g * np.log(G / g_app) / (w_m + w_g)))
        track["xy"] = _rescale(twin["object"], T, track["xy"], fac)
        twin["T_world_cam"], twin["camera_height"] = T.tolist(), float(T[2, 3])
        twin["scale"] = dict(sc, source=sc.get("source", "MoGe-2 metric depth") + ("" if conflict else " + gravity"),
                             factor=float(sc.get("factor", 1.0) * fac),
                             rel_std=float(sc["rel_std"] if conflict else 1 / np.sqrt(w_m + w_g)),
                             gravity={"apparent": g_app, "factor": G / g_app, "rel_std": g_rel, "applied": fac,
                                      "flights": len(fits), "conflict": conflict})
    else:
        fac = 1.0
    gr = twin["scale"]["gravity"]
    if gr.get("conflict", abs(np.log(gr["factor"])) > 2.5 * np.hypot(gr["rel_std"], MONOCULAR_SCALE_STD) + 0.1):
        out["warning"] = (f"gravity's scale ({gr['factor']:.2f}) and the scene's disagree: if the scene's is right the "
                          f"video runs {np.sqrt(gr['factor']):.2f} x slower than real time (meta.json slow_motion); "
                          "the scene's scale is kept as it was")
        out["slow_motion_suggested"] = float(np.sqrt(gr["factor"]))
    p, n = f["p"], f["n"]
    nodes = p[1:1 + 3 * n].reshape(n, 3)
    k0 = int(np.nonzero(f["used"])[0][0])
    P0 = f["P"][k0] * fac
    tt = t[f["k"]]
    g_twin = f["g"] * fac
    if f.get("arc"):
        V0 = (p[4:7] - np.array([0.0, 0.0, f["g"] * (tt[k0] - f["t0"])])) * fac
    elif f["fall"]:
        V0 = np.r_[p[-3:-1], p[-1] - f["g"] * (tt[k0] - nodes[0, 0])] * fac
    else:
        T0 = nodes[1, 0] - nodes[0, 0]
        V0 = np.r_[(nodes[1, 1:] - nodes[0, 1:]) / T0, f["g"] * (T0 / 2 - (tt[k0] - nodes[0, 0]))] * fac
    out["free_phase"] = {"first_frame": int(f["first"] + k0), "last_frame": int(f["last"]), "radius": f["r"] * fac,
                         "start": {"t": float(tt[k0]), "position": P0.tolist(), "velocity": V0.tolist()},
                         "nodes": [{"t": float(a), "xy": [float(b * fac), float(c * fac)]} for a, b, c in nodes],
                         "flight": {"frames": (f["first"] + np.nonzero(f["used"])[0]).tolist(),
                                    "position": (f["P"][f["used"]] * fac).tolist()},
                         "rms_px": f["rms_px"], "typical_px": f["typical_px"], "gravity_apparent": g_twin,
                         "fall": f["fall"], "touchdown_seen": not f.get("arc", False), "ok": True}
    out["bounces"] = [b for x in fits if not x.get("arc") for b in restitutions(x, dt)]
    out["parameters"]["scene.scale"] = {
        "value": twin["scale"]["factor"], "stderr": twin["scale"]["factor"] * twin["scale"]["rel_std"],
        "source": "not from gravity: it and the scene's scale disagree (warning)" if gr.get("conflict") else "measured",
        "how": f"gravity: flights look like {gr['apparent']:.2f} m/s^2 at the scale before, so that scale was off by "
               f"{gr['factor']:.3f} (+- {100 * gr['rel_std']:.0f} %); " + ("not applied" if gr.get("conflict") else
                                                                          "fused with it"),
        "gravity_factor": gr["factor"], "flights": gr["flights"]}
    bs = [b for b in out["bounces"] if 0.02 < b["restitution"] < 1.05]
    if bs:
        e = np.array([b["restitution"] for b in bs])
        w = np.array([1 / b["stderr"] ** 2 for b in bs])
        mean = float((e * w).sum() / w.sum())
        spread = float(np.sqrt((w * (e - mean) ** 2).sum() / w.sum())) if len(e) > 1 else 0.0
        out["parameters"]["object.restitution"] = {
            "value": min(mean, 1.0), "stderr": float(max(1 / np.sqrt(w.sum()), spread / np.sqrt(len(e)))),
            "source": "measured", "bounces": len(bs), "spread": spread,
            "how": "ratio of the flight times after and before a bounce (after the first one: the speed gravity "
                   "gives the fitted fall); no scale and no camera enters"}
        out["physics"]["object"]["restitution"] = min(mean, 1.0)
    else:
        out["parameters"]["object.restitution"] = {"value": pr["restitution"], "source": "default",
                                                   "how": "no bounce with a flight on both sides"}
    return out, twin, track


def describe(path, phys):
    if phys["free_phase"] is None:
        return f"{path}: {phys['note']}"
    s, e = phys["parameters"]["scene.scale"], phys["parameters"]["object.restitution"]
    bs = " ".join(f"[{b['t']:.2f} s: {b['speed_in']:.2f} m/s, e {b['restitution']:.2f}]" for b in phys["bounces"])
    return (f"{path}: restitution {e['value']:.3f}" + (f" +- {e['stderr']:.3f}" if "stderr" in e else " (default)")
            + f", gravity's scale factor {s['gravity_factor']:.3f} ({s['flights']} flights, rms "
              f"{phys['free_phase']['rms_px']:.1f} px) {bs}"
            + ("" if phys["free_phase"].get("touchdown_seen", True) else " [one arc, no touchdown in view]")
            + (f" WARNING: {phys['warning']}" if phys.get("warning") else ""))
