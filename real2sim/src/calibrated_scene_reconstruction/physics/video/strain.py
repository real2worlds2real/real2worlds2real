"""A body that a manipulator deforms while both of them are in view (meta.json "motion": "strain"; perception by
vidphys.soft): what a video shows of it, and what it does not.

    envs/recon/bin/python -m calibrated_scene_reconstruction.physics.video.strain real/<name> [...]        # physics.json

Something holds one end of a block and moves it; the other end is held still. The block's surface carries the
deformation: points on it are followed from the frame in which nothing has moved yet (corners by Lucas-Kanade, each
checked by tracking it back), and in every frame the affine map that takes their first places to their present
ones is the deformation gradient of the face the camera sees, as it appears in the picture. In the block's own axes
(its long axis in the first frame, and across it) that map is a stretch along, a stretch across, a shear and a turn.
Ratios of lengths along one direction survive the camera, so the two stretches are the material's strains.

Two things the surface's points do not give are read off the silhouette instead. A shear bends the body into an S
(its held ends stay square), which no one affine map describes: it is measured as the turn of the line from the
middle of one end to the middle of the other, the part of it that is across the line of sight. A twist turns the
faces the points are on out of view: of a body whose section is square (meta.json "section": {"shape": "square"}) it
is read off the width of the silhouette along the body, which is a (|cos| + |sin|) of the angle each section is seen
at, and so is narrowest where a face is square to the camera.

What that measures: the strain itself, and from a pull or a push along the axis Poisson's ratio, the stretch across
over the stretch along (in the middle of the block: near a held end the material cannot narrow). What it cannot: any
modulus. The deformation is the manipulator's, and no force is in the picture; physics.json says so.
"""
from calibrated_scene_reconstruction.physics.common.schema import read_document, write_document
import argparse
import json
from pathlib import Path

import numpy as np

AXIAL = 0.012  # the least stretch along the axis that counts as a pull or a push
SHEAR = 2.0  # deg: the least shear that counts as one
BEND = 0.15  # the least bow of the body's line, in its own widths, that counts as a bend
FACES_DEG = 3.0  # what the angle a square section's faces are seen at is good to, from MoGe-2's points on them
SQUARE = 0.025  # the cosine of the angle between the points' two axes under which they are still square
SLENDER = 1.5  # a body's line turns as a shear only where it has one: as long as this many of its widths (the line




POINT_PX = 0.2


def body_axes(mask):
    """(centre (2,) px, e1 the long axis (unit; pointing down the picture), e2 across it, length px, width px) of a
    mask: its principal axes, its extent along the long one and its mean width."""
    vs, us = np.nonzero(mask)
    P = np.stack([us, vs], 1).astype(float)
    c = P.mean(0)
    w, V = np.linalg.eigh(np.cov((P - c).T))
    e1 = V[:, 1] * (1 if V[1, 1] >= 0 else -1)
    e2 = np.array([-e1[1], e1[0]])
    s = (P - c) @ e1
    lo, hi = np.percentile(s, [0.5, 99.5])
    return c, e1, e2, float(hi - lo), float(len(P) / (hi - lo))


def profile(mask, c, e1, e2, L, n=24):
    """The silhouette along the axes (c, e1, e2) of the first frame: (width px (n,), middle across px (n,)) in n
    slices over the middle 90 % of the length L; NaN where a slice is empty."""
    vs, us = np.nonzero(mask)
    P = np.stack([us, vs], 1).astype(float) - c
    s, d = P @ e1, P @ e2
    edges = np.linspace(-0.45 * L, 0.45 * L, n + 1)
    w, m = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        x = d[(s >= edges[i]) & (s < edges[i + 1])]
        if len(x) > 8:
            lo, hi = np.percentile(x, [1, 99])
            w[i], m[i] = hi - lo, 0.5 * (hi + lo)
    return w, m


def square_width(a, psi):
    """The width a square section of side a shows when it is seen at the angle psi (rad) from face-on."""
    return a * (np.abs(np.cos(psi)) + np.abs(np.sin(psi)))


def faces_seen(per):
    """How a body of square section stands to the camera, from MoGe-2's points on it: seen along its own axis (the
    long direction of those points; no gravity enters, the body may lie or stand) the points of its two visible
    faces lie on the two arms of an L, and the L that fits them gives the angle (rad, 0 to pi/2) between the line
    of sight and one face's normal; the other face is seen at the rest of the right angle. Returns (angle, the
    fit's residual m, the camera's elevation over the body's section rad) or None when the points do not make an
    L (to a twelfth of the section's size)."""
    from scipy.optimize import minimize

    P = np.asarray(per["object_points"], float)
    P = P[np.all(np.isfinite(P), 1)]
    if len(P) < 200:
        return None
    c0 = np.median(P, 0)
    ax = np.linalg.svd(P - c0, full_matrices=False)[2][0]  # the body's axis, camera frame
    b1 = np.cross(ax, [0.0, 0.0, 1.0])
    if np.linalg.norm(b1) < 1e-6:
        return None  # seen along itself
    b1 /= np.linalg.norm(b1)
    b2 = np.cross(ax, b1)
    z = (P - c0) @ ax
    mid = (z > np.percentile(z, 20)) & (z < np.percentile(z, 80))  # away from its two ends
    c3 = np.median(P[mid], 0)
    Q = np.stack([(P[mid] - c3) @ b1, (P[mid] - c3) @ b2], 1)
    to_cam = -c3  # the camera is the origin
    elev = float(np.arcsin(abs(to_cam @ ax) / np.linalg.norm(to_cam)))
    cam = np.array([to_cam @ b1, to_cam @ b2])
    cam /= np.linalg.norm(cam)
    side = float(np.hypot(*np.ptp(np.percentile(Q, [2, 98], axis=0), axis=0)) / np.sqrt(2))  # the section's size

    def cost(p):
        u = np.array([np.cos(p[0]), np.sin(p[0])])
        d = Q - p[1:]
        a, b = d @ u, d @ np.array([-u[1], u[0]])
        d1 = np.where(a > 0, np.abs(b), np.hypot(a, b))  # to the arm along u, from the corner on
        d2 = np.where(b > 0, np.abs(a), np.hypot(a, b))
        return np.mean(np.minimum(np.minimum(d1, d2), 0.2 * side))

    best = min((minimize(cost, [phi, 0.0, 0.0], method="Nelder-Mead",
                         options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 600})
                for phi in np.radians(np.arange(0, 360, 20))), key=lambda r: r.fun)
    if best.fun > 0.08 * side:
        return None
    v = np.array([-np.sin(best.x[0]), np.cos(best.x[0])])  # across the first arm: that face's normal
    return float(np.arccos(np.clip(abs(v @ cam), 0, 1))), float(best.fun), elev


def twist_of(W, psi0, spread=0.0, least=0.04, fps=30.0):
    """The twist (rad, (F,)) of a square-sectioned body whose silhouette's widths along it are W (F, n), one end
    (slice 0 or n-1, whichever keeps its width) held and first seen at psi0 (faces_seen): every frame's twist is
    the one whose widths along the body fit, followed from frame to frame, both ways round. spread: the part of
    the body's length over which one row of the picture mixes sections (a camera that looks down on it). Returns
    (twist (F,) of the moved end against the held one, positive towards face-on of the face psi0 is of; the side
    a px; rms px; how far the last twist may be off, rad) or None when the widths never change by `least`."""
    F, n = W.shape
    seen = np.isfinite(W).sum(1) > 0.7 * n
    w0 = np.nanmedian(W[seen][:5], 0)
    if not np.nanmax(np.abs(W[seen] / w0 - 1)) > least:
        return None
    held_first = np.nanstd(W[seen][:, :3]) < np.nanstd(W[seen][:, -3:])
    x = np.linspace(0, 1, n) if held_first else np.linspace(1, 0, n)  # 0 at the held end
    a = float(np.nanmedian(w0) / square_width(1.0, psi0))
    thetas = np.radians(np.arange(0, 200.5, 1.0))
    k_blur = max(int(round(spread * n)), 0)
    best = None
    for sign in (-1.0, 1.0):
        model = square_width(a, psi0 + sign * thetas[:, None] * x[None])  # (angles, n)
        if k_blur:
            ker = np.ones(2 * k_blur + 1) / (2 * k_blur + 1)
            model = np.stack([np.convolve(np.pad(m, k_blur, mode="edge"), ker, "valid") for m in model])
        th, cost, prev, wide = np.zeros(F), 0.0, 0.0, 0.0
        for k in range(F):
            if not seen[k]:
                th[k] = prev
                continue
            r = np.nanmean((model - W[k]) ** 2, 1)
            near = np.abs(thetas - prev) < np.radians(max(15 * 30.0 / fps, 3))  # no faster than 450 deg/s
            j = int(np.argmin(np.where(near, r, np.inf)))
            th[k], prev = thetas[j], thetas[j]
            cost += r[j]
            # how far the twist may be off: the twists about it whose widths fit as well, a slice in four of the
            # silhouette's counting as its own (near 45 degrees from face-on a square's width hardly changes)
            fits = r <= r[j] * (1 + 4.0 / n) + 1e-12
            lo_, hi_ = j, j
            while lo_ > 0 and fits[lo_ - 1]:
                lo_ -= 1
            while hi_ < len(r) - 1 and fits[hi_ + 1]:
                hi_ += 1
            wide = 0.5 * float(thetas[hi_] - thetas[lo_])
        if best is None or cost < best[0]:
            best = (cost, -sign * th, max(wide, np.radians(0.5)))
    return best[1], a, float(np.sqrt(best[0] / seen.sum())), best[2]


def follow(frames, mask0, k0):
    """Points on the body in frame k0 followed through every frame: (P0 (N, 2), P (F, N, 2), ok (F, N)). The
    window a point is followed by is small against the body and stays clear of its outline: a window that has the
    outline in it follows the outline, the strongest edge it sees, and every point near a side then moves as that
    side does."""
    import cv2

    gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    _c, _e1, _e2, _L, W = body_axes(mask0)
    half = int(max(0.1 * W, 4))
    inner = cv2.erode(mask0.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=half + 3)
    p0 = cv2.goodFeaturesToTrack(gray[k0], 400, 0.01, max(0.06 * W, 3), mask=inner, blockSize=5)
    if p0 is None or len(p0) < 6:
        return np.zeros((0, 2)), np.zeros((len(frames), 0, 2)), np.zeros((len(frames), 0), bool)
    p0 = p0.reshape(-1, 2).astype(np.float32)
    win = (2 * half + 1,) * 2
    lk = dict(winSize=win, maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.005))
    P = np.zeros((len(frames), len(p0), 2), np.float32)
    ok = np.zeros((len(frames), len(p0)), bool)
    P[k0], ok[k0] = p0, True
    for order in (range(k0 + 1, len(frames)), range(k0 - 1, -1, -1)):
        prev, good, last = p0.copy(), np.ones(len(p0), bool), k0
        for k in order:
            nxt, st, _e = cv2.calcOpticalFlowPyrLK(gray[last], gray[k], prev, None, **lk)
            back, st2, _e = cv2.calcOpticalFlowPyrLK(gray[k], gray[last], nxt, None, **lk)
            good &= (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(back - prev, axis=1) < 0.5)
            P[k], ok[k] = nxt, good
            prev, last = nxt, k
    return p0.astype(float), P.astype(float), ok


def agrees(silhouette, points):
    """Whether the silhouette is longer (or wider) by `silhouette` as the points on the body say by `points`: the
    same way, and between 0.4 and 3 times as much (a mask that grows by ten times what its points move does not
    follow the body, whichever way it happens to go)."""
    return bool(silhouette * points > 0 and 0.4 * abs(points) < abs(silhouette) < 3.0 * abs(points))


def gradient(P0, P, ok, e1, e2, least=6):
    """The deformation of every frame in the body's axes, from the points' first places P0 to their present ones:
    {"along", "across": stretches less one, "skew": the cosine of the angle between the two axes (0 while they stay
    square), "n": points used}, each (F,). (The skew is no measure of a shear: on a body seen at a slant the axes
    turn in the picture under a shear as under a turn of the whole body, and the shear is read off the
    silhouette's line. It tells a squeeze, which leaves them square, from a twist, which does not.)"""
    import cv2

    F = len(P)
    out = {k: np.full(F, np.nan) for k in ("along", "across", "skew")}
    out["n"] = np.zeros(F, int)
    B = np.stack([e1, e2], 1)
    for k in range(F):
        m = ok[k]
        if m.sum() < least:
            continue
        A, inl = cv2.estimateAffine2D(P0[m].astype(np.float32), P[k][m].astype(np.float32), method=cv2.RANSAC,
                                      ransacReprojThreshold=1.0)
        if A is None or inl.sum() < least:
            continue
        G = B.T @ A[:, :2] @ B  # the gradient in (along, across)
        a, b = G[:, 0], G[:, 1]  # where the two axes go
        out["along"][k], out["across"][k] = np.linalg.norm(a) - 1, np.linalg.norm(b) - 1
        out["skew"][k] = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))  # how far the axes left their right angle
        out["n"][k] = int(inl.sum())
    return out


def bend_of(P0, P, ok, c, e1, e2, least=8):
    """How far the body's line has bowed, from the points: in every frame the points' shift across the body as a
    parabola of their place along it. (bow (F,) px: the middle of the points' span against the line between its
    ends, along e2; what the parabola leaves of the shifts, rms px (F,); the span, px)."""
    F = len(P)
    bow, left = np.full(F, np.nan), np.full(F, np.nan)
    s = (P0 - c) @ e1
    span = float(np.ptp(s)) if len(s) else 0.0
    for k in range(F):
        m = ok[k].copy()
        if m.sum() < least:
            continue
        v = (P[k] - P0) @ e2
        for _ in range(3):
            q = np.polyfit(s[m], v[m], 2)
            r = v - np.polyval(q, s)
            m2 = ok[k] & (np.abs(r) < max(2.5 * np.std(r[m]), 0.5))
            if m2.sum() < least or m2.sum() == m.sum():
                break
            m = m2
        bow[k], left[k] = -q[0] * (np.ptp(s[m]) / 2) ** 2, np.sqrt(np.mean(r[m] ** 2))
    return bow, left, span


def measure(per, frames, meta):
    """physics.json of a recording whose motion is "strain"."""
    obj, score, t = per["object"], per["score"], per["t"]
    out = {"motion": "strain", "fps": float(1.0 / np.median(np.diff(t)))}
    if str(per["camera"]) == "moving":
        return dict(out, measured=False, reason="the camera travels")
    seen = np.nonzero(score > 0)[0]
    if len(seen) < 10:
        return dict(out, measured=False, reason="the body is not seen")
    k0 = int(seen[0])
    c, e1, e2, L0, W0 = body_axes(obj[k0])
    n5 = max(5, int(round(out["fps"] / 6)))  # a sixth of a second, five frames at the least
    smooth = lambda y: np.convolve(np.where(np.isfinite(y), y, 0.0), np.ones(n5) / n5, "same")
    # the silhouette: its length, the line between its ends, its widths
    Wp, Mp, Ls = [], [], []
    for k in range(len(obj)):
        ok_k = score[k] > 0 and obj[k].any()
        w, m = profile(obj[k], c, e1, e2, L0) if ok_k else (np.full(24, np.nan), np.full(24, np.nan))
        Wp.append(w), Mp.append(m)
        Ls.append(body_axes(obj[k])[3] if ok_k else np.nan)
    Wp, Mp, Ls = np.array(Wp), np.array(Mp), np.array(Ls)
    chord = np.degrees(np.arctan2(np.nanmean(Mp[:, -4:], 1) - np.nanmean(Mp[:, :4], 1), 0.75 * L0))
    chord = smooth(chord - np.nanmedian(chord[k0:k0 + n5]))
    length = smooth(Ls / np.nanmedian(Ls[k0:k0 + n5]) - 1)
    wide = np.nanmedian(Wp, 1)
    width = smooth(wide / np.nanmedian(wide[k0:k0 + n5]) - 1)
    # the surface: points followed from the first frame, in the middle of the body (near a held end it cannot narrow)
    P0, P, ok = follow(frames, obj[k0], k0)
    mid = np.abs((P0 - c) @ e1) < 0.3 * L0 if len(P0) else np.zeros(0, bool)
    g = gradient(P0[mid], P[:, mid], ok[:, mid], e1, e2) if mid.sum() >= 6 else None
    good = np.isfinite(g["along"]) if g is not None else np.zeros(len(obj), bool)
    al = smooth(g["along"]) if g is not None else np.zeros(len(obj))
    ac = smooth(g["across"]) if g is not None else np.zeros(len(obj))
    kp = int(np.nanargmax(np.abs(np.where(good, al, np.nan)))) if good.sum() >= 10 else 0
    kc = int(np.nanargmax(np.abs(chord)))
    kl = int(np.nanargmax(np.abs(length)))
    s = {"points": int(mid.sum()), "length_px": L0, "width_px": W0, "first_frame": k0,
         "along_peak": float(al[kp]), "across_at_peak": float(ac[kp]), "peak_frame": kp,
         "length_change": float(length[kl]), "shear_seen_deg": float(chord[kc]), "shear_frame": kc,
         "modulus": None, "why_no_modulus": "the deformation is the manipulator's and no force is in the picture"}
    series = {"t": t.tolist(), "shear_seen_deg": np.round(chord, 3).tolist(), "length": np.round(length, 5).tolist()}
    if g is not None:
        series.update(along=np.round(g["along"], 5).tolist(), across=np.round(g["across"], 5).tolist())
    # the body's line bowed: every point at one place along it goes across it alike (round a twist they do not)
    bow, left, span = bend_of(P0, P, ok, c, e1, e2) if len(P0) >= 8 else (np.full(len(obj), np.nan),) * 2 + (0.0,)
    bent = np.isfinite(bow) & (np.abs(bow) >= BEND * W0) & (left < 0.25 * np.abs(bow)) & (span > 0.4 * L0)
    series["bow"] = np.round(bow / W0, 4).tolist()
    twist = stand = None
    if bent.sum() >= n5:
        kb = int(np.nanargmax(np.where(bent, np.abs(smooth(bow)), np.nan)))
        s.update(kind="bend", bow_over_width=float(bow[kb] / W0), bow_over_span=float(bow[kb] / span),
                 bow_end_over_width=float(np.nanmedian(bow[np.nonzero(bent)[0][-5:]]) / W0), bend_frame=kb,
                 bow_residual_px=float(left[kb]), span_over_length=float(span / L0),
                 note="the body's line bows across itself, as seen: the middle of the stretch its points cover "
                      "against the line between that stretch's ends")
        out.update(measured=True, strain=s, series=series)
        return out
    if meta.get("section", {}).get("shape") == "square":
        stand = faces_seen(per)
        if stand is not None:  # a row of the picture mixes the sections half a side, by the elevation's tangent, apart
            a_px = np.nanmedian(Wp[k0]) / square_width(1.0, stand[0])
            twist = twist_of(Wp, stand[0], 0.5 * a_px * np.tan(stand[2]) / L0, fps=out["fps"])
            if twist is not None:
                # the twist hangs on how the faces stand at rest, which the L of MoGe-2's points gives to a few
                # degrees (the same video mirrored: 2.7 off, and a twist 25 off): read again with that angle
                # FACES_DEG either way, and what the twist then differs by is part of its error
                ends = [np.median(tw[0][-5:]) for d_ in (-1.0, 1.0)
                        for tw in [twist_of(Wp, stand[0] + d_ * np.radians(FACES_DEG),
                                            0.5 * a_px * np.tan(stand[2]) / L0, fps=out["fps"])] if tw is not None]
                off = max([abs(abs(e) - abs(np.median(twist[0][-5:]))) for e in ends], default=0.0)
                twist = twist[:3] + (float(np.hypot(twist[3], off)),)
    # which axis is loaded: under a pull or a push along one axis the other changes by at most half as much
    # (Poisson's ratio), so the axis whose strain is more than twice the other's is the loaded one
    kq = int(np.nanargmax(np.abs(np.where(good, ac, np.nan)))) if good.sum() >= 10 else 0




    sk = np.abs(g["skew"]) if g is not None else np.zeros(len(obj))
    square = lambda k_: bool(np.isfinite(sk[k_]) and sk[k_] < SQUARE and g["n"][k_] >= 20)  # noqa: E731
    across_loaded = bool(good.sum() >= 10 and abs(ac[kq]) >= AXIAL and abs(ac[kq]) > 2 * abs(al[kq])
                         and (agrees(width[kq], ac[kq]) or square(kq)))
    if twist is not None and g is not None and not np.nanmax(np.where(good, sk, 0.0)) >= 2 * SQUARE:
        twist = None  # the silhouette's widths change, but the points' axes stay square: nothing turned
    if across_loaded:  # and the body's length answers it the other way (a silhouette that narrows as a square
        use = good & (np.abs(g["across"]) > 0.4 * abs(ac[kq]))  # section turns leaves the length as it was)
        nu_x = -float((g["across"][use] * g["along"][use]).sum() / (g["across"][use] ** 2).sum())
        across_loaded = 0.1 <= nu_x <= 0.6
    if abs(chord[kc]) >= SHEAR and L0 >= SLENDER * W0:
        # (a body that leans or turns as a whole turns its line the same way: neither the points on it nor its
        # silhouette's ends tell that from a shear when it is seen at a slant, so a shear is read on the
        # understanding that the body is held at its ends and they keep their direction)
        s["kind"] = "shear"
        s["note"] = ("the shear across the line of sight, of a body whose ends are held and keep their direction; "
                     "towards the camera or away from it, it is not seen")
    elif across_loaded:
        s["kind"] = "stretch" if ac[kq] > 0 else "compress"
        use = good & (np.abs(g["across"]) > 0.4 * abs(ac[kq]))
        x, y = g["across"][use], g["along"][use]
        nu = -float((x * y).sum() / (x * x).sum())
        s.update(load_axis="across the body's length", across_peak=float(ac[kq]), along_at_peak=float(al[kq]),
                 peak_frame=kq, poisson_frames=int(use.sum()), poisson=nu,
                 poisson_std=float(np.hypot(np.std(-y / x), POINT_PX / L0 / max(abs(ac[kq]), 1e-9))),
                 across_residual=float(np.sqrt(np.mean((y + nu * x) ** 2))))
    elif good.sum() >= 10 and abs(al[kp]) >= AXIAL and (
            agrees(length[kp], al[kp]) or square(kp)):
        s["kind"] = "stretch" if al[kp] > 0 else "compress"
        use = good & (np.abs(g["along"]) > 0.4 * abs(al[kp]))  # frames with enough stretch to divide by
        x, y = g["along"][use], g["across"][use]
        nu = -float((x * y).sum() / (x * x).sum())
        # frames are not independent (the points drift together): the spread of the ratio over them, not over
        # sqrt(n), and what the narrow way's strain is good to (POINT_PX)
        s.update(load_axis="along the body's length", poisson=nu,
                 poisson_std=float(np.hypot(np.std(-y / x), POINT_PX / W0 / max(abs(al[kp]), 1e-9))),
                 poisson_frames=int(use.sum()), across_residual=float(np.sqrt(np.mean((y + nu * x) ** 2))))
        if nu < -0.5:  # it grows or shrinks both ways alike: nearer or further, not strained
            s["kind"] = "none"
            s["note"] = ("the body grows or shrinks both ways alike in the picture: it comes nearer or goes further "
                         "(or the lens zooms), which is no strain")
        elif nu > 0.6 and not agrees(length[kp], al[kp]):
            # the other axis changes by more than half as much, and the silhouette does not say the body is
            # longer or shorter as the points do: which axis is loaded is not told (a cube squeezed across reads
            # so when its points widen by 1.5 % and narrow by 2 %)
            s["kind"] = "none"
            s["note"] = (f"the points change by {100 * al[kp]:+.1f} % one way and {100 * ac[kp]:+.1f} % the other, "
                         "neither twice the other, and the silhouette confirms neither: which axis is loaded is "
                         "not told, and nothing is read")
        elif nu > 0.6:  # no material narrows by more than half of what it lengthens: the points are too few or
            for k_ in ("poisson", "poisson_std", "poisson_frames", "across_residual"):  # too coarse across the body
                s.pop(k_)
            s["poisson_not_read"] = (f"the points narrow by {nu:.2f} of what they lengthen, more than any material "
                                     "does (0.5): too few of them, or too coarse a picture, across the body")
    elif twist is not None and len(P0) < 8:
        # the silhouette's widths change, as under a twist; but so they do where something presses into the body,
        # and with this few points followed on it a bend cannot be told
        s["kind"] = "none"
        s["note"] = (f"the silhouette's widths change, but only {len(P0)} points are followed on the body: too few to "
                     "tell a twist from a bend or a dent, and neither is read")
    elif twist is not None and twist[2] > 0.1 * twist[1]:
        # the widths are not those of a square section that turns (off by a tenth of its side and more): a mask
        # that does not follow the body, and no twist is read off it
        s["kind"] = "none"
        s["note"] = (f"the silhouette's widths change, but not as a square section's that turns (they are "
                     f"{twist[2]:.0f} px off, of a side of {twist[1]:.0f}): the mask does not follow the body, and "
                     "nothing is read")
    elif twist is not None:
        th, a_px, rms, th_std = twist
        kt = int(np.argmax(np.abs(th)))
        s.update(kind="twist", twist_peak_deg=float(np.degrees(th[kt])), twist_std_deg=float(np.degrees(th_std)),
                 twist_end_deg=float(np.degrees(np.median(th[-5:]))), twist_frame=kt, side_px=a_px,
                 seen_at_deg=float(np.degrees(stand[0])), faces_residual_mm=1e3 * stand[1], width_rms_px=rms,
                 note="of the body in view, from the widths of its silhouette and how its faces stand to the camera")
        series["twist_deg"] = np.round(np.degrees(th), 1).tolist()
    else:
        s["kind"] = "none"
        s["note"] = "no stretch, push, shear, twist or bend above the noise: what was done to the body is not seen"
    if s["kind"] == "none":
        for k_ in ("poisson", "poisson_std", "poisson_frames", "across_residual", "load_axis"):
            s.pop(k_, None)
    out.update(measured=s["kind"] != "none", strain=s, series=series)
    if s["kind"] == "none":
        out["reason"] = s["note"]
    return out


def fit(path):
    from .scene import read_frames

    path = Path(path)
    frames, _fps = read_frames(path)
    return measure(np.load(path / "perception.npz"), frames, read_document(path / "meta.json"))


def summary(path):
    """summary.mp4 of a measured strain: the video with the body's mask (green) and the points followed on it
    (yellow; their first places as grey dots); beside it the strains along time and what they say."""
    import subprocess

    import cv2

    from .scene import read_frames

    path = Path(path)
    phys = read_document(path / "physics.json")
    if "strain" not in phys:
        return None
    st, se = phys["strain"], phys["series"]
    per = np.load(path / "perception.npz")
    obj = per["object"]
    frames, fps = read_frames(path)
    k0 = st["first_frame"]
    P0, P, ok = follow(frames, obj[k0], k0)
    H, W = frames.shape[1:3]
    scale = 720.0 / H if H > 720 else 1.0  # the panel beside a picture no taller than 720
    h, w, PW = int(H * scale), int(W * scale), 460
    t = np.asarray(se["t"])
    curves = [(k, np.asarray(se[k], float) * (100 if k in ("along", "across", "length") else 1), col, lab)
              for k, col, lab in (("along", (80, 200, 255), "along %"), ("across", (255, 160, 80), "across %"),
                                  ("shear_seen_deg", (160, 255, 160), "shear deg"),
                                  ("twist_deg", (255, 120, 255), "twist deg"),
                                  ("bow", (120, 220, 255), "bow, in widths")) if k in se]
    if st["kind"] == "bend":
        curves = [c for c in curves if c[0] == "bow"]
        text = [f"bend: its line bows {abs(st['bow_over_width']):.2f} of its width",
                f"   ({abs(st['bow_end_over_width']):.2f} in the end; over the {100 * st['span_over_length']:.0f} % of",
                "   its length the points cover)"]
    elif st["kind"] in ("stretch", "compress"):
        curves = [c for c in curves if c[0] in ("along", "across")]
        text = [f"{st['kind']}: {100 * st['along_peak']:+.1f} % along, {100 * st['across_at_peak']:+.2f} % across",
                f"Poisson's ratio {st['poisson']:.2f} +- {st['poisson_std']:.2f}",
                f"   ({st['points']} points in the body's middle)"]
    elif st["kind"] == "shear":
        curves = [c for c in curves if c[0] == "shear_seen_deg"]
        text = [f"shear: {abs(st['shear_seen_deg']):.1f} deg across the line of sight",
                "   (the turn of the line between its two ends)"]
    elif st["kind"] == "twist":
        curves = [c for c in curves if c[0] == "twist_deg"]
        text = [f"twist: {abs(st['twist_end_deg']):.0f} deg in the end",
                f"   (a square section seen at {st['seen_at_deg']:.0f} deg; widths within {st['width_rms_px']:.1f} px)"]
    else:
        text = ["nothing done to the body is seen"]
    text += ["modulus: not measured", "   (no force is in the picture)"]
    box = (w + 20, 40, w + PW - 20, 260)
    lo = min(np.nanmin(c[1]) for c in curves) if curves else -1.0
    hi = max(np.nanmax(c[1]) for c in curves) if curves else 1.0
    lo, hi = min(lo, 0) - 0.05 * (hi - lo + 1e-6), max(hi, 0) + 0.05 * (hi - lo + 1e-6)
    px = lambda tt, y: (int(box[0] + (tt - t[0]) / max(t[-1] - t[0], 1e-6) * (box[2] - box[0])),
                        int(box[3] - (y - lo) / (hi - lo) * (box[3] - box[1])))
    tmp = path / "summary_raw.mp4"
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w + PW, h))
    for k in range(len(frames)):
        im = frames[k][:, :, ::-1].copy()
        cs, _ = cv2.findContours(obj[k].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im, cs, -1, (0, 220, 0), 1)
        for p0, p, good in zip(P0, P[k], ok[k]):
            cv2.circle(im, (int(round(p0[0])), int(round(p0[1]))), 1, (150, 150, 150), -1)
            if good:
                cv2.circle(im, (int(round(p[0])), int(round(p[1]))), 2, (0, 230, 255), -1)
        img = np.zeros((h, w + PW, 3), np.uint8)
        img[:, :w] = cv2.resize(im, (w, h)) if scale != 1.0 else im
        cv2.rectangle(img, box[:2], box[2:], (70, 70, 70), 1)
        cv2.line(img, px(t[0], 0.0), px(t[-1], 0.0), (70, 70, 70), 1)
        for j, (_k, y, col, lab) in enumerate(curves):
            pts = np.array([px(t[i], y[i]) for i in range(len(t)) if np.isfinite(y[i])], np.int32)
            cv2.polylines(img, [pts], False, col, 1, cv2.LINE_AA)
            cv2.putText(img, lab, (box[0] + 6 + 110 * j, box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1,
                        cv2.LINE_AA)
        cv2.line(img, (px(t[k], 0)[0], box[1]), (px(t[k], 0)[0], box[3]), (255, 255, 255), 1)
        for j, line in enumerate(text):
            cv2.putText(img, line, (w + 16, box[3] + 34 + 22 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (235, 235, 235), 1,
                        cv2.LINE_AA)
        cv2.putText(img, "green: the body's mask   yellow: points followed on it", (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(img)
    vw.release()
    out = path / "summary.mp4"
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-crf", "23",
                        "-pix_fmt", "yuv420p", str(out)], check=True)
        tmp.unlink()
    except (OSError, subprocess.CalledProcessError):
        tmp.rename(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording", nargs="+")
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--summary", action="store_true", help="summary.mp4 of recordings already measured")
    args = ap.parse_args()
    for p in args.recording:
        if args.summary:
            print(summary(p), flush=True)
            continue
        if args.skip_done and (Path(p) / "physics.json").exists():
            continue
        if not (Path(p) / "perception.npz").exists():
            continue
        try:
            o = fit(p)
        except Exception as e:  # one recording's failure is its own
            o = {"motion": "strain", "measured": False, "reason": f"{type(e).__name__}: {e}"}
        write_document(Path(p) / "physics.json", o, indent=1)
        s = o.get("strain", {})
        if not o.get("measured"):
            print(f"{p}: not measured: {o['reason']}", flush=True)
        elif "poisson" in s:
            print(f"{p}: {s['kind']} {100 * s['along_peak']:+.1f} % along, {100 * s['across_at_peak']:+.2f} % across: "
                  f"Poisson's ratio {s['poisson']:.2f} +- {s['poisson_std']:.2f} ({s['points']} points)", flush=True)
        elif s["kind"] == "bend":
            print(f"{p}: bend: its line bows {abs(s['bow_over_width']):.2f} of its width at most "
                  f"({100 * abs(s['bow_over_span']):.1f} % of the {100 * s['span_over_length']:.0f} % of its length "
                  f"the points cover), {abs(s['bow_end_over_width']):.2f} in the end", flush=True)
        elif s["kind"] == "twist":
            print(f"{p}: twist {abs(s['twist_peak_deg']):.0f} deg at most, {abs(s['twist_end_deg']):.0f} in the end "
                  f"(widths within {s['width_rms_px']:.1f} px)", flush=True)
        else:
            print(f"{p}: shear {s['shear_seen_deg']:+.1f} deg across the line of sight", flush=True)


if __name__ == "__main__":
    main()
