"""A body that swings from a pivot (meta.json "motion": "pendulum").

    envs/recon/bin/python -m calibrated_scene_reconstruction.physics.video.fit real/<name>     (fit.py hands "pendulum" recordings to run() here)

A pendulum is a clock that says how long it is: theta'' = -(g / L) sin(theta) - c theta' - b sign(theta'). Its
period gives L in metres whatever the camera, and with L the scene's scale: the bob moves on a circle of radius L
in a vertical plane about the pivot, and only one distance from the camera makes that circle the arc the video
shows. What decays the swing is the damping: c (1/s, a torque in proportion to the angular velocity: air at low
speed, a bearing's grease) and b (rad/s^2, a constant torque against the motion: a dry bearing). Neither needs the
mass; a twin needs them as a joint's damping and friction per unit of m L^2.

The model in the camera frame (x right, y down, z forward), d the direction of gravity there (perception: minus
the normal of a level surface, meta.json "level" or the support the agent named; with neither in view, a
pendulum before a wall, the swing itself says which way is down: an arc that hangs is its own mirror image about
the plumb line through its pivot, so the line across which the track is symmetric is gravity's in the picture
(down_in_picture), the camera is taken to look level, and the fit turns that line as the dynamics ask):

    pivot       the pixel it shows at and its distance D along that pixel's ray
    plane       the vertical plane through the pivot at azimuth phi about d
    bob         pivot + L (sin(theta) h(phi) + cos(theta) d)
    theta(t)    the equation above from (theta0, omega0) at the first free frame, integrated (RK4)

fitted to the mask's centroid in every free frame, in pixels. D and L scale together in the image; the dynamics
fix L, so D follows.
"""
from calibrated_scene_reconstruction.physics.common.schema import read_document
import json
from pathlib import Path

import numpy as np

from calibrated_scene_reconstruction.physics.common.observations import mask_stats
from .scene import G, clock_rel, correlation_time, unusable


def swing(theta0, omega0, L, c, b, t, sub=4):
    """theta (N,) at times t (N,, rising from 0) of the damped pendulum, by RK4 with `sub` steps per sample."""
    from math import sin, tanh

    w2 = G / L

    def acc(th, om):
        return -w2 * sin(th) - c * om - b * tanh(om / 0.02)

    out = [float(theta0)]
    th, om = float(theta0), float(omega0)
    tt = [float(x) for x in t]
    for i in range(1, len(tt)):
        h = (tt[i] - tt[i - 1]) / sub
        for _ in range(sub):
            k1t, k1o = om, acc(th, om)
            k2t, k2o = om + 0.5 * h * k1o, acc(th + 0.5 * h * k1t, om + 0.5 * h * k1o)
            k3t, k3o = om + 0.5 * h * k2o, acc(th + 0.5 * h * k2t, om + 0.5 * h * k2o)
            k4t, k4o = om + h * k3o, acc(th + h * k3t, om + h * k3o)
            th += h / 6 * (k1t + 2 * k2t + 2 * k3t + k4t)
            om += h / 6 * (k1o + 2 * k2o + 2 * k3o + k4o)
        out.append(th)
    return np.array(out)


def bob(K, d, pivot_uv, D, phi, L, theta):
    """Pixels (N, 2) and camera-frame points (N, 3) of the bob at angles theta."""
    K = np.asarray(K, float)
    ray = np.array([(pivot_uv[0] - K[0, 2]) / K[0, 0], (pivot_uv[1] - K[1, 2]) / K[1, 1], 1.0])
    P0 = D * ray / np.linalg.norm(ray)
    e1 = np.cross(d, [0.0, 0.0, 1.0])
    e1 /= np.linalg.norm(e1)  # horizontal, across the view
    e2 = np.cross(d, e1)      # horizontal, along the view
    h = np.cos(phi) * e1 + np.sin(phi) * e2
    X = P0 + L * (np.outer(np.sin(theta), h) + np.outer(np.cos(theta), d))
    return np.stack([K[0, 0] * X[:, 0] / X[:, 2] + K[0, 2], K[1, 1] * X[:, 1] / X[:, 2] + K[1, 2]], 1), X


def period_of(t, x):
    """The period of a signal that swings about its mean: the median time from one zero crossing to the next but
    one (None under 3 crossings: a swing and a half is the least that times itself)."""
    x = x - np.mean(x)
    k = np.nonzero((x[:-1] < 0) != (x[1:] < 0))[0]
    if len(k) < 3:
        return None
    tc = t[k] - x[k] * (t[k + 1] - t[k]) / (x[k + 1] - x[k])
    return float(np.median(tc[2:] - tc[:-2]))


def down_in_picture(uv):
    """Gravity's direction in the picture (unit (2,)) from a swing's track uv (N, 2) alone: the track is its own
    mirror image about the plumb line, so that line is the narrow one of its two principal directions, and down
    is the way from the ends of the swing (its highest points) to its middle. None for a track that is no arc."""
    uv = np.asarray(uv, float)
    uv = uv[np.all(np.isfinite(uv), 1)]
    if len(uv) < 20:
        return None
    c = uv.mean(0)
    w, v = np.linalg.eigh(np.cov((uv - c).T))
    if w[0] < 1e-9 or w[1] < 9 * w[0] and w[1] < 4.0:  # a point, or a smudge
        return None
    x, y = (uv - c) @ v[:, 1], (uv - c) @ v[:, 0]
    ends = np.abs(x) > np.percentile(np.abs(x), 80)
    mid = np.abs(x) < np.percentile(np.abs(x), 30)
    s = np.median(y[mid]) - np.median(y[ends])
    if abs(s) < 1e-6:
        return None
    return v[:, 0] * np.sign(s)


def turned(d, roll):
    """d (3,) turned by `roll` (rad) about the camera's line of sight."""
    c, s_ = np.cos(roll), np.sin(roll)
    return np.array([c * d[0] - s_ * d[1], s_ * d[0] + c * d[1], d[2]])


def fit(t, uv, K, d, most=1500, roll=False, near=None):
    """The model through pixels uv (N, 2) at times t (N,) (NaN rows are skipped). Returns a dict or None.
    roll: d is only a first guess of gravity's direction about the line of sight (down_in_picture), and the fit
    turns it (p[9], rad). near: how far from the camera the bob was seen (m), where that is known: the first
    distance the fit starts from (an arc is drawn nearly as well from further off, on a plane turned further
    from the picture, and which of the two a fit lands on hangs on where it starts)."""
    from scipy.optimize import least_squares

    ok = np.all(np.isfinite(uv), 1)
    if ok.sum() < 20:
        return None
    t = np.asarray(t, float) - t[0]
    # along the swing: the track's first principal direction in the image
    c0 = uv[ok].mean(0)
    w, v = np.linalg.eigh(np.cov((uv[ok] - c0).T))
    x = (uv[ok] - c0) @ v[:, -1]
    T = period_of(t[ok], x)
    if T is None:
        return None
    L0 = G * (T / (2 * np.pi)) ** 2
    # the pivot: straight up the image of gravity from where the bob hangs, a radius away. The bob hangs at the
    # middle of the swing (the track's point closest to the image's "down" end)
    K = np.asarray(K, float)
    down = np.array([d[0] * K[0, 0], d[1] * K[1, 1]])
    down /= max(np.linalg.norm(down), 1e-9)
    low = uv[ok][np.argmax(uv[ok] @ down)]
    amp_px = 0.5 * np.ptp(x)
    lo = [-np.inf, -np.inf, 0.05, -1.2, 0.5 * L0, -3.1, -20.0, 0.0, 0.0, -0.35 if roll else -1e-6]
    hi = [np.inf, np.inf, 100.0, 1.2, 2.0 * L0, 3.1, 20.0, 5.0, 5.0, 0.35 if roll else 1e-6]
    xs = [20, 20, 0.2, 0.2, 0.02, 0.2, 0.5, 0.02, 0.02, 0.02]
    seen = np.nonzero(ok)[0]

    lo, hi, xs = np.array(lo), np.array(hi), np.array(xs)

    def solve(p0, upto, pivot=None):
        """The model through the frames before time `upto` (at most `most` of them), from p0. pivot: (its pixel
        (2,), how far from it the fit may put it, px)."""
        k = seen[t[seen] <= upto]
        k = k[:: max(1, int(np.ceil(len(k) / most)))]
        lo_, hi_ = lo.copy(), hi.copy()
        if pivot is not None:
            lo_[:2], hi_[:2] = pivot[0] - pivot[1], pivot[0] + pivot[1]

        def res(p):
            th = swing(p[5], p[6], p[4], p[7], p[8], t[: k[-1] + 1])
            return (bob(K, turned(d, p[9]), p[:2], p[2], p[3], p[4], th[k])[0] - uv[k]).ravel()

        return least_squares(res, np.clip(p0, lo_ + 1e-9, hi_ - 1e-9), bounds=(lo_, hi_), loss="soft_l1", f_scale=2.0,
                             x_scale=xs)

    # a long swing is fitted from its start outwards: over a hundred periods a length that is a little off slips
    # a whole swing and the fit settles there. The first few periods say how long the pendulum is, the next ones
    # refine that, and only then does the decay over the whole video have a say.
    # Where the pivot is in the image is what the widest swings show, the first ones, and it is held from there
    # on: a small arc is drawn as well by a pivot further up, a longer radius and a plane turned further from the
    # image, and the hundreds of late frames, each within a few pixels whichever way, moved the pivot 40 px off
    # the bearing the rod hangs from (and the distance by 15 %) when they had a say in it. With the pivot held
    # (to 3 % of the swing's radius in the image), where the bob hangs gives the radius and how far it swings
    # sideways gives the plane
    best, first = None, min(8 * T, t[-1])
    # the swing's radius in pixels: a wide swing or a narrow one, or one let go from near the level of its pivot
    as_seen = (K[0, 0] * L0 / near,) if near and K[0, 0] * L0 / near > 1.02 * amp_px else ()
    for rho in as_seen + (1.2 * amp_px, 2.5 * amp_px, 6.0 * amp_px, 1.02 * amp_px):
        D0 = K[0, 0] * L0 / rho
        th0 = float(np.arcsin(np.clip(x[0] / rho, -0.99, 0.99)))
        for sgn in (1.0, -1.0):
            for L_ in (L0, 0.85 * L0):  # a wide swing is slower than its length says: the true one is shorter
                p0 = np.r_[low - rho * down, D0 * L_ / L0, 0.0, L_, sgn * th0, 0.0, 0.02, 0.0, 0.0]
                try:
                    r = solve(p0, first)
                except ValueError:
                    continue
                if best is None or r.cost < best.cost:
                    best = r
                if 1.2 * np.sqrt(np.median((best.fun.reshape(-1, 2) ** 2).sum(1))) < 1.5:  # this start found it
                    break
            else:
                continue
            break
        else:
            continue
        break
    if best is None:
        return None
    start = best

    def outwards(pivot):
        b_, upto = start, first
        while upto < t[-1]:
            upto = min(2.5 * upto, t[-1])
            b_ = solve(b_.x, upto, pivot)
        return b_

    def typical(b_):
        return 1.2 * np.sqrt(np.median((b_.fun.reshape(-1, 2) ** 2).sum(1)))

    best = outwards((start.x[:2].copy(), 0.03 * K[0, 0] * start.x[4] / start.x[2]))
    if typical(best) > 4.0:  # the first swings were not seen well enough to say where the pivot is: let it go
        free = outwards(None)
        if typical(free) < typical(best):
            best = free
    p = best.x
    J = best.jac
    # the residuals of a swing hold their sign over many frames (correlation_time): so many count as one
    tau = correlation_time(best.fun.reshape(-1, 2))
    cov = tau * float((best.fun ** 2).sum() / max(len(best.fun) - len(p), 1)) * np.linalg.pinv(J.T @ J)
    th = swing(p[5], p[6], p[4], p[7], p[8], t)
    d = turned(d, p[9])
    uv_m, X = bob(K, d, p[:2], p[2], p[3], p[4], th)
    e = np.linalg.norm(uv_m[ok] - uv[ok], axis=1)
    return {"p": p, "cov": cov, "theta": th, "uv": uv_m, "X": X, "rms_px": float(np.sqrt((e ** 2).mean())),
            "typical_px": float(1.2 * np.sqrt(np.median(e ** 2))), "period": T, "frames": int(ok.sum()),
            "gravity": d, "swing_px": float(amp_px), "correlation_time": tau}


def link_look(frame, plate, uv_pivot, uv_bob, bob_px):
    """What holds the bob (a rod, a string), for drawing it in the twin: (its width in pixels, its colour (3,)
    0..1) on one sharp frame, or None where nothing is seen between the pivot and the bob. Read across the line
    from the pivot to the bob, where the frame differs from the picture without the pendulum (`plate`)."""
    a, b = np.asarray(uv_pivot, float), np.asarray(uv_bob, float)
    d = b - a
    n = np.linalg.norm(d)
    if n < 3 * bob_px + 10:
        return None
    d /= n
    across = np.array([-d[1], d[0]])
    H, W = frame.shape[:2]
    widths, cols = [], []
    for s in np.linspace(0.25, 0.75, 9) * (n - bob_px):
        q = a + s * d + np.outer(np.arange(-15, 16), across)
        x, y = np.round(q[:, 0]).astype(int), np.round(q[:, 1]).astype(int)
        if x.min() < 0 or y.min() < 0 or x.max() >= W or y.max() >= H:
            continue
        diff = np.abs(frame[y, x].astype(float) - plate[y, x].astype(float)).sum(1)
        on = diff > max(40.0, 0.5 * diff.max())
        on &= np.abs(np.arange(-15, 16)) <= 10
        if on[13:18].any():
            widths.append(int(on.sum())), cols.append(np.median(frame[y, x][on], 0))
    if len(widths) < 3:
        return None
    return float(np.median(widths)), np.median(cols, 0) / 255.0


def released(uv, valid, contact, fast=0.1, least=3):
    """(first, last) frame of the free swing: the longest stretch in which no hand is at the bob, from where the
    bob begins to move in it (it may be held still before it is let go): two steps in a row above `fast` of the
    swing's fast steps (their 95th percentile; 1.5 px a frame for a swing that covers 15). None when there is no
    such stretch. (The swing after the last contact of three frames, which this was, lost forty seconds, the
    widest swings, where the segmenter took the bob for a hand in three frames of its second minute.)"""
    F = len(uv)
    # a hand at the bob: contact for `least` frames in a row (three, and half a second where that is more), not a
    # mask that flickered past it, nor the bob itself taken for a hand in a few frames of its swing
    held = np.zeros(F, bool)
    edge = np.nonzero(np.diff(np.r_[False, contact, False].astype(int)))[0]
    for a, b in zip(edge[::2], edge[1::2]):
        if b - a >= least:
            held[a + 1:b - 1] = True
    at = np.r_[-2, np.nonzero(held)[0], F + 1]
    j = int(np.argmax(np.diff(at)))
    start, stop = max(int(at[j]) + 2, 0), min(int(at[j + 1]) - 2, F - 1)
    k = np.arange(start, stop + 1)
    k = k[valid[k]]
    if len(k) < 10:
        return None
    step = np.linalg.norm(np.diff(uv[k], axis=0), axis=1) / np.diff(k)
    fast = fast * np.percentile(step, 95)
    go = np.nonzero((step[:-1] > fast) & (step[1:] > fast))[0]
    return (int(k[go[0]]), stop) if len(go) else None


def length_in_doubt(L, ratio, rel, said):
    """A period that is long for the size of its arc (the swing puts the bob `ratio` > 1 times as far as the scene's
    own scale does) is a long pendulum far away or a short one in slowed footage. The video cannot say which, and
    whoever watched it can: `said` is meta.json "slow_motion". (the length's parameter or None to leave the
    period's, the source of whatever else is by the video's clock).
      a number    the clock is known (1: the footage runs as it happened): the period's length stands
      "unknown"   slowed by a factor nobody states: the radius of the arc at the distance the scene's own scale
                  puts the bob, good to that scale (`rel`), and the factor the period then says
      nothing     no length: both are said, and which one it is is asked for"""
    slow = float(np.sqrt(ratio))
    if said == "unknown":
        return {"value": L / ratio, "stderr": L / ratio * rel, "source": "measured", "as_filmed": L, "slowed": slow,
                "how": "the radius of the arc the video shows, at the distance the scene's own scale puts the bob: "
                       "the footage is slowed by a factor nobody states, and by this swing's period it is about "
                       f"{slow:.1f} (the period alone would make it {L:.2f} m)"}, \
            "not taken: by the clock of footage that is slowed by a factor nobody states"
    if said is None:
        return {"value": None, "as_filmed": L, "if_slowed": L / ratio, "slowed": slow,
                "source": f"not taken: {L:.2f} m if the footage runs as it happened, {L / ratio:.2f} m (+- "
                          f"{100 * rel:.0f} %) if it is slowed {slow:.1f} times as the scene's own scale says; "
                          "meta.json \"slow_motion\": 1 or \"unknown\" says which"}, \
            "not taken: the clock is in doubt (the length's note)"
    return None, "measured"


def run(path, prior=None):
    """physics dict of a pendulum recording (module doc)."""
    from .fit import PRIOR

    path = Path(path)
    per, track = np.load(path / "perception.npz"), np.load(path / "track.npz")
    twin = read_document(path / "twin.json")
    pr = dict(PRIOR, **(prior or {}))
    out = {"motion": "pendulum", "free_phase": None, "parameters": {
        "object.mass": {"value": pr["mass"], "range": pr["mass_range"],
                        "source": "prior (meta.json)" if prior and "mass" in prior else "default",
                        "how": "not visible in a video of free motion"}},
        "physics": {"object": {"mass": pr["mass"]}}}
    K, t = np.asarray(per["K"], float), np.asarray(per["t"], float)
    uv = np.array([mask_stats(m)[0] for m in per["object"]])
    # only what the segmenter itself saw: where it lost the bob (behind a post, a streak at the bottom of a wide
    # swing) the frames perception filled in from the motion may be the rod or a shadow, and the model needs none
    valid = np.all(np.isfinite(uv), 1) & ~track["border"] & (per["score"] > 0.05)
    free = released(uv, valid, track["contact"], least=max(3, int(round(0.5 / float(np.median(np.diff(t)))))))
    if free is None:
        out["note"] = "the object never swings free in this video: nothing is measured"
        return out
    i0 = free[0]
    k = np.arange(i0, free[1] + 1)
    u = np.where(valid[k, None], uv[k], np.nan)
    # which way is down: a level surface; the support, when the agent named one and it is seen as a plane (a bob
    # over a table); else the swing itself
    meta = read_document(path / "meta.json") if (path / "meta.json").exists() else {}
    named = "support" in meta and not ("support_band" in per.files and bool(per["support_band"]))
    if "level_n" in per.files or (named and unusable(per) is None):
        n = np.asarray(per["level_n"] if "level_n" in per.files else per["plane_n"], float)
        d, roll = -np.median(n, 0), False
        d /= np.linalg.norm(d)
        down = "a level surface in view" if "level_n" in per.files else "the support, taken to be level"
    else:
        d2 = down_in_picture(u)
        if d2 is None:
            out["note"] = ("no level surface is in view and the track is no arc to read gravity's direction from: "
                           "nothing is measured")
            return out
        d, roll = np.array([d2[0], d2[1], 0.0]), True
        down = ("the swing itself: the line about which its arc is its own mirror image, turned as the dynamics "
                "ask; the camera is taken to look level (no level surface is in view)")
    # where MoGe-2 saw the bob: where the fit starts from, and what the distance it ends at is held against
    scale, seen_at = twin.get("scale", {}), None
    if "object_xyz" in per.files and len(per["object_xyz"]):
        Xs = np.asarray(per["object_xyz"], float) * float(scale.get("factor", 1.0))
        Xs = Xs[np.all(np.isfinite(Xs), 1)]
        seen_at = float(np.median(np.linalg.norm(Xs, axis=1))) if len(Xs) else None
    f = fit(t[k], u, K, d, roll=roll, near=seen_at)
    if f is None or f["typical_px"] > max(0.05 * f["swing_px"], 1.0):
        out["note"] = ("no swing is seen (under a period and a half), or the pendulum model does not explain the track"
                       + ("" if f is None else f" (typical residual {f['typical_px']:.1f} px of a swing of "
                                               f"{f['swing_px']:.0f})") + ": nothing is measured")
        return out
    small = float(2 * np.pi * np.sqrt(f["p"][4] / G))
    if f["period"] < 0.97 * small:  # a pendulum's period only grows with its swing (all 50 others: 1.00 to 1.07)
        out["note"] = (f"the swing's own period (between passes through the middle: {f['period']:.2f} s) is shorter "
                       f"than the fitted pendulum's at small swings ({small:.2f} s), which no pendulum's is: the model "
                       "is fitted to something else, nothing is measured")
        return out
    d = f["gravity"]
    p, cov = f["p"], f["cov"]
    L, c, b = float(p[4]), float(p[7]), float(p[8])
    sd = np.sqrt(np.maximum(np.diag(cov), 0.0))
    out["free_phase"] = {
        "first_frame": int(i0), "last_frame": int(k[-1]), "frames": f["frames"], "seconds": float(t[k[-1]] - t[i0]),
        "pivot_camera": bob(K, d, p[:2], p[2], p[3], 0.0, np.zeros(1))[1][0].tolist(), "pivot_uv": p[:2].tolist(),
        "pivot_distance": float(p[2]), "azimuth_deg": float(np.degrees(p[3])), "gravity_camera": d.tolist(),
        "gravity_from": down,
        "theta0_deg": float(np.degrees(p[5])), "omega0_deg": float(np.degrees(p[6])), "period_mean": f["period"],
        "period_small": float(2 * np.pi * np.sqrt(L / G)), "rms_px": f["rms_px"], "typical_px": f["typical_px"],
        "ok": True}
    out["parameters"]["pendulum.length"] = {
        # the fit's own error is of a tenth of a millimetre; the length goes as the period squared, and the period
        # is no better than the video's clock (scene.clock_rel)
        "value": L, "stderr": float(np.hypot(sd[4], 2 * clock_rel(path) * L)), "stderr_fit": float(sd[4]),
        "source": "measured",
        "how": "the swing's period and how it depends on the amplitude: theta'' = -(g / L) sin(theta); no scale and "
               "no camera enters. The length of the simple pendulum that swings like this one: of a bob on a light "
               "string its length, of a rod or a body that swings its equivalent length (shorter than to its end)"}
    # the scale the swing gives holds for a bob that carries the pendulum's mass and a clock that runs true: the
    # distance it puts the bob at against where MoGe-2 saw it
    conflict, timed = None, "measured"  # timed: what is by the video's clock, where the clock is in doubt
    if seen_at is not None:
        ratio = float(np.median(np.linalg.norm(f["X"], axis=1))) / seen_at
        if abs(np.log(ratio)) > 2.5 * np.hypot(sd[2] / p[2], float(scale.get("rel_std", 0.09))) + 0.1:
            conflict = (f"the swing puts the bob {ratio:.2f} times as far from the camera as the scene's own scale "
                        f"does ({seen_at:.2f} m): " + (
                            f"the footage is slowed about {np.sqrt(ratio):.1f} times (meta.json slow_motion), or the "
                            "scene's scale is off" if ratio > 1 else
                            f"what swings is not a bob on a light string (its mass is spread along it: the equivalent "
                            f"length is {ratio:.2f} of the way to the bob), or the arc is too small a piece of its "
                            "circle to say how far it is (a small swing, a pivot outside the picture), or the footage "
                            "is sped up")
                        + "; the scale the swing gives is not taken")
            out["warning"] = conflict
            if ratio > 1:
                out["slow_motion_suggested"] = float(np.sqrt(ratio))
                rel = float(np.hypot(sd[2] / p[2], scale.get("rel_std", 0.09)))
                given, timed = length_in_doubt(L, ratio, rel, meta.get("slow_motion"))
                if given is not None:
                    out["parameters"]["pendulum.length"] = given
    out["parameters"]["pendulum.damping_rate"] = {
        "value": c, "stderr": float(sd[7]), "source": timed, "unit": "1/s",
        "how": "the part of the swing's decay in proportion to the angular velocity (a joint's damping / (m L^2))"}
    out["parameters"]["pendulum.dry_friction"] = {
        "value": b, "stderr": float(sd[8]), "source": timed, "unit": "rad/s^2",
        "how": "the part of the decay at a constant rate (a joint's friction torque / (m L^2))"}
    out["parameters"]["scene.pivot_distance"] = {
        "value": float(p[2]), "stderr": float(sd[2]), "unit": "m", "seen_at": seen_at,
        "source": "measured" if conflict is None else "not taken: it and the scene's scale disagree (warning)",
        "how": "the distance from the camera at which a circle of the pendulum's length is the arc the video shows",
        "moge_scale": scale.get("factor")}
    # what a twin is given: the swing as the video shows it, and where the clock is in doubt the word that it is
    # that (a twin that replays the video runs by the video's clock; it is no calibration then)
    out["physics"]["pendulum"] = {"length": L, "damping_rate": c, "dry_friction": b}
    if timed != "measured":
        out["physics"]["pendulum"]["by_the_videos_clock"] = True
    return out


def describe(path, phys):
    if phys["free_phase"] is None:
        return f"{path}: {phys['note']}"
    q, fp = phys["parameters"], phys["free_phase"]
    if q["pendulum.length"]["value"] is None:
        return f"{path}: pendulum length {q['pendulum.length']['source']} WARNING: {phys['warning']}"
    return (f"{path}: pendulum length {q['pendulum.length']['value']:.3f} +- {q['pendulum.length']['stderr']:.3f} m "
            f"(period {fp['period_small']:.3f} s at small swings, {fp['period_mean']:.3f} s seen), damping "
            f"{q['pendulum.damping_rate']['value']:.4f} /s + {q['pendulum.dry_friction']['value']:.4f} rad/s^2, "
            f"released at {fp['theta0_deg']:.0f} deg, pivot {fp['pivot_distance']:.2f} m from the camera, "
            f"{fp['frames']} frames over {fp['seconds']:.0f} s, rms {fp['rms_px']:.1f} px"
            + (f" WARNING: {phys['warning']}" if phys.get("warning") else ""))
