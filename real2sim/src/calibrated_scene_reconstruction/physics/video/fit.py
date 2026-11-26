"""The free phase of a tracked object and what its motion measures (CPU).

    envs/recon/bin/python -m calibrated_scene_reconstruction.physics.video.fit real/<name>

Reads track.npz + twin.json, writes real/<name>/physics.json.

While something pushes the object its motion is the pusher's, and says little about the object. Once it is let go the
only horizontal force is friction, and a slide to rest measures the kinetic friction coefficient of object on support
without knowing the mass: v' = -mu g. `fit_slide` fits that model (start point, heading, release speed, deceleration,
and so the stop time) to the tracked positions of the free phase. What a video cannot give is said, not guessed:

  measured    support friction (kinetic), with a standard error from the fit and from the scale of the scene
  weak        a slide that leaves the frame before it stops shows its deceleration only as the curvature of its
              path: measured when that stands out of the noise, else reported as weakly constrained
  prior only  mass (no force is visible in a video: friction and inertia both scale with it, any mass reproduces a
              slide), static friction (a slide never shows the object starting to move), restitution: they stay at
              meta.json "prior" {"mass": kg, "mass_range": [lo, hi], "restitution", "static_ratio"}, which the agent
              writes from what the object is, or at the defaults below

physics.json: {"free_phase", "parameters": {name: {"value", "stderr" | "range", "source", "how"}},
               "physics": the same values in the layout of configs/<name>/physics.json (calibrated_scene_reconstruction.sim.physics.apply_scene)}
"""
from calibrated_scene_reconstruction.physics.common.schema import read_document, write_document
import argparse
import json
from pathlib import Path

import numpy as np

from .scene import G, camera_note, untimed, unusable


def speeds(t, xy, win=1):
    """Central-difference speed (F,) m/s over +- win frames; NaN where a neighbour is missing."""
    v = np.full(len(t), np.nan)
    for i in range(win, len(t) - win):
        v[i] = np.linalg.norm(xy[i + win] - xy[i - win]) / (t[i + win] - t[i - win])
    return v


def spikes(xy, valid, tol=0.02, edge_tol=0.05):
    """Frames whose position jumps away from its neighbours' and back (a wrong mask): farther than tol (m) from the
    per-axis median of the five frames around it, which for any motion in one direction is the frame itself. The
    first and last two frames have no such window: they are held against the line through their two neighbours."""
    F = len(xy)
    P = np.where(valid[:, None], xy, np.nan)
    bad = np.zeros(F, bool)
    for i in range(F):
        if not valid[i]:
            continue
        if 2 <= i < F - 2:
            w = P[i - 2:i + 3]
            if np.isfinite(w[:, 0]).sum() >= 4:
                bad[i] = np.linalg.norm(P[i] - np.nanmedian(w, 0)) > tol
        else:
            a, b = (P[i + 1], P[i + 2]) if i < 2 else (P[i - 1], P[i - 2])
            if np.isfinite(a).all() and np.isfinite(b).all():
                bad[i] = np.linalg.norm(P[i] - (2 * a - b)) > edge_tol
    return bad


def free_phase(t, xy, valid, contact, rest_speed=0.03):
    """(first, last) frame of the free phase: after the last frame the manipulator touched the object and not before
    the speed peaks (a pusher the masks missed is still accelerating the object), up to the end of the video.
    None when the object never moves on its own."""
    F = len(t)
    start = int(np.nonzero(contact)[0][-1]) + 1 if contact.any() else 0
    v = speeds(t, np.where(valid[:, None], xy, np.nan))
    if not np.isfinite(v[start:]).any():
        return None
    peak = start + int(np.nanargmax(v[start:]))
    if not v[peak] > rest_speed:
        return None
    first = max(start, peak - 1)
    while first < F and not valid[first]:
        first += 1
    return (first, F - 1) if F - first >= 5 else None


def free_phases(t, xy, valid, contact, rest_speed=0.03, min_speed=0.15, min_dist=0.03, min_frames=5, gap=2):
    """Every free phase of a video, [(first, last)]: a hand may launch the object several times. A phase is a run of
    frames in which the object is tracked and nothing touches it (holes of up to `gap` untracked frames are bridged),
    cut to start at its speed peak (free_phase), in which the object goes faster than min_speed and farther than
    min_dist: a run where it only sits there is not a phase."""
    F = len(t)
    free = valid & ~contact
    # bridge short tracking holes that no contact explains
    i = 0
    while i < F:
        if not free[i]:
            j = i
            while j < F and not free[j]:
                j += 1
            if 0 < i and j < F and j - i <= gap and not contact[i:j].any():
                free[i:j] = True
            i = j
        else:
            i += 1
    out, i = [], 0
    while i < F:
        if not free[i]:
            i += 1
            continue
        j = i
        while j + 1 < F and free[j + 1]:
            j += 1
        k = np.arange(i, j + 1)
        ph = free_phase(t[k], xy[k], valid[k], np.zeros(len(k), bool), rest_speed) if len(k) >= min_frames else None
        if ph is not None:
            a = i + ph[0]
            kk = np.arange(a, j + 1)
            kk = kk[valid[kk]]
            v = speeds(t[kk], xy[kk]) if len(kk) >= 3 else np.array([np.nan])
            if (len(kk) >= min_frames and np.nanmax(v) > min_speed
                    and np.linalg.norm(xy[kk[-1]] - xy[kk[0]]) > min_dist):
                out.append((int(a), int(j)))
        i = j + 1
    return out


def slide_positions(p, t):
    """Positions (N, 2) of the model at times t (from the first free frame): p = (x0, y0, heading rad, v0, a)."""
    x0, y0, th, v0, a = p
    a = max(a, 1e-6)
    tau = np.clip(t, 0.0, max(v0, 0.0) / a)
    s = v0 * tau - 0.5 * a * tau ** 2
    return np.stack([x0 + s * np.cos(th), y0 + s * np.sin(th)], 1)


def fit_slide(t, xy, weights=None):
    """Coulomb slide to rest through positions xy (N, 2) at times t (N,). Returns a dict: p (x0, y0, heading, v0, a),
    cov (5, 5), rms (m), stop_time (s, from t[0]), stopped (the stop lies inside the data, with frames at rest
    after it), distance (m)."""
    from scipy.optimize import least_squares

    t = np.asarray(t, float) - t[0]
    xy = np.asarray(xy, float)
    w = np.ones(len(t)) if weights is None else np.asarray(weights, float)
    d = xy[-3:].mean(0) - xy[0]
    D = max(np.linalg.norm(d), 1e-3)
    best = None
    for frac in (0.3, 0.5, 0.7, 1.0):  # the stop time is the nonlinear parameter: several starts
        Ts = frac * t[-1]
        p0 = [xy[0, 0], xy[0, 1], np.arctan2(d[1], d[0]), 2 * D / Ts, 2 * D / Ts ** 2]
        r = least_squares(lambda p: ((slide_positions(p, t) - xy) * w[:, None]).ravel(), p0, loss="soft_l1",
                          f_scale=0.01, x_scale=[0.1, 0.1, 0.1, 1.0, 1.0])
        if best is None or r.cost < best.cost:
            best = r
    p = best.x
    res = slide_positions(p, t) - xy
    dof = max(res.size - 5, 1)
    s2 = float((res ** 2).sum() / dof)
    J = best.jac
    cov = s2 * np.linalg.pinv(J.T @ J)
    stop = p[3] / max(p[4], 1e-6)
    # the other way to come to rest: at full speed into something. Same number of parameters; whichever explains
    # the track better says what stopped the object
    def halted(q):
        s_ = q[3] * np.clip(t, 0.0, max(q[4], 0.0))
        return np.stack([q[0] + s_ * np.cos(q[2]), q[1] + s_ * np.sin(q[2])], 1)

    alt = None
    for frac in (0.3, 0.5, 0.7, 1.0):
        Ts = frac * t[-1]
        r = least_squares(lambda q: ((halted(q) - xy) * w[:, None]).ravel(),
                          [xy[0, 0], xy[0, 1], np.arctan2(d[1], d[0]), D / Ts, Ts], loss="soft_l1", f_scale=0.01,
                          x_scale=[0.1, 0.1, 0.1, 1.0, 0.1])
        if alt is None or r.cost < alt.cost:
            alt = r
    rms_halt = float(np.sqrt(((halted(alt.x) - xy) ** 2).sum(1).mean()))
    # and the way to lose speed that is no friction: all at once, on meeting something. Along the path, one steady
    # speed up to a moment and another after it, against the parabola of a steady deceleration
    along = np.array([np.cos(p[2]), np.sin(p[2])])
    s, rms_struck = (xy - xy[0]) @ along, np.inf
    for j in range(3, len(t) - 3):
        A = np.stack([np.ones(len(t)), np.minimum(t, t[j]), np.maximum(t - t[j], 0.0)], 1)
        c, *_ = np.linalg.lstsq(A, s, rcond=None)
        rms_struck = min(rms_struck, float(np.sqrt(((A @ c - s) ** 2).mean())))
    return {"p": p, "cov": cov, "rms": float(np.sqrt((res ** 2).sum(1).mean())), "stop_time": float(stop),
            "rms_halted": rms_halt, "rms_struck": rms_struck, "rms_along": float(np.sqrt(((res @ along) ** 2).mean())),
            "stopped": bool(stop < t[-1] - 2 * np.median(np.diff(t))),
            "distance": float(p[3] ** 2 / (2 * max(p[4], 1e-6)))}


PIXELS = 1.5  # px: what a tracked centroid is good for
# two steady speeds explain a steady deceleration's parabola no better than the track's own noise and a broken line
# explains a collision to it: a break is taken for one where it leaves this part of the parabola's residual or less
STRUCK = 0.6

PRIOR = {"mass": 0.3, "mass_range": [0.02, 3.0], "restitution": 0.05, "static_ratio": 1.0 / 0.85}


MOTIONS = {  # meta.json "motion" -> the parameter a free deceleration measures
    "slide": ("support.kinetic_friction", "deceleration of the free slide / g"),
    "roll": ("support.rolling_resistance", "deceleration of the free roll / g (for a rolling body the coefficient "
                                           "of rolling resistance times a shape factor of order 1)"),
}


def run(track, twin, prior=None, motion="slide"):
    """physics dict of one recording (module doc)."""
    pr = dict(PRIOR, **(prior or {}))
    given = set(prior or {})

    def src(k):
        return "prior (meta.json)" if k in given else "default"

    t, xy = track["t"], track["xy"]
    valid = track["valid"] & ~spikes(xy, track["valid"])
    out = {"free_phase": None, "free_phases": [], "motion": motion, "parameters": {
        "object.mass": {"value": pr["mass"], "range": pr["mass_range"], "source": src("mass"),
                        "how": "not visible in a video of free motion"},
        "object.restitution": {"value": pr["restitution"], "source": src("restitution"),
                               "how": "no impact in this video"}}}
    out["physics"] = {"object": {"mass": pr["mass"], "restitution": pr["restitution"]}}
    h_cam, f_px = float(twin["camera_height"]), float(np.asarray(twin["K"])[0][0])
    rel_scale = float(twin.get("scale", {}).get("rel_std", twin["height_std"] / twin["camera_height"]))
    for i0, i1 in free_phases(t, xy, valid, track["contact"]):
        k = np.arange(i0, i1 + 1)
        k = k[valid[k] & ~track["border"][k]]
        if len(k) < 5:
            continue
        f = fit_slide(t[k], xy[k])
        a, sa = float(f["p"][4]), float(np.sqrt(max(f["cov"][4, 4], 0.0)))
        v0, seen = float(f["p"][3]), float(t[k[-1]] - t[k[0]])
        dt = float(np.median(np.diff(t)))
        # what one pixel is on the plane where the object went (along the line of sight; a grazing view sees little)
        r = np.linalg.norm(xy[k], axis=1)
        px = float(np.median((r ** 2 + h_cam ** 2) / (h_cam * f_px)))
        moved = float(np.linalg.norm(xy[k[-1]] - xy[k[0]]))
        why = ("the model does not explain the track" if not f["rms"] < max(0.01, 0.05 * f["distance"])
               else "no deceleration" if not (a > 0 and f["distance"] > 0.03)
               else "stopped at once, as by an obstacle, not by friction" if f["stopped"] and f["rms_halted"] < f["rms"]
               else "its speed changes at once, as on meeting something, and is steady before and after: no friction"
               if f["rms_struck"] < STRUCK * f["rms_along"]
               else "grazing view: a pixel is too much of the path" if PIXELS * px > 0.15 * moved
               else "implausible (a deceleration above 2 g, or a launch above 10 m/s)" if a > 2 * G or v0 > 10
               else "too short to time" if min(f["stop_time"], seen) < 4 * dt
               else "lost too little speed in view" if not f["stopped"] and a * seen < 0.3 * v0
               else None)
        out["free_phases"].append({
            "first_frame": int(i0), "last_frame": int(i1), "frames": int(len(k)), "t0": float(t[i0]),
            "release_speed": v0, "heading_deg": float(np.degrees(f["p"][2])),
            "distance": f["distance"], "stop_time": f["stop_time"], "stopped": f["stopped"], "rms": f["rms"],
            "start_xy": [float(f["p"][0]), float(f["p"][1])], "coefficient": a / G, "stderr_fit": sa / G,
            "metres_per_pixel": px, "rms_halted": f["rms_halted"], "rms_struck": f["rms_struck"],
            "rms_along": f["rms_along"], "ok": why is None, "rejected": why})
    good = [p for p in out["free_phases"] if p["ok"]]
    if not good:
        out["note"] = ("the object never moves on its own in this video: nothing is measured" if not out["free_phases"]
                       else "no free phase measures a deceleration (" + "; ".join(sorted({p["rejected"] for p in
                            out["free_phases"]})) + "): nothing is measured")
        return out
    # a slide the video sees to its end fixes the deceleration by its stop time; one that leaves the frame (or is
    # caught) still shows it as the curvature of its path, but only as well as that curvature stands out of the noise
    done = [p for p in good if p["stopped"]] or [p for p in good if p["stderr_fit"] < 0.2 * p["coefficient"]] or good
    c = np.array([p["coefficient"] for p in done])
    w = np.array([p["distance"] for p in done])  # a longer slide weighs more: its timing is better resolved
    mu = float((c * w).sum() / w.sum())
    spread = float(np.sqrt((w * (c - mu) ** 2).sum() / w.sum())) if len(c) > 1 else 0.0
    fit_se = float(np.sqrt((w ** 2 * np.array([p["stderr_fit"] for p in done]) ** 2).sum()) / w.sum())
    stat = max(fit_se, spread / np.sqrt(len(c)))
    out["free_phase"] = max(done, key=lambda p: p["distance"])
    name, how = MOTIONS[motion]
    out["parameters"][name] = {
        "value": mu, "stderr": float(np.hypot(stat, mu * rel_scale)),
        "source": ("measured" if done[0]["stopped"] else "measured (motion not seen to its end)"
                   if done[0]["stderr_fit"] < 0.2 * done[0]["coefficient"] else "weakly constrained"),
        "phases": len(done),
        "stderr_fit": fit_se, "spread": spread, "stderr_scale": mu * rel_scale,
        "how": f"{how}; scales with the camera height above the support "
               f"({twin.get('scale', {}).get('source', 'MoGe-2 metric depth')})"}
    if motion == "slide":
        out["parameters"]["support.static_friction"] = {
            "value": mu * pr["static_ratio"], "source": src("static_ratio"),
            "how": "kinetic friction x the prior static / kinetic ratio: the video never shows the object starting "
                   "to move"}
        out["physics"]["support"] = {"static_friction": mu * pr["static_ratio"], "dynamic_friction": mu}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording", nargs="+")
    args = ap.parse_args()
    for r in args.recording:
        if not (Path(r) / "track.npz").exists():  # an earlier stage failed on it: the others go on
            print(f"{r}: no track.npz, skipped", flush=True)
            continue
        r = Path(r)
        meta = read_document(r / "meta.json")
        motion = meta.get("motion", "slide")
        per = np.load(r / "perception.npz")
        # a pendulum hangs over no support; an incline seen along itself is read from the object's path
        # (incline.py); a turntable's plane is that of its own outline or of the circle the object rides (track.py)
        why = unusable(per, plane=motion not in ("pendulum", "incline", "turntable")) or untimed(meta)
        if why is not None:
            write_document(r / "physics.json",
                {"motion": motion, "measured": False, "free_phase": None, "parameters": {}, "note": why}, indent=1)
            print(f"{r}: {why}")
            continue
        phys = fit_one(r, meta)
        if camera_note(per) and isinstance(phys, dict):
            phys["camera_note"] = camera_note(per)
            write_document(r / "physics.json", phys, indent=1)


def fit_one(r, meta):
    """One recording's fit by its motion: physics.json written, a line printed; the physics dict returned."""
    prior = meta.get("prior")
    if meta.get("motion") == "drop":  # a flight: gravity's scale and the restitution (ballistic.py)
        from . import ballistic

        # the fit sets the scene's scale by gravity and reads its lengths at the scale it found: once more at
        # the new scale, until the scale stays (a second pass, as a rule)
        for _ in range(4):
            was = read_document(r / "twin.json").get("scale", {}).get("factor")
            phys, twin, track = ballistic.run(r, prior)
            write_document(r / "physics.json", phys, indent=1)
            write_document(r / "twin.json", twin, indent=1)
            np.savez_compressed(r / "track.npz", **track)
            now = twin.get("scale", {}).get("factor")
            if not was or not now or abs(now / was - 1) < 1e-6:
                break
        print(ballistic.describe(r, phys))
        return phys
    # riding a support that turns (turntable.py); a swing (pendulum.py); a support pulled along under the object
    # (carried.py); a support that is not level (incline.py)
    if meta.get("motion") in ("turntable", "pendulum", "carried", "incline"):
        import importlib

        mod = importlib.import_module(f".{meta['motion']}", __package__)
        phys = mod.run(r, prior)
        print(mod.describe(r, phys))
        if meta.get("slow_motion") == "unknown" and meta["motion"] == "incline":
            # what the tilt alone says stays (the bounds on static friction); what the slide's acceleration says,
            # and what was made of that, is by a clock nobody knows
            why = ("not taken: the footage is slowed by a factor nobody states, and a slide's acceleration is by "
                   "its clock")
            par = phys.get("parameters", {})
            if "support.kinetic_friction" in par:
                par["support.kinetic_friction"] = {"value": None, "source": why}
            m = par.get("support.static_friction", {})
            if m.get("source") == "bounded" and m.get("value") is not None:  # a guess from the kinetic one
                m.pop("value")
            phys.get("physics", {}).pop("support", None)
            print(f"{r}: {why[11:]}: kinetic friction is not given")
        write_document(r / "physics.json", phys, indent=1)
        return phys
    phys = run(np.load(r / "track.npz"), read_document(r / "twin.json"), prior, meta.get("motion", "slide"))
    write_document(r / "physics.json", phys, indent=1)
    if phys["free_phase"] is None:
        print(f"{r}: {phys['note']}")
        return phys
    name = MOTIONS[phys["motion"]][0]
    m = phys["parameters"][name]
    ph = " ".join(f"[{p['first_frame']}-{p['last_frame']}: {p['release_speed']:.2f} m/s, {p['distance']:.2f} m, "
                  f"{p['coefficient']:.3f}{'' if p['ok'] else ' rejected'}{'' if p['stopped'] else ' unfinished'}]"
                  for p in phys["free_phases"])
    print(f"{r}: {name.split('.')[1]} {m['value']:.3f} +- {m['stderr']:.3f} ({m['source']}, {m['phases']} "
          f"phase{'s' if m['phases'] > 1 else ''}) {ph}")
    return phys


if __name__ == "__main__":
    main()
