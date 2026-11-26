"""An object on a support that something pulls along under it (a cloth drawn over a table, a mat dragged over
a floor): what its ride measures (meta.json "motion": "carried"; perception and track as for a slide). The support
moves in its own plane, which is level and stands still in the picture: a tray lifted and carried is not this
(lifting and tilting it are accelerations of their own).

Whatever the support does, the one level force on the object is the support's friction, so the object's own
acceleration over g is friction it was given: a lower bound on the static friction between the two while it rides,
and the kinetic friction itself if it slips (which this module does not decide: it does not follow the support).
A support that starts gently bounds little: a cloth brought to 4 cm/s in a third of a second shows 0.015.

The acceleration is the second derivative of a spline through the tracked places (a knot every quarter second).
The fit is linear in the places, so what the track's own jitter (its residual about the spline) makes of the
acceleration is known at every time, and the bound is the part of the acceleration four such above nothing,
lowered by twice the uncertainty of the scene's scale (it is in proportion to it). No friction is read while a
manipulator touches the object, nor a knot's width before and after: what a hand does to it is not the support's.
"""
from calibrated_scene_reconstruction.physics.common.schema import read_document
import json
from pathlib import Path

import numpy as np

from .scene import G


def spline_acceleration(tk, pk, every):
    """The mean acceleration over every stretch of `every` seconds, of a least-squares cubic spline through places
    pk (N, D) at times tk with a knot every `every` seconds, and what unit noise in the places makes of it at each
    time: (acceleration (N, D), places as fitted (N, D), gain (N,) 1/s^2). A mean over a stretch never exceeds
    the largest acceleration in it (the spline's own second derivative rings past it at a sudden start). The fit
    is linear in the places, so the gain is the norm of its rows."""
    from scipy.interpolate import BSpline

    inner = np.arange(tk[0] + every, tk[-1] - every / 2, every)
    knots = np.concatenate([[tk[0]] * 4, inner, [tk[-1]] * 4])
    basis = BSpline(knots, np.eye(len(knots) - 4), 3)
    B, B1 = basis(tk), basis.derivative(1)
    solve = np.linalg.pinv(B)
    lo, hi = np.clip(tk - every / 2, tk[0], tk[-1]), np.clip(tk + every / 2, tk[0], tk[-1])
    A = (B1(hi) - B1(lo)) / np.maximum(hi - lo, 1e-9)[:, None] @ solve
    return A @ pk, B @ (solve @ pk), np.linalg.norm(A, axis=1)


def series(path, every=0.25):
    """The object's acceleration along time, as run reads it: a dict of the times kept (tk), the places (pk) and as
    fitted (pos), the acceleration (acc (N, 2), m/s^2), what the track's jitter makes of it (noise), the part four
    such above nothing where a friction is read (sure, -inf where none is), the jitter, and the manipulator's times
    (t_touch); or a dict with a note alone when there is nothing to read."""
    path = Path(path)
    tr = np.load(path / "track.npz")
    t, xy, ok = tr["t"], tr["xy"], tr["valid"].astype(bool) & np.all(np.isfinite(tr["xy"]), 1)
    if (path / "perception.npz").exists():
        # not the frames perception filled in from what moved (fill_fast, score 0.01): under an object that rides,
        # what moves is its support, and one such frame is an acceleration that never was
        ok &= np.load(path / "perception.npz")["score"] > 0.05
    touched = tr["contact"].astype(bool) if "contact" in tr.files else np.zeros(len(t), bool)
    if touched.sum() >= 0.5 * len(t):  # a hand on it most of the time is a mask that takes in the support
        touched[:] = False
    t_touch = t[touched]
    if ok.sum() < 30 or t[ok][-1] - t[ok][0] < 6 * every:
        return {"note": "the object is not followed for long enough"}
    tk, pk = t[ok], xy[ok]
    for _ in range(3):  # a mask that was something else for a frame stands far off the path the others make
        pos = spline_acceleration(tk, pk, every)[1]
        off = np.linalg.norm(pos - pk, axis=1)
        keep = off < max(8 * 1.4826 * np.median(off), 0.02 * np.linalg.norm(pos[-1] - pos[0]))
        if keep.all() or keep.sum() < 30:
            break
        tk, pk = tk[keep], pk[keep]
    acc, pos, gain = spline_acceleration(tk, pk, every)
    jitter = float(np.sqrt(np.mean(np.sum((pos - pk) ** 2, 1)) / 2))  # of one coordinate of the track
    noise = jitter * gain  # what that jitter makes of the acceleration at each time
    sure = np.linalg.norm(acc, axis=1) - 4 * noise  # the part four such away from nothing (the largest of a hundred
    # is past three); not where the track begins, ends or breaks off: there a spline is held from one side only,
    # and an object first seen half hidden and already under way makes an acceleration of how it came into view
    dt = float(np.median(np.diff(tk)))
    breaks = np.r_[tk[0], tk[-1], tk[:-1][np.diff(tk) > 3 * dt], tk[1:][np.diff(tk) > 3 * dt]]
    sure = np.where(np.abs(tk[:, None] - breaks[None]).min(1) < every, -np.inf, sure)
    if len(t_touch):  # what the object does while a manipulator is on it, and a knot's width about that, is the
        near = np.abs(tk[:, None] - t_touch[None]).min(1) <= every  # manipulator's doing: no friction is read there
        sure = np.where(near, -np.inf, sure)
    return {"tk": tk, "pk": pk, "pos": pos, "acc": acc, "noise": noise, "sure": sure, "jitter": jitter,
            "t_touch": t_touch}


def run(path, prior=None, every=0.25):
    path = Path(path)
    s = series(path, every)
    out = {"motion": "carried", "fps": float(1.0 / np.median(np.diff(np.load(path / "track.npz")["t"]))),
           "free_phase": None, "parameters": {}}
    if "note" in s:
        return dict(out, measured=False, note=s["note"])
    tk, pos, acc, noise, sure, jitter = s["tk"], s["pos"], s["acc"], s["noise"], s["sure"], s["jitter"]
    travel = float(np.linalg.norm(pos[-1] - pos[0]))
    if travel < 0.02:
        return dict(out, measured=False, note="the object goes nowhere: nothing carried it")
    if len(s["t_touch"]) and not np.isfinite(sure).any():
        return dict(out, measured=False, note="a manipulator is on the object whenever it is carried")
    a = np.linalg.norm(acc, axis=1)
    k = int(np.argmax(sure))
    resolved = bool(np.isfinite(sure[k]) and sure[k] > 0)
    speed = np.linalg.norm(np.gradient(pos, tk, axis=0), axis=1)
    twin = read_document(path / "twin.json") if (path / "twin.json").exists() else {}
    rel_scale = float(twin.get("scale", {}).get("rel_std", 0.09))
    out["parameters"]["support.static_friction"] = {
        "value": None, "lower_bound": float(sure[k] / G * max(1 - 2 * rel_scale, 0.0)) if resolved else 0.0,
        "source": "measured" if resolved else "not resolved",
        "how": "the object's acceleration while it is carried, less four times what the track's jitter makes of "
               f"it, at its largest, over g; lowered by twice the scale's uncertainty ({100 * rel_scale:.0f} %)"}
    out["carried"] = {"travel": travel, "top_speed": float(np.percentile(speed, 98)),
                      "peak_acceleration": float(a[k]), "peak_time": float(tk[k]),
                      "acceleration_noise": float(noise[k]), "position_rms": jitter, "resolved": resolved}
    out["measured"] = resolved
    if not resolved:
        out["note"] = ("the support starts too gently for this track: its acceleration is within what the track's "
                       f"jitter makes ({np.median(noise) / G:.3f} g), and no friction is bounded")
    return out


def describe(path, phys):
    if "carried" not in phys:
        return f"{path}: {phys['note']}"
    c, p = phys["carried"], phys["parameters"]["support.static_friction"]
    return (f"{path}: carried {c['travel'] * 100:.0f} cm at up to {c['top_speed'] * 100:.1f} cm/s; static friction >= "
            f"{p['lower_bound']:.3f} (acceleration {c['peak_acceleration'] / G:.3f} g, of which the track's jitter "
            f"makes {c['acceleration_noise'] / G:.3f} g)" + ("" if c["resolved"] else ": not resolved"))
