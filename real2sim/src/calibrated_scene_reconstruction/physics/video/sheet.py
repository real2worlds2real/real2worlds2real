"""A sheet that hangs from a straight edge something carries: what its trailing and its swing measure (meta.json
"motion": "sheet"; perception by vidphys.soft).

    envs/recon/bin/python -m calibrated_scene_reconstruction.physics.video.sheet real/<name> [...]        # physics.json

A cloth held by its top edge and carried sideways trails behind and, when the edge stops, swings and comes to rest.
Two forces move it, both known: gravity and the air. The air resists a wide thin body twice over: a drag on its
speed, and the air that has to move with it, which for a strip 30 cm wide weighs as much as the cloth itself does
at 280 g/m^2. So a light cloth swings slowly and dies quickly and a heavy one swings as a chain in a vacuum would,
and how slowly this one does is its mass per area. A video shows no force and still weighs the cloth, because the
air is a material whose density is known. A cloth much heavier than the air it moves swings as any heavy cloth
does: for it the video gives a lower bound.

The model is vidphys.beam's chain, hanging (its axis down), its clamp carried along the level direction the video
shows, in air (beam.Chain.swing, carry and air): a strip as wide as the cloth, bending stiffness of a fabric's
order (it does not matter here: the fit is the same over a decade of it).

What the picture has to show: the cloth hanging still, then carried, then at rest; its top edge straight and
level. Where the edge is in depth is read off its own length in the picture (an edge half as long is twice as far),
from the depth MoGe-2 gives it at rest.

meta.json: "prior": {"areal_density": kg/m^2} (what the cloth looks like; default 0.2) is where the search starts.
"""
from calibrated_scene_reconstruction.physics.common.schema import read_document, write_document
import argparse
import json
from pathlib import Path

import numpy as np

from . import beam

THICK = 5e-4  # m: the chain's own thickness; with MODULUS the bending stiffness of a fabric, 1e-5 N m
MODULUS = 1e6
ELEMENTS = 8
WIDE = 3.0  # cloths that fit as well as the best may differ by this factor for the best to count as a weight
LEAST_OVERLAP = 0.7  # of the fitted sheet's silhouettes with the masks, for the fit to say anything
CARRY_OFF = 0.12  # how far, as a part of the way (median), the carried edge may stand off one smooth move
LIGHTEST, HEAVIEST = 0.02, 20.0  # kg/m^2: what the search holds (tissue to a rubber mat)
# what the weighing is good to beyond its own fit, as a factor exp(AIR_REL) either way: the cloth is weighed against
# the air it moves, by the drag and the added mass of a flat plate (a cloth billows) and in proportion to the
# scene's size. On five fabrics the reading stood 0.7 to 1.9 times what they weigh
AIR_REL = 0.35
# the air that moves with a flat plate moved across itself, over rho pi/4 a^2 b (a its shorter side): by b / a
ADDED = ([1.0, 1.5, 2.0, 3.0, 5.0, 10.0], [0.579, 0.690, 0.757, 0.840, 0.90, 0.95])


def top_edge(mask):
    """The upper edge of a hanging thing's mask: (left end (2,), right end (2,)) px. The edge is straight but not
    level in the picture when one end of it is nearer the camera: it is the line through the mask's topmost pixel
    of every column, fitted to the middle of them and followed outwards as far as they stay on it."""
    cols = np.nonzero(mask.any(0))[0]
    if len(cols) < 12:
        return None
    top = np.argmax(mask[:, cols], 0).astype(float)
    n = len(cols)
    use = np.zeros(n, bool)
    use[int(0.2 * n):int(0.8 * n)] = True
    for _ in range(4):
        k, d = np.polyfit(cols[use], top[use], 1)
        r = top - (k * cols + d)
        ok = np.abs(r) < max(2.0, 2.5 * np.std(r[use]))
        mid = n // 2
        lo, hi = mid, mid
        while lo > 0 and ok[max(lo - 3, 0):lo].any():  # a pixel or two off the line is the mask's own roughness
            lo -= 1
        while hi < n - 1 and ok[hi + 1:hi + 4].any():
            hi += 1
        use = np.zeros(n, bool)
        use[lo:hi + 1] = ok[lo:hi + 1]
        if use.sum() < 8:
            return None
    k, d = np.polyfit(cols[use], top[use], 1)
    u0, u1 = float(cols[lo]), float(cols[hi])
    return np.array([u0, k * u0 + d]), np.array([u1, k * u1 + d])


def top_edge_towards(mask, vanish, band=2.5):
    """The upper edge of a hanging thing's mask when the edge's direction in space is known: every place of an edge
    of one direction points at one vanishing point (x, y, w) of the picture. Of the lines through that point, the
    edge is the one most of the mask's upper outline (its topmost pixel in every column) lies on; its ends are
    where the outline leaves it. A stub of rod past the cloth, or cloth flung above its own edge, is off that line
    and does not move it. (left end, right end) px or None."""
    cols = np.nonzero(mask.any(0))[0]
    if len(cols) < 12:
        return None
    P = np.stack([cols, np.argmax(mask[:, cols], 0)], 1).astype(float)
    if abs(vanish[2]) > 1e-9 * np.linalg.norm(vanish[:2]):
        v = vanish[:2] / vanish[2]
        d = P.mean(0) - v
        reach = np.linalg.norm(d)
        d = d / reach
        side = np.array([-d[1], d[0]])
        off = np.arctan2((P - v) @ side, (P - v) @ d) * reach  # each point's bearing, as pixels across at the mask
    else:  # the edge is parallel to the picture: lines of one direction
        d = vanish[:2] / np.linalg.norm(vanish[:2])
        side, v, reach = np.array([-d[1], d[0]]), np.zeros(2), None
        off = P @ side
    bins = np.round(off / band).astype(int)
    vals, counts = np.unique(bins, return_counts=True)
    best = vals[np.argmax(counts + np.roll(counts, 1) * 0.5 + np.roll(counts, -1) * 0.5)]
    on = np.abs(bins - best) <= 1
    if on.sum() < 8:
        return None
    o = float(np.median(off[on]))
    line_d = d * np.cos(o / reach) + side * np.sin(o / reach) if reach is not None else d
    origin = v if reach is not None else side * o
    along = (P[on] - origin) @ line_d
    lo, hi = np.percentile(along, [2, 98])
    ends = np.stack([origin + lo * line_d, origin + hi * line_d])
    return tuple(ends[np.argsort(ends[:, 0])])


def smooth(t, tk, yk, every=0.25):
    """A path through noisy places yk at times tk, at times t: a cubic spline with a knot every `every` seconds,
    fitted by least squares. What carries a sheet moves smoothly; the edge read off a mask jitters by a centimetre
    from frame to frame, and the swing answers to the path's second derivative."""
    from scipy.interpolate import LSQUnivariateSpline

    knots = np.arange(tk[0] + every, tk[-1] - every / 2, every)
    if len(tk) < len(knots) + 6:
        return np.interp(t, tk, yk)
    return LSQUnivariateSpline(tk, yk, knots, k=3)(np.clip(t, tk[0], tk[-1]))


def carry_path(t, y):
    """A carry from one place to another and no further, through noisy places y (0 where it starts, 1 where it
    ends): the smoothest such move (least jerk: 10 x^3 - 15 x^4 + 6 x^5) between the two times that fit best.
    The edge read off a sheet that billows is rough while it travels, and what the swing answers to is the
    carry's acceleration; what carries a sheet and sets it down does it in one move. Returns ((start, stop) s,
    the path at t, how far the places stand off it, as a part of the way: the median, since single places are
    rough)."""
    from scipy.optimize import least_squares

    def path(p):
        x = np.clip((t - p[0]) / max(p[1] - p[0], 1e-3), 0.0, 1.0)
        return 10 * x ** 3 - 15 * x ** 4 + 6 * x ** 5

    half = float(t[np.argmax(y > 0.5)])
    best = None
    for span in (0.6, 1.2, 2.5, 5.0):
        r = least_squares(lambda p: path(p) - y, [half - span / 2, half + span / 2], loss="soft_l1", f_scale=0.1)
        if best is None or r.cost < best.cost:
            best = r
    p = np.sort(best.x)
    return (float(p[0]), float(p[1])), path(p), float(np.median(np.abs(path(p) - y)))


def geometry(per, scale=1.0):
    """Where the sheet hangs and how its top edge travels: (View with its origin at the middle of the top edge at
    rest, s down, z the level direction across the sheet; width m; length m; carry (t, c across the sheet, d along
    its edge) m; the first frame it moves). (None, why) when the picture does not give it.

    The edge is level, so its line in the picture meets the horizon at its direction's vanishing point; with MoGe-2's
    depth of the sheet at rest that makes it a segment in space. Afterwards it is that segment, as long and
    pointing the same way, wherever the two rays through its ends leave room for it: its place in depth comes
    from how long it looks. `scale` is a factor on MoGe-2's metric scale (every length, the camera's height)."""
    from .scene import camera_pose

    obj, score, t, K = per["object"], per["score"], per["t"], np.asarray(per["K"], float)
    n = np.median(per["plane_n"], 0)
    T = camera_pose(n / np.linalg.norm(n), scale * float(np.median(per["plane_h"])))
    held = np.nonzero(score > 0)[0]
    if len(held) < 20:
        return None, "the sheet is not seen"
    fps = 1.0 / float(np.median(np.diff(t)))
    n_rest, n_end = max(8, int(round(0.5 * fps))), max(12, int(round(0.8 * fps)))  # half a second, most of one
    # the top edge is the mask's upper outline, column by column: for a picture in which down is down
    g_px = -(K @ (n / np.linalg.norm(n)))[:2]
    if g_px[1] < np.cos(np.radians(30)) * np.linalg.norm(g_px):
        return None, ("the picture is turned or looks steeply up or down (gravity is over 30 deg off its downward "
                      "direction): the sheet's upper outline is not the edge it hangs from")
    k0 = int(held[0])
    first = [top_edge(obj[k]) for k in held[:n_rest]]
    first = [np.stack(e) for e in first if e is not None]
    if len(first) < 3:
        return None, "the sheet's top edge is not seen while it hangs still"
    rest = np.median(first, 0)
    ray = lambda uv: np.stack([(uv[..., 0] - K[0, 2]) / K[0, 0], (uv[..., 1] - K[1, 2]) / K[1, 1],
                               np.ones(uv.shape[:-1])], -1)
    # the edge is level, so in the picture it points at the horizon: where its line at rest meets the horizon is
    # the vanishing point of its direction, with no depth asked of anything
    n_plane = n / np.linalg.norm(n)
    horizon = np.linalg.inv(K).T @ n_plane
    line = np.cross(np.append(rest[0], 1.0), np.append(rest[1], 1.0))
    vanish = np.cross(line, horizon)
    n_cam = np.linalg.inv(K) @ vanish
    if np.linalg.norm(n_cam) < 1e-9:
        return None, "the sheet's top edge lies on the horizon: its direction is not seen"
    n_cam = n_cam / np.linalg.norm(n_cam)
    r1, r2 = ray(rest[0]), ray(rest[1])
    mid_ray = (r1 + r2) / np.linalg.norm(r1 + r2)
    if abs(n_cam @ mid_ray) > np.cos(np.radians(10)):  # it points at the camera: its length is all depth
        return None, "the sheet's top edge is seen end-on"
    # the segment between the two rays that points that way, at the depth MoGe-2 gives the sheet near its top
    uv, P = per["object_uv"], per["object_points"]
    near = uv[:, 1] < np.percentile(uv[:, 1], 25)
    if near.sum() < 20:
        return None, "no depth of the sheet at rest"
    Z0 = scale * float(np.median(P[near, 2]))
    lam = np.linalg.svd(np.stack([r2, -r1, -n_cam], 1))[2][-1]  # (to the right end, to the left end, its length)
    lam = lam * Z0 / (0.5 * (lam[0] * r2[2] + lam[1] * r1[2]))
    if lam[2] < 0:
        n_cam, lam[2] = -n_cam, -lam[2]
    width = float(lam[2])
    along_edge = T[:3, :3] @ n_cam
    X0 = 0.5 * (lam[0] * r2 + lam[1] * r1) @ T[:3, :3].T + T[:3, 3]
    # every frame: the edge on the line towards that vanishing point, and the segment of that length and direction
    # between the two rays through its ends
    X = np.full((len(t), 3), np.nan)
    for k in held:
        e = top_edge_towards(obj[k], vanish)
        if e is None:
            continue
        r1, r2 = ray(e[0]), ray(e[1])
        lam = np.linalg.lstsq(np.stack([r2, -r1], 1), width * n_cam, rcond=None)[0]
        if lam.min() > 0:
            X[k] = (0.5 * (lam[0] * r2 + lam[1] * r1)) @ T[:3, :3].T + T[:3, 3]
    ok = np.isfinite(X[:, 0])
    if ok.sum() < 20:
        return None, "the sheet's top edge is not followed"
    across = np.cross([0, 0, 1.0], along_edge)  # level, across the sheet
    D = X - np.nanmedian(X[k0:k0 + n_rest], 0)
    raw_c, raw_d = (D @ across)[ok], (D @ along_edge)[ok]
    end_c, end_d = float(np.median(raw_c[-n_end:])), float(np.median(raw_d[-n_end:]))  # where it has come to rest
    if abs(end_c) < 0.1 * width:
        return None, "the edge the sheet hangs from is not carried across the sheet, or not left there"
    if end_c < 0:  # z points the way it is carried
        across, raw_c, end_c = -across, -raw_c, -end_c
    # the model swings the sheet by one move from one place to another, where the edge then stays: that it is
    # one is for whoever watched the video to say (the edge read off a sheet that billows overshoots and wanders by
    # a third of the way while it travels, so a carry there and back is not told from one that is only rough);
    # only an edge that half the time stands far off any such move is refused here
    (t_go, t_stop), c, off = carry_path(t[ok], raw_c / end_c)
    if off > CARRY_OFF:
        return None, (f"the edge the sheet hangs from does not go from one place to another in one smooth move (half "
                      f"its places stand over {off:.2f} of the way off such a move): it goes in steps, or it is not "
                      "followed")
    c, d = end_c * np.interp(t, t[ok], c), end_d * np.interp(t, t[ok], c)
    start = int(np.searchsorted(t, t_go))
    w_axis = along_edge if along_edge @ (T[:3, 3] - X0) > 0 else -along_edge
    d = d * (1.0 if w_axis is along_edge else -1.0)
    # its length: how far below the edge the lowest of the mask at rest is, in the sheet's own plane
    vs, us = np.nonzero(obj[k0])
    low = vs > np.percentile(vs, 97)
    dirs = ray(np.stack([us[low], vs[low]], 1).astype(float)) @ T[:3, :3].T
    lam = ((X0 - T[:3, 3]) @ across) / (dirs @ across)
    length = float(X0[2] - np.median(T[2, 3] + lam * dirs[:, 2]))
    view = beam.View(K, T, np.nanmedian(X[k0:k0 + n_rest], 0), across, w_axis, -np.pi / 2)
    return (view, width, length, (t, c, d), start, off), None


def chain_of(length, width, areal_density):
    ch = beam.Chain([{"length": length, "soft": True, "density": areal_density / THICK}], THICK, width, ELEMENTS)
    ch.tilt = -np.pi / 2
    return ch


def stated_range(lo, hi):
    """The range a weighing is stated with: the cloths that fit as well as the best (lo to hi, kg/m^2), widened
    by what the air's part is good to (AIR_REL)."""
    return [float(lo * np.exp(-AIR_REL)), float(hi * np.exp(AIR_REL))]


def added_of(length, width):
    a, b = min(length, width), max(length, width)
    return float(np.interp(b / a, *ADDED)) * (a / width) ** 2  # per rho pi/4 width^2 per length, as beam.swing takes it


def pictures(chain, q, view, c, d, shape):
    """The sheet's silhouettes (T, H, W) for states q (T, 3 n) in its clamp's frame, the clamp at c (T,) m across
    the sheet and d (T,) along its edge."""
    return np.stack([beam.silhouette(chain, x, view.moved(dz=float(ck), dw=float(dk)), shape)
                     for x, ck, dk in zip(q, c, d)])


def measure(per, meta, search=(48, 6), seconds=8.0, seed=0, scale=None):
    """physics.json of a recording whose motion is "sheet". meta.json "prior": {"size": the sheet's longer side in
    metres as the agent judges it, "size_rel_std"} is weighed against MoGe-2's scale by their variances, as
    track.metric_scale does for an object; `scale` sets the factor on MoGe-2's scale outright (an ablation)."""
    from .track import MONOCULAR_SCALE_STD, SIZE_PRIOR_STD, two_scales

    obj, t = per["object"], per["t"]
    out = {"motion": "sheet", "fps": float(1.0 / np.median(np.diff(t)))}
    if str(per["camera"]) == "moving":
        return dict(out, measured=False, reason="the camera travels")
    geo, why = geometry(per)
    if geo is None:
        return dict(out, measured=False, reason=why)
    sc = {"source": "MoGe-2 metric depth", "factor": 1.0, "rel_std": MONOCULAR_SCALE_STD}
    size = meta.get("prior", {}).get("size")
    if scale is not None:
        sc = {"source": "given", "factor": float(scale), "rel_std": 0.0}
    elif size:
        ps = float(meta["prior"].get("size_rel_std", SIZE_PRIOR_STD))
        f_size = float(size) / max(geo[1], geo[2])
        f, rel, apart = two_scales(f_size, MONOCULAR_SCALE_STD, ps)
        sc = {"source": "MoGe-2 metric depth + the sheet's size prior", "rel_std": rel, "rel_std_apart": apart,
              "factor": f, "factor_from_size": f_size,
              "consistent": bool(abs(np.log(f_size)) < 2.5 * np.hypot(MONOCULAR_SCALE_STD, ps))}
    if sc["factor"] != 1.0:
        geo, why = geometry(per, sc["factor"])
        if geo is None:
            return dict(out, measured=False, reason=why)
    out["scale"] = sc
    view, width, length, (tc, c, d), start, carry_off = geo
    prior = float(meta.get("prior", {}).get("areal_density", 0.2))
    chain = chain_of(length, width, prior)
    added = added_of(length, width)
    t0 = float(t[max(start - 8, 0)])
    use = np.nonzero((t >= t0) & (t <= t0 + seconds) & (per["score"] > 0))[0][::2]  # the carry and the first swings
    ts = t[use]

    def loss_of(scale, cd):
        q = chain.swing(ts, MODULUS, 0.3, release=t0, mass_scale=scale, carry=(tc, c),
                        air={"width": width, "cd": cd, "added": added})
        q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)  # a rollout that blew up shows nothing
        return np.array([1.0 - np.mean([beam.iou(m, obj[k]) for m, k in
                                        zip(pictures(chain, qp, view, c[use], d[use], obj[0].shape), use)])
                         for qp in q])

    rng = np.random.default_rng(seed)
    mu, sd, best, tried = np.array([0.0, np.log(1.3)]), np.array([1.5, 0.35]), None, []
    for _ in range(search[1]):
        X = mu + sd * rng.standard_normal((search[0], 2))
        X[0] = mu
        X[:, 0] = np.clip(X[:, 0], np.log(LIGHTEST / prior), np.log(HEAVIEST / prior))
        X[:, 1] = np.clip(X[:, 1], np.log(0.5), np.log(3.0))
        L = loss_of(np.exp(X[:, 0]), np.exp(X[:, 1]))
        tried += [(float(prior * np.exp(x[0])), float(np.exp(x[1])), float(l)) for x, l in zip(X, L)]
        o = np.argsort(L)
        mu, sd = X[o[:search[0] // 6]].mean(0), X[o[:search[0] // 6]].std(0) + 0.02
        if best is None or L[o[0]] < best[0]:
            best = (float(L[o[0]]), X[o[0]].copy())
    loss, x = best
    sigma, cd = float(prior * np.exp(x[0])), float(np.exp(x[1]))
    # what the picture says about heavier cloths: the loss up a ladder of them, the drag as fitted
    ladder = sigma * np.array([1.0, 1.5, 2.25, 3.5, 5.0, 10.0, 30.0])
    Ll = loss_of(ladder / prior, np.full(len(ladder), cd))
    slack = max(0.01, 0.05 * loss)  # a loss this close to the best is as good
    Tried = np.array(tried)
    good = Tried[Tried[:, 2] <= loss + slack]
    # weighed: a cloth thirty times as heavy does worse (the air shows), and the cloths that do as well as the best
    # are within a factor of WIDE of one another; a wider run of them is a bound from below and no weight
    bounded = bool(Ll[-1] > loss + slack and good[:, 0].max() <= WIDE * good[:, 0].min())
    s = {"length": length, "width": width, "travel": float(np.max(c)), "start_frame": int(start),
         "areal_density": sigma if bounded else None,
         "areal_density_range": stated_range(good[:, 0].min(), good[:, 0].max()) if bounded else None,
         "areal_density_fit_range": [float(good[:, 0].min()), float(good[:, 0].max())],
         "areal_density_at_least": None if bounded else float(good[:, 0].min() * np.exp(-2 * AIR_REL)),
         "drag_coefficient": cd, "drag_at_bound": bool(cd > 2.95 or cd < 0.51), "carry_off": carry_off,
         "added_mass_coefficient": added, "air_density": beam.AIR,
         "loss": loss, "iou": 1.0 - loss, "loss_heavy": float(Ll[-1]),
         "ladder": [[float(a), float(b)] for a, b in zip(ladder, Ll)], "best_density": sigma,
         "note": ("mass per area from how the air slows it" if bounded else
                  "cloths several times as heavy fit as well: the air does not show enough, and the video bounds "
                  "its mass per area from below"),
         "plane": {"origin": view.o.tolist(), "along": view.a.tolist(), "across": view.n.tolist()},
         "carry": {"t": np.round(tc, 4).tolist(), "c": np.round(c, 4).tolist(), "d": np.round(d, 4).tolist()}}
    out.update(measured=True, sheet=s, K=np.asarray(per["K"]).tolist(), T_world_cam=view.T.tolist())
    if sigma < 1.05 * LIGHTEST:
        out.update(measured=False, reason=f"the sheet swings as the lightest the search holds ({1e3 * LIGHTEST:.0f} "
                                          "g/m^2) or lighter: a film or a tissue, which this model does not weigh")
    if s["iou"] < LEAST_OVERLAP:
        out.update(measured=False, reason=f"the fitted sheet and the masks overlap by {s['iou']:.2f}: the sheet or the "
                                          "edge it hangs from is not followed well enough to say how it swings")
    return out


def fit(path):
    path = Path(path)
    return measure(np.load(path / "perception.npz"), read_document(path / "meta.json"))


def summary(path):
    """summary.mp4 of a measured sheet: the video with the sheet's mask (green) and the fitted chain's picture
    (yellow), and the same sheet as a heavy cloth would hang and swing (red, thin): where the two part, the air
    shows; beside it the sheet from the side in its own frame and the numbers."""
    import subprocess

    import cv2

    from .scene import read_frames

    path = Path(path)
    phys = read_document(path / "physics.json")
    if "sheet" not in phys:
        return None
    s = phys["sheet"]
    per = np.load(path / "perception.npz")
    obj, t = per["object"], per["t"]
    frames, fps = read_frames(path)
    view = beam.View(phys["K"], phys["T_world_cam"], s["plane"]["origin"], s["plane"]["along"], s["plane"]["across"],
                     -np.pi / 2)
    tc, c, d = (np.asarray(s["carry"][k]) for k in ("t", "c", "d"))
    chain = chain_of(s["length"], s["width"], s["best_density"])
    air = {"width": s["width"], "cd": s["drag_coefficient"], "added": s["added_mass_coefficient"]}
    q = chain.swing(t, MODULUS, 0.3, release=float(t[0]), mass_scale=[1.0, 30.0], carry=(tc, c), air=air)
    q = np.nan_to_num(q)
    H, W = frames.shape[1:3]
    PW = 440
    box = (W + 20, 40, W + PW - 20, 300)
    sc = (box[3] - box[1] - 20) / (1.5 * s["length"])
    side = lambda z, x: (int(0.5 * (box[0] + box[2]) + z * sc), int(box[1] + 14 + x * sc))
    weight = (f"{1e3 * s['areal_density']:.0f} g/m2 ({1e3 * s['areal_density_range'][0]:.0f} to "
              f"{1e3 * s['areal_density_range'][1]:.0f})" if s["areal_density"] else
              f"at least {1e3 * s['areal_density_at_least']:.0f} g/m2")
    text = [f"sheet {100 * s['length']:.0f} x {100 * s['width']:.0f} cm, carried {100 * s['travel']:.0f} cm",
            f"mass per area: {weight}", f"   ({s['note']})",
            f"drag coefficient {s['drag_coefficient']:.2f}; air {s['air_density']} kg/m3",
            f"overlap of chain and mask {s['iou']:.2f}; a heavy cloth: {1 - s['loss_heavy']:.2f}"]
    tmp = path / "summary_raw.mp4"
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W + PW, H))
    first, last = max(s["start_frame"] - 15, 0), min(len(t), s["start_frame"] + int(10 * fps))
    for k in range(first, last):
        im = frames[k][:, :, ::-1].copy()
        cs, _ = cv2.findContours(obj[k].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im, cs, -1, (0, 220, 0), 2)
        vk = view.moved(dz=float(c[k]), dw=float(d[k]))
        for qq, col, th in ((q[1, k], (60, 60, 255), 1), (q[0, k], (0, 230, 255), 2)):
            cs, _ = cv2.findContours(beam.silhouette(chain, qq, vk, obj[k].shape).astype(np.uint8), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(im, cs, -1, col, th)
        img = np.zeros((H, W + PW, 3), np.uint8)
        img[:, :W] = im
        cv2.putText(img, "green: the video's mask   yellow: the fitted sheet   red: the same, heavy", (10, H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(img, box[:2], box[2:], (70, 70, 70), 1)
        cv2.putText(img, "from the side, in the frame of the edge it hangs from", (box[0] + 4, box[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
        for qq, col in ((q[1, k], (60, 60, 255)), (q[0, k], (0, 230, 255))):
            pts = np.array([side(0.0, 0.0)] + [side(z, x) for x, z in zip(qq[0::3], qq[1::3])], np.int32)
            cv2.polylines(img, [pts], False, col, 2, cv2.LINE_AA)
        for j, line in enumerate(text):
            cv2.putText(img, line, (W + 14, box[3] + 30 + 21 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (235, 235, 235), 1,
                        cv2.LINE_AA)
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
            from .scene import untimed

            why = untimed(read_document(Path(p) / "meta.json"))
            o = fit(p) if why is None else {"motion": "sheet", "measured": False, "reason": why}
        except Exception as e:  # one recording's failure is its own
            o = {"motion": "sheet", "measured": False, "reason": f"{type(e).__name__}: {e}"}
        write_document(Path(p) / "physics.json", o, indent=1)
        if not o.get("measured"):
            print(f"{p}: not measured: {o['reason']}", flush=True)
            continue
        s = o["sheet"]
        what = (f"{1e3 * s['areal_density']:.0f} g/m^2 ({1e3 * s['areal_density_range'][0]:.0f} to "
                f"{1e3 * s['areal_density_range'][1]:.0f})" if s["areal_density"] else
                f"at least {1e3 * s['areal_density_at_least']:.0f} g/m^2")
        print(f"{p}: sheet {s['length'] * 100:.0f} x {s['width'] * 100:.0f} cm carried {s['travel'] * 100:.0f} cm: "
              f"{what}, drag {s['drag_coefficient']:.2f}, overlap {s['iou']:.2f} (a heavy cloth: "
              f"{1 - s['loss_heavy']:.2f})", flush=True)


if __name__ == "__main__":
    main()
