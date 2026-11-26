"""The object's planar pose in every frame, from its masks and the support plane (CPU).

    envs/recon/bin/python -m calibrated_scene_reconstruction.physics.video.track real/<name>

Reads perception.npz, writes real/<name>/track.npz and twin.json:
  twin.json   {"K", "T_world_cam" (scene.camera_pose: the support plane is z = 0), "camera_height", "height_std",
               "scale": where the metric scale comes from and how uncertain it is, "ref_frame",
               "object": an upright primitive standing on the plane (real2sim.tabletop's object dict)}
  track.npz   t (F,), xy (F, 2) m on the plane, yaw (F,) deg, iou (F,) of the primitive's silhouette with the mask,
              valid (F,), contact (F,) the manipulator's mask touches the object's, border (F,) the mask is cut by
              the image edge

The primitive's size is fitted on one frame (by default the last one where the object is whole and untouched: at
rest, no motion blur) and held; every other frame fits only where it stands, the previous frame's answer as the
start. A mask alone cannot tell a near small object from a far large one, but the plane can: the object stands on it.
"""
from calibrated_scene_reconstruction.physics.common.schema import read_document, write_document
import argparse
import json
from pathlib import Path

import numpy as np

from calibrated_scene_reconstruction.physics.common.observations import convex_silhouette, iou, mask_stats
from calibrated_scene_reconstruction.physics.replay.tabletop import fit_object, object_points
from .scene import camera_pose, ray_plane


def touches(a, b, px=6):
    """Per frame: do masks a and b (F, H, W) come within px pixels of each other?"""
    from scipy.ndimage import binary_dilation

    out = np.zeros(len(a), bool)
    for i in range(len(a)):
        if a[i].any() and b[i].any():
            out[i] = (binary_dilation(a[i], iterations=px) & b[i]).any()
    return out


def on_border(masks, px=2):
    return (masks[:, :px].any((1, 2)) | masks[:, -px:].any((1, 2)) | masks[:, :, :px].any((1, 2))
            | masks[:, :, -px:].any((1, 2)))


def init_object(mask, K, T, shape):
    """A first guess of the upright primitive from the mask alone: it stands where the mask's lowest pixels meet the
    plane, as wide as the mask at that depth, as tall as the mask's top above its centre. -> fit_object's init."""
    ys, xs = np.nonzero(mask)
    low = ys >= np.percentile(ys, 97)
    base = ray_plane(K, T, np.array([xs[low].mean(), ys[low].mean()]))  # the nearest point of the footprint
    c = np.asarray(T, float)[:3, 3]
    away = (base - c)[:2] / np.linalg.norm((base - c)[:2])
    zc = np.linalg.norm(base - c)
    width = (xs.max() - xs.min() + 1) * zc / K[0, 0]
    centre = base[:2] + away * width / 2
    # the top pixel's ray, where it passes over the centre
    top = np.array([xs[ys <= np.percentile(ys, 3)].mean(), ys.min()])
    d = ray_plane(K, T, top, z=-1e3) - c  # any point on the ray below the camera gives its direction
    s = np.linalg.norm(centre - c[:2]) / np.linalg.norm(d[:2])
    height = float(np.clip(np.nan_to_num(c[2] + s * d[2], nan=width), 0.02, 1.0))
    if shape == "cylinder":
        return [centre[0], centre[1], width / 2, height]
    return [centre[0], centre[1], width, width, height]


def fit_pose(mask, K, T, obj, x0, yaw=False):
    """Where the fixed-size primitive stands to best cover `mask`: (x, y[, yaw_deg]) from x0; returns (pose, IoU)."""
    from scipy.optimize import minimize

    o = dict(obj)

    def cost(v):
        o["pose"] = [v[0], v[1], obj["pose"][2], v[2] if yaw else obj["pose"][3]]
        return 1.0 - iou(convex_silhouette(K, T, object_points(o), mask.shape), mask)

    x0 = np.asarray(x0, float)[: 3 if yaw else 2]
    step = np.array([0.03, 0.03, 10.0])[: len(x0)]
    sim = np.vstack([x0] + [x0 + np.eye(len(x0))[i] * step[i] for i in range(len(x0))])
    r = minimize(cost, x0, method="Nelder-Mead", options={"initial_simplex": sim, "xatol": 2e-4, "fatol": 1e-4,
                                                          "maxiter": 200})
    return r.x, 1.0 - float(r.fun)


def centroid_height(mask, K, T, shape="cylinder"):
    """Height above the plane (m) of the point the mask's centroid shows, for an object standing or lying on the
    plane: its centroid's ray, where it passes over the middle of the footprint (init_object). Also returns that
    middle (2,), the mask's width in metres and the object's height."""
    init = init_object(mask, K, T, shape)
    c = np.asarray(T, float)[:3, 3]
    d = ray_plane(K, T, mask_stats(mask)[0], z=-1e3) - c
    s = np.linalg.norm(np.asarray(init[:2]) - c[:2]) / np.linalg.norm(d[:2])
    width = 2 * init[2] if shape == "cylinder" else init[2]
    hc = np.nan_to_num(c[2] + s * d[2], nan=init[-1] / 2)
    return float(np.clip(hc, 0.002, 1.0)), np.asarray(init[:2]), float(width), float(init[-1])


def run_blob(per, ref=None, prior=None):
    """The shape-free tracker: no primitive is fitted. The object is the point its mask's centroid shows, at the
    height above the plane it has on the reference frame (centroid_height), and every frame's centroid ray is cut
    with the plane at that height. Right for anything whose silhouette keeps its shape while it moves: a body that
    slides without turning much, a ball or a can that rolls. The twin's object is a stand-in upright cylinder as
    wide and tall as the reference mask (for the simulation; `shape` "blob" marks it as such)."""
    obj_m, manip, score = per["object"], per["manipulator"], per["score"]
    F = len(obj_m)
    h = float(np.median(per["plane_h"]))
    n = per["plane_n"][np.argsort(per["plane_h"])[len(per["plane_h"]) // 2]]
    K, T = np.asarray(per["K"], float), camera_pose(n, h)
    contact, border = touches(obj_m, manip), on_border(obj_m)
    on = over_support(obj_m, per["support"]) if "support" in per else None
    ref = reference_frame(score, contact, border, on) if ref is None else ref
    hc, mid, width, tall = centroid_height(obj_m[ref], K, T)
    view = view_scale_std(K, n, mask_stats(obj_m[ref])[0])
    xy = np.full((F, 2), np.nan)
    for i in range(F):
        if score[i] > 0:
            xy[i] = ray_plane(K, T, mask_stats(obj_m[i])[0], z=hc)[:2]
    q = (score > 0).astype(float)  # no silhouette is fitted: nothing to score a frame by but its being tracked
    obj = {"shape": "cylinder", "dims": {"radius": width / 2, "height": tall}, "pose": [mid[0], mid[1], 0.0, 0.0],
           "first_frame_iou": float("nan"), "blob": True, "centroid_height": hc}
    scale = metric_scale(per, prior, view)
    xy = _rescale(obj, T, xy, scale["factor"])
    twin = {"K": K.tolist(), "T_world_cam": T.tolist(), "camera_height": float(T[2, 3]), "scale": scale,
            "height_std": float(np.std(per["plane_h"])), "object": obj, "ref_frame": int(ref)}
    track = {"t": np.asarray(per["t"], float), "xy": xy, "yaw": np.full(F, np.nan), "iou": np.nan_to_num(q),
             "valid": (score > 0) & np.isfinite(xy[:, 0]), "contact": contact,
             "border": border}
    return twin, track


def over_support(obj_m, support, px=10):
    """Which frames show the object over the support (F,): its mask's centroid inside the support mask's convex
    outline in the image. An object that has left its support (slid off a board, fell to the floor) is not."""
    import cv2

    cs, _ = cv2.findContours(support.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return np.ones(len(obj_m), bool)
    hull = cv2.convexHull(np.concatenate(cs))
    out = np.zeros(len(obj_m), bool)
    for i, m in enumerate(obj_m):
        if m.any():
            c = mask_stats(m)[0]
            out[i] = cv2.pointPolygonTest(hull, (float(c[0]), float(c[1])), True) > -px
    return out


def support_frame(mask, K, T, prev=None):
    """The frame a support carries with it, from its mask on one image: (the middle of its outline in the plane
    (2,), its longer side's direction (2,), the outline's convex hull (N, 2)) in the plane frame T; the direction
    keeps the sense of `prev`. None when the mask gives no outline."""
    import cv2
    from scipy.spatial import ConvexHull

    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    X = ray_plane(K, T, np.concatenate([c.reshape(-1, 2) for c in cs]).astype(float))[:, :2]
    X = X[np.all(np.isfinite(X), 1)]
    if len(X) < 8:
        return None
    hull = X[ConvexHull(X).vertices]
    x, y = hull[:, 0], hull[:, 1]
    cr = x * np.roll(y, -1) - np.roll(x, -1) * y
    A = cr.sum() / 2
    c = np.array([((x + np.roll(x, -1)) * cr).sum(), ((y + np.roll(y, -1)) * cr).sum()]) / (6 * A)
    ax = np.linalg.eigh(np.cov((hull - c).T))[1][:, -1]
    if (prev is not None and ax @ prev < 0) or (prev is None and ax[0] < 0):
        ax = -ax
    return c, ax, hull


def along_support(obj_m, sup_t):
    """Where the object is along its support in the image, per frame (F,): its mask's centroid along the support
    mask's longer axis, from the mask's middle, in units of the mask's half length there. An object that rides a
    support that moves keeps this number, whichever way the support lies to the camera (seen from its edge, where
    its plane cannot be fitted, too)."""
    out, prev = np.full(len(obj_m), np.nan), None
    for i, (o, s) in enumerate(zip(obj_m, sup_t)):
        if not o.any() or s.sum() < 50:
            continue
        ys, xs = np.nonzero(s)
        P = np.stack([xs, ys], 1).astype(float)
        c = P.mean(0)
        a = np.linalg.eigh(np.cov((P - c).T))[1][:, -1]
        if (prev is not None and a @ prev < 0) or (prev is None and a[0] < 0):
            a = -a
        prev = a
        lo, hi = np.percentile((P - c) @ a, [2, 98])
        out[i] = ((mask_stats(o)[0] - c) @ a - (lo + hi) / 2) / max((hi - lo) / 2, 1.0)
    return out


def run_blob_moving(per, ref=None, prior=None):
    """run_blob on a support that moves (perception plane_n_t, plane_h_t, support_t): every frame's centroid ray is
    cut with that frame's plane, and the position is given in the support's own frame (support_frame), in which an
    object that rides the support stays where it is. track["on_support"] says where it is still over it."""
    from .incline import inside

    obj_m, manip, score = per["object"], per["manipulator"], per["score"]
    F = len(obj_m)
    K, n_t, h_t, sup_t = np.asarray(per["K"], float), per["plane_n_t"], per["plane_h_t"], per["support_t"]
    contact, border = touches(obj_m, manip), on_border(obj_m)
    over = np.array([over_support(obj_m[i:i + 1], sup_t[i])[0] for i in range(F)])
    ref = reference_frame(score, contact, border, over) if ref is None else ref
    T = camera_pose(n_t[ref], h_t[ref])
    hc, mid, width, tall = centroid_height(obj_m[ref], K, T)
    xy, on, prev = np.full((F, 2), np.nan), np.zeros(F, bool), None
    half = np.full(F, np.nan)  # half the support's length (m): what one unit of track "along" is
    for i in range(F):
        if score[i] <= 0 or not np.all(np.isfinite(n_t[i])):
            continue
        Ti = camera_pose(n_t[i], h_t[i])
        sf = support_frame(sup_t[i], K, Ti, prev)
        if sf is not None:
            half[i] = 0.5 * np.ptp((sf[2] - sf[0]) @ sf[1])
        p = ray_plane(K, Ti, mask_stats(obj_m[i])[0], z=hc)[:2]
        if sf is None or not np.all(np.isfinite(p)):
            continue
        c, prev, hull = sf
        xy[i] = (p - c) @ prev, (p - c) @ np.array([-prev[1], prev[0]])
        on[i] = inside(hull, p[None])[0]
    obj = {"shape": "cylinder", "dims": {"radius": width / 2, "height": tall}, "pose": [mid[0], mid[1], 0.0, 0.0],
           "first_frame_iou": float("nan"), "blob": True, "centroid_height": hc}
    scale = metric_scale(per, prior, view_scale_std(K, n_t[ref], mask_stats(obj_m[ref])[0]))
    xy = _rescale(obj, T, xy, scale["factor"])
    twin = {"K": K.tolist(), "T_world_cam": T.tolist(), "camera_height": float(T[2, 3]), "scale": scale,
            "height_std": float(np.std(per["plane_h"])), "object": obj, "ref_frame": int(ref),
            "support_frame": "the support's own (it moves): origin the middle of its outline, x along its length",
            # read where the support is seen steeply enough for its plane to be sure (the upper half of its frames)
            "support_half_length": float(np.nanmedian(half[np.nan_to_num(per["plane_graze_t"], nan=0.0)
                                                           >= np.nanmedian(per["plane_graze_t"])]) * scale["factor"])
            if "plane_graze_t" in per.files else float(np.nanmedian(half) * scale["factor"])}
    track = {"t": np.asarray(per["t"], float), "xy": xy, "yaw": np.full(F, np.nan),
             "iou": (score > 0).astype(float), "valid": (score > 0) & np.isfinite(xy[:, 0]), "contact": contact,
             "border": border, "on_support": on, "along": along_support(obj_m, sup_t)}
    return twin, track


def circle_through(X):
    """(centre (2,), radius, rms distance from it) of the circle through points X (N, 2), least squares."""
    A = np.c_[2 * X, np.ones(len(X))]
    sol = np.linalg.lstsq(A, (X ** 2).sum(1), rcond=None)[0]
    c, r = sol[:2], float(np.sqrt(max(sol[2] + sol[:2] @ sol[:2], 1e-12)))
    return c, r, float(np.sqrt(np.mean((np.linalg.norm(X - c, axis=1) - r) ** 2)))


def round_support(mask, K, n, h):
    """A support that is round (a turntable, a plate, a lazy Susan), from its mask on one image: the plane
    (n, h) turned about MoGe-2's so that the support's outline is a disc's, and the disc {"centre" (2,),
    "radius", "thickness"} in the plane's frame. A camera that looks along a disc sees its plane poorly in a depth
    map and exactly in its outline: a circle is an ellipse in the image only from one direction (of two, and
    MoGe-2's plane says which). The far half of the outline is the rim of the top face; the near half is where
    the disc's side ends, the same circle a thickness lower: both halves are fitted, with the thickness. The
    plane keeps its distance from the camera along the ray to the disc's middle, so the scene's scale is MoGe-2's
    still."""
    import cv2
    from scipy.optimize import least_squares

    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    hull = cv2.convexHull(max(cs, key=cv2.contourArea)).reshape(-1, 2).astype(float)  # what stands on it notches it
    uv = np.concatenate([a + (b - a) * np.linspace(0, 1, max(2, int(np.linalg.norm(b - a) / 2)), endpoint=False)[:, None]
                         for a, b in zip(hull, np.roll(hull, -1, 0))])
    uv = uv[:: max(1, len(uv) // 600)]
    n0 = np.asarray(n, float) / np.linalg.norm(n)
    ok = np.all(np.isfinite(ray_plane(K, camera_pose(n0, h), uv)), 1)
    if ok.sum() < 30:
        return None
    uv = uv[ok]
    le, ri = uv[np.argmin(uv[:, 0])], uv[np.argmax(uv[:, 0])]  # the widest chord parts the far rim from the near one
    far = uv[:, 1] < le[1] + (uv[:, 0] - le[0]) * (ri[1] - le[1]) / max(ri[0] - le[0], 1e-6)
    mid = np.linalg.inv(K) @ np.r_[uv.mean(0), 1.0]
    mid /= np.linalg.norm(mid)
    depth = h / max(-(n0 @ mid), 1e-6)  # along the middle ray to the plane
    e1 = np.cross(n0, [1.0, 0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n0, e1)

    def points(q):
        nn = n0 + q[0] * e1 + q[1] * e2
        nn /= np.linalg.norm(nn)
        hh = float(depth * max(-(nn @ mid), 1e-6))
        T = camera_pose(nn, hh)
        X = np.where(far[:, None], ray_plane(K, T, uv)[:, :2], ray_plane(K, T, uv, z=-abs(q[2]))[:, :2])
        return nn, hh, X

    def res(q):
        X = points(q)[2]
        if not np.all(np.isfinite(X)):
            return np.full(len(uv), 1.0)
        c, r, _ = circle_through(X)
        return (np.linalg.norm(X - c, axis=1) - r) / r

    best = None
    for tk in (0.0, 0.02):
        r_ = least_squares(res, [0.0, 0.0, tk], bounds=([-0.4, -0.4, 0.0], [0.4, 0.4, 0.12]), loss="soft_l1",
                           f_scale=0.01, x_scale=[0.05, 0.05, 0.01])
        if best is None or r_.cost < best.cost:
            best = r_
    nn, hh, X = points(best.x)
    c, r, rms = circle_through(X)
    return {"n": nn, "h": hh, "centre": c, "radius": r, "thickness": float(abs(best.x[2])), "rms_rel": rms / r,
            "turned_deg": float(np.degrees(np.arccos(np.clip(nn @ n0, -1, 1))))}


def reference_frame(score, contact, border, on=None):
    """The last frame where the object is whole, untouched and over its support (else the last whole and untouched
    one, else the best-scoring one)."""
    ok = (score > 0.05) & ~contact & ~border  # 0.01: a frame perception filled in from the motion (fill_fast)
    if on is not None and (ok & on).any():
        ok = ok & on
    return int(np.nonzero(ok)[0][-1]) if ok.any() else int(np.argmax(score))


MONOCULAR_SCALE_STD = 0.09



SIZE_PRIOR_STD = 0.10        # relative: an agent's estimate of an everyday object's largest dimension from its look
SIZE_MEASURE_STD = 0.15      # relative: the largest dimension measured in one view's MoGe-2 points against the object's



MONOCULAR_TILT_STD_DEG = 0.7  # MoGe-2's support plane against the calibrated camera pose: rms 0.67 deg over the



def view_scale_std(K, n, uv):
    """(relative std, elevation deg) of distances on the support that the plane's tilt makes: a distance d = h cot e
    on a plane seen at elevation e (the line of sight to the object, from the plane) is off by 2 dt / tan e along the
    line of sight when the plane's normal is off by dt (dt / tan e across it), and no size or depth scale corrects
    that: it stretches the plane, not the scene. Along the line of sight, as the object's heading is not known yet.
    dt is MoGe-2's tilt error (not the scatter of the planes of several frames: a camera that turns sees one plane
    at as many normals). e is between the plane (normal n, camera frame) and the ray through uv (the object)."""
    ray = np.linalg.solve(np.asarray(K, float), np.array([uv[0], uv[1], 1.0]))
    n = np.asarray(n, float)
    e = float(np.arcsin(np.clip(abs(ray @ n) / (np.linalg.norm(ray) * np.linalg.norm(n)), 1e-3, 1.0)))
    return 2 * np.radians(MONOCULAR_TILT_STD_DEG) / np.tan(e), float(np.degrees(e))


def two_scales(f_size, moge_std, size_std):
    """(factor, relative std, the part of it that is the two's disagreement) of one scale from MoGe-2's (factor 1, moge_std) and the size prior's (f_size,
    size_std): their mean weighted by their variances in log space, and its doubt: the two's own, and how far apart
    they are, p_m p_s ln(f_size)^2, as either may be the one that is off (the spread of the two about their mean).
    Agreement reduces the combined uncertainty.
    It is the doubt by variances alone times sqrt(1 + chi^2), chi how many of their joint sigmas they are apart."""
    w_m, w_s = 1 / moge_std ** 2, 1 / size_std ** 2
    p_s, d = w_s / (w_m + w_s), float(np.log(f_size))
    apart = float(np.sqrt(p_s * (1 - p_s)) * abs(d))
    return float(np.exp(p_s * d)), float(np.sqrt(1 / (w_m + w_s) + apart ** 2)), apart


def metric_scale(per, prior=None, view=(0.0, None)):
    """The factor on MoGe-2's metric scale and what is known about it. Without a ruler in the scene two things say
    how large it is: MoGe-2's metric depth, and what the object is. prior {"size": the object's largest dimension
    in metres as the agent judges it from what the object is, "size_rel_std": how sure (default SIZE_PRIOR_STD)}
    is compared with the same dimension measured in MoGe-2's points (perception object_size): two independent
    estimates of one scale, combined by their variances in log space and doubted by how far apart they are
    (two_scales). What the plane's tilt makes of distances on it
    (view: view_scale_std) is apart from either and is added after: rel_std is both, rel_std_scale the scale's
    alone."""
    h = float(np.median(per["plane_h"]))
    moge_std = float(np.hypot(MONOCULAR_SCALE_STD, np.std(per["plane_h"]) / h))
    view, elev = view

    def with_view(sc):
        return dict(sc, rel_std=float(np.hypot(sc["rel_std"], view)), rel_std_scale=sc["rel_std"],
                    rel_std_view=view, elevation_deg=elev)

    out = {"source": "MoGe-2 metric depth", "factor": 1.0, "rel_std": moge_std}
    size = np.asarray(per["object_size"], float) if "object_size" in per else np.array([np.nan])
    size = size[np.isfinite(size)]
    if not prior or not prior.get("size") or len(size) == 0:
        return with_view(out)
    # the prior's own doubt and that of measuring the same dimension in one view's points
    ps = float(np.hypot(prior.get("size_rel_std", SIZE_PRIOR_STD), SIZE_MEASURE_STD))
    f_size = float(prior["size"]) / float(np.median(size))
    f, rel, apart = two_scales(f_size, moge_std, ps)
    return with_view({"source": "MoGe-2 metric depth + object size prior", "factor": f,
            "rel_std": rel, "rel_std_apart": apart, "factor_from_size": f_size, "size_prior": float(prior["size"]),
            "size_measured": float(np.median(size)),
            "consistent": bool(abs(np.log(f_size)) < 2.5 * np.hypot(moge_std, ps))})


def _rescale(twin_obj, T, xy, f):
    """Everything metric scaled by f about the camera's foot on the plane (every silhouette stays as it was)."""
    T[2, 3] *= f
    twin_obj["dims"] = {k: v * f for k, v in twin_obj["dims"].items()}
    twin_obj["pose"][:2] = [twin_obj["pose"][0] * f, twin_obj["pose"][1] * f]
    if "centroid_height" in twin_obj:
        twin_obj["centroid_height"] *= f
    return xy * f


def run(per, shape, ref=None, prior=None):
    """perception arrays -> (twin dict, track dict). `prior`: meta.json "prior" (metric_scale)."""
    obj_m, manip, score = per["object"], per["manipulator"], per["score"]
    F = len(obj_m)
    h = float(np.median(per["plane_h"]))
    n = per["plane_n"][np.argsort(per["plane_h"])[len(per["plane_h"]) // 2]]
    K, T = np.asarray(per["K"], float), camera_pose(n, h)
    contact, border = touches(obj_m, manip), on_border(obj_m)
    on = over_support(obj_m, per["support"]) if "support" in per else None
    ref = reference_frame(score, contact, border, on) if ref is None else ref
    obj, ref_iou = fit_object(obj_m[ref], K, T, 0.0, shape, init_object(obj_m[ref], K, T, shape))
    obj["first_frame_iou"] = ref_iou
    use_yaw = shape == "box"
    xy, yaw, q = np.full((F, 2), np.nan), np.full(F, np.nan), np.zeros(F)
    for order in (range(ref, F), range(ref, -1, -1)):
        x = np.array([obj["pose"][0], obj["pose"][1], obj["pose"][3]])
        for i in order:
            if score[i] <= 0:
                continue
            # the mask's centroid moved since the last fitted frame: start from the pose moved by the same ray shift
            x, q[i] = fit_pose(obj_m[i], K, T, obj, _shift(x, obj_m[i], K, T, obj, use_yaw), use_yaw)
            x = np.r_[x, obj["pose"][3]][:3]
            xy[i], yaw[i] = x[:2], x[2]
    scale = metric_scale(per, prior, view_scale_std(K, n, mask_stats(obj_m[ref])[0]))
    xy = _rescale(obj, T, xy, scale["factor"])
    twin = {"K": K.tolist(), "T_world_cam": T.tolist(), "camera_height": float(T[2, 3]), "scale": scale,
            "height_std": float(np.std(per["plane_h"])), "object": obj, "ref_frame": int(ref)}
    track = {"t": np.asarray(per["t"], float), "xy": xy, "yaw": yaw, "iou": q, "valid": (score > 0) & (q > 0.5),
             "contact": contact, "border": border}
    return twin, track


def outline_lean(mask, K, up, hinge):
    """(how far a long support leans out of the level, deg; how long its mask is for its width). The mask's long
    axis in the image is a line; the support's length lies in the plane through the camera and that line, and at
    right angles to `hinge`, the level direction across it. So its direction in space is their cross product, and
    its lean the angle that makes with the level (`up`: the level's normal). Only where the support points comes
    from 3D; how far it leans is read off the image, which a depth map that stretches the scene cannot bend."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        return np.nan, 0.0
    P = np.stack([xs, ys], 1).astype(float)
    c = P.mean(0)
    w, v = np.linalg.eigh(np.cov((P - c).T))
    Ki = np.linalg.inv(np.asarray(K, float))
    m = np.cross(Ki @ np.r_[c, 1.0], Ki @ np.r_[c + 50 * v[:, -1], 1.0])
    d = np.cross(hinge, m)
    d /= max(np.linalg.norm(d), 1e-12)
    return float(np.degrees(np.arcsin(min(abs(d @ up), 1.0)))), float(np.sqrt(w[-1] / max(w[0], 1e-9)))


def hinge_of(up, normals, axes, least_deg=15.0):
    """The level direction across a tilted support (unit, camera frame): where its plane meets the level, from
    the plane normals that lean by more than least_deg (median); a plane that leans less says little of it, and
    then it is the level direction at right angles to the support's long axis."""
    up = up / np.linalg.norm(up)
    normals = np.asarray(normals, float)
    lean = np.degrees(np.arccos(np.clip(normals @ up / np.linalg.norm(normals, axis=1), -1, 1)))
    use = np.isfinite(lean) & (lean > least_deg)
    H = np.cross(up, normals[use]) if use.sum() >= 1 else np.cross(up, np.asarray(axes, float)[np.all(np.isfinite(axes), 1)])
    if len(H) == 0:
        return None
    H = H / np.linalg.norm(H, axis=1, keepdims=True)
    H = H * np.where(H @ H[0] < 0, -1.0, 1.0)[:, None]
    h = np.median(H, 0)
    return h / np.linalg.norm(h)


def gravity_of(per, twin):
    """{"gravity": unit vector of gravity in the twin's frame, "tilt_deg"} of a support that is not level. The tilt
    is the angle between its plane and a level surface's in the same MoGe-2 point maps (the median over the frames;
    two planes of one map are off together), or, of a support that is long in the image (a board, a ramp), what
    its outline says (outline_lean: MoGe-2's planes were 2.8 deg over at 10 deg and 15 % over on a board tilting
    from the level, the outline 0.5 deg and 1 % off). With "plane_n_t" (a support that moves) also the tilt of
    every frame, "tilt_deg_t"."""
    ns, nl = np.asarray(per["plane_n"], float), np.asarray(per["level_n"], float)
    tilt = np.arccos(np.clip(np.sum(ns * nl, 1) / np.linalg.norm(ns, axis=1) / np.linalg.norm(nl, axis=1), -1, 1))
    th = float(np.median(tilt))
    out = {"tilt_plane_deg": float(np.degrees(th)), "tilt_deg_frames": np.degrees(tilt).tolist(), "tilt_from": "planes"}
    K = np.asarray(per["K"], float)
    up = np.median(nl, 0)
    up /= np.linalg.norm(up)
    axes = np.asarray(per["plane_axis"], float) if "plane_axis" in per.files else np.zeros((0, 3))
    moving = "plane_n_t" in per.files and "level_n_t" in per.files
    if moving:  # the hinge from every frame's plane
        up = np.nanmedian(per["level_n_t"], 0)
        up /= np.linalg.norm(up)
        hinge = hinge_of(up, per["plane_n_t"][np.all(np.isfinite(per["plane_n_t"]), 1)], per["plane_axis_t"])
    else:
        hinge = hinge_of(up, ns, axes) if len(axes) == len(ns) else None
    if hinge is not None:
        lean, long_ = outline_lean(per["support"] if not moving else per["support_t"][twin["ref_frame"]], K, up, hinge)
        out.update(tilt_outline_deg=lean, support_aspect=long_)
        if long_ >= 3.0 and np.isfinite(lean):
            th, out["tilt_from"] = float(np.radians(lean)), "outline"
    j = int(np.argsort(per["plane_h"])[len(per["plane_h"]) // 2])
    upw = np.asarray(twin["T_world_cam"], float)[:3, :3] @ (nl[j] / np.linalg.norm(nl[j]))
    d = -upw[:2] / max(np.linalg.norm(upw[:2]), 1e-9)  # downhill, in the plane
    out.update(gravity=np.r_[np.sin(th) * d, -np.cos(th)].tolist(), tilt_deg=float(np.degrees(th)))
    if moving:
        from scipy.ndimage import median_filter

        def smooth(a):
            return median_filter(np.nan_to_num(a, nan=float(np.nanmedian(a))), size=5, mode="nearest")

        a = np.degrees(np.arccos(np.clip(np.sum(per["plane_n_t"] * per["level_n_t"], 1), -1, 1)))
        out["tilt_plane_deg_t"] = smooth(a).tolist()
        if hinge is not None and out["tilt_from"] == "outline":
            a = np.array([outline_lean(m, K, up, hinge)[0] for m in per["support_t"]])
        out["tilt_deg_t"] = smooth(a).tolist()
        out["tilt_deg"] = float(out["tilt_deg_t"][twin["ref_frame"]])
    return out


def _shift(x, mask, K, T, obj, yaw):
    """x moved so that the primitive's projected centre lands on the mask's centroid: the start of a frame's fit when
    the object moved more than its own size since the last one."""
    o = dict(obj)
    o["pose"] = [x[0], x[1], obj["pose"][2], x[2]]
    hz = obj["dims"]["height"] / 2
    c = ray_plane(K, T, mask_stats(mask)[0], z=hz)
    if not np.all(np.isfinite(c)):
        return x
    return np.r_[c[:2], x[2]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording", nargs="+")
    ap.add_argument("--ref", type=int, help="frame the primitive's size is fitted on")
    ap.add_argument("--shape", choices=("blob", "cylinder", "box"), help="override meta.json shape")
    ap.add_argument("--no-size", action="store_true", help="ignore meta.json prior.size: MoGe-2's scale alone")
    args = ap.parse_args()
    for r in args.recording:
        if not (Path(r) / "perception.npz").exists():  # an earlier stage failed on it: the others go on
            print(f"{r}: no perception.npz, skipped", flush=True)
            continue
        r = Path(r)
        meta = read_document(r / "meta.json")
        shape = args.shape or meta.get("shape", "blob")
        prior = None if args.no_size else meta.get("prior")
        per = np.load(r / "perception.npz")
        circle = None
        if meta.get("motion") == "turntable" or meta.get("support_round"):
            j = int(np.argsort(per["plane_h"])[len(per["plane_h"]) // 2])
            circle = round_support(per["support"], np.asarray(per["K"], float), per["plane_n"][j], float(per["plane_h"][j]))
            if circle is not None and circle["rms_rel"] < 0.03:  # the rim is a circle to 3 % of its radius
                per = {k: per[k] for k in per.files}
                per["plane_n"] = np.tile(circle["n"], (len(per["plane_h"]), 1))
                per["plane_h"] = np.full(len(per["plane_h"]), circle["h"])
            else:
                circle = None
        files = per.files if hasattr(per, "files") else list(per)
        if "plane_n_t" in files:
            twin, track = run_blob_moving(per, args.ref, prior)
        else:
            twin, track = run_blob(per, args.ref, prior) if shape == "blob" else run(per, shape, args.ref, prior)
        if meta.get("motion") == "pendulum":  # a bob hangs over no support: nothing is measured along its plane
            twin["scale"] = dict(twin["scale"], rel_std=twin["scale"]["rel_std_scale"], rel_std_view=0.0)
        if "level_n" in files:
            twin.update(gravity_of(per, twin))
        if circle is not None:
            f = twin["scale"]["factor"]
            twin["support_circle"] = {"centre": (circle["centre"] * f).tolist(), "radius": circle["radius"] * f,
                                      "thickness": circle["thickness"] * f, "rms_rel": circle["rms_rel"],
                                      "plane_turned_deg": circle["turned_deg"]}
        write_document(r / "twin.json", twin, indent=1)
        np.savez_compressed(r / "track.npz", **track)
        o = twin["object"]
        print(f"{r}: {'blob as ' if o.get('blob') else ''}{o['shape']} "
              f"{({k: round(v, 3) for k, v in o['dims'].items()})} fitted on frame "
              f"{twin['ref_frame']} (IoU {o['first_frame_iou']:.2f}); {int(track['valid'].sum())}/{len(track['t'])} "
              f"frames, median IoU {np.median(track['iou'][track['valid']]):.2f}, path "
              f"{np.nansum(np.linalg.norm(np.diff(track['xy'], axis=0), axis=1)):.3f} m")


if __name__ == "__main__":
    main()
