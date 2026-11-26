"""Discrepancy between a simulated rollout and the video: everything is compared in the real camera's image, on the
video frames' clock, so reconstruction scale errors and depth noise stay out of the loss.

sim  {"t": video seconds (F,), "uv": projected object centre (F, 2), optional "mask": projected silhouettes (F, h, w),
      optional "uv_tcp": projected TCP (F, 2), optional "ee_err": distance (m) of the replayed end-effector link from
      the recorded one per trajectory sample}
obs  observe.track_from_masks / track.npz: t, valid, centroid, optional masks (same stride as sim masks), uv_tcp
"""
import numpy as np

from .observe import iou

WEIGHTS = {"track": 1.0, "iou": 20.0, "events": 30.0, "arm": 1.0}  # px; px per unit IoU, per s of event mistiming,
# per mm of arm deviation


def sample_at(t_src, x, t):
    """Columns of x (N, ...) sampled at times t by linear interpolation (held at the ends)."""
    x = np.asarray(x, float)
    x2 = x.reshape(len(t_src), -1)
    y = np.stack([np.interp(t, t_src, c) for c in x2.T], 1)
    return y.reshape((len(t),) + x.shape[1:])


def huber(r, delta):
    a = np.abs(r)
    return np.where(a <= delta, 0.5 * a ** 2 / delta, a - 0.5 * delta)


def track_error(uv_sim, uv_obs, valid, delta=10.0):
    """Mean Huber pixel distance over frames where both tracks exist (a pixel-like scale: ~distance above delta)."""
    d = np.linalg.norm(np.asarray(uv_sim, float) - np.asarray(uv_obs, float), axis=-1)
    m = np.asarray(valid, bool) & np.isfinite(d)
    return float(huber(d[m], delta).mean()) if m.any() else np.inf


def first_motion(uv, valid, px=4.0, hold=3, ref_frames=5):
    """First frame from which the track stays > px away from its initial position for `hold` valid frames (lift-off /
    push start), or None. Invalid frames (the hand occludes the object) neither count nor break the run: the event is
    then the first valid frame after the gap, a bias both tracks share only if both lose the object."""
    uv, valid = np.asarray(uv, float), np.asarray(valid, bool)
    idx = np.flatnonzero(valid)
    if len(idx) < ref_frames + hold:
        return None
    ref = np.median(uv[idx[:ref_frames]], 0)
    away = np.linalg.norm(uv[idx] - ref, axis=1) > px
    run = 0
    for k, a in enumerate(away):
        run = run + 1 if a else 0
        if run == hold:
            return int(idx[k - hold + 1])
    return None


def settle(uv, valid, px=1.5, hold=5, end_frames=5):
    """First frame from which the object stays within px of its final resting place (median of the last valid
    frames) until the end, or None if it never moved or is still moving; the last motion of a drop or a slide. A
    position band rather than a speed threshold: mask centroids jitter by a pixel from frame to frame."""
    uv, valid = np.asarray(uv, float), np.asarray(valid, bool)
    if first_motion(uv, valid) is None:
        return None
    idx = np.flatnonzero(valid)
    rest = np.median(uv[idx[-end_frames:]], 0)
    off = np.flatnonzero(np.linalg.norm(uv[idx] - rest, axis=1) > px)
    if len(off) == 0 or len(idx) - 1 - off[-1] < hold:
        return None
    return int(idx[off[-1] + 1])


def slip(uv_obj, uv_tcp, valid, start, px=6.0):
    """First frame after `start` (grasp closed) at which the object has moved > px relative to the TCP, or None."""
    rel = np.asarray(uv_obj, float) - np.asarray(uv_tcp, float)
    ok = np.asarray(valid, bool) & np.all(np.isfinite(rel), 1)
    if start is None or start >= len(rel) or not ok[start]:
        return None
    d = np.linalg.norm(rel - rel[start], axis=1)
    hit = np.flatnonzero(ok & (d > px) & (np.arange(len(rel)) > start))
    return int(hit[0]) if len(hit) else None


def events(tr, uv_key="centroid", grasp_frame=None):
    """{event: frame or None} of a track dict (obs) or a sim dict (uv_key="uv")."""
    uv, valid = tr[uv_key], tr.get("valid", np.ones(len(tr[uv_key]), bool))
    ev = {"move": first_motion(uv, valid), "settle": settle(uv, valid)}
    if "uv_tcp" in tr and grasp_frame is not None:
        ev["slip"] = slip(uv, tr["uv_tcp"], valid, grasp_frame)
    return ev


def event_error(ev_sim, ev_obs, fps, miss_s=1.0):
    """Mean |timing difference| in seconds over events either side detected; an event only one side shows costs
    miss_s."""
    errs = []
    for k in set(ev_sim) | set(ev_obs):
        a, b = ev_sim.get(k), ev_obs.get(k)
        if a is None and b is None:
            continue
        errs.append(miss_s if a is None or b is None else abs(a - b) / fps)
    return float(np.mean(errs)) if errs else 0.0


def discrepancy(sim, obs, fps, weights=None, grasp_frame=None):
    """Weighted sum of the track error, 1 - mean silhouette IoU (when both have masks), event mistiming and, when the
    sim has ee_err, the arm's mean deviation from the recording in mm: a replay with velocity feed-forward follows the
    recording to a fraction of a mm, so a deviation means the twin obstructs the arm where the real one moved freely
    (a support raised into the fingers). Returns (total, parts)."""
    w = dict(WEIGHTS, **(weights or {}))
    valid = np.asarray(obs["valid"], bool)
    parts = {"track": track_error(sim["uv"], obs["centroid"], valid)}
    if "mask" in sim and "masks" in obs:
        ious = [iou(a, b) for a, b, v in zip(sim["mask"], obs["masks"], valid) if v]
        parts["iou"] = 1.0 - float(np.mean(ious)) if ious else 1.0
    sim_tr = {"uv": sim["uv"], "valid": np.all(np.isfinite(sim["uv"]), 1)}
    if "uv_tcp" in sim:
        sim_tr["uv_tcp"] = sim["uv_tcp"]
    parts["events"] = event_error(events(sim_tr, "uv", grasp_frame), events(obs, grasp_frame=grasp_frame), fps)
    if "ee_err" in sim:
        parts["arm"] = 1000.0 * float(np.mean(sim["ee_err"]))
    total = sum(w[k] * v for k, v in parts.items())
    return float(total), parts
