"""GPU half of video physics: what the video shows, before any geometry or physics is fitted.

    envs/recon/bin/python -m calibrated_scene_reconstruction.physics.video.perceive real/<name> [--moge-frames 5]

real/<name>/meta.json (written by the agent after watching the video):
  {"fps": 30, "object": SAM 3 prompt of the object, "support": prompt of the surface it moves on,
   "manipulator": [prompts of whatever touches the object: "hand", "wooden stick", "robot gripper"],
   "shape": "cylinder" | "box", optional "K": 3x3 when the camera's intrinsics are known,
   optional "object_at": [x, y], a point on the object in the first frame (fractions of the picture) when the
   prompt names other things too}
with the video as frames/*.png or video.mp4.

Writes real/<name>/perception.npz:
  t (F,) s; object (F, H, W) bool masks of the tracked object (all False where it was not found), score (F,);
  manipulator (F, H, W) bool, the union of the manipulator prompts' instances;
  camera "static" | "stabilised" (frames and masks were warped onto the reference frame: stabilise.py) | "moving"
  (every fit refuses the recording: scene.unusable) | "unknown" (a background with nothing to follow),
  camera_shift (F,) px of background motion against the reference frame;
  support (H, W) bool on the reference frame; K (3, 3) from MoGe-2 (or meta.json);
  plane_ok: whether MoGe-2 saw that support as one plane in every frame it looked at; support_band: whether the
  support is no surface a prompt named, only the pixels about the object's path;
  plane_n (M, 3), plane_h (M,), plane_frames (M,): the support plane in the camera frame from MoGe-2 on M frames
  spread over the video (its metric scale drifts between frames: the spread is the scale's noise);
  object_size (M,): the object's largest dimension at MoGe-2's scale on those frames (NaN where it is hidden);
  object_xyz (M, 3): where MoGe-2 sees the object on those frames (camera frame, m): what places a thing that is on
  no plane that is seen (a ball in flight, a book on a board seen along itself);
  object_at_frames (N,), object_at_xyz (N, 3): the same on up to 32 frames that show the object clear of the
  manipulator: a slide of half a second is in few of the frames the plane is fitted on.
"""
from calibrated_scene_reconstruction.physics.common.schema import read_document, write_document
import argparse
import json
import os
from pathlib import Path

import numpy as np

from calibrated_scene_reconstruction.physics.common.observations import associate, iou, mask_stats
from .scene import fit_plane, plane_is_one, read_frames, repeated_frames
from .stabilise import camera_motion, verdict, warp


class Sam3:
    """SAM 3 for several text prompts per frame: the image encoder (nine tenths of the time) runs once per frame, in
    half precision, and each prompt only decodes from its features."""

    def __init__(self, device="cuda", threshold=0.4, batch=8):
        import torch
        from calibrated_scene_reconstruction.core.perception.sam import load_sam3

        self.torch, self.device, self.threshold, self.batch = torch, device, threshold, batch
        # VIDPHYS_SAM_HALF=1: the weights in half precision too (1.6 GB instead of 3.4), for a GPU that is shared
        half = {"torch_dtype": torch.float16} if os.environ.get("VIDPHYS_SAM_HALF") else {}
        self.batch = 1 if half else batch
        self.model, self.proc = load_sam3(device, **half)

    def __call__(self, frames, prompts):
        """{prompt: [[{"mask" (H, W) bool, "score"}] per frame]} for RGB frames (F, H, W, 3)."""
        from PIL import Image

        torch, out = self.torch, {p: [] for p in prompts}
        for b in range(0, len(frames), self.batch):
            ims = [Image.fromarray(f) for f in frames[b:b + self.batch]]
            feats = None
            for p in prompts:
                inputs = self.proc(images=ims, text=[p] * len(ims), return_tensors="pt").to(self.device)
                sizes = inputs.get("original_sizes").tolist()
                with torch.no_grad():
                    if feats is None:
                        with torch.autocast("cuda", dtype=torch.float16):
                            feats = self.model.get_vision_features(pixel_values=inputs["pixel_values"])
                    with torch.autocast("cuda", dtype=torch.float16):
                        res = self.model(vision_embeds=feats, **{k: v for k, v in inputs.items()
                                                                 if k != "pixel_values"})
                for r in self.proc.post_process_instance_segmentation(res, threshold=self.threshold,
                                                                      mask_threshold=0.5, target_sizes=sizes):
                    out[p].append([{"mask": m.cpu().numpy().astype(bool), "score": float(sc)}
                                   for m, sc in zip(r["masks"], r["scores"])])
        return out

    def close(self):
        del self.model
        self.torch.cuda.empty_cache()


def track_object(cands, shape, max_jump_frac=0.15):
    """One instance per frame from SAM 3's per-frame candidates (observe.associate); (masks (F, H, W), scores)."""
    H, W = shape
    masks, scores, prev = np.zeros((len(cands), H, W), bool), np.zeros(len(cands)), None
    for i, c in enumerate(cands):
        c = associate(c, prev, max_jump_frac * np.hypot(H, W))
        if c is not None:
            masks[i], scores[i], prev = c["mask"], c["score"], mask_stats(c["mask"])[0]
    return masks, scores


def tracklets(cands, max_jump, gap=5):
    """SAM 3's per-frame instances linked over time: each instance joins the nearest tracklet seen in the last `gap`
    frames whose centroid is within max_jump px (nearest pairs first, one instance per tracklet), else starts one.
    Returns [{"frames": [f], "index": [i in cands[f]], "centroid": [(2,)]}]."""
    out = []
    for f, cs in enumerate(cands):
        cents = [mask_stats(c["mask"])[0] for c in cs]
        live = [k for k, tr in enumerate(out) if f - tr["frames"][-1] <= gap]
        pairs = sorted((np.linalg.norm(cents[i] - out[k]["centroid"][-1]), k, i) for k in live for i in range(len(cs)))
        used_k, used_i = set(), set()
        for d, k, i in pairs:
            if d > max_jump * (f - out[k]["frames"][-1]) or k in used_k or i in used_i:
                continue
            used_k.add(k), used_i.add(i)
            out[k]["frames"].append(f), out[k]["index"].append(i), out[k]["centroid"].append(cents[i])
        for i in range(len(cs)):
            if i not in used_i:
                out.append({"frames": [f], "index": [i], "centroid": [cents[i]]})
    return out


def select_object(cands, shape, max_jump_frac=0.15, static_frac=0.03):
    """The object of the video among everything the prompt matches: the one that moves. Instances that stay where
    they are for the whole video (two more tubes on a tray, the same mug on a shelf) are set aside; of the rest the
    tracklet that travels farthest is the object, and where it is missing (hidden by the hand, then picked up
    elsewhere) the best-scoring other moving instance stands in. When nothing moves, the instance SAM 3 is surest of
    on the first frame is followed (track_object). Returns (masks (F, H, W), scores (F,), info)."""
    H, W = shape
    diag = float(np.hypot(H, W))
    trs = tracklets(cands, max_jump_frac * diag)
    for tr in trs:
        c = np.array(tr["centroid"])
        tr["travel"] = float(np.linalg.norm(c.max(0) - c.min(0)))
    moving = [tr for tr in trs if tr["travel"] > static_frac * diag and len(tr["frames"]) >= 3]
    info = {"instances": len(trs), "moving": len(moving)}
    if not moving:
        masks, scores = track_object(cands, shape, max_jump_frac)
        return masks, scores, info
    moving.sort(key=lambda tr: -tr["travel"])
    masks, scores = np.zeros((len(cands), H, W), bool), np.zeros(len(cands))
    for tr in reversed(moving):  # the farthest traveller is written last: it wins the frames it shares
        for f, i in zip(tr["frames"], tr["index"]):
            if tr is moving[0] or scores[f] == 0 or cands[f][i]["score"] > scores[f]:
                masks[f], scores[f] = cands[f][i]["mask"], cands[f][i]["score"]
    for f, i in zip(moving[0]["frames"], moving[0]["index"]):
        masks[f], scores[f] = cands[f][i]["mask"], cands[f][i]["score"]
    return masks, scores, info


def fill_fast(frames, obj, score, busy, max_gap=8, level=28.0):
    """The object in the frames between two sightings where the segmenter lost it: a thing that falls is a streak
    the prompt no longer names, but it is still the one place that differs from the scene without it. Per gap of
    up to max_gap frames: the pixels far from the video's median image, near the straight line between the two
    sightings, in the component closest to where that line puts the object, when its size is the object's
    (a streak is longer). Needs a camera that stands still. Returns (obj, score, frames filled)."""
    import cv2

    seen = np.nonzero(score > 0)[0]
    if len(seen) < 4:
        return obj, score, 0
    bg = np.median(frames[:: max(1, len(frames) // 40)], 0).astype(np.float32)
    area = float(np.median(obj[seen].reshape(len(seen), -1).sum(1)))
    cen = {int(k): mask_stats(obj[k])[0] for k in seen}
    obj, score, n = obj.copy(), score.copy(), 0
    for a, b in zip(seen[:-1], seen[1:]):
        if not 1 < b - a <= max_gap + 1:
            continue
        ca, cb = cen[int(a)], cen[int(b)]
        reach = max(0.75 * np.linalg.norm(cb - ca), 3 * np.sqrt(area))
        for k in range(a + 1, b):
            pred = ca + (cb - ca) * (k - a) / (b - a)
            d = np.abs(frames[k].astype(np.float32) - bg).max(-1) > level
            d &= ~busy[k]
            nlab, lab, st, cc = cv2.connectedComponentsWithStats(d.astype(np.uint8), connectivity=8)
            best = None
            for j in range(1, nlab):
                if not 0.3 * area < st[j, cv2.CC_STAT_AREA] < 8 * area:
                    continue
                dist = float(np.linalg.norm(cc[j] - pred))
                if dist < reach and (best is None or dist < best[0]):
                    best = (dist, j)
            if best is not None:
                obj[k], score[k] = lab == best[1], 0.01
                n += 1
    return obj, score, n


def union(cands, shape):
    out = np.zeros((len(cands),) + tuple(shape), bool)
    for i, c in enumerate(cands):
        for d in c:
            out[i] |= d["mask"]
    return out


from calibrated_scene_reconstruction.core.perception.moge import load_moge as load_moge  # noqa: E402,F401


SUPPORT_FALLBACKS = ("table", "floor", "ground", "surface")


def find_support(sam, frame, prompt, obj, drop=False):
    """The support's candidate masks on one frame, best first: the largest instance of the prompt that reaches under
    the object's path, then of each generic prompt, then (no prompt names it: ice, a board, a mat) the band of
    pixels just below the object's masks over the whole video, which is support wherever the object has been.
    The first whose plane MoGe-2 sees the same in every frame is the support (plane_is_one). Of an object that
    falls (`drop`) only the lowest part of the path is on the support: the rest of it is air."""
    from scipy.ndimage import binary_dilation

    path = obj.any(0)
    if drop and path.any():
        ys = np.nonzero(path.any(1))[0]
        path = path & (np.arange(len(path))[:, None] > ys.max() - 0.2 * (ys.max() - ys.min()))
    under = binary_dilation(path, iterations=12) & ~path
    prompts = [prompt] + [p for p in SUPPORT_FALLBACKS if p != prompt]
    cands = sam(frame[None], prompts)
    out = []
    for p in prompts:
        c = [d["mask"] for d in cands[p][0] if (d["mask"] & under).sum() > 0.2 * under.sum()]
        if c:
            out.append(max(c, key=lambda m: m.sum()))
    # the surface the agent named, wherever the object's path touches it at all, before any generic one: a cube
    # that leaves a board and lands on the table spends most of its path off the board
    named = sorted(((d["mask"] & under).sum(), i) for i, d in enumerate(cands[prompt][0]))[::-1]
    named = [cands[prompt][0][i]["mask"] for n, i in named if n >= max(150, 0.03 * under.sum())]
    out = named + [m for m in out if not any(m is q for q in named)]
    return out + [binary_dilation(path, iterations=40) & ~path]


def follow_support(sam, frames, prompt, obj, f0):
    """The support's mask on every frame (F, H, W) when it moves (a board that tilts, a tray carried): the instance
    of its prompt right under the object on frame f0, followed from frame to frame by overlap."""
    from scipy.ndimage import binary_dilation

    cands = sam(frames, [prompt])[prompt]
    below = binary_dilation(obj[f0], iterations=14) & ~obj[f0]
    seed = max(cands[f0], key=lambda d: (d["mask"] & below).sum(), default=None)
    if seed is None or not (seed["mask"] & below).any():
        return None
    out = np.zeros((len(frames),) + obj.shape[1:], bool)
    out[f0] = seed["mask"]
    for order in (range(f0 + 1, len(frames)), range(f0 - 1, -1, -1)):
        prev = out[f0]
        for k in order:
            best = max(cands[k], key=lambda d: iou(d["mask"], prev), default=None)
            if best is not None and iou(best["mask"], prev) > 0.2:
                prev = best["mask"]
            out[k] = prev
    return out


def planes_in_time(frames, support_t, level, busy, K, device="cuda", model=None, most=160):
    """The support's plane on every frame, (n (F, 3), h (F,)), and a level surface's normal (F, 3) or None: MoGe-2
    on at most `most` frames spread over the video, the rest between them."""
    import torch
    from scipy.ndimage import binary_dilation, binary_erosion

    F, H, W = frames.shape[:3]
    fov = float(np.degrees(2 * np.arctan(W / 2 / K[0, 0])))
    ks = np.arange(0, F, int(np.ceil(F / most)))
    n, h, ln = np.full((len(ks), 3), np.nan), np.full(len(ks), np.nan), np.full((len(ks), 3), np.nan)
    ax, gz = np.full((len(ks), 3), np.nan), np.full(len(ks), np.nan)
    lev = None if level is None else binary_erosion(level, iterations=3)
    for j, k in enumerate(ks):
        x = torch.tensor(frames[k] / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
        with torch.no_grad():
            pts = model.infer(x, fov_x=fov)["points"].cpu().numpy()
        ok = np.all(np.isfinite(pts), -1) & ~binary_dilation(busy[k], iterations=5)
        for mask, dst in ((binary_erosion(support_t[k], iterations=3), "s"), (lev, "l")):
            if mask is None:
                continue
            P = pts[mask & ok]
            P = P[:: max(1, len(P) // 20000)]
            try:
                nn, hh, inl = fit_plane(P)
            except ValueError:
                continue
            if dst == "s":
                n[j], h[j] = nn, hh
                ax[j], gz[j] = long_axis(P, inl)
                if j and np.all(np.isfinite(ax[j - 1])) and ax[j] @ ax[j - 1] < 0:
                    ax[j] = -ax[j]
            else:
                ln[j] = nn

    def spread(a):
        a = np.asarray(a, float)
        good = np.all(np.isfinite(a.reshape(len(ks), -1)), 1)
        if good.sum() < 2:
            return np.full((F,) + a.shape[1:], np.nan)
        cols = [np.interp(np.arange(F), ks[good], c) for c in a.reshape(len(ks), -1)[good].T]
        return np.stack(cols, 1).reshape((F,) + a.shape[1:])

    n_t, ln_t, ax_t = spread(n), spread(ln), spread(ax)
    n_t /= np.linalg.norm(n_t, axis=1, keepdims=True)
    ln_t /= np.linalg.norm(ln_t, axis=1, keepdims=True)
    ax_t /= np.linalg.norm(ax_t, axis=1, keepdims=True)
    return n_t, spread(h), (None if level is None else ln_t), ax_t, spread(gz)


def long_axis(P, inl):
    """(direction (3,) of the longest extent of a plane's inlier points, how far the camera stands off the plane
    for its distance: sine of the angle it is seen at). Of a board seen along its surface the plane is known only
    up to a turn about its length, but its length points where it points."""
    Q = P[inl]
    c = Q.mean(0)
    w, v = np.linalg.eigh(np.cov((Q - c).T))
    n = v[:, 0]
    return v[:, -1], float(abs(n @ c) / max(np.linalg.norm(c), 1e-9))


def level_mask(sam, frame, prompt, support, busy):
    """The largest instance of a level surface's prompt on one frame, without what the support and the moving
    things cover of it; None when there is none."""
    c = [d["mask"] & ~support & ~busy for d in sam(frame[None], [prompt])[prompt][0]]
    c = [m for m in c if m.sum() > 0.004 * m.size]  # 2000 px of a 960 x 540 picture
    return max(c, key=lambda m: m.sum()) if c else None


def support_planes(frames, idx, support, obstacles, K=None, device="cuda", model=None, objects=None, extra=None):
    """MoGe-2 on frames[idx]: the support plane (camera frame) of each and the intrinsics (median over the frames
    when K is not given). `support` (H, W) masks the surface, `obstacles` (F, H, W) what stands on or moves over it."""
    import torch
    from scipy.ndimage import binary_dilation, binary_erosion

    H, W = frames.shape[1:3]
    fov = None if K is None else float(np.degrees(2 * np.arctan(W / 2 / K[0, 0])))
    model = model or load_moge(device)
    ns, hs, Ks, sizes = [], [], [], []
    for i in idx:
        sup = binary_erosion(support if support.ndim == 2 else support[i], iterations=3)
        x = torch.tensor(frames[i] / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
        with torch.no_grad():
            out = model.infer(x) if fov is None else model.infer(x, fov_x=fov)
        pts = out["points"].cpu().numpy()
        k = out["intrinsics"].cpu().numpy() * np.array([[W], [H], [1.0]])  # normalised -> pixels
        m = sup & ~binary_dilation(obstacles[i], iterations=5) & np.all(np.isfinite(pts), -1)
        P = pts[m]
        P = P[:: max(1, len(P) // 20000)]
        n, h, inl = fit_plane(P)
        ns.append(n), hs.append(h), Ks.append(k)
        if extra is not None:  # the plane's long axis and how steeply it is seen (long_axis)
            a, g = long_axis(P, inl)
            extra.setdefault("axis", []).append(a), extra.setdefault("graze", []).append(g)
            extra.setdefault("object_xyz", []).append(object_place(pts, objects[i]) if objects is not None
                                                      else np.full(3, np.nan))
        sizes.append(object_size(pts, objects[i]) if objects is not None else np.nan)
    return (np.array(ns), np.array(hs), (np.median(Ks, 0) if K is None else np.asarray(K, float)),
            np.array(sizes))


def object_places(frames, idx, K, device, model, objects):
    """Where MoGe-2 sees the object on frames[idx] ((len(idx), 3), camera frame, m; NaN rows where it shows too
    little): object_place on more frames than the support's plane is fitted on, for what moves on no plane that is
    seen and has to be followed by its own places."""
    import torch

    W = frames.shape[2]
    fov = float(np.degrees(2 * np.arctan(W / 2 / K[0, 0])))
    out = []
    for i in idx:
        x = torch.tensor(frames[i] / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
        with torch.no_grad():
            pts = model.infer(x, fov_x=fov)["points"].cpu().numpy()
        out.append(object_place(pts, objects[i]))
    return np.array(out)


def object_place(pts, mask):
    """Where MoGe-2 sees the object (camera frame, m at its scale): the median of the points of its mask, NaN
    when it shows too little. What stands on no plane that is seen (a ball in flight, a book on a board seen
    along itself) is placed by this."""
    from scipy.ndimage import binary_erosion

    m = binary_erosion(mask, iterations=2) & np.all(np.isfinite(pts), -1)
    if m.sum() < 30:
        m = mask & np.all(np.isfinite(pts), -1)
    return np.median(pts[m], 0).astype(float) if m.sum() >= 10 else np.full(3, np.nan)


def object_size(pts, mask):
    """The object's largest dimension (m, at MoGe-2's scale) from the points of its mask: their extent along their
    principal axis, 1st to 99th percentile. Only the surface the camera sees is measured, which for the largest
    dimension of most objects (a bottle's height, a lying can's length) is the whole of it."""
    from scipy.ndimage import binary_erosion

    m = binary_erosion(mask, iterations=2) & np.all(np.isfinite(pts), -1)
    P = pts[m]
    if len(P) < 30:
        return np.nan
    P = P.astype(np.float64) - np.median(P, 0)
    P = P[np.linalg.norm(P, axis=1) < 10 * np.median(np.linalg.norm(P, axis=1)) + 1e-9]  # depth flying off an edge
    try:
        ax = np.linalg.eigh(np.cov(P.T))[1][:, -1]
    except np.linalg.LinAlgError:
        return np.nan
    lo, hi = np.percentile(P @ ax, [1, 99])
    return float(hi - lo)


OBJECT_FRAMES = 32  # on how many frames the object is placed in depth by its own points


def run(path, moge_frames=5, device="cuda", use_meta_K=True, sam=None, moge=None):
    """Perception of one recording; pass `sam` (observe.Sam3) and `moge` (load_moge) to share them between runs."""
    path = Path(path)
    meta = read_document(path / "meta.json")
    (path / "timing.json").unlink(missing_ok=True)
    rep = repeated_frames(read_frames(path, stabilised=False, raw=True)[0])
    if rep is not None:
        write_document(path / "timing.json", {"drop": [int(k) for k in rep]})
    frames, fps = read_frames(path, stabilised=False)
    (path / "stabilised.npz").unlink(missing_ok=True)
    F, H, W = frames.shape[:3]
    sam = sam or Sam3(device)
    prompts = [meta["object"]] + [p for p in meta.get("manipulator", []) if p != meta["object"]]
    cands = sam(frames, prompts)
    obj, score, picked = select_object(cands[meta["object"]], (H, W))
    if meta.get("object_at"):  # the agent's point on the object in the first frame: that instance, followed
        from .soft import follow_from

        obj, score = follow_from(cands[meta["object"]], (meta["object_at"][0] * W, meta["object_at"][1] * H), (H, W))
    manip = np.zeros_like(obj)
    for p in prompts[1:]:
        manip |= union(cands[p], (H, W))
    manip &= ~obj
    Hs, shift, inl = camera_motion(frames, obj | manip, ref=int(np.argmax(score > 0)))
    camera = verdict(shift, inl, width=W)
    if camera == "stabilised":  # every later stage sees the reference frame's camera
        Hs[inl == 0] = np.eye(3)
        frames, (obj, manip) = warp(frames, Hs, (obj, manip))
        np.savez_compressed(path / "stabilised.npz", H=Hs)
    ref = int(np.argmax(score > 0))
    filled = 0
    if camera != "moving":
        obj, score, filled = fill_fast(frames, obj, score, manip)
    seen = np.nonzero(score > 0.05)[0]  # MoGe-2 on frames that show the object (its size is measured on them)
    idx = np.unique(seen[np.linspace(0, len(seen) - 1, moge_frames).round().astype(int)]) if len(seen) else np.array([0])
    K0 = np.asarray(meta["K"], float) if use_meta_K and "K" in meta else None
    moge = moge or load_moge(device)
    from scipy.ndimage import binary_dilation

    # where the object is on its own: carried in a hand it is over no support
    alone = np.array([not (binary_dilation(o, iterations=6) & m).any() for o, m in zip(obj, manip)])
    support_t = None
    if meta.get("support_moves"):  # followed from the first frame the object is on it by itself
        own = np.nonzero(alone & (score > 0.05))[0]
        if len(own):
            support_t = follow_support(sam, frames, meta.get("support", "table"), obj, int(own[0]))
    cand = ([support_t] if support_t is not None else
            find_support(sam, frames[ref], meta.get("support", "table"),
                         obj[alone] if alone.sum() >= 3 else obj, meta.get("motion") == "drop"))
    # the first candidate that MoGe-2 sees as one plane in every frame; none: the stages that stand the object on
    # the support refuse (plane_ok), and nothing is said of a plane that is a guess. The last candidate of
    # find_support is no surface anyone named, only the pixels about the object's path (support_band)
    plane_ok, band, ex, n, h, K, size, support = False, False, {}, None, None, None, None, cand[0]
    for j, c in enumerate(cand):
        try:
            ex_ = {}
            n_, h_, K_, size_ = support_planes(frames, idx, c, obj | manip, K0, device, moge, obj & ~manip, ex_)
        except ValueError:  # no finite points under this mask
            continue
        if n is None or plane_is_one(n_, h_):
            support, n, h, K, size, ex = c, n_, h_, K_, size_, ex_
            band = support_t is None and j == len(cand) - 1
        if plane_is_one(n_, h_):
            plane_ok = True
            break
    if n is None:
        raise ValueError("MoGe-2 gives no points on any surface under the object")
    level = {}
    if meta.get("level"):  # a surface that is level, beside a support that is not: gravity's direction (incline.py)
        from scipy.ndimage import binary_dilation

        sup_all = support if support.ndim == 2 else support.any(0)
        lm = level_mask(sam, frames[ref], meta["level"], binary_dilation(sup_all, iterations=8),
                        obj.any(0) | manip.any(0))
        if lm is not None:
            try:
                ln, lh, _K, _s = support_planes(frames, idx, lm, obj | manip, K, device, moge)
                if plane_is_one(ln, lh):
                    level = {"level": lm, "level_n": ln, "level_h": lh}
            except ValueError:
                pass
    if support_t is not None:  # every frame's plane: track.py puts the object in the support's own frame
        n_t, h_t, ln_t, ax_t, gz_t = planes_in_time(frames, support_t, level.get("level"), obj | manip, K, device, moge)
        level.update({"support_t": support_t, "plane_n_t": n_t, "plane_h_t": h_t, "plane_axis_t": ax_t,
                      "plane_graze_t": gz_t},
                     **({} if ln_t is None else {"level_n_t": ln_t}))
        support = support_t[ref]
    # the object's own places on up to OBJECT_FRAMES frames that show it clear of the manipulator
    clear = np.nonzero((score > 0.05) & alone)[0]
    at = np.unique(clear[np.linspace(0, len(clear) - 1, OBJECT_FRAMES).round().astype(int)]) if len(clear) else idx[:0]
    at_xyz = object_places(frames, at, K, device, moge, obj & ~manip) if len(at) else np.zeros((0, 3))
    out = {"t": np.arange(F) / fps, "object": obj, "score": score, "manipulator": manip, "support": support, "K": K,
           "object_at_frames": at, "object_at_xyz": at_xyz,
           **level, "plane_axis": np.array(ex.get("axis", [])), "plane_graze": np.array(ex.get("graze", [])),
           "object_xyz": np.array(ex.get("object_xyz", [])),
           "plane_n": n, "plane_h": h, "plane_frames": idx, "object_size": size, "ref_frame": ref, "camera": camera, "camera_shift": shift,
           "camera_inliers": inl, "instances": picked["instances"], "moving_instances": picked["moving"],
           "filled": filled, "plane_ok": plane_ok, "support_band": band}
    np.savez_compressed(path / "perception.npz", **out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording", nargs="+")
    ap.add_argument("--moge-frames", type=int, default=5)
    ap.add_argument("--skip-done", action="store_true", help="skip recordings that have perception.npz")
    ap.add_argument("--estimate-K", action="store_true", help="ignore meta.json K: intrinsics from MoGe-2")
    args = ap.parse_args()
    sam, moge = Sam3(), load_moge()
    for r in args.recording:
        if args.skip_done and (Path(r) / "perception.npz").exists():
            continue
        try:
            o = run(r, args.moge_frames, use_meta_K=not args.estimate_K, sam=sam, moge=moge)
        except Exception as e:  # noqa: BLE001  one unusable video does not stop the batch
            print(f"{r}: FAILED {type(e).__name__}: {e}", flush=True)
            continue
        if o["camera"] != "static":
            print(f"{r}: camera {o['camera']}: background moves up to {o['camera_shift'].max():.1f} px")
        print(f"{r}: object in {int((o['score'] > 0).sum())}/{len(o['t'])} frames, manipulator in "
              f"{int(o['manipulator'].any((1, 2)).sum())}, camera height {np.median(o['plane_h']):.3f} m "
              f"(+- {np.std(o['plane_h']):.3f} over {len(o['plane_h'])} frames), fx {o['K'][0, 0]:.0f} px")


if __name__ == "__main__":
    main()
