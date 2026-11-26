"""GPU half of video physics for bodies that deform: what the video shows of them.

    envs/recon/bin/python -m calibrated_scene_reconstruction.physics.video.soft real/<name>

A body that deforms is not on a support the way a sliding box is, so perceive.py's search for the surface under the
object's path does not apply. meta.json names instead
  "object": SAM 3 prompt of the whole deformable body (or several: their instances together),
  "object_at": [x, y], a point on the body in the first frame as fractions of the picture's width and height, for
      a scene in which the prompt names other things too (a black cloth before black curtains): the agent that
      watched the video points at the one that is meant, and it is followed from there,
  "manipulator": [...] as in perceive.py,
  "level": prompt of a surface that is level (a table, the floor): gravity's direction and the camera's height,
  "parts": {"soft": prompt of the parts that deform, when the body is made of stiff and soft parts by turns}.

A level surface gives gravity's direction (a beam, a sheet); a strain is read without one, and its plane_n,
plane_h are then NaN. A camera that only shakes is undone as in perceive.py (stabilised.npz).

Writes perception.npz with perceive.py's fields (t, object, score, manipulator, K, plane_n, plane_h, plane_frames,
camera, ref_frame; `support` is the level surface) and
  object_points (N, 3), object_uv (N, 2): MoGe-2's points of the object on the reference frame (camera frame, m) and
  their pixels: where in depth the body is, which no plane gives for a thing held in the air;
  soft_parts (P, H, W) bool: the soft parts on the reference frame, one mask each.
"""
from calibrated_scene_reconstruction.physics.common.schema import read_document
import argparse
import json
from pathlib import Path

import numpy as np

from .perceive import Sam3, fill_fast, load_moge, plane_is_one, select_object, support_planes, union
from .scene import read_frames
from .stabilise import camera_motion, verdict, warp


def level_surface(sam, frame, prompt, busy):
    """The largest instance of the level surface's prompt, less what stands on it."""
    c = sam(frame[None], [prompt])[prompt][0]
    if not c:
        return None
    m = max((d["mask"] for d in c), key=lambda m: (m & ~busy).sum())
    return m & ~busy


def fill_still(cands, obj, score, least=0.5):
    """The object in the frames where select_object, which follows what moves, has not got it: a body held still
    before it is let go is another instance to it. From every frame that has the object outwards, the candidate
    that overlaps the neighbouring frame's mask by `least` is the same thing. Returns (obj, score, frames added)."""
    from calibrated_scene_reconstruction.physics.common.observations import iou

    obj, score, n = obj.copy(), score.copy(), 0
    for order in (range(len(obj) - 2, -1, -1), range(1, len(obj))):
        step = 1 if order.start > order.stop else -1  # the neighbour already passed
        for k in order:
            if score[k] > 0 or not score[k + step] > 0:
                continue
            best = max(cands[k], key=lambda d: iou(d["mask"], obj[k + step]), default=None)
            if best is not None and iou(best["mask"], obj[k + step]) > least:
                obj[k], score[k] = best["mask"], best["score"]
                n += 1
    return obj, score, n


def soft_parts(sam, frames, obj, score, ref, prompt, threshold=0.2):
    """The soft parts of the body at rest, one mask each (P, H, W): the prompt's instances that lie on the body,
    are not the whole of it and do not repeat one another, on the reference frame or, where the segmenter finds
    fewer there, on a frame after it while the body has not moved. The prompt names a part of a thing, which SAM 3
    is less sure of than of things: its threshold is lowered for this question."""
    from calibrated_scene_reconstruction.physics.common.observations import iou

    F, H, W = obj.shape
    best = np.zeros((0, H, W), bool)
    if not prompt:
        return best
    same = [k for k in range(ref, min(ref + 30, F)) if score[k] > 0 and
            (obj[k] & obj[ref]).sum() > 0.9 * (obj[k] | obj[ref]).sum()]
    keep, sam.threshold = sam.threshold, threshold
    try:
        for k in same[:1] + same[2::4][:5]:
            c = sorted(sam(frames[k][None], [prompt])[prompt][0], key=lambda d: -d["score"])
            on = []
            for d in c:
                m = d["mask"]
                if (m & obj[ref]).sum() > 0.6 * m.sum() and m.sum() < 0.7 * obj[ref].sum() and \
                        all(iou(m, q) < 0.3 for q in on):
                    on.append(m)
            if len(on) > len(best):
                best = np.stack(on)
    finally:
        sam.threshold = keep
    return best


def redo_parts(path, sam):
    """perception.npz's soft_parts again, after the agent changed the parts' prompt in meta.json."""
    path = Path(path)
    meta = read_document(path / "meta.json")
    per = dict(np.load(path / "perception.npz"))
    frames, _fps = read_frames(path)
    per["soft_parts"] = soft_parts(sam, frames, per["object"], per["score"], int(per["ref_frame"]),
                                   meta.get("parts", {}).get("soft"))
    np.savez_compressed(path / "perception.npz", **per)
    return per


def follow_from(cands, at, shape, least=0.25):
    """The instance that has the pixel `at` (x, y) in the first frame that shows one there, followed frame by frame
    as the candidate that overlaps it most (at least `least`): (masks (F, H, W), scores (F,)). Of several
    instances on the point the surest, of equally sure ones the smallest (a thing before a backdrop). A point
    beside a thin thing counts as on it within a hundredth of the picture's diagonal."""
    from calibrated_scene_reconstruction.physics.common.observations import iou

    H, W = shape
    u, v = int(np.clip(at[0], 0, W - 1)), int(np.clip(at[1], 0, H - 1))
    r = max(int(0.01 * np.hypot(H, W)), 1)
    v0, v1, u0, u1 = max(v - r, 0), v + r + 1, max(u - r, 0), u + r + 1
    obj, score, prev = np.zeros((len(cands), H, W), bool), np.zeros(len(cands)), None
    for k, cs in enumerate(cands):
        if prev is None:
            on = [d for d in cs if d["mask"][v, u]] or [d for d in cs if d["mask"][v0:v1, u0:u1].any()]
            if on:
                best = max(on, key=lambda d: (round(d["score"], 1), -d["mask"].sum()))
                obj[k], score[k], prev = best["mask"], best["score"], best["mask"]
            continue
        best = max(cs, key=lambda d: iou(d["mask"], prev), default=None)
        if best is not None and iou(best["mask"], prev) > least:
            obj[k], score[k], prev = best["mask"], best["score"], best["mask"]
            continue
        # it moved by more than its own size since it was last seen (it falls, it is thrown): the candidate of its
        # size nearest to where it was, within what a thing crosses of the picture between two sightings
        c0, a0 = np.argwhere(prev).mean(0), prev.sum()
        near = [(np.linalg.norm(np.argwhere(d["mask"]).mean(0) - c0), d) for d in cs
                if d["mask"].any() and 0.5 * a0 < d["mask"].sum() < 2 * a0]
        near = [x for x in near if x[0] < 0.15 * np.hypot(H, W)]
        if near:
            best = min(near, key=lambda x: x[0])[1]
            obj[k], score[k], prev = best["mask"], best["score"], best["mask"]
    return obj, score


def run(path, moge_frames=3, device="cuda", sam=None, moge=None):
    import torch
    from scipy.ndimage import binary_dilation, binary_erosion

    path = Path(path)
    meta = read_document(path / "meta.json")
    frames, fps = read_frames(path, stabilised=False)
    F, H, W = frames.shape[:3]
    sam = sam or Sam3(device)
    names = [meta["object"]] if isinstance(meta["object"], str) else list(meta["object"])
    hands = [p for p in meta.get("manipulator", []) if p not in names]
    cands = sam(frames, names + hands)
    body = [sum((cands[p][k] for p in names), []) for k in range(F)]  # every prompt's instances, frame by frame
    if meta.get("object_at"):
        obj, score = follow_from(body, (meta["object_at"][0] * W, meta["object_at"][1] * H), (H, W))
        picked, still = {"instances": 1, "moving": 1}, 0
    else:
        obj, score, picked = select_object(body, (H, W))
        obj, score, still = fill_still(body, obj, score)
    if not (score > 0).any():
        raise ValueError("the body is not found: no instance of its prompt" + (" at the point meta.json gives"
                                                                             if meta.get("object_at") else ""))
    manip = np.zeros_like(obj)
    for p in hands:
        manip |= union(cands[p], (H, W))
    manip &= ~obj
    ref = int(np.argmax(score > 0))
    Hs, shift, inl = camera_motion(frames, obj | manip, ref=ref)
    camera = verdict(shift, inl, width=W)
    (path / "stabilised.npz").unlink(missing_ok=True)
    if camera == "stabilised":  # every later stage sees the reference frame's camera (as perceive.py does it)
        Hs[inl == 0] = np.eye(3)
        frames, (obj, manip) = warp(frames, Hs, (obj, manip))
        np.savez_compressed(path / "stabilised.npz", H=Hs)
    obj, score, filled = fill_fast(frames, obj, score, manip)
    moge = moge or load_moge(device)
    busy = binary_dilation(obj.any(0) | manip.any(0), iterations=6)
    lm = level_surface(sam, frames[ref], meta.get("level", "table"), busy)
    seen = np.nonzero(score > 0.05)[0]
    idx = np.unique(seen[np.linspace(0, len(seen) - 1, moge_frames).round().astype(int)])
    planes, why = None, "no level surface is seen: nothing gives gravity's direction"
    if lm is not None:
        try:
            planes = support_planes(frames, idx, lm, obj | manip, None, device, moge, obj & ~manip)
            if not plane_is_one(planes[0], planes[1]):
                planes, why = None, "the level surface is not one plane in MoGe-2's points"
        except ValueError as e:  # no finite points under its mask
            why = str(e)
    if planes is not None:
        n, h, K, size = planes
    elif meta.get("motion") == "strain":
        # a strain is read in the picture and about the body's own axis: gravity does not enter, and a body held
        # up before a wall, or in a machine, has no level surface beside it (or one that is no plane in the
        # points). The intrinsics and the body's size from MoGe-2 alone
        from .perceive import object_size

        Ks, sizes = [], []
        for i in idx:
            with torch.no_grad():
                o = moge.infer(torch.tensor(frames[i] / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1))
            Ks.append(o["intrinsics"].cpu().numpy() * np.array([[W], [H], [1.0]]))
            sizes.append(object_size(o["points"].cpu().numpy(), obj[i] & ~manip[i]))
        n, h, K, size = np.full((len(idx), 3), np.nan), np.full(len(idx), np.nan), np.median(Ks, 0), np.array(sizes)
        lm = np.zeros((H, W), bool)
    else:
        raise ValueError(why)
    fov = float(np.degrees(2 * np.arctan(W / 2 / K[0, 0])))
    x = torch.tensor(frames[ref] / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
    with torch.no_grad():
        pts = moge.infer(x, fov_x=fov)["points"].cpu().numpy()
    m = binary_erosion(obj[ref], iterations=2) & np.all(np.isfinite(pts), -1)
    vs, us = np.nonzero(m)
    keep = slice(None, None, max(1, len(us) // 5000))
    soft = soft_parts(sam, frames, obj, score, ref, meta.get("parts", {}).get("soft"))
    out = {"t": np.arange(F) / fps, "object": obj, "score": score, "manipulator": manip, "support": lm, "K": K,
           "plane_n": n, "plane_h": h, "plane_frames": idx, "object_size": size, "ref_frame": ref, "camera": camera,
           "camera_shift": shift, "camera_inliers": inl, "instances": picked["instances"],
           "moving_instances": picked["moving"], "filled": filled, "still": still,
           "object_points": pts[vs[keep], us[keep]], "object_uv": np.stack([us[keep], vs[keep]], 1).astype(float),
           "soft_parts": soft}
    np.savez_compressed(path / "perception.npz", **out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording", nargs="+")
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--parts", action="store_true", help="only the soft parts again (meta.json's prompt changed)")
    args = ap.parse_args()
    sam = Sam3()
    if args.parts:
        for p in args.recording:
            print(f"{p}: {len(redo_parts(p, sam)['soft_parts'])} soft parts", flush=True)
        return
    moge = load_moge()
    for p in args.recording:
        if args.skip_done and (Path(p) / "perception.npz").exists():
            continue
        try:
            o = run(p, sam=sam, moge=moge)
            print(f"{p}: object in {int((o['score'] > 0).sum())}/{len(o['score'])} frames ({o['filled']} filled), "
                  f"camera {o['camera']}, height {np.median(o['plane_h']):.2f} m, "
                  f"{len(o['soft_parts'])} soft parts", flush=True)
        except Exception as e:  # one recording's failure is its own
            print(f"{p}: FAILED {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
