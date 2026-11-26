"""Object observations from the fixed real camera, and the camera model shared with the sim side.

    python -m calibrated_scene_reconstruction.physics.replay.observe real/<name> [--every 1] [--depth]

Writes real/<name>/track.npz: t (F,) video seconds, valid (F,), centroid (F, 2) px, bbox (F, 4) x0 y0 x1 y1, area (F,),
score (F,), masks (F, H/s, W/s) bool at stride s, optional p_world (F, 3) depth centroid. Only track_video() needs the
GPU (SAM 3 per frame, MoGe-2 for --depth); everything else is numpy.
"""
from calibrated_scene_reconstruction.physics.common.observations import world_to_cam as world_to_cam, project as project, backproject as backproject, cam_to_world as cam_to_world, mask_stats as mask_stats, iou as iou, depth_centroid as depth_centroid, silhouette as silhouette, mesh_silhouette as mesh_silhouette, convex_silhouette as convex_silhouette, associate as associate, track_from_masks as track_from_masks, read_video as read_video  # noqa: F401
import argparse

import numpy as np


# --- camera -----------------------------------------------------------------------------------------------------
from calibrated_scene_reconstruction.core.geometry import mask_iou









# --- masks ------------------------------------------------------------------------------------------------------
















# --- GPU ---------------------------------------------------------------------------------------------------------


class Sam3:
    """SAM 3 loaded once (recon.perceive.run_sam3 loads it per call): text-prompted instances of a batch of frames."""

    def __init__(self, device="cuda", threshold=0.4, batch=8):
        import torch
        from calibrated_scene_reconstruction.core.perception.sam import load_sam3

        self.torch, self.device, self.threshold, self.batch = torch, device, threshold, batch
        self.model, self.proc = load_sam3(device)

    def __call__(self, frames, prompt):
        """[[{"mask" (H, W) bool, "score"}] per frame] for RGB frames (F, H, W, 3)."""
        from PIL import Image

        out = []
        for b in range(0, len(frames), self.batch):
            ims = [Image.fromarray(f) for f in frames[b:b + self.batch]]
            inputs = self.proc(images=ims, text=[prompt] * len(ims), return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                res = self.model(**inputs)
            for r in self.proc.post_process_instance_segmentation(
                    res, threshold=self.threshold, mask_threshold=0.5,
                    target_sizes=inputs.get("original_sizes").tolist()):
                out.append([{"mask": m.cpu().numpy().astype(bool), "score": float(s)}
                            for m, s in zip(r["masks"], r["scores"])])
        return out

    def close(self):
        del self.model
        self.torch.cuda.empty_cache()


def track_video(rec, prompt=None, every=1, depth=False, device="cuda", stride=2, threshold=0.4, sam=None):
    """SAM 3 masks of the recording's object in every `every`-th video frame (one model load, frames in batches) ->
    track dict (module doc). The first frame's instance is the highest-scoring one; later frames follow it (associate).
    TODO: SAM 3's video tracker (temporal memory) instead of per-frame detection; undistort frames when camera.json has
    `dist`."""
    from calibrated_scene_reconstruction.core.perception.moge import run_moge

    frames, fps = read_video(rec.video, every)
    prompt = prompt or rec.meta["object"]
    own = sam is None
    sam = sam or Sam3(device, threshold)
    cands = sam(frames, prompt)
    if own:
        sam.close()
    masks, scores, prev = [], [], None
    max_jump = 0.08 * np.hypot(*frames.shape[1:3])  # px per (sampled) frame: 29 at 320 x 180, 64 at 640 x 480
    for c in cands:
        c = associate(c, prev, max_jump)
        masks.append(None if c is None else c["mask"])
        scores.append(0.0 if c is None else c["score"])
        prev = prev if c is None else mask_stats(c["mask"])[0]
    tr = track_from_masks(masks, scores, np.arange(len(frames)) / fps)
    empty = np.zeros(frames.shape[1:3], bool)
    tr["masks"] = np.stack([empty if m is None else m for m in masks])[:, ::stride, ::stride]
    tr["mask_stride"] = stride
    if depth:  # MoGe-2 metric scale drifts per frame: use its centroid for event timing and sanity checks, not as truth
        tr["p_world"] = np.stack([depth_centroid(m, run_moge(f, device)["points"], rec.T_world_cam) if m is not None
                                  else np.full(3, np.nan) for f, m in zip(frames, masks)])
    return tr


def main():
    from .data import load

    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--prompt")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--depth", action="store_true")
    ap.add_argument("--stride", type=int, default=2, help="mask downsampling in track.npz (1 for small images)")
    args = ap.parse_args()
    rec = load(args.recording)
    tr = track_video(rec, args.prompt, args.every, args.depth, stride=args.stride)
    out = rec.path / "track.npz"
    np.savez_compressed(out, **tr)
    print(f"{out}: {int(tr['valid'].sum())}/{len(tr['valid'])} frames tracked")


if __name__ == "__main__":
    main()
