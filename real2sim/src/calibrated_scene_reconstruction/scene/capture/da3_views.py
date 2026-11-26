"""Depth Anything 3 on the session photo and its views: camera poses and depth of all of them in one frame.

    PYTHONPATH=src envs/da3/bin/python -m calibrated_scene_reconstruction.recon.da3_views sessions/<name> [--process-res 1008] [--metric] [--fp32]

Runs in envs/da3 (setup/setup_da3_env.sh), called by recon.multiview. Reads capture/rgb.png (always the first image)
and capture/views/*.png. DA3 takes one image size for the whole batch (images of other sizes it centre-crops to the
smallest common square), so every image is brought here to the photo's processing size first: turned by 90 degrees
when its orientation differs from the photo's (`rot`: np.rot90 / PIL counter-clockwise quarter turns), scaled to
cover that size (`scale`) and centre-cropped (`x0`, `y0`). Writes parsing/multiview/da3.npz: names, depth (N, h, w),
conf, K (N, 3, 3, of the prepared images), w2c (N, 3, 4, OpenCV), prep (N, 4: rot, scale, x0, y0), size (N, 2: the
captured W, H), is_metric.

Memory: the nested checkpoint holds two giant models, the any-view one (poses, depth up to scale) and a metric one
whose scale only sets the depth's units; recon.multiview takes its units from the robot instead, so the metric
branch is left out unless `--metric`. The any-view backbone runs in bf16
weights (`--fp32` keeps them: depth differs by 0.3 % median, poses by under 0.1 degree) and its DPT head two views
at a time instead of eight: 16 views of a 1008 px video then take 7.2 GiB instead of more than the card, and the
stage can share the card with another job (SCENE_RECONSTRUCTION_GPU_FRACTION, calibrated_scene_reconstruction.gpu)."""
import argparse
import functools
import gc
from pathlib import Path

import numpy as np
from PIL import Image

from calibrated_scene_reconstruction.core.gpu import device, share


def prepare(im, Wt, Ht):
    rot = int((im.width > im.height) != (Wt > Ht))
    if rot:
        im = im.rotate(90, expand=True)
    s = max(Wt / im.width, Ht / im.height)
    w, h = round(im.width * s), round(im.height * s)
    x0, y0 = (w - Wt) // 2, (h - Ht) // 2
    return im.resize((w, h), Image.LANCZOS).crop((x0, y0, x0 + Wt, y0 + Ht)), (rot, s, x0, y0)


def load_model(name, metric=False, fp32=False, head_chunk=2):
    import torch
    import torch.utils._pytree as pt
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(name).to(device()).eval()
    nested = hasattr(model.model, "da3")
    if nested and not metric:
        model.model = model.model.da3
        gc.collect()
        torch.cuda.empty_cache()
    nets = [model.model.da3, model.model.da3_metric] if hasattr(model.model, "da3") else [model.model]
    for net in nets:
        net.head.forward = functools.partial(net.head.forward, chunk_size=head_chunk)
        if not fp32:  # bf16 weights in the backbone, its features back to float for the heads
            net.backbone.to(torch.bfloat16)
            fwd = net.backbone.forward
            net.backbone.forward = (lambda f: lambda *a, **k: pt.tree_map(
                lambda t: t.float() if torch.is_tensor(t) and t.dtype == torch.bfloat16 else t, f(*a, **k)))(fwd)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--model", default="depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    ap.add_argument("--process-res", type=int, default=1008)
    ap.add_argument("--metric", action="store_true", help="run the metric branch too (DA3's own depth units)")
    ap.add_argument("--fp32", action="store_true", help="fp32 backbone weights (twice the memory)")
    a = ap.parse_args()
    import torch

    share()
    s = Path(a.session)
    paths = [s / "capture" / "rgb.png"] + sorted((s / "capture" / "views").glob("*.png"))
    names = ["photo"] + [p.stem for p in paths[1:]]
    imgs = [Image.open(p).convert("RGB") for p in paths]
    W0, H0 = imgs[0].size
    f = a.process_res / max(W0, H0)
    Wt, Ht = round(W0 * f / 14) * 14, round(H0 * f / 14) * 14
    prepped, prep = zip(*[prepare(im, Wt, Ht) for im in imgs])
    model = load_model(a.model, a.metric, a.fp32 or device() == "cpu")  # CPU autocast rejects bf16 weights
    if device() == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        pred = model.inference(list(prepped), process_res=max(Wt, Ht), process_res_method="upper_bound_resize",
                               ref_view_strategy="first")
    assert pred.depth.shape[1:] == (Ht, Wt), pred.depth.shape
    is_metric = int(pred.is_metric) if pred.is_metric not in (None, {}) else 0
    out = s / "parsing" / "multiview"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "da3.npz", names=np.array(names), depth=pred.depth.astype(np.float32),
                        conf=pred.conf.astype(np.float32), K=pred.intrinsics.astype(np.float32),
                        w2c=pred.extrinsics[:, :3, :4].astype(np.float32), prep=np.array(prep, float),
                        size=np.array([im.size for im in imgs]), is_metric=is_metric)
    for n, d, K, p in zip(names, pred.depth, pred.intrinsics, prep):
        print(f"[da3] {n}: {Wt}x{Ht} (turned {90 * p[0]} deg, x{p[1]:.3f}, crop {p[2]},{p[3]}) fx {K[0, 0]:.1f} "
              f"depth {np.percentile(d, 5):.2f}-{np.percentile(d, 95):.2f}")
    peak = f", peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB" if device() == "cuda" else ", on the CPU"
    print(f"done: {len(names)} views, metric {is_metric}{peak}")


if __name__ == "__main__":
    main()
