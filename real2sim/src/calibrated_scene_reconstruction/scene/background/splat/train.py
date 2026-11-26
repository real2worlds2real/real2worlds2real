"""Splat background, step 2: 3DGRUT trains the scene's Gaussians on splat/data (recon.splat_capture's frames and
poses): 3DGUT (rasterized Gaussians with the unscented transform, which NuRec renders the same way in Isaac Sim),
the MCMC strategy (a fixed budget, `max_gaussians`, relocated to where the images need it, fewer floaters than
densification), every `test_every`-th frame held out and scored (PSNR, SSIM, LPIPS) at the end.

    tools/run_splat.sh -m calibrated_scene_reconstruction.recon.splat_train sessions/<name>

Writes the checkpoint and test renders to splat/runs/bg/<run>/, the training record to splat/train.json,
and the held-out and coverage previews to qa/splat/.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time

import numpy as np

from calibrated_scene_reconstruction.core.paths import ROOT
from calibrated_scene_reconstruction.scene.session import Session
from calibrated_scene_reconstruction.scene.background.splat.config import folder, settings


def valid_regions(img_dir, n=80, thr=12, erode=6):
    """{video: bool mask} of the pixels an undistorted frame really has: the image's own border, where undistortion
    pulled in nothing, is black in every frame of a video (the max over n of its frames stays under `thr`)."""
    import cv2

    out = {}
    by_video = {}
    for p in sorted(img_dir.glob("*.jpg")):
        by_video.setdefault(p.name.split("_")[0], []).append(p)
    for v, ps in by_video.items():
        mx = None
        for p in ps[:: max(1, len(ps) // n)]:
            g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            mx = g if mx is None else np.maximum(mx, g)
        ok = (mx > thr).astype(np.uint8)
        out[v] = cv2.erode(ok, np.ones((2 * erode + 1, 2 * erode + 1), np.uint8)) > 0
    return out


def loss_masks(img_dir, masked, session=None):
    """Every frame's <image>_mask.png for 3DGRUT (255 learnt, 0 left out of the loss): the frame's valid region, and, `masked`,
    the SAM 3 masks of recon.splat_masks (kept as <image>_sam.png)."""
    import cv2

    marker = img_dir.parent / "loss_masks.json"
    if marker.exists():  # the masks of an earlier training: written here, not SAM's
        for f in img_dir.glob("*_mask.png"):
            f.unlink()
    for f in img_dir.glob("*_mask.png"):  # recon.splat_masks writes <image>_mask.png
        f.rename(f.with_name(f.name.replace("_mask.png", "_sam.png")))
    from calibrated_scene_reconstruction.scene.capture.valid_pixels import ValidPixels
    validity=ValidPixels(session) if session is not None else None
    valid = valid_regions(img_dir) if validity is None else None
    shares = []
    for p in sorted(img_dir.glob("*.jpg")):
        m = (validity.get(p.name,cv2.imread(str(p),0).shape).copy() if validity is not None
             else valid[p.name.split("_")[0]].copy())
        sam = p.with_name(p.stem + "_sam.png")
        if masked and sam.exists():
            m &= cv2.imread(str(sam), cv2.IMREAD_GRAYSCALE) > 127
        cv2.imwrite(str(p.with_name(p.stem + "_mask.png")), m.astype(np.uint8) * 255)
        shares.append(float(m.mean()))
    marker.write_text(json.dumps({"masked": bool(masked), "learnt_share": round(float(np.mean(shares)), 4),
                    'validity_policy':'camera remap where available; full raster otherwise' if validity else 'legacy image-colour heuristic',
                    'rectification_inputs':validity.inputs if validity else {}}) + "\n")
    print(f"[train] loss masks: {100 * np.mean(shares):.1f} % of the pixels learnt"
          + (" (SAM 3 masks included)" if masked else " (the frames' valid regions)"), flush=True)


def background_metrics(data, run, cfg,session=None):
    """PSNR on the held-out frames over the background alone and over the task zone's things (the SAM 3 masks of
    recon.splat_masks, whichever suffix they carry): the whole-frame PSNR falls with the zone's things, which the
    background need not learn, and a masked run scores 14 dB there while its background is as before."""
    from PIL import Image

    img_dir = data / "images"
    names = sorted(p.name for p in img_dir.glob("*.jpg"))[::cfg["test_every"]]
    renders = sorted((run / f"ours_{cfg['iterations']}" / "renders").glob("*.png"))
    bg, zone = [], []
    from calibrated_scene_reconstruction.scene.capture.valid_pixels import ValidPixels
    validity=ValidPixels(session) if session is not None else None
    for k, n in enumerate(names[:len(renders)]):
        m_path = img_dir / (n[:-4] + "_sam.png")  # SAM 3's zone masks (loss_masks keeps them under this name)
        m_path = m_path if m_path.exists() else None
        if m_path is None:
            return {}
        r = np.asarray(Image.open(renders[k]).convert("RGB"), float) / 255
        g = np.asarray(Image.open(img_dir / n).convert("RGB").resize((r.shape[1], r.shape[0])), float) / 255
        m = np.asarray(Image.open(m_path).resize((r.shape[1], r.shape[0]), Image.NEAREST)) > 0
        v_path = img_dir / (n[:-4] + "_mask.png")  # the frame's valid region (its black border out)
        v = np.asarray(Image.open(v_path).resize((r.shape[1], r.shape[0]), Image.NEAREST)) > 0 if v_path.exists() \
            else np.ones(m.shape, bool)
        if validity is not None:
            raw=Image.open(img_dir/n)
            from calibrated_scene_reconstruction.scene.capture.valid_pixels import resize_valid
            v=resize_valid(validity.get(n,(raw.height,raw.width)),(r.shape[1],r.shape[0]))
        e = ((r - g) ** 2).mean(-1)
        m, z_ = m & v, ~m & v
        if m.any():bg.append(10 * np.log10(1 / max(e[m].mean(), 1e-10)))
        if z_.any():zone.append(10 * np.log10(1 / max(e[z_].mean(), 1e-10)))
    return {"background_psnr": round(float(np.mean(bg)), 2) if bg else None,
            "zone_psnr": round(float(np.mean(zone)), 2) if zone else None,
            "background_views":len(bg),"zone_views":len(zone)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    a = ap.parse_args()
    s = Session(a.session)
    if (s/'qa/independent_freeze.json').exists():raise ValueError('Frozen delivery: use a background derivative')
    allcfg = settings(s)
    cfg = allcfg["train"]
    out = folder(s)
    src = (ROOT / "envs" / "splat").resolve() / "3dgrut"
    runs = out / "runs"
    from .training_data import prepare,validate,seal_masks
    data=prepare(s,allcfg)
    loss_masks(data/'images',cfg['masked'],session=s)
    snapshot=seal_masks(data)
    cmd = [sys.executable, "train.py", "--config-name", cfg["config"], f"path={data.resolve()}",
           f"out_dir={runs.resolve()}", "experiment_name=bg", f"dataset.test_split_interval={cfg['test_every']}",
           f"n_iterations={cfg['iterations']}", f"strategy.add.max_n_gaussians={cfg['max_gaussians']}",
           f"dataset.normalize_world_space={'true' if cfg.get('normalize', True) else 'false'}", "num_workers=8"]
    t0 = time.time()
    subprocess.run(cmd, cwd=src, check=True)
    run = max((runs / "bg").glob("*/ckpt_last.pt"), key=lambda p: p.stat().st_mtime).parent
    validate(data)
    metrics = json.loads((run / "metrics.json").read_text())
    rec = {"checkpoint": str(run.relative_to(s.dir) / "ckpt_last.pt"), "minutes": round((time.time() - t0) / 60, 1),
           "data": str(data.relative_to(s.dir)), "training_snapshot":snapshot,
           "settings": cfg, "held_out": {k.replace("mean_", ""): round(v, 4) for k, v in metrics.items()
                                         if k in ("mean_psnr", "mean_ssim", "mean_lpips")}}
    rec["held_out"].update(background_metrics(data, run, cfg,session=s))
    (out / "train.json").write_text(json.dumps(rec, indent=1) + "\n")
    # the review's first sheet: held-out frames spread over the walks, the real one beside the splat's
    from PIL import Image

    names = sorted(p.name for p in (data / "images").glob("*.jpg"))[::cfg["test_every"]]
    renders = sorted((run / f"ours_{cfg['iterations']}" / "renders").glob("*.png"))
    pick = np.linspace(0, len(renders) - 1, min(8, len(renders))).astype(int)
    qa = s / "qa" / "splat"
    qa.mkdir(parents=True, exist_ok=True)
    sheet = Image.new("RGB", (1280, 360 * len(pick)))
    for r, k in enumerate(pick):
        sheet.paste(Image.open(data / "images" / names[k]).convert("RGB").resize((640, 360)), (0, 360 * r))
        sheet.paste(Image.open(renders[k]).convert("RGB").resize((640, 360)), (640, 360 * r))
    sheet.save(qa / "held_out.jpg", quality=88)
    if (out/"coverage.png").is_file():shutil.copy(out/"coverage.png",qa/"coverage.png")
    print(f"done: {rec['checkpoint']} in {rec['minutes']} min, held out {rec['held_out']}")


if __name__ == "__main__":
    main()
