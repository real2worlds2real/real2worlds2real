"""The background splat sharpened where the walk did not look from: novel views round the task zone rendered from the
splat, cleaned by Difix (NVIDIA's single-step diffusion fixer, with the nearest walk frame as its reference) and
trained back into the splat beside the walk's own frames (Difix3D's distillation) -- no new capture, and the result is
still one splat, rendered by Isaac Sim as before.

    tools/run_splat.sh -m calibrated_scene_reconstruction.recon.splat_fix sessions/<name> prepare   # views, their renders (splat env)
    envs/difix/bin/python -m calibrated_scene_reconstruction.recon.splat_fix sessions/<name> fix   # Difix on them (setup/setup_difix_env.sh)
    tools/run_splat.sh -m calibrated_scene_reconstruction.recon.splat_fix sessions/<name> train     # fine-tuned from the checkpoint (splat env)

The views (`views`): eyes on rings round the support's middle (radii RADII, heights HEIGHTS, every 360/AZ degrees),
each looking at the support and out at the room (the robot's and a tour's directions), kept where a walk camera stood
within NEAR m looking within COVER degrees of the same way (Difix restores, it does not invent what nobody saw). All
cameras are written in the trained model's own frame (the training's normalization applied once, then off), so the
fine-tuning resumes the checkpoint's Gaussians where they are; splat/train.json then names the new checkpoint and
the original data, and recon.splat_export runs as after any training.

Writes splat/fix/{novel/ (the views' dataset, their renders and fixes), train/ (walk + fixed views), fix.json} and
qa/splat/fix.jpg (render | fixed | reference).
"""
import argparse

from calibrated_scene_reconstruction.core.paths import ROOT
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from calibrated_scene_reconstruction.scene.session import Session
from calibrated_scene_reconstruction.scene.background.splat.config import folder, sim3, train_data

RADII, HEIGHTS, AZ = (1.0, 1.5, 2.0), (1.3, 1.7), 24
NEAR, COVER = 2.0, 40.0  # m, deg
W, H, FOV = 1024, 576, 70.0
EXTRA = 10000  # fine-tuning iterations


def model_transform(sp):
    """The training's COLMAP -> model similarity (GSplat's, from sparse/0 of the data trained on)."""
    import pycolmap
    from threedgrut.datasets.colmap_gsplat import normalize_world_space

    rec = pycolmap.Reconstruction(str(train_data(sp) / "sparse" / "0"))
    c2w = []
    for img in sorted(rec.images.values(), key=lambda i: i.name):
        M = np.eye(4)
        M[:3] = np.asarray(img.cam_from_world().inverse().matrix())
        c2w.append(M)
    pts = np.array([p.xyz for p in rec.points3D.values()])
    return normalize_world_space(np.array(c2w), pts)[2], rec


def look(eye, at):
    """OpenCV camera-to-world (x right, y down, z forward) at eye towards at, +Z world up."""
    f = (at - eye) / np.linalg.norm(at - eye)
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    M = np.eye(4)
    M[:3, :3] = np.stack([r, np.cross(f, r), f], 1)
    M[:3, 3] = eye
    return M


def views(s):
    """Novel cameras (world, OpenCV c2w) the walk's cameras cover."""
    sg = s.scene_graph
    lo, hi = (np.asarray(b, float) for b in sg["furniture"][sg["task"]["support"]]["box"])
    c = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[2]])
    z = np.load(s / "parsing" / "posed_frames" / "cameras.npz")
    cc, ax = z["centre"], z["R_world_cam"][:, :, 2]
    out = []
    for r in RADII:
        for h in HEIGHTS:
            for a in np.radians(np.arange(AZ) * 360 / AZ):
                eye = np.array([c[0] + r * np.cos(a), c[1] + r * np.sin(a), h])
                for at in (c, eye + (eye - c) * [1, 1, 0] / r * 2.0 - [0, 0, 0.3]):
                    M = look(eye, at)
                    d = np.linalg.norm(cc - eye, axis=1)
                    ang = np.degrees(np.arccos(np.clip(ax @ M[:3, 2], -1, 1)))
                    if ((d < NEAR) & (ang < COVER)).any():
                        out.append(M)
    return np.array(out)


def write_model(dst, rec, cams, images, points):
    """A COLMAP model: `cams` {id: pycolmap camera}, `images` [(name, camera id, c2w 4x4)], points (n, 3) + colours."""
    import pycolmap

    m = pycolmap.Reconstruction()
    for cam in cams.values():
        m.add_camera_with_trivial_rig(cam)
    for k, (name, cid, c2w) in enumerate(images, 1):
        R = c2w[:3, :3].T
        t = -R @ c2w[:3, 3]
        img = pycolmap.Image(name=name, keypoints=np.zeros((0, 2)), camera_id=cid, image_id=k)
        m.add_image_with_trivial_frame(img, pycolmap.Rigid3d(pycolmap.Rotation3d(R), t))
    for xyz, col in zip(*points):
        m.add_point3D(xyz, pycolmap.Track(), col)
    (dst / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    m.write(str(dst / "sparse" / "0"))


def prepare(s):
    import pycolmap
    import torch
    from PIL import Image
    from threedgrut.datasets.colmap_gsplat import transform_cameras, transform_points

    sp = folder(s)
    fx = folder(s) / "fix"
    shutil.rmtree(fx, ignore_errors=True)
    T, rec = model_transform(sp)
    c, R, t = sim3(s)
    Mw = np.eye(4)  # world -> COLMAP: the inverse of splat/align.json's similarity
    Mw[:3, :3], Mw[:3, 3] = R.T / c, -R.T @ t / c
    V = views(s)
    Vc = np.einsum("ij,njk->nik", Mw, V)
    Vc[:, :3, :3] *= c  # rotations back to orthonormal (the similarity's scale is the position's alone)
    Vm = transform_cameras(T, Vc)
    f = 0.5 * W / np.tan(np.radians(FOV) / 2)
    cam = pycolmap.Camera(model="PINHOLE", width=W, height=H, params=[f, f, W / 2, H / 2], camera_id=1)
    nov = fx / "novel"
    (nov / "images").mkdir(parents=True)
    names = [f"n_{k:04d}.png" for k in range(len(Vm))]
    for n in names:
        Image.new("RGB", (W, H)).save(nov / "images" / n)
    pts = np.array([p.xyz for p in rec.points3D.values()])
    cols = np.array([p.color for p in rec.points3D.values()])
    write_model(nov, rec, {1: cam}, [(n, 1, M) for n, M in zip(names, Vm)], (transform_points(T, pts), cols))
    # render them: the checkpoint's renderer on this dataset, every image a test image, no normalization
    ckpt = s / json.loads((sp / "train.json").read_text())["checkpoint"]
    orig = torch.load

    def load(*a, **k):
        d = orig(*a, **k)
        if isinstance(d, dict) and "config" in d:
            d["config"].dataset.test_split_interval = -1
            d["config"].dataset.normalize_world_space = False
        return d
    torch.load = load
    from threedgrut.render import Renderer

    r = Renderer.from_checkpoint(checkpoint_path=str(ckpt), path=str(nov), out_dir=str(fx / "renders"), save_gt=False,
                                 computes_extra_metrics=True)  # (False fails on its own ssim key)
    r.render_all()
    torch.load = orig
    out = max((fx / "renders").rglob("renders"), key=lambda p: len(list(p.glob("*.png"))))
    rend = sorted(out.glob("*.png"))
    (nov / "render").mkdir()
    for n, p in zip(names, rend):
        shutil.move(str(p), nov / "render" / n)
    # each view's reference: the walk frame standing nearest and looking the same way
    z = np.load(s / "parsing" / "posed_frames" / "cameras.npz")
    refs = []
    for M in V:
        d = np.linalg.norm(z["centre"] - M[:3, 3], axis=1)
        ang = np.degrees(np.arccos(np.clip(z["R_world_cam"][:, :, 2] @ M[:3, 2], -1, 1)))
        refs.append(str(z["names"][int(np.argmin(d + ang / 20.0))]))
    (fx / "fix.json").write_text(json.dumps({"views": len(V), "names": names, "refs": refs,
                                             "transform": T.tolist(), "checkpoint": str(ckpt.relative_to(s.dir)),
                                             "world_c2w": V.tolist(), "model_c2w": Vm.tolist()}, indent=0))
    print(f"done: {len(V)} views rendered -> {nov}", flush=True)


def fix(s):
    from PIL import Image

    root = ROOT / "third_party" / "Difix3D" / "src"
    sys.path.insert(0, str(root))
    from pipeline_difix import DifixPipeline

    fx = folder(s) / "fix"
    meta = json.loads((fx / "fix.json").read_text())
    pipe = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True).to("cuda")
    img_dir = Path(str(np.load(s / "parsing" / "posed_frames" / "cameras.npz")["source"])) / "splat" / "data" / "images"
    nov = fx / "novel"
    (nov / "fixed").mkdir(exist_ok=True)
    for n, ref in zip(meta["names"], meta["refs"]):
        im = Image.open(nov / "render" / n).convert("RGB").resize((W, H))
        rf = Image.open(img_dir / ref).convert("RGB").resize((W, H))
        out = pipe("remove degradation", image=im, ref_image=rf, num_inference_steps=1, timesteps=[199],
                   guidance_scale=0.0).images[0]
        out.resize((W, H)).save(nov / "fixed" / n)
    qa = s / "qa" / "splat"
    qa.mkdir(parents=True, exist_ok=True)
    rows = []
    for k in np.linspace(0, len(meta["names"]) - 1, 6).astype(int):
        n = meta["names"][k]
        tiles = [Image.open(nov / d / n).convert("RGB").resize((W // 2, H // 2)) for d in ("render", "fixed")]
        tiles.append(Image.open(img_dir / meta["refs"][k]).convert("RGB").resize((W // 2, H // 2)))
        rows.append(np.hstack([np.asarray(t) for t in tiles]))
    Image.fromarray(np.vstack(rows)).save(qa / "fix.jpg", quality=88)
    print(f"done: {len(meta['names'])} views fixed -> {nov / 'fixed'}", flush=True)


def train(s):
    import pycolmap
    from threedgrut.datasets.colmap_gsplat import transform_cameras, transform_points

    sp = folder(s)
    fx = sp / "fix"
    meta = json.loads((fx / "fix.json").read_text())
    T = np.asarray(meta["transform"])
    rec = pycolmap.Reconstruction(str(train_data(sp) / "sparse" / "0"))
    data = train_data(sp)
    td = fx / "train"
    shutil.rmtree(td, ignore_errors=True)
    (td / "images").mkdir(parents=True)
    cams = {cid: cam for cid, cam in rec.cameras.items()}
    nid = max(cams) + 1
    f = 0.5 * W / np.tan(np.radians(FOV) / 2)
    cams[nid] = pycolmap.Camera(model="PINHOLE", width=W, height=H, params=[f, f, W / 2, H / 2], camera_id=nid)
    images = []
    for img in sorted(rec.images.values(), key=lambda i: i.name):  # the walk's frames, in the model's frame
        M = np.eye(4)
        M[:3] = np.asarray(img.cam_from_world().inverse().matrix())
        images.append((img.name, img.camera_id, transform_cameras(T, M[None])[0]))
        (td / "images" / img.name).symlink_to((data / "images" / img.name).resolve())
        mk = data / "images" / (img.name[:-4] + "_mask.png")  # its loss mask (splat_train.loss_masks: the black
        if mk.exists():  # border of an undistorted frame left out)
            (td / "images" / mk.name).symlink_to(mk.resolve())
    for n, M in zip(meta["names"], np.asarray(meta["model_c2w"])):  # the fixed views
        images.append((n, nid, M))
        (td / "images" / n).symlink_to((fx / "novel" / "fixed" / n).resolve())
    pts = np.array([p.xyz for p in rec.points3D.values()])
    cols = np.array([p.color for p in rec.points3D.values()])
    write_model(td, rec, cams, images, (transform_points(T, pts), cols))
    tj = json.loads((sp / "train.json").read_text())
    cfg = tj["settings"]
    ckpt = s / meta["checkpoint"]
    import torch

    step = int(torch.load(str(ckpt), weights_only=False, map_location="cpu")["global_step"])
    src = (ROOT / "envs" / "splat").resolve() / "3dgrut"
    runs = sp / "runs"
    cmd = [sys.executable, "train.py", "--config-name", cfg["config"], f"path={td.resolve()}",
           f"out_dir={runs.resolve()}", "experiment_name=fix", f"resume={ckpt.resolve()}",
           f"dataset.test_split_interval={cfg['test_every']}", f"n_iterations={step + EXTRA}",
           f"strategy.add.max_n_gaussians={cfg['max_gaussians']}", "dataset.normalize_world_space=false",
           "num_workers=8"]
    t0 = time.time()
    subprocess.run(cmd, cwd=src, check=True)
    run = max((runs / "fix").glob("*/ckpt_last.pt"), key=lambda p: p.stat().st_mtime).parent
    metrics = json.loads((run / "metrics.json").read_text())
    rec_ = {**tj, "checkpoint": str(run.relative_to(s.dir) / "ckpt_last.pt"),
            "minutes": round(tj.get("minutes", 0) + (time.time() - t0) / 60, 1),
            "fix": {"views": meta["views"], "iterations": EXTRA, "from": meta["checkpoint"],
                    "held_out": {k.replace("mean_", ""): round(v, 4) for k, v in metrics.items()
                                 if k in ("mean_psnr", "mean_ssim", "mean_lpips")}}}
    (sp / "train.json").write_text(json.dumps(rec_, indent=1) + "\n")
    print(f"done: {rec_['checkpoint']}, held out (walk + fixed views) {rec_['fix']['held_out']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("step", choices=["prepare", "fix", "train"])
    a = ap.parse_args()
    {"prepare": prepare, "fix": fix, "train": train}[a.step](Session(a.session))


if __name__ == "__main__":
    main()
