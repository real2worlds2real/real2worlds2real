"""Splat background, step 3: the trained Gaussians as the twin's background asset for Isaac Sim.

    tools/run_splat.sh -m calibrated_scene_reconstruction.recon.splat_export sessions/<name>

From the checkpoint (recon.splat_train) to a PLY in COLMAP's frame (3DGRUT's exporter, no normalizing transform),
then, in the twin's world (splat/align.json):
  task zone   Gaussians the meshes and the simulated robot stand for are removed (splat.task_zone: the support's
              footprint + margin up to above the tallest piece on it, the robot's column), the floor's own layer
              kept (above `keep_floor`: cutting it left the mesh floor showing in a hole under the table)
  unseen      Gaussians in fewer than `min_views` frames' view (frustum, 0.1-30 m): they only show from where no
              frame looked, as black-backed needles
  faint, huge opacity under 0.005 (they cost and show nothing), a largest axis over 0.5 m (blurred blobs)
  near_cameras (crop.near_cameras, m) Gaussians that close to a training camera: never fitted, drawn by NuRec
  undrawn     (crop.min_drawn) Gaussians 3DGRUT drew in fewer training frames: never fitted (drawn)
  floaters    (crop.free_space: a share) Gaussians the frames look past -- in front of the surface the splat renders
              there in over that share of the frames that see them (free_space)
and back to NuRec USDZ (scene/splat/background.usdz: the Omniverse volume Isaac Sim 5.0-6.0 renders with RTX) and a
ParticleField USD (scene/splat/background_particlefield.usdz, the OpenUSD schema of Isaac Sim 6+), both still in
COLMAP's frame: scene/splat/transform.json carries the similarity that places them (sim.splat applies it).
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from calibrated_scene_reconstruction.scene.session import Session
from calibrated_scene_reconstruction.scene.background.splat.config import folder, settings, sim3, task_zone, train_data
from calibrated_scene_reconstruction.scene.background.splat.selection import appearance_filters, prepare_asset_directory, prepare_work_directory


def unnormalize(out, P, v):
    """The training's world normalization (3DGRUT dataset.normalize_world_space: GSplat's similarity from the
    cameras and points of sparse/0, deterministic) undone on the Gaussians: positions, scales and rotations."""
    import pycolmap
    from scipy.spatial.transform import Rotation
    from threedgrut.datasets.colmap_gsplat import normalize_world_space

    rec = pycolmap.Reconstruction(str(train_data(out) / "sparse" / "0"))  # the model trained on
    c2w = []
    for img in sorted(rec.images.values(), key=lambda i: i.name):
        M = np.eye(4)
        M[:3] = np.asarray(img.cam_from_world().inverse().matrix())
        c2w.append(M)
    pts = np.array([p.xyz for p in rec.points3D.values()])
    _, _, T = normalize_world_space(np.array(c2w), pts)  # COLMAP -> model
    Ti = np.linalg.inv(T)
    A = Ti[:3, :3]
    k = float(np.cbrt(np.linalg.det(A)))
    Rm = A / k
    P = P @ A.T + Ti[:3, 3]
    v = v.copy()
    v["x"], v["y"], v["z"] = P[:, 0], P[:, 1], P[:, 2]
    for n in ("scale_0", "scale_1", "scale_2"):
        v[n] = v[n] + np.log(k)
    q = np.stack([v["rot_1"], v["rot_2"], v["rot_3"], v["rot_0"]], 1)  # PLY wxyz -> scipy xyzw
    q2 = (Rotation.from_matrix(Rm) * Rotation.from_quat(q)).as_quat()
    v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"] = q2[:, 3], q2[:, 0], q2[:, 1], q2[:, 2]
    from calibrated_scene_reconstruction.scene.background.splat.radiance import rotate_ply_radiance
    v = rotate_ply_radiance(v, Rm)
    print(f"[export] normalization undone: scale {k:.4f}", flush=True)
    return P, v


def free_space(s, X, ckpt, every=4, near=0.85, far=1.10, min_seen=5):
    """Each Gaussian's share of the frames that look past it: of the frames it is in view of and not hidden in (its
    depth under `far` x the splat's rendered depth there), those whose surface lies well behind it (its depth under
    `near` x the rendered). A surface's Gaussians are that surface (share ~0); a floater hangs in air the frames saw
    through -- the smears that drew themselves across the tour off the walk. Every `every`-th posed frame; X in the
    twin's world. Returns (share, seen)."""
    from calibrated_scene_reconstruction.scene.background.code_room import frame_depths

    z = np.load(s / "parsing" / "posed_frames" / "cameras.npz")
    allnames = [str(n) for n in z["names"]]
    names = allnames[::every]
    dep, ds = frame_depths(s, set(names), checkpoint=ckpt, pool="min")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
    seen = torch.zeros(len(X), dtype=torch.int32, device=dev)
    front = torch.zeros(len(X), dtype=torch.int32, device=dev)
    for n in names:
        if n not in dep:
            continue
        i = allnames.index(n)
        K = z["K"][i]
        R = torch.as_tensor(z["R_world_cam"][i], dtype=torch.float32, device=dev)
        Xc = (Xt - torch.as_tensor(z["centre"][i], dtype=torch.float32, device=dev)) @ R
        zc = Xc[:, 2]
        D = torch.as_tensor(dep[n], dtype=torch.float32, device=dev)
        u = (float(K[0, 0]) * Xc[:, 0] / zc + float(K[0, 2])) / ds
        w = (float(K[1, 1]) * Xc[:, 1] / zc + float(K[1, 2])) / ds
        ok = (zc > 0.1) & (u >= 0) & (u < D.shape[1] - 1) & (w >= 0) & (w < D.shape[0] - 1)
        d = torch.full_like(zc, float("inf"))
        d[ok] = D[w[ok].long(), u[ok].long()]
        vis = ok & (zc < far * d + 0.03)
        seen += vis.int()
        front += (vis & (zc < near * d - 0.05)).int()
    seen, front = seen.cpu().numpy(), front.cpu().numpy()
    return front / np.maximum(seen, 1), seen


def drawn(s, ckpt, n, every=2):
    """Each Gaussian's number of training frames 3DGRUT drew it in (its own renderer's per-particle visibility, every
    `every`-th posed frame). One never drawn was never fitted: no loss ever reached it, and Isaac's NuRec, which
    culls differently near the camera and at the image's edge, draws it -- the white and orange blobs in front of the
    cameras and the smears on the floor that 3DGRUT's own render of the same frame does not have."""
    from calibrated_scene_reconstruction.scene.background.code_room import frame_depths

    z = np.load(s / "parsing" / "posed_frames" / "cameras.npz")
    seen = {}
    frame_depths(s, set([str(x) for x in z["names"]][::every]), ds=8, checkpoint=ckpt, seen=seen)
    count = seen["count"].cpu().numpy()
    if len(count) != n:
        raise SystemExit(f"splat_export: {len(count)} Gaussians drawn against {n} exported (not the same order)")
    return count


def run(*args):
    subprocess.run([sys.executable, "-m", *args], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--cousin", default="", help="a splat cousin's checkpoint (recon.splat_cousin): "
                    "scene/splat/cousins/<name>/ instead of the twin's background")
    a = ap.parse_args()
    s = Session(a.session)
    if (s/'qa/independent_freeze.json').exists():raise ValueError('Frozen delivery: use a background derivative')
    cfg = settings(s)
    crop = cfg["crop"]
    out = folder(s)
    work = out / "export" if not a.cousin else out / "cousins" / a.cousin / "export"
    prepare_work_directory(work)
    tj = (out / "cousins" / a.cousin if a.cousin else out) / "train.json"
    ckpt = s / json.loads(tj.read_text())["checkpoint"]
    run("threedgrut.export.scripts.export_usd", "--checkpoint", str(ckpt), "--output", str(work / "full.usdz"),
        "--dataset", str(train_data(out)), "--format", "standard", "--up-axis", "z", "--no-cameras", "--no-background",
        "--no-transform")
    run("threedgrut.export.scripts.transcode", str(work / "full.usdz"), "-o", str(work / "full.ply"), "-f", "ply")

    v = PlyData.read(str(work / "full.ply"))["vertex"].data
    c, R, t = sim3(s)
    P = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    if json.loads((out / "train.json").read_text())["settings"].get("normalize", True):
        P, v = unnormalize(out, P, v)  # the model's frame -> COLMAP's (the training's GSplat normalization undone)
    # Preserve the same row IDs before any destructive selection, allowing
    # controlled export ablations without re-running reconstruction or training.
    PlyData([PlyElement.describe(v, "vertex")], text=False).write(str(work / "full_colmap.ply"))
    X = c * P @ R.T + t
    inside, zone = task_zone(s, crop)
    in_zone = inside(X)
    opacity = 1 / (1 + np.exp(-v["opacity"].astype(np.float64)))
    size = c * np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1).astype(np.float64)).max(1)

    # frames whose view each Gaussian lies in (COLMAP's poses, its pinhole; on the GPU, 2M points x 900 frames)
    import pycolmap

    rec = pycolmap.Reconstruction(str(train_data(out) / "sparse" / "0"))  # the model trained on
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Pt = torch.as_tensor(P, dtype=torch.float32, device=dev)
    views = torch.zeros(len(P), dtype=torch.int32, device=dev)
    for img in rec.images.values():
        cam = rec.cameras[img.camera_id]  # each video its own camera (splat_capture)
        fx, fy, cx, cy = (float(x) for x in cam.params[:4])
        M = torch.as_tensor(np.asarray(img.cam_from_world().matrix()), dtype=torch.float32, device=dev)
        Xc = Pt @ M[:, :3].T + M[:, 3]
        z = Xc[:, 2]
        u, w = fx * Xc[:, 0] / z + cx, fy * Xc[:, 1] / z + cy
        d = z * c
        views += ((d > 0.1) & (d < 30) & (u >= 0) & (u < cam.width) & (w >= 0) & (w < cam.height)).int()
    views = views.cpu().numpy()
    from scipy.spatial import cKDTree

    centres = np.array([img.projection_center() for img in rec.images.values()])
    near_cam = cKDTree(centres).query(P)[0] * c < crop.get("near_cameras", 0)
    unseen = views < crop["min_views"]
    faint, huge = appearance_filters(opacity, size, crop)
    undrawn = np.zeros(len(X), bool)
    if crop.get("min_drawn"):
        undrawn = drawn(s, ckpt, len(X)) < crop["min_drawn"]
    floater = np.zeros(len(X), bool)
    if crop.get("free_space"):  # (on what the other tests keep)
        rest = ~(in_zone | unseen | faint | huge | near_cam | undrawn)
        share, seen = free_space(s, X[rest], ckpt)
        floater[np.flatnonzero(rest)] = (share > crop["free_space"]) & (seen >= 5)
    keep = ~(in_zone | unseen | faint | huge | near_cam | undrawn | floater)
    masks_path = work / 'selection_masks.npz'
    np.savez_compressed(masks_path, in_zone=in_zone, unseen=unseen, faint=faint,
                        huge=huge, near_cameras=near_cam, undrawn=undrawn,
                        floaters=floater, kept=keep, opacity=opacity, world_sigma_m=size)
    engine = work/'engine'
    engine.mkdir()
    for module in (Path(__file__), Path(__file__).with_name('selection.py'), Path(__file__).with_name('radiance.py')):
        shutil.copyfile(module, engine/module.name)
    def sha(path):
        value = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                value.update(block)
        return value.hexdigest()
    evidence = {'scope': 'Background export rows only. Task meshes, calibration and video selection unchanged.',
                'checkpoint': str(ckpt), 'checkpoint_sha256': sha(ckpt), 'crop': crop,
                'gaussians': len(v), 'kept': int(keep.sum()), 'radiance_frame': 'colmap',
                'free_space_evaluated': 'Only rows not rejected by earlier rules; retained low-opacity rows need new review in ablations.',
                'files': {str(p): sha(p) for p in [work/'full_colmap.ply', masks_path, *engine.glob('*.py')]}}
    (work/'selection.json').write_text(json.dumps(evidence, indent=2)+'\n')
    PlyData([PlyElement.describe(v[keep], "vertex")], text=False).write(str(work / "background.ply"))
    dst = s / "scene" / "splat" / ("cousins/" + a.cousin if a.cousin else "")
    prepare_asset_directory(dst)
    run("threedgrut.export.scripts.transcode", str(work / "background.ply"), "-o", str(dst / "background.usdz"),
        "-f", "nurec")
    run("threedgrut.export.scripts.transcode", str(work / "background.ply"), "-o",
        str(dst / "background_particlefield.usdz"), "-f", "lightfield")
    # the training cameras in the twin's world (centre, viewing axis, horizontal and vertical half field of view): what
    # a viewer may look at -- apps/scene_tour turns only where they looked (a capture filmed facing the table saw
    # little behind it)
    cc, ax, hf, names = [], [], [], []
    for img in sorted(rec.images.values(), key=lambda i: i.name):  # by name: each video's frames in order
        names.append(img.name)
        cam = rec.cameras[img.camera_id]
        T = np.asarray(img.cam_from_world().matrix())
        cc.append(c * R @ (-T[:, :3].T @ T[:, 3]) + t)
        ax.append(R @ T[:, :3].T[:, 2])
        hf.append([np.arctan(cam.width / 2 / float(cam.params[0])), np.arctan(cam.height / 2 / float(cam.params[1]))])
    np.savez_compressed(dst / "cameras.npz", centre=np.array(cc), axis=np.array(ax), half_fov=np.array(hf),
                        names=np.array(names))
    M = np.eye(4)
    M[:3, :3] = c * R
    M[:3, 3] = t
    stats = {"gaussians": int(len(v)), "kept": int(keep.sum()), "task_zone": int(in_zone.sum()),
             "unseen": int((unseen & ~in_zone).sum()), "faint": int((faint & ~in_zone & ~unseen).sum()),
             "huge": int((huge & ~in_zone & ~unseen & ~faint).sum()), "near_cameras": int((near_cam & ~in_zone & ~unseen & ~faint & ~huge).sum()), "undrawn": int((undrawn & ~in_zone & ~unseen & ~faint & ~huge).sum()), "floaters": int(floater.sum())}
    (dst / "transform.json").write_text(json.dumps({
        "_note": "COLMAP frame -> the twin's world: x_world = M x (column vectors); the USD files are in COLMAP's frame",
        "matrix": M.tolist(), "scale": c, "zone": zone, "crop": crop, "stats": stats}, indent=1) + "\n")
    print(f"done: {stats['kept']} of {stats['gaussians']} Gaussians kept (task zone {stats['task_zone']}, unseen "
          f"{stats['unseen']}, faint {stats['faint']}, huge {stats['huge']}, at a camera {stats['near_cameras']}, never drawn {stats['undrawn']}, floaters {stats['floaters']}) -> {dst}")


if __name__ == "__main__":
    main()
