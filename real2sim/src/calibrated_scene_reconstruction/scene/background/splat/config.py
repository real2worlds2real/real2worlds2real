"""The optional Gaussian-splat background of a twin (run_session.sh --splat, or configs/<name>/splat.json): what the
video saw around the task, rendered by Isaac Sim's RTX as a NuRec volume beside the twin's meshes; the meshes keep
the task (the support, its objects, the robot) and every collider. Off by default: without it the twin is the mesh
twin, unchanged. Shared here: the settings, the session's splat folder, the COLMAP -> world similarity, the task zone.

  splat/data/images, sparse/0   frames of the session's videos and their poses (recon.splat_capture)
  splat/align.json              COLMAP -> twin world (scale, R, t) and its checks
  splat/coverage.{json,png}     the scene's coverage: frames and viewing spread per 25 cm column
  splat/runs/, train.json       3DGRUT training (recon.splat_train), held-out metrics
  scene/splat/                  background.usdz (NuRec), transform.json (recon.splat_export)

The twin is one fixed asset: everything here is judged on the scene (held-out frames, the coverage map, scene-level
viewpoints), never tuned to a policy camera's viewpoint.
"""
import copy
import json
from pathlib import Path

import numpy as np

DEFAULTS = {
    'colmap_workers': 4,
    "videos": [],  # more videos of the same scene, besides the session's own (capture/keyframes.json)
    "step": 2,  # every step-th frame read
    "keep": 0.85,  # the sharpest share of each second's frames kept
    "cross_every_s": 0.5,  # pairs between two videos: one frame per this many seconds of each, every pair matched
    # loop pairs within a video: one frame per this many seconds, every pair beyond the sequential window matched
    "loop_every_s": 1.0,
    "train": {"config": "apps/colmap_3dgut_mcmc.yaml", "iterations": 60000, "max_gaussians": 4000000,
              "test_every": 8, "masked": False,  # masked: the task zone's things out of the loss (splat_masks)
              # Normalize camera positions so that MCMC learning rates are independent of the capture's extent.
              "normalize": True},
    # the task zone the meshes take: the support's footprint + margin up to the tallest piece on it + above, the
    # robot's column; the floor's own layer kept (above keep_floor), Gaussians seen by fewer than min_views frames
    # (floaters before cameras that never looked there) dropped
    "crop": {"margin": 0.10, "above": 0.25, "robot_r": 0.5, "robot_h": 1.9, "keep_floor": 0.03, "min_views": 4,
             "near_cameras": 0.4, "min_drawn": 0, "free_space": 0.5},
    # in splat mode these mesh visuals are hidden (their colliders kept): the splat shows what they stood for
    "hide": ["walls", "ceiling", "static", "synthetic"],
}


def settings(s):
    """Legacy splat settings plus camera-only overrides, without enabling GS."""
    cfg = copy.deepcopy(DEFAULTS)
    for k, v in s.config("splat.json", {}).items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    capture = s.config('camera_reconstruction.json', {})
    allowed = {'colmap_workers', 'step', 'keep', 'motion', 'max_gap',
               'cross_every_s', 'loop_every_s'}
    unknown = set(capture)-allowed
    if unknown:
        raise ValueError('Unknown camera reconstruction setting: '+str(sorted(unknown)))
    for key, value in capture.items():
        if key in {'colmap_workers', 'step', 'max_gap'}:
            if type(value) is not int or value < 1 or key == 'colmap_workers' and value > 32:
                raise ValueError('Invalid positive integer camera setting: '+key)
        elif (isinstance(value, bool) or not isinstance(value, (int, float))
              or not np.isfinite(value) or value < 0
              or key == 'keep' and not 0 < value <= 1
              or key == 'motion' and value > 1):
            raise ValueError('Invalid camera sampling setting: '+key)
    cfg.update(capture)
    return cfg


def enabled(s):
    return (s.config_dir / "splat.json").exists()


def folder(s):
    d = s / "splat"
    d.mkdir(exist_ok=True)
    return d


def sim3(s):
    """(scale, R, t): COLMAP point x -> world scale R x + t."""
    a = json.loads((s / "splat" / "align.json").read_text())["sim3"]
    return float(a["scale"]), np.asarray(a["R"], float), np.asarray(a["t"], float)


def umeyama(x, y, fixed_scale=None):
    """(c, R, t) with y ~ c R x + t; `fixed_scale` keeps c."""
    mx, my = x.mean(0), y.mean(0)
    U, S, Vt = np.linalg.svd((y - my).T @ (x - mx) / len(x))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    c = fixed_scale or float(np.trace(np.diag(S) @ D) / ((x - mx) ** 2).sum(1).mean())
    return c, R, my - c * R @ mx


def train_data(out):
    """The data the splat was trained on: splat/data, or the subset splat_train made of it (train.skip_videos)."""
    t = out / "train.json"
    if t.exists():
        d = json.loads(t.read_text()).get("data")
        if d:
            return out.parent / d
    return out / "data"


def subset_model(src, dst, keep):
    """A COLMAP model of the registered images whose name `keep` accepts, their poses as they are (one capture's
    poses and scale, a subset trained on), every 3D point still seen by two of them kept."""
    import pycolmap

    rec = pycolmap.Reconstruction(str(src))
    sub = pycolmap.Reconstruction()
    used = {img.camera_id for img in rec.images.values() if keep(img.name)}
    for cam in rec.cameras.values():  # only the kept frames' cameras (3DGRUT looks up an image for every camera)
        if cam.camera_id in used:
            sub.add_camera_with_trivial_rig(cam)
    kept = {}
    for iid, img in rec.images.items():
        if not keep(img.name):
            continue
        new = pycolmap.Image(name=img.name, keypoints=np.array([p.xy for p in img.points2D], float).reshape(-1, 2),
                             camera_id=img.camera_id, image_id=iid)
        sub.add_image_with_trivial_frame(new, img.cam_from_world())
        kept[iid] = True
    for p in rec.points3D.values():
        tr = pycolmap.Track()
        for el in p.track.elements:
            if el.image_id in kept:
                tr.add_element(el.image_id, el.point2D_idx)
        if tr.length() >= 2:
            sub.add_point3D(p.xyz, tr, p.color)
    Path(dst).mkdir(parents=True, exist_ok=True)
    sub.write(str(dst))
    return len(kept), sub.num_points3D()


def task_zone(s, crop):
    """A test X (n, 3) world points -> bool: in the zone the meshes and the robot take. With twin.json the meshes are
    only what the task touches (twin.meshed_structures): the support up to its top, each manipulated object's box,
    the robot's column -- a frame over the support, its cameras and cables stay the splat's. Without, the support's
    footprint with all that stands on it, up to above the tallest."""
    from calibrated_scene_reconstruction.scene.twin.gates import meshed_structures

    sg = s.scene_graph
    lo, hi = (np.asarray(b, float) for b in sg["furniture"][sg["task"]["support"]]["box"])
    if not meshed_structures(s):
        return touched_zone(s, sg, lo, hi, crop)
    on = [np.asarray(f["box"], float) for f in sg["furniture"].values()
          if np.all(np.asarray(f["box"])[0][:2] < hi[:2]) and np.all(np.asarray(f["box"])[1][:2] > lo[:2])]
    top = max(b[1][2] for b in on) + crop["above"]
    # the footprint holds what stands on the support where it reaches past it
    lo[:2] = np.min([b[0][:2] for b in on], 0)
    hi[:2] = np.max([b[1][:2] for b in on], 0)
    base = (sg.get("photo_robot") or {}).get("base")

    def inside(X):
        z = (np.all(X[:, :2] >= lo[:2] - crop["margin"], 1) & np.all(X[:, :2] <= hi[:2] + crop["margin"], 1)
             & (X[:, 2] < top) & (X[:, 2] > crop["keep_floor"]))
        if base:
            z |= ((np.hypot(X[:, 0] - base["xy"][0], X[:, 1] - base["xy"][1]) < crop["robot_r"])
                  & (X[:, 2] < crop["robot_h"]) & (X[:, 2] > crop["keep_floor"]))
        return z
    return inside, {"support": [lo.tolist(), hi.tolist()], "top": top, "robot_base": base}


def touched_zone(s, sg, lo, hi, crop):
    """task_zone's test when only what the task touches is a mesh: the support's box (to `top_skin` over its top),
    each manipulated object's kit (its radius and height, + `object_margin`), the robot's column."""
    from calibrated_scene_reconstruction.assets import kits

    base = (sg.get("photo_robot") or {}).get("base")
    objs = []
    for o in (sg.get("dynamic_objects") or {}).values():
        fx = o.get("fix") or {}
        if fx.get("xy") and fx.get("params"):
            if fx.get("model"):
                from calibrated_scene_reconstruction.scene.objects.blender.export import asset
                k = asset(fx)
            else:
                k = kits.build(fx["kit"], fx["params"])
            objs.append((np.r_[fx["xy"], fx.get("support_z", hi[2])], float(np.hypot(*k["half"])), k["height"]))
    m = crop.get("object_margin", 0.03)

    def inside(X):
        z = (np.all(X[:, :2] >= lo[:2] - crop["margin"], 1) & np.all(X[:, :2] <= hi[:2] + crop["margin"], 1)
             & (X[:, 2] < hi[2] + crop.get("top_skin", 0.01)) & (X[:, 2] > crop["keep_floor"]))
        for c_, r_, h_ in objs:
            z |= ((np.hypot(X[:, 0] - c_[0], X[:, 1] - c_[1]) < r_ + m) & (X[:, 2] > c_[2] - 0.01)
                  & (X[:, 2] < c_[2] + h_ + m))
        if base:
            z |= ((np.hypot(X[:, 0] - base["xy"][0], X[:, 1] - base["xy"][1]) < crop["robot_r"])
                  & (X[:, 2] < crop["robot_h"]) & (X[:, 2] > crop["keep_floor"]))
        return z
    return inside, {"support": [lo.tolist(), hi.tolist()], "top": float(hi[2]), "objects": len(objs),
                    "robot_base": base, "only_what_the_task_touches": True}


def grade(frames, spread):
    """A column's coverage: unseen, thin (< 30 frames), one-sided (seen within < 40 deg), good."""
    return "unseen" if frames == 0 else "thin" if frames < 30 else "one-sided" if spread < 40 else "good"


def angular_spread(a):
    """The arc (deg) a set of directions spans: 360 minus the largest gap between them."""
    if len(a) < 2:
        return 0.0
    a = np.sort(np.mod(a, 2 * np.pi))
    return float(np.degrees(2 * np.pi - np.diff(np.r_[a, a[0] + 2 * np.pi]).max()))
