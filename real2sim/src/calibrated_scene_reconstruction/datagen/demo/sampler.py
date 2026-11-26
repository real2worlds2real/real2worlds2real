"""Augmentation axes for demo re-targeting: one scene variant per draw, configured per task (configs/demo_tasks/).

    objects      planar pose jitter of each listed object on its support (xy box, yaw), kept on the support and apart
    augment      the target replaced by assets.augment's shape / colour / finish variant of itself (probability)
    distractors  number of library distractors (cousins.py places them off the approach corridor)
    scenes       twin and / or cousins (themes, furniture swaps, per-cousin rooms), uniformly
    light        intensity scale and colour temperature ranges; hdri: a random theme sky
    camera       jitter of every recorded camera: translation (m) and rotation (deg)
    base         robot base jitter around the demo's base: xy (m) and yaw (deg)
"""
import copy
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT = {
    "task": None,                       # text for the dataset; default: the demo's
    "objects": {},                      # {name: {"xy": [dx, dy] m half-ranges, "yaw_deg": half-range}}
    "gap": 0.02,                        # clearance between perturbed objects' footprints (m)
    "augment_target": 0.5,
    "distractors": [0, 3],
    "scenes": {"twin": 1.0, "cousins": 3.0},  # relative weights
    "light": {"scale": [0.7, 1.3], "temp": [3500, 6500]},
    "hdri": True,
    "camera": {"pos": 0.02, "rot_deg": 2.0},
    "base": {"xy": [0.05, 0.05], "yaw_deg": 8.0},
    "success": {"pos_tol": 0.03, "rot": "upright", "rot_tol_deg": 15.0, "lift": 0.05},
    # grasp (demo.functional.hold): "type" "auto" = the demo's hold measured on the object's mesh decides whether it
    # is functional (thin part / off its axis / a separate part / a tool's end: candidates only in its region, ranked
    # by in-hand pose error within the tolerances) or generic (any candidate, demo-like first); "functional" | "any"
    # override the measurement. "symmetry": "auto" (measured on the mesh) | null | "z"; "top_k": plans draw among the
    # best ranked; "hold": classify() thresholds (functional.HOLD)
    "grasp": {"type": "auto", "radius": 0.03, "inhand_pos_tol": 0.02, "inhand_rot_tol_deg": 25.0,
              "symmetry": "auto", "top_k": 8, "hold": {}},
    # held segments by index: "transport" (object path replayed in the reference frame) | "pour" (object turned about
    # its spout over the target: "spout" in the object frame, "axis" horizontal, "height"; each missing one taken
    # from the demo's own pour) | "insert" ("axis", "depth"); held paths are re-targeted as object paths with the
    # chosen grasp's in-hand pose (functional.retarget_segments)
    "skills": {},
    "near": 0.15,                       # radius of a segment's object-centric part (segment.segment)
    "speed": 0.25,                      # bridge cruise speed (m/s)
}


def merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_cfg(path=None):
    return merge(DEFAULT, json.loads(Path(path).read_text()) if path else {})


def perturb_planar(T, dxy, dyaw):
    """Translate in world xy and turn about the world z through the object's own origin (it stays on its support)."""
    T = np.asarray(T, float).copy()
    T[:3, :3] = Rotation.from_euler("z", dyaw).as_matrix() @ T[:3, :3]
    T[:2, 3] += dxy
    return T


def _on(xy, r, box):
    (x0, y0, _), (x1, y1, _) = box
    return x0 + r <= xy[0] <= x1 - r and y0 + r <= xy[1] <= y1 - r


def perturb_objects(poses, cfg, rng, radii=None, support_box=None, tries=100):
    """Jitter the configured objects jointly (rejection: on the support, footprints apart). The input poses when no
    draw within `tries` is valid."""
    radii = radii or {}
    for _ in range(tries):
        out = dict(poses)
        for name, j in cfg["objects"].items():
            if name not in poses:
                continue
            h = np.asarray(j.get("xy", [0.0, 0.0]), float)
            dyaw = np.radians(rng.uniform(-1, 1) * j.get("yaw_deg", 0))
            out[name] = perturb_planar(poses[name], rng.uniform(-h, h), dyaw)
        moved = [n for n in cfg["objects"] if n in poses]
        # objects on the task's support stay on it (TODO: the other objects' own supports)
        on = [n for n in moved if support_box is not None and _on(poses[n][:2, 3], 0.0, support_box)]
        ok = all(_on(out[n][:2, 3], radii.get(n, 0.0), support_box) for n in on)
        for a in moved:
            for b in out:
                gap = radii.get(a, 0) + radii.get(b, 0) + cfg["gap"]
                ok = ok and (b == a or np.linalg.norm(out[a][:2, 3] - out[b][:2, 3]) >= gap)
        if ok:
            return out
    return dict(poses)


def sample(cfg, rng, scenes):
    """One variant: scene tag (from `scenes`: "twin" and cousin tags), target augmentation, distractors, lights,
    camera and base jitter. Object poses come per scene (perturb_objects), they depend on the scene's layout."""
    w = np.asarray([cfg["scenes"]["twin"] if s == "twin" else cfg["scenes"]["cousins"] / max(1, len(scenes) - 1)
                    for s in scenes], float)
    scene = scenes[rng.choice(len(scenes), p=w / w.sum())]
    cam, base = cfg["camera"], cfg["base"]
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    bxy = np.asarray(base["xy"], float)
    return {
        "scene": scene,
        "augment_target": bool(rng.random() < cfg["augment_target"]),
        "distractors": int(rng.integers(cfg["distractors"][0], cfg["distractors"][1] + 1)),
        "light": {"scale": float(rng.uniform(*cfg["light"]["scale"])),
                  "temp": float(rng.uniform(*cfg["light"]["temp"])), "hdri": bool(cfg["hdri"])},
        "camera": {"dp": rng.uniform(-cam["pos"], cam["pos"], 3).tolist(),
                   "drot": (axis * np.radians(rng.uniform(0, cam["rot_deg"]))).tolist()},
        "base": {"dxy": rng.uniform(-bxy, bxy).tolist(),
                 "dyaw": float(np.radians(rng.uniform(-1, 1) * base["yaw_deg"]))},
    }


def base_pose(T_world_base, v):
    """The demo's base with the variant's jitter (about the base's own vertical axis)."""
    return perturb_planar(T_world_base, np.asarray(v["base"]["dxy"]), v["base"]["dyaw"])
