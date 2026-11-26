"""The task target as the simulator instantiates it, and where its synthesized grasps live.

Pure numpy / trimesh (no cuRobo), so both envs/grasp (synthesis) and the Isaac apps (episodes) use it. The target's
local frame is sim.scene.TwinScene.add_object's: the mesh's bounding-box bottom centre at the origin, scaled, unrotated
(the prim's pose carries the placement and yaw). Synthesized grasp records are expressed in that frame, so
world = T_prim @ record.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh

from calibrated_scene_reconstruction.core.paths import rel, resolve
from calibrated_scene_reconstruction.core.contracts.scene import SceneInputs as Session

from calibrated_scene_reconstruction.assets.object_geometry import (  # noqa: F401 -- retained public helpers
    collision_parts as _collision_parts, level, mesh_key, object_label, scene_key,
)

from calibrated_scene_reconstruction.assets.grasp_records import GRASP_DIR as GRASP_DIR  # noqa: E402,F401


def collision_parts(s: Session, key: str):
    return _collision_parts(s, key, directory=GRASP_DIR)


def target_spec(s: Session, variant: dict) -> dict:
    """The target's object spec in a (prepared) variant, as sim.scene.build_scene resolves it."""
    target = s.scene_graph["task"]["target"]
    default = [{"name": n, "mass": spec.get("mass", 0.3)} for n, spec in s.manifest["dynamic_objects"].items()]
    return next(o for o in variant.get("objects", default) if o.get("as", o.get("name")) == target)




def local_mesh(s: Session, spec: dict):
    """(vertices, faces) of the target in its prim frame, or None for a library (Poly Haven) model."""
    if spec.get("ph_model"):
        return None
    fix = ((s.scene_graph.get("dynamic_objects") or {}).get(spec["name"]) or {}).get("fix") or {}
    if fix.get("model") and fix.get("mesh") and not spec.get("mesh_path") and not spec.get("as"):
        # an agent-modelled object: the simulator instantiates the fitted fix.mesh in its own frame, the bottom
        # bounding-box centre at the origin (sim.scene_stage._add_kit_object), not the single-view mesh_cam.glb
        with np.load(resolve(fix["mesh"])) as d:
            return np.asarray(d["V"], float) * float(spec.get("scale", 1.0)), np.asarray(d["F"])
    if spec.get("mesh_path"):
        mesh = trimesh.load(resolve(spec["mesh_path"]), force="mesh")
        V = level(np.asarray(mesh.vertices))  # an augmented twin keeps its source's tilt
    else:
        mesh = trimesh.load(s / "objects" / spec["name"] / "mesh_cam.glb", force="mesh")
        T = np.asarray(s.manifest["photo_camera"]["T_world_cam_cv"])
        V = level(np.asarray(mesh.vertices) @ T[:3, :3].T + T[:3, 3])  # as sim.scene.add_object builds it
    bottom = np.array([(V[:, 0].min() + V[:, 0].max()) / 2, (V[:, 1].min() + V[:, 1].max()) / 2, V[:, 2].min()])
    return (V - bottom) * float(spec.get("scale", 1.0)), np.asarray(mesh.faces)






def solid_mesh(V, F, pitch=0.002):
    """The target with its cavities filled: voxelized at `pitch`, every horizontal slice's enclosed holes filled (a
    cup's opening, a bowl's well), re-meshed (marching cubes). Grasps are synthesized on, and the simulator collides
    with, this solid: a hand holding a cup from inside its cavity passes force closure but lets go as the fingers
    squeeze, and is not how the object is picked up. A handle's loop and the gap beside it stay open. Needs scipy and
    scikit-image (envs/grasp)."""
    from scipy import ndimage

    vg = trimesh.Trimesh(V, F, process=False).voxelized(pitch)
    M = ndimage.binary_closing(np.pad(vg.matrix, 1), iterations=1)  # seal pinholes in thin walls
    S = np.stack([ndimage.binary_fill_holes(M[:, :, k]) for k in range(M.shape[2])], axis=2)
    out = trimesh.voxel.ops.matrix_to_marching_cubes(S, pitch=1.0)
    T = vg.transform.copy()
    T[:3, 3] -= T[:3, :3] @ np.ones(3)  # the padding
    out.apply_transform(T)
    return np.asarray(out.vertices), np.asarray(out.faces)






def grasp_key(mesh: str, facing=None, env=None) -> str:
    """Grasp set key: the target mesh, plus the arm direction the seeds faced (rounded: nearby placements share)
    and the environment the synthesis respected (obstacle boxes in the target frame, 1 cm)."""
    key = mesh
    if facing is not None:
        key += f"__arm{hashlib.sha1(np.round(np.asarray(facing, float), 1).tobytes()).hexdigest()[:6]}"
    if env:
        flat = np.round([v for o in env for v in (*o["center"], *o["half"], *np.ravel(o["R"]))], 2)
        key += f"__env{hashlib.sha1(flat.tobytes()).hexdigest()[:6]}"
    return key


def scene_env_world(s: Session, tag: str):
    """[(centre, half, R, name)] obstacle boxes in the world frame (apps/scene_obstacles.py), [] without an export."""
    f = s / GRASP_DIR / "_env" / f"{tag}.json"
    if not f.exists():
        return []
    return [(np.asarray(o["center"], float), np.asarray(o["half"], float), np.asarray(o["R"], float), o["name"])
            for o in json.loads(f.read_text())["obstacles"]]


def scene_env(s: Session, tag: str):
    """(T_world_target, obstacle boxes in the target's frame) from apps/scene_obstacles.py, or (None, None)."""
    f = s / GRASP_DIR / "_env" / f"{tag}.json"
    if not f.exists():
        return None, None
    from scipy.spatial.transform import Rotation

    d = json.loads(f.read_text())
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(d["target"]["quat_wxyz"], scalar_first=True).as_matrix()
    T[:3, 3] = d["target"]["pos"]
    Ti = np.linalg.inv(T)
    local = []
    for o in d["obstacles"]:
        R = np.asarray(o["R"], float)
        local.append({"name": o["name"], "center": (Ti[:3, :3] @ np.asarray(o["center"]) + Ti[:3, 3]).tolist(),
                      "half": list(o["half"]), "R": (Ti[:3, :3] @ R).tolist()})
    return T, local


def hand_name(spec: dict, side: str) -> str | None:
    """BODex hand asset name of a robot spec's arm (`ee.bodex.asset` YAML stem), None for analytic grasp families."""
    b = spec["arms"][side]["ee"].get("bodex")
    return Path(b["asset"]).stem if b else None


def arm_direction(s: Session, spec: dict, side: str, tspec: dict, tag: str | None = None, photo_base: bool = False):
    """Unit direction from the target to the executing arm's shoulder, in the target's prim frame, averaged over the
    robot's accepted placements (shifted with the target, as apps/episode.py does) or, `photo_base`, from where the
    photo shows the real robot (scene graph photo_robot.base); None without placements. With
    `tag` and its environment export (grasps/_env/<tag>.json), the target is where the simulator settled it in its settled orientation."""
    name = spec["name"]
    f = s / "placements" / f"{name}.json"
    if not f.exists() and name == "vega_1p_sharpa":
        f = s / "placements" / "placements.json"
    shoulder = spec["arms"][side].get("shoulder")
    if not f.exists() or shoulder is None:
        return None
    pl = json.loads(f.read_text())
    acc = [c for c in pl["candidates"] if c["accepted"]]
    sg = s.scene_graph
    pb = (sg.get("photo_robot") or {}).get("base")
    if photo_base and pb:  # the robot where the photo shows it (apps/episode.py --photo-base): that base alone
        acc = [{"base": [float(pb["xy"][0]), float(pb["xy"][1]), float(pb["yaw_deg"])], "fixed": True}]
    if not acc:
        return None
    top = float(sg["furniture"][sg["task"]["support"]]["box"][1][2])
    h = spec["base"].get("height", 0.0)
    mount = top if h == "support" else float(h)
    T_env = scene_env(s, tag)[0] if tag else None
    xy = T_env[:2, 3] if T_env is not None else np.asarray(tspec.get("xy") or pl["object"][:2], float)[:2]
    shift = xy - np.asarray(pl["object"][:2], float)
    dirs = []
    for c in acc:
        sx, sy_ = (0.0, 0.0) if c.get("fixed") else shift  # the photo's base does not follow the target
        bx, by, yaw = c["base"][0] + sx, c["base"][1] + sy_, np.radians(c["base"][2])
        cy, sy = np.cos(yaw), np.sin(yaw)
        sh = np.array([bx + cy * shoulder[0] - sy * shoulder[1], by + sy * shoulder[0] + cy * shoulder[1],
                       mount + shoulder[2]])
        v = sh - np.array([xy[0], xy[1], top + 0.05])
        dirs.append(v / np.linalg.norm(v))
    v = np.mean(dirs, axis=0)
    if T_env is not None:  # world -> the settled target's frame
        v = T_env[:3, :3].T @ v
        return v / np.linalg.norm(v)
    oy = float(tspec.get("yaw", 0.0))  # world -> the target's prim frame (yaw only)
    c, sn = np.cos(-oy), np.sin(-oy)
    v = np.array([c * v[0] - sn * v[1], sn * v[0] + c * v[1], v[2]])
    return v / np.linalg.norm(v)


def robot_axes(s: Session, spec: dict, tag: str | None = None, photo_base: bool = False):
    """(forward, left, up) of the robot's base in the target's frame: the circular mean heading of the accepted
    placements (`photo_base`: the heading of the real robot in the photo, scene graph photo_robot.base), turned into
    the settled target's frame (grasps/_env/<tag>.json; the world without it); None without placements."""
    name = spec["name"]
    f = s / "placements" / f"{name}.json"
    if not f.exists() and name == "vega_1p_sharpa":
        f = s / "placements" / "placements.json"
    if not f.exists():
        return None
    yaws = [np.radians(c["base"][2]) for c in json.loads(f.read_text())["candidates"] if c["accepted"]]
    pb = (s.scene_graph.get("photo_robot") or {}).get("base")
    if photo_base and pb:
        yaws = [np.radians(float(pb["yaw_deg"]))]
    if not yaws:
        return None
    yaw = float(np.arctan2(np.mean(np.sin(yaws)), np.mean(np.cos(yaws))))
    T_env = scene_env(s, tag)[0] if tag else None
    Ri = T_env[:3, :3].T if T_env is not None else np.eye(3)
    return (Ri @ [np.cos(yaw), np.sin(yaw), 0.0], Ri @ [-np.sin(yaw), np.cos(yaw), 0.0], Ri @ [0.0, 0.0, 1.0])


def grasp_dir(s: Session, hand: str, key: str) -> Path:
    return s / GRASP_DIR / hand / key


def load_records(s: Session, hand: str, key: str, successful_only=True):
    """Ranked grasp records (best first) of one hand on one target mesh."""
    d = grasp_dir(s, hand, key)
    recs = []
    for p in sorted(d.glob("grasp_*.json")):
        r = json.loads(p.read_text())
        if r.get("success") or not successful_only:
            r["_path"] = rel(p)
            recs.append(r)
    return recs
