"""Robot specs: configs/robots/<name>.json describes a body + end effector(s) for the generic `Robot` wrapper.

Schema (paths relative to the repo root; angles in radians):
  name                  identifier (also the built USD's folder under assets/robots/built/)
  usd | urdf            a ready USD ("usd_env" names an environment variable that overrides it), or
                        {"body", "ee"?, "flange"?, "prefix"?, "xyz"?, "rpy"?, "drop"? (body links removed first, with
                        what they carry)} assembled + imported by
                        apps/robot_build.py
  base                  {"type": "fixed|floating", "mount": "floor|pedestal", "height": m | "support" (base origin
                        above the floor: pedestal top, a humanoid's pelvis; "support" = the support's top surface),
                        "standoff": [m, m], "lateral": [m, m], "yaw_tol_deg": deg}
                        standoff/lateral/yaw sample the base in the task frame (see calibrated_scene_reconstruction.sim.taskframe)
  drives                [{"match": regex, "kp", "kd"?, "max_force"}], first match wins; kd defaults to 0.1 kp
  home                  {joint: q} (others 0); authored as the reset state so the first step cannot cut through
                        nearby furniture
  arms                  {side: {"ik_joints": [...], "tcp_link", "tcp_offset": [x,y,z] (tcp link frame), "tcp_axes":
                        {"approach": "+x", "closing": "+y"}, "ee": {...}}}
  arms.<side>.ik_weights  {joint regex: weight} for the weighted IK (default 1; a torso at 0.2 moves less)
  arms.<side>.ik_limits   {joint: [lo, hi]} a working range the IK keeps within the joint limits (a torso that may
                          lean a little, not fold over the support)
  arms.<side>.shoulder  [x, y, z] of the arm's first link at home in the base frame (z above the base mount): BODex
                        grasp seeds face it (calibrated_scene_reconstruction.grasping.synth)
  arms.<side>.ee        {"type": "parallel|dexterous", "joints": regex, "close": {"mode": "frac_upper", "fraction":
                        {suffix: f}, "cap": 0.95} | {"mode": "targets", "open": {joint: q}, "closed": {joint: q}},
                        "grasp": family name (calibrated_scene_reconstruction.robots.grasps), "min_grasp_height": m above the support
                        (power_lateral: a large palm held thumb-up must not reach into the slab), "max_width": m
                        (parallel: stroke; wider targets are swapped for a YCB cup that fits), "collision":
                        USD approximation of the finger links, e.g. "convexDecomposition" (default: importer hulls),
                        "contact_material": {"static_friction", "dynamic_friction", "restitution"?, "combine"?
                        ("average|min|multiply|max"), "source": where the values come from} bound to every collider of
                        the links the ee joints connect (default: none, so PhysX's 0.5 / 0.5 averaged with the object)}
  cameras               [{"name", "link", "pos" (in the link frame: outside its mesh), "quat_wxyz", "K": 3x3 at
                        "size": [w, h], "role": "head|wrist|..."}] (head: coverage scoring in apps/placements.py)
  physics               {"disable_gravity": bool, "self_collision": bool, "solver_iters": [pos, vel]}
"""
import json
import os
from pathlib import Path

from calibrated_scene_reconstruction.core.environment import setting
from calibrated_scene_reconstruction.core.paths import CONFIGS, ROBOTS_DIR, robot_asset_path

BUILT_DIR = ROBOTS_DIR / "built"
SPEC_DIR = CONFIGS / "robots"


def load_spec(name_or_path) -> dict:
    p = Path(name_or_path)
    if not p.suffix:
        p = SPEC_DIR / f"{name_or_path}.json"
    spec = json.loads(Path(p).read_text())
    spec.setdefault("drives", [])
    spec.setdefault("home", {})
    spec.setdefault("cameras", [])
    spec.setdefault("physics", {})
    spec.setdefault("base", {"type": "fixed", "mount": "floor"})
    if spec.get("urdf"):
        u = spec["urdf"]
        for k in ("body", "ee"):
            if u.get(k):
                u[k] = str(robot_asset_path(u[k]))
        spec["usd"] = str(BUILT_DIR / spec["name"] / f"{spec['name']}.usd")
    else:
        key = spec.get("usd_env", "")
        for prefix in ("SCENE_RECONSTRUCTION_", "R2S2R_"):
            if key.startswith(prefix):
                source = setting(key.removeprefix(prefix), spec["usd"])
                break
        else:
            source = os.environ.get(key, spec["usd"])
        spec["usd"] = str(robot_asset_path(source))
    for arm in spec.get("arms", {}).values():
        bodex = arm.get("ee", {}).get("bodex")
        if bodex and bodex.get("asset"):
            bodex["asset"] = str(robot_asset_path(bodex["asset"]))
    return spec


def list_specs():
    return sorted(p.stem for p in SPEC_DIR.glob("*.json"))
