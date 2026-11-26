"""The room written as code (twin.json `background: code`, recon.code_scene): the photo's room and everything of it
outside the task zone switched off (their visuals and their colliders: the photo's room is a box round what one
picture saw), and in their place the pieces of configs/<name>/code_scene.json -- the shell (floor, walls, ceiling)
and the furniture, each a procedural kit (assets.office_kits) at its measured place and size, in the colours the walk
saw -- with the room's ceiling light panels. The task zone (the support, the rig, the task objects, their marks) is
the twin's as built.

Apps take it with `--background code`.
"""
import json

import numpy as np
from pxr import UsdGeom, UsdLux

from .setpiece_stage import add_set_piece
from .zone_stage import keep_task_zone
from .usd import set_pose


def add_code_background(stage, session):
    """Returns the paths switched off."""
    s = session.session
    spec = json.loads((s.config_dir / "code_scene.json").read_text())
    off = keep_task_zone(stage, session, set(spec.get("keep", [])))
    UsdGeom.Scope.Define(stage, "/World/Code")
    pieces = {**spec["pieces"], **spec.get("manual", {})}  # the agent's corrections win; null removes a piece
    for name, p in pieces.items():
        if p is None or name.startswith("_"):
            continue
        params = {k: v for k, v in p.items() if k not in ("kit", "center", "yaw_deg", "z", "seed", "_from")}
        add_set_piece(stage, f"/World/Code/{name}", p["kit"], params, [*p["center"], p.get("z", 0.0)],
                      np.radians(p.get("yaw_deg", 0.0)), seed=p.get("seed", 0))
    li = {**spec.get("lights", {}), **(spec.get("manual", {}).get("_lights") or {})}
    if li.get("panels"):  # the photo's recessed disks off, the room's panels on (lights.json's other lights stay)
        lights = stage.GetPrimAtPath("/World/Lights")
        for p in (lights.GetChildren() if lights.IsValid() else []):
            if p.GetName().startswith("ceiling_") and p.GetName() != "ceiling_bounce":
                p.SetActive(False)
        zc = float(li.get("z", 2.7)) - 0.01
        for k, (x, y, w, h) in enumerate(li["panels"]):
            r = UsdLux.RectLight.Define(stage, f"/World/Code/light_{k}")
            r.CreateWidthAttr(float(w))
            r.CreateHeightAttr(float(h))
            r.CreateIntensityAttr(float(li.get("intensity", 6000.0)))
            r.CreateEnableColorTemperatureAttr(True)
            r.CreateColorTemperatureAttr(float(li.get("temp", 4500.0)))
            set_pose(r.GetPrim(), [x, y, zc])  # a rect light emits along its local -z: down
    return off
