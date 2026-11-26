"""The task zone alone (twin.json `background: zone`): the twin's room and everything of it outside the task zone
switched off (visuals and colliders), a plain floor under the zone in their place. What stays is what the robot
works with, as the frames measured it: the support, the rig over it, the task objects and their marks. For a capture
whose point is the task zone's accuracy, before (or without) any background.

Apps take it with `--background zone` (sim.splat.add_background).
"""
import os

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from .receptacle import look_material

from calibrated_scene_reconstruction.core.environment import setting

FLOOR = {"color": (0.42, 0.42, 0.42), "roughness": 0.9}
DOME = 1000.0  # the dome's intensity (SCENE_RECONSTRUCTION_ZONE_DOME overrides): a white of 0.8 at about 0.8


def keep_task_zone(stage, inputs, keep_names=()):
    """Switches off the children of the room's roots outside the task zone (the support and the rig of the room
    stay, and every kit -- the zone's set pieces, its tape). Returns the paths switched off."""
    inside = inputs.inside
    # every purpose: a box collider is a guide, and with no bound it read as outside the zone and was switched off --
    # the desk's own collider, the task objects fell through it
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"])
    off = []

    def in_zone(prim):
        r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        return not r.IsEmpty() and inside(np.asarray([(np.asarray(r.GetMin()) + np.asarray(r.GetMax())) / 2]))[0]

    for root in ("/World/Room", "/World/Static", "/World/Synthetic", "/World/Support"):
        top = stage.GetPrimAtPath(root)
        if not top.IsValid():
            continue
        for p in top.GetChildren():
            if p.GetName() in keep_names or p.GetCustomDataByKey("calibrated_scene_reconstruction:kit"):  # the task zone's set pieces (tape)
                continue
            if root == "/World/Room" and in_zone(p):  # the support and the rig
                continue
            p.SetActive(False)
            off.append(str(p.GetPath()))
    return off


def add_zone_background(stage, session, half=6.0, floor_visible=True):
    """The task zone alone on a plain floor (a 2 x `half` m square at the floor's height, with its collider).
    `floor_visible` False: the floor a collider the camera does not see (a splat shows the floor). Returns the paths
    switched off."""
    s = session.session
    off = keep_task_zone(stage, session)
    zf = float(s.scene_graph.get("floor_z", 0.0))
    UsdGeom.Scope.Define(stage, "/World/Zone")
    floor = UsdGeom.Mesh.Define(stage, "/World/Zone/floor")
    floor.CreatePointsAttr([Gf.Vec3f(x, y, zf - 0.001) for x, y in ((-half, -half), (half, -half), (half, half), (-half, half))])
    floor.CreateFaceVertexCountsAttr([4])
    floor.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    floor.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(floor.GetPrim()).Bind(look_material(stage, "/World/Looks/zone_floor", FLOOR))
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    if not floor_visible:
        UsdGeom.PrimvarsAPI(floor.GetPrim()).CreatePrimvar("hideForCamera", Sdf.ValueTypeNames.Bool,
                                                          UsdGeom.Tokens.constant).Set(True)
    # its light: a uniform white dome (the room that lit it is gone), at a fixed exposure with linear tone mapping --
    # a white of 0.8 renders near 0.8, as the frames show it (the room's photo lights left it dark: the close-ups'
    # cups a brown grey against the frames' white, contrast -1.1)
    from pxr import UsdLux

    from .splat_stage import fixed_exposure

    dome = UsdLux.DomeLight.Define(stage, "/World/Zone/dome")
    dome.CreateIntensityAttr(float(setting("ZONE_DOME", DOME)))
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    lights = stage.GetPrimAtPath("/World/Lights")
    for p in (lights.GetChildren() if lights.IsValid() and not setting("ZONE_KEEP_LIGHTS") else []):
        p.SetActive(False)
        off.append(str(p.GetPath()))
    fixed_exposure()
    return off
