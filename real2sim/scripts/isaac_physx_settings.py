"""Activate runtime support for opt-in PhysX properties retained in asset USDs."""

from pxr import PhysxSchema, UsdPhysics


def enable_requested_ccd(stage):
    """Sweep-based CCD requires both the body flag and the physics-scene flag."""
    bodies = [str(prim.GetPath()) for prim in stage.Traverse()
              if prim.HasAPI(UsdPhysics.RigidBodyAPI)
              and (attr := prim.GetAttribute("physxRigidBody:enableCCD")) and attr.Get()]
    scenes = []
    if bodies:
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                gpu = prim.GetAttribute("physxScene:enableGPUDynamics")
                if gpu and gpu.Get():
                    raise RuntimeError("Requested CCD is unsupported by this GPU dynamics adapter; use CPU physics or a supported collision/timestep configuration")
                PhysxSchema.PhysxSceneAPI.Apply(prim).CreateEnableCCDAttr(True)
                scenes.append(str(prim.GetPath()))
        if not scenes:
            raise RuntimeError("Asset requests CCD but no physics scene exists")
    return {"requested_body_paths": bodies, "enabled_scene_paths": scenes}
