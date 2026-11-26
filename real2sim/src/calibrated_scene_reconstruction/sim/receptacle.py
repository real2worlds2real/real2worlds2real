"""Place targets in the simulator (assets.receptacles): static triangle-mesh colliders with their own looks --
Poly Haven PBR textures, generated texture sets (woven fibre, cork, slate) or a glaze / metal / tape colour for the
procedural kinds, the scanned textures of library models.

    add_receptacle(stage, path, rec, origin, yaw) -> prim     (rec: a Receptacle; origin: world xyz of its frame)

The prim is rebuilt per episode (each plan has its own goal): remove_receptacle() first, author before world.reset()
so the physics scene is parsed with it.
"""
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

from ..assets.receptacles import texture_set
from calibrated_scene_reconstruction.core.paths import POLYHAVEN_DIR, ROOT
from .assets import add_library_model
from .usd import add_mesh, color_material, glass_material, pbr_material, set_pose


def _physics_material(stage):
    path = "/World/Looks/receptacle_physics"
    if not stage.GetPrimAtPath(path):
        m = UsdShade.Material.Define(stage, path)
        pm = UsdPhysics.MaterialAPI.Apply(m.GetPrim())
        pm.CreateStaticFrictionAttr(0.8)
        pm.CreateDynamicFrictionAttr(0.6)
        pm.CreateRestitutionAttr(0.0)
    return UsdShade.Material(stage.GetPrimAtPath(path))


def _collide(stage, prim):
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(_physics_material(stage), UsdShade.Tokens.weakerThanDescendants,
                                                 "physics")


def look_material(stage, path, m):
    tint = tuple(m.get("tint", (1, 1, 1)))
    if "tex" in m:
        return pbr_material(stage, path, POLYHAVEN_DIR / "textures" / m["tex"], tint)
    if "proc" in m:
        return pbr_material(stage, path, ROOT / texture_set(m), tint)
    if m.get("glass"):
        return glass_material(stage, path, m["color"], m.get("ior", 1.5), m.get("roughness", 0.0),
                              m.get("thin_walled", False))
    return color_material(stage, path, m["color"], m.get("roughness", 0.6), m.get("metallic", 0.0))


def remove_receptacle(stage, path):
    if stage.GetPrimAtPath(path):
        stage.RemovePrim(path)


def add_receptacle(stage, path, rec, origin, yaw):
    remove_receptacle(stage, path)
    root = UsdGeom.Xform.Define(stage, path)
    root.GetPrim().SetCustomDataByKey("calibrated_scene_reconstruction:receptacle", rec.kind)
    name = rec.source.partition(":")[2]
    if rec.parts:
        for k, part in enumerate(rec.parts):
            look = f"/World/Looks/receptacle_{path.strip('/').replace('/', '_')}_{k}"
            stage.RemovePrim(look)  # the previous plan's material: a stale texture input would win over a colour
            mat = look_material(stage, look, part["material"])
            geom = add_mesh(stage, f"{path}/{part['name']}", part["V"], part["F"], part["UV"], mat)
            _collide(stage, geom.GetPrim())
    else:  # a library model, built at the identity first: add_library_model measures world points
        add_library_model(stage, f"{path}/model", name, (0.0, 0.0), 0.0, scale=rec.scale, collision="none")
        for p in Usd.PrimRange(root.GetPrim()):
            if p.IsA(UsdGeom.Mesh):
                _collide(stage, p)
    set_pose(root.GetPrim(), [float(v) for v in origin], (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)))
    return root.GetPrim()
