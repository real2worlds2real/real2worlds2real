"""A recording's tabletop twin (calibrated_scene_reconstruction.physics.replay.tabletop, real/<name>/twin.json) as a USD stage, for recordings without
a reconstructed session. Same layout as sim.scene, so apps/real2sim_replay.py and sim.physics treat both alike:

/World/PhysicsScene   gravity 9.81, CCD
/World/Support/top    static box: the support, its top at the fitted plane
/World/Room/floor     static box 0.75 m below the support (an object pushed off the edge lands)
/World/Objects/<name> rigid body: the fitted primitive as a mesh with a convex-hull collider, origin at its bottom
author_twin re-writes the support and the object geometry under a root (a batch copy), e.g. for another depth scale
(calibrated_scene_reconstruction.physics.replay.tabletop.scaled); like every USD edit it takes effect at the next world.reset().
"""
import numpy as np
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, Vt

from calibrated_scene_reconstruction.assets.primitive_mesh import primitive_mesh
from .usd import add_box_collider, add_mesh, color_material, set_pose


def object_mesh(obj):
    if obj['shape'] == 'mesh':
        from calibrated_scene_reconstruction.assets.model_artifact import read_mesh
        from calibrated_scene_reconstruction.core.paths import resolve
        return read_mesh(resolve(obj['mesh']))
    return primitive_mesh(obj['shape'], obj['dims'])


def build_tabletop(stage, twin):
    """Populate an empty stage from a twin.json dict; returns the object's prim path."""
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)
    PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim()).CreateEnableCCDAttr(True)
    for scope in ("/World/Looks", "/World/Support", "/World/Room", "/World/Objects"):
        UsdGeom.Scope.Define(stage, scope)
    lo, hi = (np.asarray(v, float) for v in twin["support"]["box"])
    add_box_collider(stage, "/World/Support/top", lo, hi)
    add_box_collider(stage, "/World/Room/floor", [lo[0] - 2, lo[1] - 2, hi[2] - 0.8], [hi[0] + 2, hi[1] + 2, hi[2] - 0.75])
    o = twin["object"]
    V, F = object_mesh(o)
    path = f"/World/Objects/{o['name']}"
    xf = UsdGeom.Xform.Define(stage, path)
    geom = add_mesh(stage, f"{path}/geom", V, F, material=color_material(stage, "/World/Looks/twin_object",
                                                                           (0.8, 0.3, 0.2)))
    UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
    agent_model = o.get('route') == 'agent_blender'
    UsdPhysics.MeshCollisionAPI.Apply(geom.GetPrim()).CreateApproximationAttr('sdf' if agent_model else 'convexHull')
    if agent_model:
        PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(geom.GetPrim()).CreateSdfResolutionAttr(256)
    UsdPhysics.RigidBodyAPI.Apply(xf.GetPrim())
    UsdPhysics.MassAPI.Apply(xf.GetPrim()).CreateMassAttr(float(o.get("mass", 0.3)))
    author_twin(stage, twin)
    return path


def author_twin(stage, twin, root="/World"):
    """Support box, object pose, mesh and centre of mass of the twin under `root` (/World or a batch copy)."""
    lo, hi = (np.asarray(v, float) for v in twin["support"]["box"])
    set_pose(stage.GetPrimAtPath(f"{root}/Support/top"), (lo + hi) / 2, scale=hi - lo)
    o = twin["object"]
    path = f"{root}/Objects/{o['name']}"
    x, y, z, yaw = o["pose"]
    h = np.radians(yaw) / 2
    set_pose(stage.GetPrimAtPath(path), [x, y, z + 0.001], (np.cos(h), 0, 0, np.sin(h)))
    V, _F = object_mesh(o)
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(f"{path}/geom"))
    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(V, np.float32)))
    mesh.GetExtentAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*V.min(0).tolist()), Gf.Vec3f(*V.max(0).tolist())]))
    UsdPhysics.MassAPI(stage.GetPrimAtPath(path)).CreateCenterOfMassAttr().Set(Gf.Vec3f(*((V.max(0) + V.min(0)) / 2)))
