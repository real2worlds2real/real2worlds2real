"""Library assets: Poly Haven PBR textures, converted library models (Poly Haven + ABO) and YCB scans."""
import numpy as np
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

from ..assets.library import usd_index
from ..assets.physprops import ROLLING_DAMPING, mass_props, rest_pose, rolls
from calibrated_scene_reconstruction.core.paths import POLYHAVEN_DIR
from .clearance import world_points
from .usd import pbr_material, quat_from_matrix, set_pose

_INDEX = None


def library_index():
    global _INDEX
    if _INDEX is None:
        _INDEX = usd_index()
    return _INDEX


def polyhaven_material(stage, path, name, tint=(1, 1, 1)):
    return pbr_material(stage, path, POLYHAVEN_DIR / "textures" / name, tint)


def planar_uv(V, tile_m):
    """Planar UVs (metres / tile size) along a quad's two dominant axes."""
    ax = np.argsort(V.max(0) - V.min(0))[::-1][:2]
    return (V[:, ax] - V[:, ax].min(0)) / tile_m


def add_library_model(stage, path, model, xy, support_z, yaw=0.0, height=None, scale=None, dynamic=False, mass=0.3,
                      collision=None):
    """Reference a converted library model, fix its up axis, scale to a target height and rest it on a support.
    Same convention as recon.retrieve.place_library_mesh (z-up, yaw about +z, bbox centred on xy). `collision`: the
    mesh collision approximation (default convex decomposition when dynamic, else the convex hull; "none" = the
    triangle mesh, static only: a basket stays hollow)."""
    idx = library_index()
    meta = idx["meta"][model]
    s = scale if scale is not None else (height / meta["size"][2] if height else 1.0)
    q_up = Gf.Quatd(0.7071068, 0.7071068, 0, 0) if meta["up"] == "Y" else Gf.Quatd(1, 0, 0, 0)
    q = Gf.Quatd(float(np.cos(yaw / 2)), 0, 0, float(np.sin(yaw / 2))) * q_up
    root = UsdGeom.Xform.Define(stage, path)
    inner = stage.DefinePrim(f"{path}/model", "Xform")
    set_pose(inner, [0, 0, 0], (q.GetReal(), *q.GetImaginary()), scale=s * meta["mpu"])
    # the referenced asset keeps its own root transform (e.g. a GLB node scale): pose one level above it
    stage.DefinePrim(f"{path}/model/asset", "Xform").GetReferences().AddReference(idx["usd"][model])
    parent = root.GetPrim().GetParent()

    def points():
        """The model's points (from the points: authored extents of converted assets can be stale) in the parent's
        frame: a model built under an offset (a batch copy) is placed relative to it."""
        P = world_points(inner, max_pts=20000)
        if parent and parent.IsA(UsdGeom.Xformable):
            M = np.asarray(UsdGeom.Xformable(parent).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse())
            P = P @ M[:3, :3] + M[3, :3]
        return P

    P = points()
    mn, mx = P.min(0), P.max(0)
    c = (mn[:2] + mx[:2]) / 2
    set_pose(root.GetPrim(), [xy[0] - c[0], xy[1] - c[1], support_z - mn[2] + 0.002])
    for prim in Usd.PrimRange(inner):
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(
                collision or ("convexDecomposition" if dynamic else "convexHull"))
    if dynamic and rolls(model):  # round produce: on its steadiest flat spot, with rolling resistance
        R, _ = rest_pose(P, mass_props(model, P)["com"])
        if not np.allclose(R, np.eye(3)):
            q = Gf.Quatd(*quat_from_matrix(R)) * q
            set_pose(inner, [0, 0, 0], (q.GetReal(), *q.GetImaginary()), scale=s * meta["mpu"])
            set_pose(root.GetPrim(), [0, 0, 0])
            P = points()
            mn, mx = P.min(0), P.max(0)
            c = (mn[:2] + mx[:2]) / 2
            set_pose(root.GetPrim(), [xy[0] - c[0], xy[1] - c[1], support_z - mn[2] + 0.002])
        PhysxSchema.PhysxRigidBodyAPI.Apply(root.GetPrim()).CreateAngularDampingAttr(ROLLING_DAMPING)
    if dynamic:  # centre of mass by the model's class (assets.physprops), in the root frame the points are in
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
        m = UsdPhysics.MassAPI.Apply(root.GetPrim())
        m.CreateMassAttr(float(mass))
        m.CreateCenterOfMassAttr(Gf.Vec3f(*mass_props(model, P)["com"]))
    return root.GetPrim(), mx - mn
