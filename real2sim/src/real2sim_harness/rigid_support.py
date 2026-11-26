"""Ground support from authored colliders, independent of a body's visual AABB."""

from __future__ import annotations

import itertools

import numpy as np
from scipy.spatial.transform import Rotation


def _sphere_height(center, radius, transform):
    """Analytic support of a sphere under a rigid, uniformly scaled transform."""
    linear = transform[:3, :3]
    gram = linear @ linear.T
    scale2 = float(np.trace(gram) / 3)
    if scale2 <= 0 or not np.allclose(gram, np.eye(3) * scale2, rtol=1e-6, atol=1e-12):
        raise ValueError("Sphere support requires uniform scale without shear")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Invalid collider sphere radius")
    return float(np.asarray(center) @ transform[:3, 2] + transform[3, 2] - radius * np.sqrt(scale2))


def _cospherical_mesh(points):
    """Recognize an authored sphere, not an arbitrary backend sphere fit.

    All vertices must be on one sphere and include its six axial extrema.
    The latter provides antipodal witnesses for the minimum enclosing radius.
    General meshes and nonuniformly scaled spheres remain unsupported.
    """
    center = (points.min(0) + points.max(0)) / 2
    distances = np.linalg.norm(points - center, axis=1)
    radius = float(distances.max())
    tolerance = max(radius * 1e-5, 1e-8)
    extrema = center + np.r_[np.eye(3), -np.eye(3)] * radius
    if radius <= 0 or (np.abs(distances - radius) > tolerance).any() or any(
        np.linalg.norm(points - point, axis=1).min() > tolerance for point in extrema
    ):
        raise ValueError("Unsupported boundingSphere fit: mesh is not a witnessed authored sphere")
    return center, radius


def collider_support(stage, body_path: str, pose_xyzw) -> dict:
    """Measure collider height at a measured rigid pose (metres, XYZW).

    Mesh vertices give the exact support of the *authored* convex hull; cooked
    hull simplification, decomposition, SDF discretization and contact/rest
    offsets can differ. Report that basis explicitly, rather than claiming a
    query of the solver's cooked geometry. Unknown shapes fail closed.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    pose = np.asarray(pose_xyzw, dtype=float)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("Expected one finite XYZ + XYZW rigid pose")
    body = stage.GetPrimAtPath(body_path)
    if not body.IsValid():
        raise ValueError(f"Missing rigid body: {body_path}")
    cache = UsdGeom.XformCache()
    body_transform = cache.GetLocalToWorldTransform(body)
    inverse = np.asarray(body_transform.GetInverse())
    basis = np.asarray(body_transform)[:3, :3]
    scales = np.linalg.norm(basis, axis=1)
    if (np.any(scales <= 0) or not np.isfinite(scales).all()
            or np.linalg.det(basis) <= 0
            or not np.allclose((basis / scales[:, None]) @ (basis / scales[:, None]).T, np.eye(3), atol=1e-6)):
        raise ValueError("Rigid support requires a nonsingular, unreflected body transform without shear")
    scale_transform = np.diag([*scales, 1.])
    world = np.eye(4)
    world[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix().T
    world[3, :3] = pose[:3]
    records = []
    for prim in Usd.PrimRange(body):
        if prim != body and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError("Nested rigid bodies require their own measured poses")
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            continue
        transform = np.asarray(cache.GetLocalToWorldTransform(prim)) @ inverse @ scale_transform @ world
        approximation = None
        if prim.IsA(UsdGeom.Mesh):
            points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
            approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() or "none"
            if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
                raise ValueError(f"Invalid collider points: {prim.GetPath()}")
            if approximation == "boundingCube":
                points = np.array(list(itertools.product(*zip(points.min(0), points.max(0), strict=True))))
            elif approximation == "boundingSphere":
                center, radius = _cospherical_mesh(points)
                minimum = _sphere_height(center, radius, transform)
                records.append({"path": str(prim.GetPath()), "min_z_m": minimum,
                                "basis": "authored_cospherical_mesh", "approximation": approximation})
                continue
            elif approximation not in {"none", "convexHull", "convexDecomposition", "sdf"}:
                raise ValueError(f"Unsupported collider support approximation: {approximation}")
            basis = "authored_mesh_vertices" if approximation != "boundingCube" else "authored_bounding_cube"
        elif prim.IsA(UsdGeom.Cube):
            half = float(UsdGeom.Cube(prim).GetSizeAttr().Get()) / 2
            points = np.array(list(itertools.product([-half, half], repeat=3)))
            basis = "authored_cube"
        elif prim.IsA(UsdGeom.Sphere):
            minimum = _sphere_height(np.zeros(3), float(UsdGeom.Sphere(prim).GetRadiusAttr().Get()), transform)
            records.append({"path": str(prim.GetPath()), "min_z_m": minimum,
                            "basis": "authored_sphere", "approximation": None})
            continue
        else:
            raise ValueError(f"Unsupported collider support shape: {prim.GetTypeName()}")
        height = points @ transform[:3, 2] + transform[3, 2]
        if not np.isfinite(height).all():
            raise ValueError(f"Non-finite collider transform: {prim.GetPath()}")
        records.append({"path": str(prim.GetPath()), "min_z_m": float(height.min()),
                        "basis": basis, "approximation": approximation})
    if not records:
        raise ValueError(f"No enabled supported colliders below {body_path}")
    return {"min_z_m": min(row["min_z_m"] for row in records), "colliders": records,
            "limits": "Authored geometry support; solver cooking and contact/rest offsets are not measured."}
