"""Editable capsule-chain ropes in an existing PhysX world, without a policy.

This optional backend complements the Newton rod model. D6 limits/drives are
discrete joint assumptions and must not be called equivalent rod coefficients.
"""
from __future__ import annotations

import numpy as np


def chain_geometry(points_m, radius_m, density_kg_m3):
    p = np.asarray(points_m, float)
    if (p.ndim != 2 or p.shape[1] != 3 or not 2 <= len(p) <= 513
            or not np.isfinite(p).all()):
        raise ValueError("An ordered finite centerline with 2..513 nodes required")
    if (not np.isfinite([radius_m, density_kg_m3]).all()
            or radius_m <= 0 or density_kg_m3 <= 0):
        raise ValueError("Positive finite radius and density required")
    vectors = np.diff(p, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(lengths <= 2 * radius_m):
        raise ValueError("Segments must exceed twice the radius; resample the centerline")
    masses = np.pi * radius_m**2 * lengths * density_kg_m3
    return {"points": p, "centers": (p[1:] + p[:-1]) / 2,
            "axes": vectors / lengths[:, None], "lengths": lengths, "masses": masses}


def author_rope(stage, path, points_m, *, radius_m, density_kg_m3,
                bend_limit_deg=110.0, bend_stiffness=0.0, bend_damping=1e-6,
                static_friction=0.8, dynamic_friction=0.6, contact_offset_m=None,
                color=(0.93, 0.42, 0.12)):
    """Author dynamic capsules and D6 joints; no anchors or prescribed motion.

    Centerline segment endpoints are the native joint anchors. Total mass is
    rho*pi*r^2*centerline_length; overlapping capsule endcaps add no extra mass.
    Adjacent capsule contact is disabled, nonadjacent self-contact stays active.
    """
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    g = chain_geometry(points_m, radius_m, density_kg_m3)
    offset = radius_m * 0.2 if contact_offset_m is None else contact_offset_m
    numbers = [bend_limit_deg, bend_stiffness, bend_damping, static_friction, dynamic_friction, offset]
    if (not np.isfinite(numbers).all() or not 0 < bend_limit_deg < 180
            or min(numbers[1:]) < 0 or offset <= 0 or dynamic_friction > static_friction):
        raise ValueError("Invalid bend/contact settings")
    if UsdGeom.GetStageMetersPerUnit(stage) != 1:
        raise ValueError("Author ropes in a metre stage")
    if not Sdf.Path(path).IsAbsolutePath() or stage.GetPrimAtPath(path):
        raise ValueError("Choose a new absolute asset prim path")
    colors = np.asarray(color, float)
    if colors.shape != (3,) or not np.isfinite(colors).all() or np.any((colors < 0) | (colors > 1)):
        raise ValueError("Color must be RGB in [0,1]")
    bodies, joints, local_anchors = [], [], []
    try:
        UsdGeom.Xform.Define(stage, path).SetResetXformStack(True)
        material = UsdShade.Material.Define(stage, path + "/ContactMaterial")
        phys = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        phys.CreateStaticFrictionAttr(static_friction)
        phys.CreateDynamicFrictionAttr(dynamic_friction)
        phys.CreateRestitutionAttr(0.0)
        for i, (center, axis, length, mass) in enumerate(zip(
                g["centers"], g["axes"], g["lengths"], g["masses"], strict=True)):
            body_path = f"{path}/Segment_{i:03d}"
            cap = UsdGeom.Capsule.Define(stage, body_path)
            cap.CreateAxisAttr("X")
            cap.CreateHeightAttr(float(length))
            cap.CreateRadiusAttr(radius_m)
            cap.AddTranslateOp().Set(Gf.Vec3d(*center))
            rotation = Gf.Rotation(Gf.Vec3d(1, 0, 0), Gf.Vec3d(*axis)).GetQuat()
            cap.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(rotation)
            cap.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([Gf.Vec3f(*colors)])
            prim = cap.GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(prim)
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(mass))
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")
            from pxr import PhysxSchema
            collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            collision.CreateRestOffsetAttr(0.0)
            collision.CreateContactOffsetAttr(float(offset))
            body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            body.CreateSolverPositionIterationCountAttr(16)
            body.CreateSolverVelocityIterationCountAttr(4)
            bodies.append(body_path)
        for i in range(len(bodies) - 1):
            joint_path = f"{path}/Joint_{i:03d}"
            joint = UsdPhysics.Joint.Define(stage, joint_path)
            joint.CreateBody0Rel().SetTargets([Sdf.Path(bodies[i])])
            joint.CreateBody1Rel().SetTargets([Sdf.Path(bodies[i + 1])])
            a, b = float(g["lengths"][i] / 2), -float(g["lengths"][i + 1] / 2)
            joint.CreateLocalPos0Attr(Gf.Vec3f(a, 0, 0))
            joint.CreateLocalPos1Attr(Gf.Vec3f(b, 0, 0))
            joint.CreateCollisionEnabledAttr(False)
            for dof in ("transX", "transY", "transZ", "rotX"):
                limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), dof)
                limit.CreateLowAttr(1.0)
                limit.CreateHighAttr(-1.0)
            for dof in ("rotY", "rotZ"):
                limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), dof)
                limit.CreateLowAttr(-bend_limit_deg)
                limit.CreateHighAttr(bend_limit_deg)
                drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), dof)
                drive.CreateTypeAttr("force")
                drive.CreateStiffnessAttr(bend_stiffness)
                drive.CreateDampingAttr(bend_damping)
            joints.append(joint_path)
            local_anchors.append([[a, 0, 0], [b, 0, 0]])
    except Exception:
        stage.RemovePrim(path)
        raise
    return {"path": path, "bodies": bodies, "joints": joints, "anchors": local_anchors,
            "centerline_length_m": float(g["lengths"].sum()), "radius_m": radius_m,
            "total_mass_kg": float(g["masses"].sum()),
            "rest_segment_lengths_m": g["lengths"].tolist(),
            "settings": {"bend_limit_deg": bend_limit_deg, "bend_stiffness": bend_stiffness,
                         "bend_damping": bend_damping, "contact_offset_m": offset}}


def read_rope_state(stage, asset):
    """Sample USD transforms with CPU writeback; no Fabric state is inferred."""
    from pxr import Gf, Usd, UsdGeom

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    transforms = [cache.GetLocalToWorldTransform(stage.GetPrimAtPath(p)) for p in asset["bodies"]]
    centers = np.array([m.Transform(Gf.Vec3d(0)) for m in transforms])
    endpoints = np.array([[m.Transform(Gf.Vec3d(-length/2, 0, 0)),
                           m.Transform(Gf.Vec3d(length/2, 0, 0))]
                          for m, length in zip(transforms, asset["rest_segment_lengths_m"], strict=True)])
    gaps = []
    for i, anchors in enumerate(asset["anchors"]):
        a = transforms[i].Transform(Gf.Vec3d(*anchors[0]))
        b = transforms[i + 1].Transform(Gf.Vec3d(*anchors[1]))
        gaps.append(float((a - b).GetLength()))
    return {"centers_m": centers, "joint_gaps_m": np.asarray(gaps),
            "segment_endpoints_m": endpoints,
            "minimum_surface_z_m": float(endpoints[:, :, 2].min() - asset["radius_m"]),
            "finite": bool(np.isfinite(centers).all() and np.isfinite(gaps).all()
                           and np.isfinite(endpoints).all())}
