"""Add coarse PhysX links and joints to an exported Blender USD asset."""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from real2sim_harness.physx_mass_properties import author_mass_properties
from real2sim_harness.rigid_frames import rotation_only
from real2sim_harness.sim_parameters import load_parameters


def safe_name(name):
    value = re.sub(r"[^A-Za-z0-9_]", "_", name)
    return value if value and not value[0].isdigit() else f"joint_{value}"


def find_blender_objects(stage):
    result = {}
    for prim in stage.Traverse():
        attr = prim.GetAttribute("userProperties:blender:object_name")
        if attr and attr.IsValid() and attr.Get():
            result[attr.Get()] = prim
    return result


def add_box(stage, body_path, bounds, name="Collision", collision=True):
    if bounds is None:
        return None
    low = bounds["min"]
    high = bounds["max"]
    center = [(a + b) / 2 for a, b in zip(low, high, strict=True)]
    size = [max(0.005, b - a) for a, b in zip(low, high, strict=True)]
    cube = UsdGeom.Cube.Define(stage, f"{body_path}/{name}")
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(*center))
    xform.AddScaleOp().Set(Gf.Vec3f(*size))
    cube.MakeInvisible()
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return math.prod(size)


def add_body(prim, volume, *, mass_kg=None, density=100.0, enable_ccd=False):
    rigid = UsdPhysics.RigidBodyAPI.Apply(prim)
    rigid.CreateRigidBodyEnabledAttr(True)
    if enable_ccd:
        prim.AddAppliedSchema("PhysxRigidBodyAPI")
        prim.CreateAttribute("physxRigidBody:enableCCD", Sdf.ValueTypeNames.Bool).Set(True)
    mass = UsdPhysics.MassAPI.Apply(prim)
    mass.CreateMassAttr(float(mass_kg) if mass_kg is not None else max(0.01, min(20.0, volume * density)))


def add_component_colliders(stage, body, components, material, enabled=True):
    count = 0
    for index, component in enumerate(components):
        if not component["faces"]:
            continue
        path = f"{body.GetPath()}/Colliders/part_{index:04d}"
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in component["vertices"]])
        mesh.CreateFaceVertexCountsAttr([3] * len(component["faces"]))
        mesh.CreateFaceVertexIndicesAttr([i for face in component["faces"] for i in face])
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.MakeInvisible()
        prim = mesh.GetPrim()
        prim.CreateAttribute("remo:sourceObject", Sdf.ValueTypeNames.String).Set(component["object_name"])
        if enabled:
            UsdPhysics.CollisionAPI.Apply(prim)
            approximation = component["approximation"]
            if approximation not in {"convexHull", "convexDecomposition", "boundingCube", "boundingSphere", "sdf"}:
                raise ValueError(f"Unsupported dynamic collision approximation: {approximation}")
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(approximation)
            if approximation == "sdf":
                prim.AddAppliedSchema("PhysxSDFMeshCollisionAPI")
                prim.CreateAttribute("physxSDFMeshCollision:sdfResolution", Sdf.ValueTypeNames.Int).Set(component.get("sdf_resolution", 256))
            if "contact_offset_m" in component or "rest_offset_m" in component:
                prim.AddAppliedSchema("PhysxCollisionAPI")
                for key, attr in (("contact_offset_m", "contactOffset"), ("rest_offset_m", "restOffset")):
                    if key in component:
                        prim.CreateAttribute("physxCollision:" + attr, Sdf.ValueTypeNames.Float).Set(component[key])
            if approximation == "convexDecomposition":
                prim.AddAppliedSchema("PhysxConvexDecompositionCollisionAPI")
                prim.CreateAttribute("physxConvexDecompositionCollision:maxConvexHulls", Sdf.ValueTypeNames.Int).Set(16)
                prim.CreateAttribute("physxConvexDecompositionCollision:hullVertexLimit", Sdf.ValueTypeNames.Int).Set(64)
                prim.CreateAttribute("physxConvexDecompositionCollision:voxelResolution", Sdf.ValueTypeNames.Int).Set(100000)
            binding = UsdShade.MaterialBindingAPI.Apply(prim)
            binding.Bind(material, materialPurpose="physics")
        count += 1
    return count


def author_angular_drive(joint, settings):
    """Convert SI gains to USD's degree-based angular drive coefficients."""
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    # USD multiplies these by degree/degree-per-second errors. PhysX tensor
    # views expose the corresponding radian-based coefficients instead.
    drive.CreateStiffnessAttr(math.radians(settings["stiffness"]))
    drive.CreateDampingAttr(math.radians(settings["damping"]))
    drive.CreateMaxForceAttr(settings["max_force"])
    drive.CreateTargetPositionAttr(0.0)
    return drive


def author_collision_pairs(stage, root, bodies, pairs):
    """Author symmetric link filtering and explicitly enable articulation contact."""
    allowed = {frozenset((bodies[a], bodies[b])) for a, b in pairs}
    if root.HasAPI(UsdPhysics.ArticulationRootAPI):
        root.AddAppliedSchema("PhysxArticulationAPI")
        root.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(bool(allowed))
    for path in bodies.values():
        filtered = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(path))
        filtered.CreateFilteredPairsRel().SetTargets([
            other for other in bodies.values() if other != path and frozenset((path, other)) not in allowed
        ])
    return allowed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("--no-collision", action="store_true", help="Diagnostic variant that isolates joint drives")
    args = parser.parse_args()
    folder = Path(args.folder).resolve()
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["asset_kind"] not in {"rigid", "articulated"}:
        raise ValueError("expected rigid or articulated asset")
    articulated = manifest["asset_kind"] == "articulated"
    components = json.loads((folder / manifest["collision_meshes"]).read_text()) if manifest.get("collision_meshes") else []
    parameters = load_parameters(folder, create=True)
    mass_overrides = parameters["physx"]["mass_properties"]
    known_bodies = {"base", *("joint:" + spec["name"] for spec in manifest["joints"])}
    if set(mass_overrides) - known_bodies:
        raise ValueError("Mass properties name a body absent from the exported asset")
    collision_pairs = parameters["physx"]["enabled_collision_pairs"]
    if any(set(pair) - known_bodies for pair in collision_pairs):
        raise ValueError("Collision pair names a body absent from the exported asset")
    assumptions = parameters["rigid"] | parameters["material"]
    output = folder / ("physx_no_collision.usda" if args.no_collision else "physx.usda")
    stage = Usd.Stage.CreateNew(str(output))
    root = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
    root.GetReferences().AddReference("./visual.usdc")
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    objects = find_blender_objects(stage)
    material = UsdShade.Material.Define(stage, "/Asset/PhysicsMaterial")
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    material_api.CreateStaticFrictionAttr(assumptions.get("friction", 0.6))
    material_api.CreateDynamicFrictionAttr(assumptions.get("friction", 0.6))
    material_api.CreateRestitutionAttr(assumptions.get("restitution", 0.0))
    if parameters["physx"]["friction_combine_mode"] is not None:
        material.GetPrim().AddAppliedSchema("PhysxMaterialAPI")
        material.GetPrim().CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set(
            parameters["physx"]["friction_combine_mode"])
    base = UsdGeom.Xform.Define(stage, "/Asset/Physics/BaseLink").GetPrim() if articulated else root
    low, high = manifest["base_bounds"]["min"], manifest["base_bounds"]["max"]
    base_volume = math.prod(max(0.005, b - a) for a, b in zip(low, high, strict=True))
    collider_count = 0
    if components:
        collider_count += add_component_colliders(stage, base, [c for c in components if c["owner_joint"] is None], material, not args.no_collision)
    else:
        add_box(stage, base.GetPath(), manifest["base_bounds"], collision=not args.no_collision)
    add_body(base, base_volume, mass_kg=assumptions.get("mass_kg"), density=assumptions.get("density_kg_m3", 100.0),
             enable_ccd=assumptions["enable_ccd"])
    if "base" in mass_overrides:
        author_mass_properties(base, mass_overrides["base"])
    if articulated:
        UsdPhysics.ArticulationRootAPI.Apply(root)
        if manifest.get("fixed_base", True):
            fixed = UsdPhysics.FixedJoint.Define(stage, "/Asset/Physics/WorldFixed")
            fixed.CreateBody1Rel().SetTargets([base.GetPath()])

    joint_paths = {}
    for spec in manifest["joints"]:
        if spec["joint_type"] not in {"revolute", "continuous"}:
            raise ValueError(f"Joint type {spec['joint_type']!r} needs a matching USD adapter; refusing to silently convert it to revolute")
        prim = objects.get(spec["name"])
        if prim is None:
            raise RuntimeError(f"Blender joint missing from USD: {spec['name']}")
        joint_paths[spec["name"]] = prim.GetPath()
        if spec["bounds"] is None:
            raise RuntimeError(f"joint has no moving geometry: {spec['name']}")
        low, high = spec["bounds"]["min"], spec["bounds"]["max"]
        volume = math.prod(max(0.005, b - a) for a, b in zip(low, high, strict=True))
        if components:
            collider_count += add_component_colliders(stage, prim, [c for c in components if c["owner_joint"] == spec["name"]], material, not args.no_collision)
        else:
            add_box(stage, prim.GetPath(), spec["bounds"], collision=not args.no_collision)
        add_body(prim, volume, mass_kg=spec.get("mass_kg"), density=assumptions.get("density_kg_m3", 100.0),
                 enable_ccd=assumptions["enable_ccd"])
        if "joint:" + spec["name"] in mass_overrides:
            author_mass_properties(prim, mass_overrides["joint:" + spec["name"]])

    # Legacy coarse proxies can overlap near hinges. Enable only explicitly
    # selected internal contacts, after reviewing the corresponding colliders.
    bodies = {"base": base.GetPath(), **{"joint:" + name: path for name, path in joint_paths.items()}}
    allowed_pairs = author_collision_pairs(stage, root, bodies, collision_pairs)

    from real2sim_harness.usd_visual_ownership import bind_base_visuals
    base_visual_bindings = bind_base_visuals(stage, manifest, base) if articulated else []

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for spec in manifest["joints"]:
        body1_path = joint_paths[spec["name"]]
        body0_path = joint_paths.get(spec["parent_joint"], base.GetPath())
        body0 = stage.GetPrimAtPath(body0_path)
        body1 = stage.GetPrimAtPath(body1_path)
        world0 = cache.GetLocalToWorldTransform(body0)
        world1 = cache.GetLocalToWorldTransform(body1)
        local_frame0 = world1 * world0.GetInverse()
        position0 = local_frame0.ExtractTranslation()
        # Joint rotations are rigid frames, even when the source hierarchy has
        # a uniform object scale. Extracting a quaternion from a scaled matrix
        # directly produces a non-unit quaternion and changes the hinge axis.
        rigid_frame = np.eye(4)
        rigid_frame[:3, :3] = rotation_only(local_frame0)
        rotation0 = Gf.Matrix4d(rigid_frame.tolist()).ExtractRotationQuat().GetNormalized()
        quat0 = Gf.Quatf(float(rotation0.GetReal()), Gf.Vec3f(*rotation0.GetImaginary()))
        joint_path = f"/Asset/Physics/{safe_name(spec['name'])}"
        joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
        joint.CreateBody0Rel().SetTargets([body0_path])
        joint.CreateBody1Rel().SetTargets([body1_path])
        joint.CreateCollisionEnabledAttr(frozenset((body0_path, body1_path)) in allowed_pairs)
        joint.CreateAxisAttr(spec["axis"])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*position0))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        joint.CreateLocalRot0Attr(quat0)
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
        # USD joint coordinates start from the current authored pose.
        joint.CreateLowerLimitAttr(spec["lower_degrees"] - spec["current_degrees"])
        joint.CreateUpperLimitAttr(spec["upper_degrees"] - spec["current_degrees"])
        # Export a passive object for manipulation. Validation can enable motors
        # through Isaac Lab without giving the delivered object a hidden servo.
        author_angular_drive(joint, parameters["joint_drive"])
        joint.GetPrim().CreateAttribute("remo:blenderJointName", Sdf.ValueTypeNames.String).Set(spec["name"])
    stage.GetRootLayer().Save()
    (folder / "physics_manifest.json").write_text(json.dumps({
        "backend": "physx", "asset_kind": manifest["asset_kind"],
        "collider_source": "per_component_mesh" if components else "legacy_link_box",
        "collider_count": collider_count,
        "passive_joint_drives": parameters["joint_drive"]["stiffness"] == 0.0,
        "fixed_base": manifest.get("fixed_base", articulated),
        "physics_assumptions": assumptions, "robot_interaction_tested": False,
        "explicit_mass_properties": mass_overrides,
        "friction_combine_mode": parameters["physx"]["friction_combine_mode"],
        "enabled_collision_pairs": collision_pairs,
        "base_visual_bindings": base_visual_bindings,
    }, indent=2))
    print(f"PHYSX_ASSET_AUTHORED links={len(joint_paths) + 1} joints={len(joint_paths)} path={output}")


if __name__ == "__main__":
    main()
