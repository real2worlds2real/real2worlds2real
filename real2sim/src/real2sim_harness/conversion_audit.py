"""Check Blender metadata against authored USD before running any dynamics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from .artifacts import sha256_file
from .rigid_frames import rotation_only


def named_objects(stage):
    result = {}
    for prim in stage.Traverse():
        attr = prim.GetAttribute("userProperties:blender:object_name")
        if attr and attr.Get():
            result[str(attr.Get())] = prim
    return result


def world_points(prim, points, cache):
    transform = np.asarray(cache.GetLocalToWorldTransform(prim))
    return np.asarray(points) @ transform[:3, :3] + transform[3, :3]


def audit_conversion(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    failures, notes = [], []
    visual = Usd.Stage.Open(str(folder / "visual.usdc"))
    kind = manifest["asset_kind"]
    physics = Usd.Stage.Open(str(folder / ("newton_surface.usda" if kind == "deformable" else "physx.usda")))
    if visual is None or physics is None:
        raise ValueError("Exported visual and authored physics stages must both open")
    for label, stage in (("visual", visual), ("physics", physics)):
        if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
            failures.append(f"{label}:up_axis_changed")
        if not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0, atol=1e-9):
            failures.append(f"{label}:meters_per_unit_changed")
    objects, cache = named_objects(visual), UsdGeom.XformCache()
    material_check = None
    if manifest.get("material_export"):
        from .blender_glass import audit_native_glass
        receipt = (folder / manifest["material_export"]).resolve()
        if not receipt.is_relative_to(folder.resolve()):
            raise ValueError("Material export receipt must remain inside the bundle")
        material_check = audit_native_glass(folder / "visual.usdc", receipt)
        failures.extend(material_check["failures"])
    normal_check = None
    if manifest.get("normal_transfer"):
        from .blender_normal_transfer import audit_normal_transfer
        receipt = (folder / manifest["normal_transfer"]).resolve()
        if not receipt.is_relative_to(folder.resolve()):
            raise ValueError("Normal transfer receipt must remain inside the bundle")
        normal_check = audit_normal_transfer(folder / "visual.usdc", receipt)
        failures.extend(normal_check["failures"])
    expected_visuals = manifest.get("visual_objects", [])
    for item in [*expected_visuals, *manifest.get("joints", [])]:
        name = item["name"]
        if name not in objects:
            failures.append(f"missing_visual_object:{name}")
        elif not np.allclose(np.asarray(cache.GetLocalToWorldTransform(objects[name])).T,
                             item["matrix_world"], atol=2e-5, rtol=1e-5):
            failures.append(f"visual_transform_changed:{name}")
    if not expected_visuals:
        notes.append("Legacy export has no per-visual source transforms; re-export for full visual correspondence checks")

    collider_count = joint_count = control_nodes = 0
    visual_ownership = None
    physics_cache = UsdGeom.XformCache()
    if kind in {"rigid", "articulated"}:
        if expected_visuals:
            from .usd_visual_ownership import audit_visual_ownership
            visual_ownership = audit_visual_ownership(visual, physics, manifest)
            failures.extend(visual_ownership["failures"])
        components = json.loads((folder / "collision_meshes.json").read_text())
        physics_objects = named_objects(physics)
        base_path = "/Asset/Physics/BaseLink" if kind == "articulated" else "/Asset"
        colliders = {}
        for prim in physics.Traverse():
            attr = prim.GetAttribute("remo:sourceObject")
            if attr and attr.Get() and prim.IsA(UsdGeom.Mesh):
                colliders.setdefault(str(attr.Get()), []).append(prim)
        owner_frames = {joint["name"]: np.asarray(joint["matrix_world"]) for joint in manifest["joints"]}
        for component in components:
            name = component["object_name"]
            candidates = colliders.get(name, [])
            if len(candidates) != 1:
                failures.append(f"missing_or_duplicate_collider:{name}")
                continue
            prim = candidates[0]
            mesh = UsdGeom.Mesh(prim)
            source = np.asarray(component["vertices"])
            actual = world_points(prim, mesh.GetPointsAttr().Get(), physics_cache)
            transform = owner_frames.get(component["owner_joint"], np.eye(4))
            expected = source @ transform[:3, :3].T + transform[:3, 3]
            if actual.shape != expected.shape or not np.allclose(actual, expected, atol=2e-5, rtol=1e-5):
                failures.append(f"collider_geometry_or_transform_changed:{name}")
            indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
            faces = np.asarray(component["faces"])
            if not np.array_equal(indices, faces.reshape(-1)) or not np.all(np.asarray(mesh.GetFaceVertexCountsAttr().Get()) == 3):
                failures.append(f"collider_topology_changed:{name}")
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                failures.append(f"collider_not_enabled:{name}")
            owner = prim.GetParent()
            while owner and not owner.IsPseudoRoot() and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
                owner = owner.GetParent()
            source_owner = component["owner_joint"]
            expected_owner = str(physics_objects[source_owner].GetPath()) if source_owner in physics_objects else base_path
            if not owner or str(owner.GetPath()) != expected_owner:
                failures.append(f"collider_rigid_owner_changed:{name}")
            collider_count += 1
        if set(colliders) - {item["object_name"] for item in components}:
            failures.append("unexpected_source_colliders")
        authored_joints = {}
        for prim in physics.Traverse():
            attr = prim.GetAttribute("remo:blenderJointName")
            if attr and attr.Get():
                authored_joints[str(attr.Get())] = prim
        for spec in manifest["joints"]:
            name = spec["name"]
            prim = authored_joints.get(name)
            if prim is None or not prim.IsA(UsdPhysics.RevoluteJoint):
                failures.append(f"missing_joint:{name}")
                continue
            joint = UsdPhysics.RevoluteJoint(prim)
            parent_name = spec["parent_joint"]
            expected_parent = physics_objects[parent_name].GetPath() if parent_name in physics_objects else base_path
            if [str(path) for path in joint.GetBody0Rel().GetTargets()] != [str(expected_parent)]:
                failures.append(f"joint_parent_body_changed:{name}")
            if name not in physics_objects or [str(path) for path in joint.GetBody1Rel().GetTargets()] != [str(physics_objects[name].GetPath())]:
                failures.append(f"joint_child_body_changed:{name}")
            if joint.GetAxisAttr().Get() != spec["axis"]:
                failures.append(f"joint_axis_changed:{name}")
            expected_limits = [spec["lower_degrees"] - spec["current_degrees"], spec["upper_degrees"] - spec["current_degrees"]]
            if not np.allclose([joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()], expected_limits, atol=1e-4):
                failures.append(f"joint_limits_changed:{name}")
            frames = []
            for relation, position, rotation in (
                (joint.GetBody0Rel(), joint.GetLocalPos0Attr(), joint.GetLocalRot0Attr()),
                (joint.GetBody1Rel(), joint.GetLocalPos1Attr(), joint.GetLocalRot1Attr()),
            ):
                targets = relation.GetTargets()
                if len(targets) != 1 or not physics.GetPrimAtPath(targets[0]):
                    failures.append(f"joint_body_missing:{name}")
                    break
                quaternion = rotation.Get()
                if not np.isclose(quaternion.GetLength(), 1.0, atol=1e-5):
                    failures.append(f"joint_rotation_not_normalized:{name}")
                local = Gf.Matrix4d().SetRotate(Gf.Quatd(quaternion.GetReal(), Gf.Vec3d(quaternion.GetImaginary())))
                local.SetTranslateOnly(Gf.Vec3d(position.Get()))
                frames.append(np.asarray(local * physics_cache.GetLocalToWorldTransform(physics.GetPrimAtPath(targets[0]))))
            if len(frames) == 2:
                if not np.allclose(frames[0][3, :3], frames[1][3, :3], atol=2e-5, rtol=1e-5):
                    failures.append(f"joint_rest_positions_disagree:{name}")
                try:
                    if not np.allclose(rotation_only(frames[0]), rotation_only(frames[1]), atol=2e-5, rtol=1e-5):
                        failures.append(f"joint_rest_rotations_disagree:{name}")
                except ValueError as exc:
                    failures.append(f"invalid_joint_frame:{name}:{exc}")
            if not prim.HasAPI(UsdPhysics.DriveAPI, "angular"):
                failures.append(f"joint_drive_schema_missing:{name}")
            joint_count += 1
    elif kind == "deformable":
        source = json.loads((folder / "sim_mesh.json").read_text())
        mesh = UsdGeom.Mesh.Get(physics, "/Asset/SimMesh")
        expected = np.asarray(source["vertices"])
        actual = world_points(mesh.GetPrim(), mesh.GetPointsAttr().Get(), physics_cache)
        if actual.shape != expected.shape or not np.allclose(actual, expected, atol=2e-5, rtol=1e-5):
            failures.append("cloth_control_geometry_changed")
        if not np.array_equal(np.asarray(mesh.GetFaceVertexIndicesAttr().Get()), np.asarray(source["faces"]).reshape(-1)):
            failures.append("cloth_control_topology_changed")
        if list(mesh.GetPrim().GetAttribute("remo:pinIndices").Get()) != source["pin_indices"]:
            failures.append("cloth_pin_indices_changed")
        binding = json.loads((folder / "surface_binding.json").read_text())
        if not binding.get("ok") or any(row["rest_error_max_m"] > 1e-5 for row in binding["meshes"]):
            failures.append("cloth_visual_rest_shape_changed")
        if {row["source_object"] for row in binding["meshes"]} != set(manifest["deformable_visual_objects"]):
            failures.append("cloth_visual_followers_changed")
        for field, filename in (("visual_usd_sha256", "visual.usdc"), ("sim_mesh_sha256", "sim_mesh.json"), ("binding_sha256", "surface_binding.npz")):
            if binding[field] != sha256_file(folder / filename):
                failures.append(f"stale_cloth_binding:{filename}")
        control_nodes = len(expected)
    else:
        failures.append(f"unsupported_asset_kind:{kind}")
    return {"ok": not failures, "failure_scope": "conversion_consistency" if failures else None,
            "failures": failures, "notes": notes, "asset_kind": kind,
            "native_material_check": material_check,
            "normal_transfer_check": normal_check,
            "visual_ownership_check": visual_ownership,
            "source_visual_transforms_recorded": bool(expected_visuals),
            "checked_visual_objects": len(expected_visuals), "checked_colliders": collider_count,
            "checked_joints": joint_count, "checked_control_nodes": control_nodes}
