"""Bind rendered geometry to its rigid body and audit the composed result."""
from __future__ import annotations

import numpy as np
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


def object_name(prim):
    while prim and not prim.IsPseudoRoot():
        name = prim.GetAttribute("userProperties:blender:object_name").Get()
        if name:
            return str(name)
        prim = prim.GetParent()
    return None


def visual_meshes(stage, manifest):
    objects = {row["name"]: row for row in manifest.get("visual_objects", [])}
    joints = {row["name"] for row in manifest["joints"]}
    rows = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        name = object_name(prim)
        if name not in objects:
            continue
        owner = objects[name].get("owner_joint")
        if "owner_joint" not in objects[name]:
            ancestor = prim
            while ancestor and not ancestor.IsPseudoRoot():
                source_name = ancestor.GetAttribute("userProperties:blender:object_name").Get()
                if source_name in joints:
                    owner = source_name
                    break
                ancestor = ancestor.GetParent()
        rows.append((prim, name, owner))
    return rows


def bind_base_visuals(stage, manifest, base):
    """Copy only base-owned mesh leaves, retaining joints and material paths.

    The composed source meshes stay in place but are hidden. Copying mesh specs
    rather than their object ancestors avoids moving a nested joint with them.
    """
    cache = UsdGeom.XformCache()
    body_inverse = cache.GetLocalToWorldTransform(base).GetInverse()
    pending = [(prim, name, cache.GetLocalToWorldTransform(prim) * body_inverse)
               for prim, name, owner in visual_meshes(stage, manifest) if owner is None]
    if not pending:
        return []
    flattened = stage.Flatten()
    parent = base.GetPath().AppendChild("Visuals")
    UsdGeom.Xform.Define(stage, parent)
    retained = []
    for index, (source, name, local) in enumerate(pending):
        destination = parent.AppendChild(f"mesh_{index:04d}")
        if not Sdf.CopySpec(flattened, source.GetPath(), stage.GetRootLayer(), destination):
            raise ValueError(f"Cannot bind base visual: {source.GetPath()}")
        prim = stage.GetPrimAtPath(destination)
        xform = UsdGeom.Xformable(prim)
        xform.MakeMatrixXform().Set(local)
        xform.SetResetXformStack(False)
        # Material bindings inherited above the source mesh are not part of a
        # mesh-only copy. Retain their resolved material identities explicitly.
        for old in Usd.PrimRange(source):
            new_path = old.GetPath().ReplacePrefix(source.GetPath(), destination)
            material, _ = UsdShade.MaterialBindingAPI(old).ComputeBoundMaterial()
            if material:
                UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(new_path)).Bind(material)
        prim.CreateAttribute("remo:sourceVisualMesh", Sdf.ValueTypeNames.String).Set(str(source.GetPath()))
        prim.CreateAttribute("remo:sourceVisualObject", Sdf.ValueTypeNames.String).Set(name)
        UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited)
        UsdGeom.Imageable(source).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        retained.append({"source_mesh": str(source.GetPath()), "mesh": str(destination), "body": str(base.GetPath())})
    return retained


def audit_visual_ownership(visual, physics, manifest):
    expected = {str(prim.GetPath()): (prim, name, owner)
                for prim, name, owner in visual_meshes(visual, manifest)}
    physics_objects = {}
    actual = {}
    for prim in physics.Traverse():
        name = prim.GetAttribute("userProperties:blender:object_name").Get()
        if name:
            physics_objects[str(name)] = prim
        if prim.IsA(UsdGeom.Mesh) and UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.invisible:
            key = prim.GetAttribute("remo:sourceVisualMesh").Get() or str(prim.GetPath())
            actual.setdefault(str(key), []).append(prim)
    failures = []
    source_cache, target_cache = UsdGeom.XformCache(), UsdGeom.XformCache()
    base_path = "/Asset/Physics/BaseLink" if manifest["asset_kind"] == "articulated" else "/Asset"
    for path, (source, name, owner) in expected.items():
        matches = actual.get(path, [])
        if len(matches) != 1:
            failures.append("missing_or_duplicate_physics_visual:" + name)
            continue
        target = matches[0]
        body = target
        while body and not body.IsPseudoRoot() and not body.HasAPI(UsdPhysics.RigidBodyAPI):
            body = body.GetParent()
        if owner is not None and owner not in physics_objects:
            failures.append("visual_joint_owner_missing:" + name)
        wanted = str(physics_objects[owner].GetPath()) if owner in physics_objects else base_path
        if not body or str(body.GetPath()) != wanted:
            failures.append("visual_rigid_owner_changed:" + name)
        # A reset stack below the body would look attached in the namespace
        # while preventing it from following that body at runtime.
        ancestor = target
        while ancestor and ancestor != body and not ancestor.IsPseudoRoot():
            if UsdGeom.Xformable(ancestor).GetResetXformStack():
                failures.append("visual_resets_rigid_transform:" + name)
            ancestor = ancestor.GetParent()
        a, b = UsdGeom.Mesh(source), UsdGeom.Mesh(target)
        points_a, points_b = np.asarray(a.GetPointsAttr().Get()), np.asarray(b.GetPointsAttr().Get())
        ma = np.asarray(source_cache.GetLocalToWorldTransform(source))
        mb = np.asarray(target_cache.GetLocalToWorldTransform(target))
        wa, wb = points_a @ ma[:3, :3] + ma[3, :3], points_b @ mb[:3, :3] + mb[3, :3]
        if wa.shape != wb.shape or not np.allclose(wa, wb, atol=2e-5, rtol=1e-5):
            failures.append("physics_visual_rest_geometry_changed:" + name)
        if (list(a.GetFaceVertexCountsAttr().Get()) != list(b.GetFaceVertexCountsAttr().Get())
                or list(a.GetFaceVertexIndicesAttr().Get()) != list(b.GetFaceVertexIndicesAttr().Get())):
            failures.append("physics_visual_topology_changed:" + name)
        old_material, _ = UsdShade.MaterialBindingAPI(source).ComputeBoundMaterial()
        new_material, _ = UsdShade.MaterialBindingAPI(target).ComputeBoundMaterial()
        if bool(old_material) != bool(new_material) or (old_material and old_material.GetPath() != new_material.GetPath()):
            failures.append("physics_visual_material_changed:" + name)
        for old_subset in UsdGeom.Subset.GetGeomSubsets(a):
            old = old_subset.GetPrim()
            new = physics.GetPrimAtPath(old.GetPath().ReplacePrefix(source.GetPath(), target.GetPath()))
            if (not new or list(old_subset.GetIndicesAttr().Get()) !=
                    list(UsdGeom.Subset(new).GetIndicesAttr().Get())):
                failures.append("physics_visual_subset_changed:" + name)
                continue
            before, _ = UsdShade.MaterialBindingAPI(old).ComputeBoundMaterial()
            after, _ = UsdShade.MaterialBindingAPI(new).ComputeBoundMaterial()
            if bool(before) != bool(after) or (before and before.GetPath() != after.GetPath()):
                failures.append("physics_visual_subset_material_changed:" + name)
    return {"ok": not failures, "failures": sorted(set(failures)), "checked_meshes": len(expected)}
