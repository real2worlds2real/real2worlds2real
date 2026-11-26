"""Export an accepted Blender asset and its simulator metadata to USD."""

import hashlib
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


if not __package__:  # Blender also executes this file directly.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from real2sim_harness.asset_metadata import asset_key, asset_value, is_harness_datablock


def matrix_rows(matrix):
    return [[float(value) for value in row] for row in matrix]


def source_hash():
    with open(bpy.data.filepath, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def owner_joint(obj):
    parent = obj.parent
    while parent is not None:
        if parent.get("joint_type"):
            return parent
        parent = parent.parent
    return None


def add_bounds(bounds, point):
    if bounds is None:
        return {"min": list(point), "max": list(point)}
    for axis in range(3):
        bounds["min"][axis] = min(bounds["min"][axis], point[axis])
        bounds["max"][axis] = max(bounds["max"][axis], point[axis])
    return bounds


def mesh_bundle(obj):
    vertices = [list(obj.matrix_world @ vertex.co) for vertex in obj.data.vertices]
    obj.data.calc_loop_triangles()
    faces = [list(triangle.vertices) for triangle in obj.data.loop_triangles]
    pin_indices = []
    source_physics = []
    for modifier in obj.modifiers:
        if modifier.type != "CLOTH":
            continue
        source_physics.append({
            "backend": "blender_cloth",
            "mass_parameter": modifier.settings.mass,
            "tension_stiffness": modifier.settings.tension_stiffness,
            "compression_stiffness": modifier.settings.compression_stiffness,
            "shear_stiffness": modifier.settings.shear_stiffness,
            "bending_stiffness": modifier.settings.bending_stiffness,
            "air_damping": modifier.settings.air_damping,
            "gravity_scale": modifier.settings.effector_weights.gravity,
            "self_collision": modifier.collision_settings.use_self_collision,
            "self_friction": modifier.collision_settings.self_friction,
            "collision_distance_m": modifier.collision_settings.distance_min,
            "conversion": "recorded source values; Newton parameters require a separate material model",
        })
        group = obj.vertex_groups.get(modifier.settings.vertex_group_mass)
        if group is not None:
            for vertex in obj.data.vertices:
                if any(entry.group == group.index and entry.weight > 0.5 for entry in vertex.groups):
                    pin_indices.append(vertex.index)
    return {"object_name": obj.name, "vertices": vertices, "faces": faces,
            "pin_indices": pin_indices, "source_physics": source_physics}


def exportable(obj):
    return (
        obj.type in {"MESH", "EMPTY"}
        and not obj.hide_render
        and not asset_value(obj, "real2sim_sim_proxy")
        and not asset_value(obj, "real2sim_collision_proxy")
        and not is_harness_datablock(obj.name)
        and not asset_value(obj, "real2sim_ground")
        and not asset_value(obj, "real2sim_fixture")
    )


def collision_bundle(obj, owner):
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        transform = (owner.matrix_world.inverted() if owner else Matrix.Identity(4)) @ evaluated.matrix_world
        result = {
            "object_name": obj.name,
            "owner_joint": owner.name if owner else None,
            "vertices": [list(transform @ vertex.co) for vertex in mesh.vertices],
            "faces": [list(face.vertices) for face in mesh.loop_triangles],
            "approximation": str(asset_value(obj, "real2sim_collision_approximation", "convexDecomposition")),
        }
        if asset_key(obj, "real2sim_collision_sdf_resolution") in obj:
            resolution = obj[asset_key(obj, "real2sim_collision_sdf_resolution")]
            if type(resolution) is not int or not 16 <= resolution <= 1024:
                raise ValueError("Collision SDF resolution must be an integer within 16..1024")
            result["sdf_resolution"] = resolution
        for name in ("contact_offset_m", "rest_offset_m"):
            if asset_key(obj, "real2sim_collision_" + name) in obj:
                value = obj[asset_key(obj, "real2sim_collision_" + name)]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"Collision {name} must be finite")
                result[name] = float(value)
        if result.get("contact_offset_m", 0.001) <= max(0.0, result.get("rest_offset_m", 0.0)):
            raise ValueError("Collision contact offset must exceed zero and the rest offset")
        return result
    finally:
        evaluated.to_mesh_clear()


def rigid_collision_proxy(obj, kind):
    return (
        kind in {"rigid", "articulated", "hybrid"}
        and obj.type == "MESH"
        and bool(asset_value(obj, "real2sim_collision_proxy") or (kind != "hybrid" and asset_value(obj, "real2sim_sim_proxy")))
        and not is_harness_datablock(obj.name)
        and not asset_value(obj, "real2sim_ground")
        and not asset_value(obj, "real2sim_fixture")
    )


def hybrid_configuration(scene, control, objects, joints, mesh_data):
    attachments = json.loads(asset_value(control, "real2sim_attachments", "[]"))
    if not attachments:
        for modifier in control.modifiers:
            if modifier.type == "HOOK" and modifier.object and modifier.vertex_group:
                target = modifier.object
                owner = target if target.get("joint_type") else owner_joint(target)
                attachments.append({"group": modifier.vertex_group, "body": owner.name if owner else "__base__"})
    records = []
    for spec in attachments:
        group = control.vertex_groups.get(spec["group"])
        if group is None:
            raise ValueError(f"Hybrid attachment group missing from control mesh: {spec['group']}")
        indices = [vertex.index for vertex in control.data.vertices
                   if any(entry.group == group.index and entry.weight > 0.5 for entry in vertex.groups)]
        records.append({"body": spec["body"], "particle_indices": indices,
                        "stiffness": float(spec.get("stiffness", 1e4)), "damping": float(spec.get("damping", 1.0))})
    configuration = {
        "schema_version": 1, "attachments": records,
        "motion": json.loads(asset_value(scene, "real2sim_hybrid_motion", "[]")),
        "loop_constraints": json.loads(asset_value(scene, "real2sim_loop_constraints", "[]")),
        "solver": "Newton 1.7 unified VBD with native two-way particle-body attachments",
        "control_object": control.name,
    }
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from real2sim_harness.hybrid_contract import validate_hybrid_contract
    configuration["contract"] = validate_hybrid_contract(configuration, joints, mesh_data)
    return configuration


def main():
    args = sys.argv[sys.argv.index("--") + 1 :]
    output = Path(args[0]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.frame_set(1)
    objects = list(bpy.context.scene.objects)
    unsupported = [obj.name for obj in objects
                   if obj.type in {"CURVE", "FONT", "SURFACE", "META", "VOLUME", "POINTCLOUD", "GREASEPENCIL"}
                   and not obj.hide_render and not asset_value(obj, "real2sim_fixture")
                   and not asset_value(obj, "real2sim_sim_proxy") and not is_harness_datablock(obj.name)]
    if unsupported:
        raise ValueError(f"Visible non-mesh objects require mesh conversion before USD integration; refusing to drop them: {unsupported}")
    original_unit_scale = float(bpy.context.scene.unit_settings.scale_length)
    if original_unit_scale <= 0:
        raise ValueError("Blender scene unit scale must be positive")
    if abs(original_unit_scale - 1.0) > 1e-9:
        # Normalize the in-memory export scene so visual and collision coordinates
        # share meters. The saved Blender source remains untouched.
        for obj in objects:
            if obj.parent is None:
                obj.matrix_world = Matrix.Scale(original_unit_scale, 4) @ obj.matrix_world
        bpy.context.scene.unit_settings.scale_length = 1.0
        bpy.context.view_layer.update()
    joints = [obj for obj in objects if obj.get("joint_type")]
    kind = asset_value(bpy.context.scene, "real2sim_asset_kind")
    explicit_kind = kind is not None
    has_deformation = any(mod.type in {"CLOTH", "SOFT_BODY"} for obj in objects for mod in obj.modifiers)
    if kind is None:
        kind = "hybrid" if joints and has_deformation else "articulated" if joints else "deformable" if has_deformation else "rigid"
    if kind in {"rigid", "articulated"} and has_deformation:
        raise ValueError("The declared rigid/articulated kind conflicts with native deformable physics")
    if kind == "rigid" and joints:
        raise ValueError("A rigid declaration would discard the source joints; declare articulated instead")
    if kind == "articulated" and not joints:
        raise ValueError("No joint metadata found for the declared articulated asset")
    if kind == "deformable" and joints:
        raise ValueError("Joint and cloth geometry requires a hybrid adapter")
    if kind == "hybrid" and not joints:
        raise ValueError("A hybrid asset needs articulated rigid links as well as cloth")
    # Rigid collision proxies can use the same visibility marker; they are not
    # cloth controls and may legitimately be separate meshes for each body.
    deformable = [] if kind in {"rigid", "articulated"} else [obj for obj in objects if obj.type == "MESH" and (asset_value(obj, "real2sim_sim_proxy") or any(m.type == "CLOTH" for m in obj.modifiers))]
    if len(deformable) > 1:
        raise ValueError("Multiple cloth control meshes require separate deformable entries; refusing to silently export only one")
    driven, visual_followers = set(), []
    if deformable:
        deformable.sort(key=lambda obj: (not bool(asset_value(obj, "real2sim_sim_proxy")), -len(obj.data.vertices)))
        driven = {deformable[0].name}
        while True:
            previous = set(driven)
            for obj in objects:
                for modifier in obj.modifiers:
                    target = getattr(modifier, "target", None) if modifier.type == "SURFACE_DEFORM" else getattr(modifier, "object", None) if modifier.type == "MESH_DEFORM" else None
                    if target is not None and target.name in driven:
                        driven.add(obj.name)
            if driven == previous:
                break
        visual_followers = sorted(obj.name for obj in objects if obj.name in driven and exportable(obj))
    if kind == "hybrid" and not deformable:
        raise ValueError("A hybrid asset needs one cloth control mesh")
    collision_meshes = []
    mesh_groups = {obj.name: {"objects": [], "bounds": None} for obj in joints}
    mesh_groups["__base__"] = {"objects": [], "bounds": None}
    for obj in objects:
        proxy = rigid_collision_proxy(obj, kind)
        if obj.type != "MESH" or not (exportable(obj) or proxy):
            continue
        if kind == "hybrid" and obj.name in driven:
            continue
        owner = owner_joint(obj)
        if kind in {"rigid", "articulated", "hybrid"} and asset_value(obj, "real2sim_collision", True):
            collision_meshes.append(collision_bundle(obj, owner))
        if proxy:
            continue
        name = owner.name if owner else "__base__"
        group = mesh_groups[name]
        group["objects"].append(obj.name)
        inverse = owner.matrix_world.inverted() if owner else Matrix.Identity(4)
        for corner in obj.bound_box:
            point = inverse @ obj.matrix_world @ Vector(corner)
            group["bounds"] = add_bounds(group["bounds"], point)

    if deformable:
        sim_mesh = mesh_bundle(deformable[0])
        (output / "sim_mesh.json").write_text(json.dumps(sim_mesh, separators=(",", ":")))

    joint_data = []
    for obj in joints:
        if str(obj.get("joint_axis", "Z")).strip().startswith("-"):
            raise ValueError("Current USD integration requires a positive joint axis; put the fixed axis orientation on a parent Empty")
        axis = str(obj.get("joint_axis", "Z")).upper().lstrip("+-")
        axis_index = "XYZ".index(axis)
        parent = owner_joint(obj)
        joint_data.append({
            "name": obj.name,
            "parent_joint": parent.name if parent else None,
            "joint_type": str(obj.get("joint_type")),
            "axis": axis,
            "lower_degrees": float(obj.get("lower_limit_degrees", 0.0)),
            "upper_degrees": float(obj.get("upper_limit_degrees", 0.0)),
            "current_degrees": math.degrees(obj.rotation_euler[axis_index]),
            "lower_m": float(obj.get("lower_limit_m", 0.0)),
            "upper_m": float(obj.get("upper_limit_m", 0.0)),
            "drive_stiffness": float(asset_value(obj, "real2sim_drive_stiffness", 1000.0)),
            "drive_damping": float(asset_value(obj, "real2sim_drive_damping", 20.0)),
            "matrix_world": matrix_rows(obj.matrix_world),
            "bounds": mesh_groups[obj.name]["bounds"],
            "mesh_count": len(mesh_groups[obj.name]["objects"]),
            "mass_kg": asset_value(obj, "real2sim_mass_kg"),
        })

    if kind == "hybrid":
        configuration = hybrid_configuration(bpy.context.scene, deformable[0], objects, joint_data, sim_mesh)
        (output / "hybrid_configuration.json").write_text(json.dumps(configuration, indent=2))

    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        if exportable(obj):
            obj.select_set(True)
    bpy.ops.wm.usd_export(
        filepath=str(output / "visual.usdc"),
        selected_objects_only=True,
        export_animation=False,
        export_materials=True,
        export_custom_properties=True,
        author_blender_name=True,
        export_lights=False,
        export_cameras=False,
        export_textures_mode="NEW",
        overwrite_textures=True,
        relative_paths=True,
        convert_scene_units="METERS",
        meters_per_unit=1.0,
        root_prim_path="/Asset",
    )
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from real2sim_harness.blender_glass import export_native_glass
    from real2sim_harness.blender_normal_transfer import export_normal_transfer

    export_native_glass(output / "visual.usdc", [o for o in objects if exportable(o)])
    export_normal_transfer(output / "visual.usdc", [o for o in objects if exportable(o)])
    manifest = {
        "schema_version": 2,
        "source_blend": bpy.data.filepath,
        "source_blend_sha256": source_hash(),
        "visual_usd": "visual.usdc",
        "material_export": "visual.materials.json",
        "normal_transfer": "visual.normals.json",
        "asset_kind": kind,
        "asset_kind_source": "declared" if explicit_kind else "inferred_from_native_physics_and_joint_metadata",
        "up_axis": "+Z",
        "units": "m",
        "original_blender_unit_scale": original_unit_scale,
        "visual_objects": [{"name": obj.name, "matrix_world": matrix_rows(obj.matrix_world),
                            "owner_joint": owner_joint(obj).name if owner_joint(obj) else None}
                           for obj in objects if obj.type == "MESH" and exportable(obj)],
        "joints": joint_data,
        "base_bounds": mesh_groups["__base__"]["bounds"],
        "base_mesh_count": len(mesh_groups["__base__"]["objects"]),
        "sim_mesh": "sim_mesh.json" if deformable else None,
        "deformable_visual_objects": visual_followers if deformable else [],
        "collision_meshes": "collision_meshes.json" if collision_meshes else None,
        "fixed_base": bool(asset_value(bpy.context.scene, "real2sim_fixed_base", kind in {"articulated", "hybrid"})),
        "physics_assumptions": {
            "gravity": [float(value) * (sim_mesh["source_physics"][0].get("gravity_scale", 1.0) if deformable and sim_mesh["source_physics"] else 1.0)
                        for value in (bpy.context.scene.gravity if bpy.context.scene.use_gravity else (0, 0, 0))],
            "mass_source": "real2sim_mass_kg when supplied; otherwise bounded density estimate",
            "density_kg_m3": float(asset_value(bpy.context.scene, "real2sim_density_kg_m3", 100.0)),
            "friction": float(asset_value(bpy.context.scene, "real2sim_friction", 0.6)),
            "restitution": float(asset_value(bpy.context.scene, "real2sim_restitution", 0.0)),
            "mass_kg": asset_value(bpy.context.scene, "real2sim_mass_kg"),
            "enable_ccd": bool(asset_value(bpy.context.scene, "real2sim_enable_ccd", False)),
        },
    }
    if collision_meshes:
        (output / "collision_meshes.json").write_text(json.dumps(collision_meshes, separators=(",", ":")))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"EXPORT_BLEND_FOR_ISAAC_OK kind={manifest['asset_kind']} joints={len(joints)} output={output}")


if __name__ == "__main__":
    main()
