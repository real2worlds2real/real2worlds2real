"""Executed inside Blender, not imported by the normal Python environment."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import struct
import sys
import traceback
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

if not __package__:  # Blender also executes this file directly.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real2sim_harness.asset_metadata import asset_value

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blender_structure import snapshot_structure  # noqa: E402
from blender_surface import is_asset_object, portable_image_uv_scales  # noqa: E402


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def load_builder(path: Path):
    spec = importlib.util.spec_from_file_location("real2sim_agent_scene", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load submitted script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    builder = getattr(module, "build_scene", None)
    if not callable(builder):
        raise RuntimeError("Submitted script does not define callable build_scene()")
    return builder


def mesh_bounds(objects):
    points = []
    for obj in objects:
        points.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
    lo = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    hi = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    return lo, hi


def look_at(camera, target):
    camera.rotation_euler = (Vector(target) - camera.location).to_track_quat("-Z", "Y").to_euler()


def neutral_world():
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.035, 0.035, 0.035, 1.0)
    background.inputs["Strength"].default_value = 0.35


def principled_material(name, color, roughness=0.8):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1.0)
    shader.inputs["Roughness"].default_value = roughness
    material.diffuse_color = (*color, 1.0)
    return material


def ensure_lights(center, radius):
    if any(obj.type == "LIGHT" for obj in bpy.context.scene.objects):
        return
    for name, offset, energy, size in (
        ("Key", (2.5, 3.0, 3.2), 650, 2.2),
        ("Fill", (-2.8, 1.2, 2.0), 350, 2.8),
        ("Rim", (0.5, -3.0, 3.8), 500, 2.0),
    ):
        data = bpy.data.lights.new(name, "AREA")
        data.energy = energy
        data.shape = "DISK"
        data.size = size * max(radius, 0.5)
        obj = bpy.data.objects.new(name, data)
        bpy.context.collection.objects.link(obj)
        obj.location = center + Vector(offset) * max(radius, 0.5)
        look_at(obj, center)


def ensure_ground(lo, hi):
    if any(asset_value(obj, "real2sim_ground") for obj in bpy.context.scene.objects):
        return
    size = max(hi.x - lo.x, hi.y - lo.y, hi.z - lo.z, 0.5) * 4
    bpy.ops.mesh.primitive_plane_add(
        size=size, location=((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z)
    )
    ground = bpy.context.object
    ground.name = "Real2Sim_Ground"
    ground["real2sim_ground"] = True
    material = principled_material("Real2Sim_GroundMaterial", (0.11, 0.11, 0.11), 0.82)
    ground.data.materials.append(material)


def axis_vector(value):
    return {
        "+X": Vector((1.0, 0.0, 0.0)),
        "-X": Vector((-1.0, 0.0, 0.0)),
        "+Y": Vector((0.0, 1.0, 0.0)),
        "-Y": Vector((0.0, -1.0, 0.0)),
    }[value]


def joint_axis(joint):
    value = str(joint.get("joint_axis", "Z")).strip().upper()
    sign = -1.0 if value.startswith("-") else 1.0
    key = value.lstrip("+-")
    if key not in {"X", "Y", "Z"}:
        raise ValueError(f"Unsupported joint axis {value!r} on {joint.name}")
    return {"X": 0, "Y": 1, "Z": 2}[key], sign, key


def joint_limits(joint):
    kind = str(joint.get("joint_type", "")).lower()
    if kind == "revolute":
        return (
            float(joint.get("lower_limit_degrees", 0.0)),
            float(joint.get("upper_limit_degrees", 0.0)),
            "degrees",
        )
    if kind == "prismatic":
        return (
            float(joint.get("lower_limit_m", 0.0)),
            float(joint.get("upper_limit_m", 0.0)),
            "m",
        )
    return 0.0, 0.0, "unsupported"


def joint_schema_errors(joint):
    kind = str(joint.get("joint_type", "")).lower()
    fields = {
        "revolute": ("lower_limit_degrees", "upper_limit_degrees"),
        "prismatic": ("lower_limit_m", "upper_limit_m"),
    }.get(kind)
    if fields is None:
        return ["joint_type must be revolute or prismatic"]
    errors = [f"Missing custom property {field}" for field in fields if field not in joint]
    lower, upper, _ = joint_limits(joint)
    if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
        errors.append("Joint limits must be finite and upper must exceed lower")
    if joint.rotation_mode in {"QUATERNION", "AXIS_ANGLE"}:
        errors.append(
            "The current pose sampler expects an Euler joint. Put fixed mounting orientation on a parent Empty, and use Euler rotation for the variable joint."
        )
    return errors


def set_joint_value(joint, value, original_rotation, original_location):
    axis_index, sign, _ = joint_axis(joint)
    joint.rotation_euler = original_rotation.copy()
    joint.location = original_location.copy()
    kind = str(joint.get("joint_type", "")).lower()
    if kind == "revolute":
        joint.rotation_euler[axis_index] = sign * math.radians(value)
    elif kind == "prismatic":
        joint.location[axis_index] += sign * value
    bpy.context.view_layer.update()


def mask_material():
    material = bpy.data.materials.new("Real2Sim_MaskMaterial")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    emission.inputs["Strength"].default_value = 1.0
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def fit_camera(scene, camera, center, direction, radius, points):
    distance = max(radius * 2.5, 0.15)
    for _ in range(6):
        camera.location = center + direction.normalized() * distance
        look_at(camera, center)
        bpy.context.view_layer.update()
        projected = [world_to_camera_view(scene, camera, point) for point in points]
        if any(point.z <= 0 for point in projected):
            distance *= 1.5
            continue
        width = max(point.x for point in projected) - min(point.x for point in projected)
        height = max(point.y for point in projected) - min(point.y for point in projected)
        scale = max(width, height) / 0.80
        if 0.98 <= scale <= 1.02:
            break
        distance = max(distance * min(max(scale, 0.55), 1.8), radius * 1.1)


def render_views(out: Path, objects, center, radius, front_axis):
    scene = bpy.context.scene
    camera_data = bpy.data.cameras.new("Real2Sim_Camera")
    camera = bpy.data.objects.new("Real2Sim_Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera
    camera.data.lens = 52
    # Solved reference cameras of small objects can sit closer than Blender's 0.1 m default near plane.
    camera.data.clip_start = 0.001
    front = axis_vector(front_axis)
    right = Vector((front.y, -front.x, 0.0))
    up = Vector((0.0, 0.0, 1.0))
    views = {
        "front": front + up * 0.28,
        "front_left": front + right * 0.72 + up * 0.38,
        "front_right": front - right * 0.72 + up * 0.38,
        "back": -front + right * 0.55 + up * 0.42,
    }
    lo, hi = mesh_bounds(objects)
    horizontal = (hi.x - lo.x, hi.y - lo.y)
    if max(horizontal) / max(min(horizontal), 1e-6) > 2.0:
        views.update(left=right + up * 0.18, right=-right + up * 0.18)
    if hi.z - lo.z < min(horizontal) * 0.4:
        views["top"] = up + front * 0.15
    points = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    rendered = []
    mask = mask_material()
    view_layer = bpy.context.view_layer
    ground = next((obj for obj in scene.objects if asset_value(obj, "real2sim_ground")), None)
    for name, direction in views.items():
        fit_camera(scene, camera, center, direction, radius, points)
        scene.render.filepath = str(out / f"preview_{name}.png")
        bpy.ops.render.render(write_still=True)
        rendered.append(f"preview_{name}.png")

        previous_override = view_layer.material_override
        previous_transparent = scene.render.film_transparent
        if ground is not None:
            ground.hide_render = True
        view_layer.material_override = mask
        scene.render.film_transparent = True
        scene.render.filepath = str(out / f"mask_{name}.png")
        bpy.ops.render.render(write_still=True)
        view_layer.material_override = previous_override
        scene.render.film_transparent = previous_transparent
        if ground is not None:
            ground.hide_render = False
        previous_format = scene.render.image_settings.file_format
        previous_depth = scene.render.image_settings.color_depth
        if ground is not None:
            ground.hide_render = True
        scene.render.film_transparent = True
        scene.render.image_settings.file_format = "OPEN_EXR_MULTILAYER"
        scene.render.image_settings.color_depth = "32"
        scene.render.filepath = str(out / f"passes_{name}.exr")
        bpy.ops.render.render(write_still=True)
        scene.render.image_settings.file_format = previous_format
        scene.render.image_settings.color_depth = previous_depth
        scene.render.film_transparent = previous_transparent
        if ground is not None:
            ground.hide_render = False
    return rendered


def render_articulation_previews(out, objects, front_axis):
    camera = asset_value(bpy.data.objects, "Real2Sim_Camera")
    if camera is None:
        return []
    front = axis_vector(front_axis)
    right = Vector((front.y, -front.x, 0.0))
    direction = front + right * 0.55 + Vector((0.0, 0.0, 0.38))
    rendered = []
    joints = [obj for obj in bpy.data.objects if obj.get("joint_type")]
    for index, joint in enumerate(joints):
        original_rotation = joint.rotation_euler.copy()
        original_location = joint.location.copy()
        lower, upper, _ = joint_limits(joint)
        for label, value in (
            ("lower", lower),
            ("middle", (lower + upper) / 2),
            ("upper", upper),
        ):
            set_joint_value(joint, value, original_rotation, original_location)
            lo, hi = mesh_bounds(objects)
            center = (lo + hi) / 2
            radius = max((hi - lo).length / 2, 0.1)
            points = [
                obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box
            ]
            fit_camera(bpy.context.scene, camera, center, direction, radius, points)
            name = f"joint_{index:02d}_{label}.png"
            bpy.context.scene.render.filepath = str(out / name)
            bpy.ops.render.render(write_still=True)
            rendered.append(name)
        joint.rotation_euler = original_rotation
        joint.location = original_location
        bpy.context.view_layer.update()
    return rendered


def render_reference_cameras(out, run_path):
    if run_path is None:
        return []
    solution_dir = run_path.parent / "camera_solutions"
    if not solution_dir.is_dir():
        return []
    scene = bpy.context.scene
    camera = asset_value(bpy.data.objects, "Real2Sim_Camera")
    if camera is None:
        return []
    view_layer = bpy.context.view_layer
    ground = next((obj for obj in scene.objects if asset_value(obj, "real2sim_ground")), None)
    mask = asset_value(bpy.data.materials, "Real2Sim_MaskMaterial") or mask_material()
    original_resolution = (scene.render.resolution_x, scene.render.resolution_y)
    original_aspect = (scene.render.pixel_aspect_x, scene.render.pixel_aspect_y)
    rendered = []
    provenance = {}
    for solution_path in sorted(solution_dir.glob("reference_*.json")):
        solution = json.loads(solution_path.read_text())
        index = int(solution["reference_index"])
        width, height = solution["image_size"]
        scale = min(1.0, 1024.0 / max(width, height))
        scene.render.resolution_x = max(1, round(width * scale))
        scene.render.resolution_y = max(1, round(height * scale))
        camera.matrix_world = Matrix(solution["blender"]["matrix_world"])
        camera.data.sensor_width = float(solution["blender"]["sensor_width_mm"])
        camera.data.sensor_fit = solution['blender'].get('sensor_fit', 'HORIZONTAL')
        scene.render.pixel_aspect_x = solution['blender'].get('pixel_aspect_x', 1.0)
        scene.render.pixel_aspect_y = solution['blender'].get('pixel_aspect_y', 1.0)
        camera.data.lens = float(solution["blender"]["lens_mm"])
        camera.data.shift_x = float(solution["blender"]["shift_x"])
        camera.data.shift_y = float(solution["blender"]["shift_y"])
        bpy.context.view_layer.update()

        preview_name = f"aligned_reference_{index:03d}.png"
        scene.render.filepath = str(out / preview_name)
        bpy.ops.render.render(write_still=True)
        rendered.append(preview_name)

        previous_override = view_layer.material_override
        previous_transparent = scene.render.film_transparent
        if ground is not None:
            ground.hide_render = True
        view_layer.material_override = mask
        scene.render.film_transparent = True
        mask_name = f"aligned_mask_reference_{index:03d}.png"
        scene.render.filepath = str(out / mask_name)
        bpy.ops.render.render(write_still=True)
        view_layer.material_override = previous_override
        scene.render.film_transparent = previous_transparent
        if ground is not None:
            ground.hide_render = False
        rendered.append(mask_name)
        provenance[str(index)] = {
            'camera_sha256': hashlib.sha256(solution_path.read_bytes()).hexdigest(),
            'images': {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                       for name in (preview_name, mask_name)},
        }
    scene.render.resolution_x, scene.render.resolution_y = original_resolution
    scene.render.pixel_aspect_x, scene.render.pixel_aspect_y = original_aspect
    (out / 'aligned_camera_provenance.json').write_text(json.dumps(provenance, indent=2))
    return rendered


def configure_render():
    scene = bpy.context.scene
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = -0.55
    scene.eevee.taa_render_samples = 24
    view_layer = bpy.context.view_layer
    view_layer.use_pass_z = True
    view_layer.use_pass_normal = True
    view_layer.use_pass_object_index = True


def exportable_curve_meshes():
    converted = []
    sources = [
        obj
        for obj in bpy.context.scene.objects
        if obj.type in {"CURVE", "FONT"} and is_asset_object(obj)
    ]
    for source in sources:
        duplicate = source.copy()
        duplicate.data = source.data.copy()
        source.users_collection[0].objects.link(duplicate)
        duplicate.name = f"{source.name} export mesh"
        duplicate["real2sim_export_copy"] = True
        bpy.ops.object.select_all(action="DESELECT")
        duplicate.select_set(True)
        bpy.context.view_layer.objects.active = duplicate
        bpy.ops.object.convert(target="MESH")
        converted.append(bpy.context.object)
    return converted


def export_asset_glb(path, meshes):
    export_objects = set([*meshes, *exportable_curve_meshes()])
    for obj in list(export_objects):
        parent = obj.parent
        while parent is not None:
            if asset_value(parent, "real2sim_sim_proxy") or asset_value(parent, "real2sim_collision_proxy"):
                raise RuntimeError("Simulation/collision proxies cannot be visual transform parents; keep visual hierarchy separate")
            export_objects.add(parent)
            parent = parent.parent
    bpy.ops.object.select_all(action="DESELECT")
    for obj in export_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    with portable_image_uv_scales(export_objects) as uv_conversions:
        bpy.ops.export_scene.gltf(
            filepath=str(path), export_format="GLB", use_selection=True, export_extras=True,
            export_apply=True,
        )
    with path.open("rb") as stream:
        stream.seek(12)
        length, _ = struct.unpack("<II", stream.read(8))
        document = json.loads(stream.read(length))
    nodes = document.get("nodes", [])
    expected = {obj.name for obj in export_objects if obj.get("joint_type")}
    exported = {
        node.get("name") for node in nodes if node.get("extras", {}).get("joint_type")
    }
    missing = sorted(expected - exported)
    if missing:
        raise RuntimeError(f"GLB export lost joint nodes or metadata: {missing}")
    return {
        "ok": True,
        "nodes": len(nodes),
        "nodes_with_children": sum(bool(node.get("children")) for node in nodes),
        "joint_nodes": sorted(exported),
        "joint_metadata_preserved": True,
        "embedded_images": sum("bufferView" in entry for entry in document.get("images", [])),
        "external_image_uris": [entry["uri"] for entry in document.get("images", []) if entry.get("uri") and not entry["uri"].startswith("data:")],
        "normal_mapped_materials": [entry.get("name", str(index)) for index, entry in enumerate(document.get("materials", [])) if "normalTexture" in entry],
        "uv_scale_normalizations": uv_conversions,
        "normal_texture_transforms": [
            {"material": entry.get("name"), **entry["normalTexture"]["extensions"]["KHR_texture_transform"]}
            for entry in document.get("materials", [])
            if "KHR_texture_transform" in entry.get("normalTexture", {}).get("extensions", {})
        ],
    }


def geometry_report(objects, lo, hi):
    return {
        "mesh_objects": len(objects),
        "vertices": sum(len(obj.data.vertices) for obj in objects),
        "polygons": sum(len(obj.data.polygons) for obj in objects),
        "materials": len(
            {slot.material.name for obj in objects for slot in obj.material_slots if slot.material}
        ),
        "names": [obj.name for obj in objects],
        "parents": {obj.name: obj.parent.name if obj.parent else None for obj in objects},
        "bounds_min": [round(v, 6) for v in lo],
        "bounds_max": [round(v, 6) for v in hi],
        "size": [round(hi[i] - lo[i], 6) for i in range(3)],
    }


def descendants(root):
    pending = list(root.children)
    found = []
    while pending:
        obj = pending.pop()
        found.append(obj)
        pending.extend(obj.children)
    return found


def articulation_report(required):
    joints = [obj for obj in bpy.data.objects if obj.get("joint_type")]
    rows = []
    for joint in joints:
        moving = [obj for obj in descendants(joint) if obj.type == "MESH"]
        lower, upper, units = joint_limits(joint)
        original_rotation = joint.rotation_euler.copy()
        original_location = joint.location.copy()

        def samples(
            value,
            joint=joint,
            moving=moving,
            original_rotation=original_rotation,
            original_location=original_location,
        ):
            set_joint_value(joint, value, original_rotation, original_location)
            return {
                obj.name: [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
                for obj in moving
            }

        configurations = [samples(value) for value in (lower, (lower + upper) / 2, upper)]
        joint.rotation_euler = original_rotation
        joint.location = original_location
        bpy.context.view_layer.update()
        motions = []
        for first_index in range(len(configurations)):
            for second_index in range(first_index + 1, len(configurations)):
                first = configurations[first_index]
                second = configurations[second_index]
                motions.extend(
                    (a - b).length
                    for name in first
                    for a, b in zip(first[name], second[name], strict=True)
                )
        max_motion = max(motions, default=0.0)
        kind = str(joint.get("joint_type", "")).lower()
        schema_errors = joint_schema_errors(joint)
        valid = (
            kind in {"revolute", "prismatic"}
            and bool(moving)
            and upper > lower
            and max_motion > 1e-4
            and not schema_errors
        )
        axis_index, axis_sign, axis_name = joint_axis(joint)
        rows.append(
            {
                "name": joint.name,
                "joint_type": kind,
                "schema_errors": schema_errors,
                "joint_axis": ("-" if axis_sign < 0 else "+") + axis_name,
                "limit_lower": lower,
                "limit_upper": upper,
                "limit_units": units,
                "lower_limit_degrees": lower if units == "degrees" else None,
                "upper_limit_degrees": upper if units == "degrees" else None,
                "lower_limit_m": lower if units == "m" else None,
                "upper_limit_m": upper if units == "m" else None,
                "current_rotation_degrees": [
                    round(math.degrees(value), 4) for value in original_rotation
                ],
                "current_axis_value": round(
                    math.degrees(original_rotation[axis_index])
                    if units == "degrees"
                    else original_location[axis_index],
                    6,
                ),
                "moving_mesh_count": len(moving),
                "moving_meshes": sorted(obj.name for obj in moving),
                "max_sampled_point_motion_m": round(max_motion, 6),
                "valid": valid,
            }
        )
    return {
        "required": required,
        "joint_count": len(joints),
        "joints": rows,
        "ok": (not required and not joints) or (bool(joints) and all(row["valid"] for row in rows)),
    }


def contract_report(expected):
    scene = bpy.context.scene
    declared = {
        "kind": asset_value(scene, "real2sim_asset_kind"),
        "front_axis": asset_value(scene, "real2sim_front_axis"),
        "up_axis": asset_value(scene, "real2sim_up_axis"),
        "units": asset_value(scene, "real2sim_units"),
    }
    contract_keys = ("kind", "front_axis", "up_axis", "units")
    mismatches = {
        key: {"expected": value, "declared": declared.get(key)}
        for key, value in expected.items()
        if key in contract_keys
        if declared.get(key) not in (None, value)
    }
    missing = [key for key in contract_keys if declared.get(key) is None]
    object_names = [
        obj.name.lower() for obj in bpy.data.objects
        if not asset_value(obj, "real2sim_fixture") and not asset_value(obj, "real2sim_ground") and not asset_value(obj, "real2sim_sim_proxy")
        and not asset_value(obj, "real2sim_collision_proxy")
    ]
    missing_parts = [
        part
        for part in expected.get("required_parts", [])
        if not any(part.lower() in name for name in object_names)
    ]
    return {
        "expected": expected,
        "declared": declared,
        "missing": missing,
        "mismatches": mismatches,
        "missing_parts": missing_parts,
        "ok": not missing and not mismatches and not missing_parts,
    }


def main():
    separator = sys.argv.index("--")
    script = Path(sys.argv[separator + 1]).resolve()
    out = Path(sys.argv[separator + 2]).resolve()
    run_path = Path(sys.argv[separator + 3]).resolve() if len(sys.argv) > separator + 3 else None
    run = json.loads(run_path.read_text()) if run_path and run_path.is_file() else {}
    expected = run.get(
        "asset_contract",
        {"kind": "rigid", "front_axis": "+Y", "up_axis": "+Z", "units": "m"},
    )
    report = {"ok": False}
    try:
        clear_scene()
        load_builder(script)()
        for obj in bpy.context.scene.objects:
            if asset_value(obj, "real2sim_sim_proxy"):
                obj.hide_render = True
        meshes = [
            obj
            for obj in bpy.context.scene.objects
            if obj.type == "MESH" and is_asset_object(obj)
        ]
        if not meshes:
            raise RuntimeError("build_scene() created no mesh objects")
        for index, obj in enumerate(sorted(meshes, key=lambda value: value.name), start=1):
            obj.pass_index = index
        lo, hi = mesh_bounds(meshes)
        size = hi - lo
        if min(size) <= 1e-7 or max(size) > 1000:
            raise RuntimeError(f"Invalid object bounds: {tuple(size)}")
        center = (lo + hi) / 2
        radius = max(size.length / 2, 0.1)
        neutral_world()
        ensure_ground(lo, hi)
        ensure_lights(center, radius)
        configure_render()
        review_images = render_views(out, meshes, center, radius, expected["front_axis"])
        articulation_previews = render_articulation_previews(
            out, meshes, expected["front_axis"]
        )
        camera_alignment_previews = render_reference_cameras(out, run_path)
        (out / "part_map.json").write_text(
            json.dumps(
                {
                    str(obj.pass_index): {
                        "name": obj.name,
                        "parent": obj.parent.name if obj.parent else None,
                    }
                    for obj in meshes
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        bpy.ops.wm.save_as_mainfile(filepath=str(out / "scene.blend"))
        export_report = export_asset_glb(out / "object.glb", meshes)
        report = {
            "ok": True,
            "export": export_report,
            "geometry": geometry_report(meshes, lo, hi),
            "contract": contract_report(expected),
            "articulation": articulation_report(expected["kind"] == "articulated"),
            "functional_structure": snapshot_structure(),
            "articulation_previews": articulation_previews,
            "camera_alignment_previews": camera_alignment_previews,
            "review_images": review_images,
            "artifacts": {
                path.name: path.name
                for path in out.iterdir()
                if path.name in {"scene.blend", "object.glb", "part_map.json"}
                or path.name.startswith(
                    ("preview_", "mask_", "passes_", "joint_", "aligned_")
                )
            },
        }
        if run.get("cousin"):
            from cousin_contract import validate_cousin
            cousin = run["cousin"]
            report["cousin_validation"] = validate_cousin(
                cousin["parent"]["functional_structure"], report["functional_structure"],
                allow_surface_only=cousin["allow_surface_only"],
                sibling_geometry=[row["geometry_fingerprint"] for row in cousin["accepted_siblings"]],
            )
            if not report["cousin_validation"]["ok"]:
                report["ok"] = False
                report["error"] = "Cousin invariant check failed: " + ", ".join(report["cousin_validation"]["failures"])
        if not report["articulation"]["ok"]:
            issues = [
                f"{joint['name']}: {'; '.join(joint.get('schema_errors', [])) or 'no valid sampled motion'}"
                for joint in report["articulation"]["joints"]
                if not joint["valid"]
            ]
            report["error"] = "Articulation gate failed. " + " | ".join(issues)
    except BaseException as exc:
        report = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-12000:],
        }
        raise
    finally:
        (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
