"""Render editable assets at one physical scale, camera and studio configuration."""

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector

if not __package__:  # Blender also executes this file directly.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from real2sim_harness.asset_metadata import asset_value

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/real2sim_harness"))
from blender_final_harness import configure_cycles, fit_camera  # noqa: E402
from blender_harness import joint_limits, set_joint_value  # noqa: E402
from blender_surface import is_asset_object  # noqa: E402


def asset_bounds(joint_fraction=None):
    bpy.context.scene.frame_set(1)
    if joint_fraction is not None:
        for obj in bpy.context.scene.objects:
            if obj.get("joint_type") in {"revolute", "prismatic"}:
                lower, upper, _ = joint_limits(obj)
                set_joint_value(
                    obj,
                    lower + (upper - lower) * joint_fraction,
                    obj.rotation_euler.copy(),
                    obj.location.copy(),
                )
    bpy.context.view_layer.update()
    return [
        obj.matrix_world @ Vector(corner)
        for obj in bpy.context.scene.objects
        if obj.type == "MESH" and is_asset_object(obj) and not obj.hide_render
        for corner in obj.bound_box
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument(
        "--joint-fraction",
        type=float,
        help="Optional shared normalized presentation pose; source files stay unchanged",
    )
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    if args.joint_fraction is not None and not 0 <= args.joint_fraction <= 1:
        raise ValueError("Normalized joint pose must be within [0, 1]")
    args.output.mkdir(parents=True, exist_ok=True)
    all_points, records = [], []
    for source in args.source:
        bpy.ops.wm.open_mainfile(filepath=str(source.resolve()))
        points = asset_bounds(args.joint_fraction)
        if not points:
            raise ValueError("A comparison source has no visible geometry")
        low = Vector(tuple(min(p[i] for p in points) for i in range(3)))
        high = Vector(tuple(max(p[i] for p in points) for i in range(3)))
        all_points.extend(points)
        records.append({"source": str(source.resolve()), "extent_m": list(high - low)})
    low = Vector(tuple(min(p[i] for p in all_points) for i in range(3)))
    high = Vector(tuple(max(p[i] for p in all_points) for i in range(3)))
    center = (low + high) / 2
    radius = max((high - low).length / 2, 0.001)
    for number, source in enumerate(args.source):
        bpy.ops.wm.open_mainfile(filepath=str(source.resolve()))
        asset_bounds(args.joint_fraction)
        scene = bpy.context.scene
        for obj in list(scene.objects):
            if obj.type in {"LIGHT", "CAMERA"} or asset_value(obj, "real2sim_ground") or asset_value(obj, "real2sim_fixture"):
                bpy.data.objects.remove(obj, do_unlink=True)
        world = bpy.data.worlds.new("Comparison studio")
        world.use_nodes = True
        world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.08, 0.08, 0.08, 1)
        world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.4
        scene.world = world
        bpy.ops.mesh.primitive_plane_add(
            size=radius * 20, location=(center.x, center.y, low.z - 0.0001)
        )
        floor = bpy.context.object
        floor.name = "Real2Sim_ComparisonFloor"
        mat = bpy.data.materials.new("Comparison floor")
        mat.use_nodes = True
        shader = mat.node_tree.nodes.get("Principled BSDF")
        shader.inputs["Base Color"].default_value = (0.16, 0.17, 0.19, 1)
        shader.inputs["Roughness"].default_value = 0.8
        floor.data.materials.append(mat)
        for direction, power, size in [
            ((2, 1, 3), 1800, 2),
            ((-2, 1, 2), 800, 2.4),
            ((0, -2, 3), 1200, 1.8),
        ]:
            bpy.ops.object.light_add(type="AREA", location=center + Vector(direction) * radius)
            light = bpy.context.object
            light.data.energy = power * radius * radius
            light.data.size = size * radius
            light.rotation_euler = (center - light.location).to_track_quat("-Z", "Y").to_euler()
        bpy.ops.object.camera_add()
        camera = bpy.context.object
        scene.camera = camera
        camera.data.lens = 55
        camera.data.sensor_width = 36
        camera.data.clip_start = max(radius * 0.001, 0.00001)
        configure_cycles(args.resolution, args.samples)
        scene.render.resolution_y = round(args.resolution * 3 / 4)
        fit_camera(scene, camera, center, Vector((1.1, 1.7, 0.9)), radius, all_points, fill=0.78)
        target = args.output / f"asset_{number:02d}.png"
        scene.render.filepath = str(target)
        bpy.ops.render.render(write_still=True)
        records[number]["image"] = target.name
        print("COMPARISON_RENDER", number + 1, len(args.source), flush=True)
    (args.output / "comparison.json").write_text(
        json.dumps(
            {
                "same_camera_and_scale": True,
                "objects_scaled": False,
                "presentation_joint_fraction": args.joint_fraction,
                "sources": records,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
