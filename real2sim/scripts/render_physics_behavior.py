"""Render retained simulator states with the original editable Blender appearance."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

if not __package__:  # Blender also executes this file directly.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from real2sim_harness.asset_metadata import asset_value, is_harness_datablock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/real2sim_harness"))
from blender_final_harness import configure_cycles, fit_camera  # noqa: E402
from blender_hybrid_harness import ancestors, pose_matrix  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("behavior", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--aspect", type=float, default=4/3)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--sample-frames", help="Additional original trajectory indices for preview, as a JSON integer list")
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:])
    report = json.loads((args.behavior / "behavior_report.json").read_text())
    if not report["ok"]:
        raise ValueError("A failed physical behavior cannot become a final demonstration")
    source = Path(bpy.data.filepath)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != report["source_scene_sha256"]:
        raise ValueError("Blender appearance source changed after the simulator trajectory was recorded")
    data = np.load(args.behavior / "behavior_motion.npz", allow_pickle=False)
    bpy.context.scene.frame_set(1)
    for obj in bpy.context.scene.objects:
        obj.animation_data_clear()
    objects = {name: bpy.data.objects[name] for name in report["body_ids"] if name != "__base__"}
    order = sorted(objects, key=lambda name: len(list(ancestors(objects[name]))))
    control = bpy.data.objects[report["control_object"]] if report.get("control_object") else None
    followers = set(report.get("visual_followers", []))
    if control:
        followers.add(control.name)
        if control.data.shape_keys:
            control.shape_key_clear()
        if asset_value(control, "real2sim_sim_proxy"):
            control.hide_render = True
        for modifier in control.modifiers:
            if modifier.type in {"CLOTH", "SOFT_BODY", "HOOK", "ARMATURE"}:
                modifier.show_viewport = modifier.show_render = False
    rest = {obj.name: obj.matrix_world.copy() for obj in bpy.context.scene.objects}
    base = [obj for obj in bpy.context.scene.objects if obj.type == "MESH" and obj.name not in followers
            and not is_harness_datablock(obj.name) and not asset_value(obj, "real2sim_fixture")
            and not any(parent.name in objects for parent in ancestors(obj))]
    base.sort(key=lambda obj: len(list(ancestors(obj))))

    def apply(frame):
        poses = data["bodies"][frame]
        transform = pose_matrix(poses[report["body_ids"]["__base__"]])
        for obj in base:
            obj.matrix_world = transform @ rest[obj.name]
            bpy.context.view_layer.update()
        for name in order:
            scale = np.linalg.norm(np.asarray(report["rest_matrices"][name])[:3, :3], axis=0)
            objects[name].matrix_world = pose_matrix(poses[report["body_ids"][name]]) @ Matrix.Diagonal((*scale, 1.0))
            bpy.context.view_layer.update()
        if control:
            points = data["particles"][frame]
            if len(points) != len(control.data.vertices) or not np.isfinite(points).all():
                raise ValueError("Cloth trajectory does not match the source control nodes")
            inverse = control.matrix_world.inverted()
            for vertex, point in zip(control.data.vertices, points, strict=True):
                vertex.co = inverse @ Vector(point)
            control.data.update()
            bpy.context.view_layer.update()

    count = len(data["times"])
    selected = sorted(set(np.linspace(0, count-1, 7).round().astype(int).tolist()))
    if args.sample_frames:
        extra = json.loads(args.sample_frames)
        if not isinstance(extra, list) or any(type(frame) is not int or not 0 <= frame < count for frame in extra):
            raise ValueError("Preview indices must lie in the original trajectory")
        selected = sorted(set(selected+extra))
    bounds = []
    for frame in selected:
        apply(frame)
        for obj in bpy.context.scene.objects:
            if obj.type == "MESH" and not obj.hide_render and not asset_value(obj, "real2sim_fixture") and not is_harness_datablock(obj.name):
                bounds.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
    low = Vector(tuple(min(point[i] for point in bounds) for i in range(3)))
    high = Vector(tuple(max(point[i] for point in bounds) for i in range(3)))
    center, radius = (low+high)/2, max((high-low).length/2, .02)
    scene = bpy.context.scene
    camera = scene.camera
    camera.data.type = "PERSP"
    camera.data.lens = 55
    camera.data.sensor_width = 36
    camera.data.clip_start = .001
    camera.data.shift_x = camera.data.shift_y = 0
    configure_cycles(args.resolution, args.samples)
    scene.render.resolution_y = round(args.resolution/args.aspect)
    front_axis = asset_value(scene, "real2sim_front_axis", "+Y")
    front = {"+X": Vector((1,0,0)), "-X": Vector((-1,0,0)),
             "+Y": Vector((0,1,0)), "-Y": Vector((0,-1,0))}[front_axis]
    direction = Vector(report.get("control_plan", {}).get("camera_direction", front*1.5+front.cross(Vector((0,0,1)))*.8+Vector((0,0,.7))))
    fit_camera(scene, camera, center, direction, radius, bounds, fill=.82)
    args.output.mkdir(parents=True, exist_ok=True)
    target_folder = args.output if args.preview else args.output / "frames"
    target_folder.mkdir(exist_ok=True)
    plan = selected if args.preview else range(count)
    for number, frame in enumerate(plan):
        apply(frame)
        target = target_folder / (f"preview_{number:02d}.png" if args.preview else f"frame_{number:04d}.png")
        scene.render.filepath = str(target)
        bpy.ops.render.render(write_still=True)
        print("PHYSICS_RENDER", number+1, len(plan), flush=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output / "simulation.blend"))
    fps = report["fps"]
    result = {"ok": True, "preview_only": args.preview, "frames": len(plan), "fps": fps,
              "duration_seconds": len(plan)/fps, "resolution": args.resolution, "samples": args.samples,
              "motion": "Actual retained simulator trajectory", "physics_backend": report["backend"],
              "source_scene_sha256": digest, "source_behavior": str(args.behavior.resolve()),
              "motion_cache_sha256": hashlib.sha256((args.behavior / "behavior_motion.npz").read_bytes()).hexdigest(),
              "real_physics_calibrated": False, "robot_task_implemented": False}
    if args.preview:
        result["sample_frames"] = selected
    (args.output / ("preview_report.json" if args.preview else "animation_report.json")).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
