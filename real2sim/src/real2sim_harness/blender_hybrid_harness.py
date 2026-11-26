"""Replay actual Newton rigid/cloth states with the original Blender materials."""

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

if not __package__:  # Blender also executes this file directly.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real2sim_harness.asset_metadata import asset_value, is_harness_datablock, matches_datablock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blender_final_harness import configure_cycles, fit_camera  # noqa: E402


def pose_matrix(pose):
    result = Quaternion((pose[6], *pose[3:6])).to_matrix().to_4x4()
    result.translation = Vector(pose[:3])
    return result


def ancestors(obj):
    parent = obj.parent
    while parent is not None:
        yield parent
        parent = parent.parent


def load_player(folder):
    """Load one replay implementation for rendering and geometric inspection."""
    folder = Path(folder).resolve()
    configuration = json.loads((folder / "hybrid_configuration.json").read_text())
    manifest = json.loads((folder / "manifest.json").read_text())
    report = json.loads((folder / "hybrid_validation.json").read_text())
    data = np.load(folder / "hybrid_motion.npz", allow_pickle=False)
    control = bpy.data.objects[configuration["control_object"]]
    if control.data.shape_keys:
        control.shape_key_clear()
    if asset_value(control, "real2sim_sim_proxy"):
        control.hide_render = True
    for modifier in control.modifiers:
        if modifier.type in {"CLOTH", "SOFT_BODY", "HOOK", "ARMATURE"}:
            modifier.show_viewport = modifier.show_render = False
    followers = set(manifest["deformable_visual_objects"]) | {control.name}
    rest = {obj.name: obj.matrix_world.copy() for obj in bpy.context.scene.objects}
    body_objects = {name: bpy.data.objects[name] for name in report["body_ids"] if name != "__base__"}
    order = sorted(body_objects, key=lambda name: len(list(ancestors(body_objects[name]))))
    base_objects = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or obj.name in followers or asset_value(obj, "real2sim_fixture") or is_harness_datablock(obj.name):
            continue
        if not any(parent.name in body_objects for parent in ancestors(obj)):
            base_objects.append(obj)
    base_objects.sort(key=lambda obj: len(list(ancestors(obj))))

    def apply(frame):
        poses, points = data["bodies"][frame], data["particles"][frame]
        if len(points) != len(control.data.vertices) or not np.isfinite(points).all():
            raise ValueError("Cached cloth points do not match the source control mesh")
        base = pose_matrix(poses[report["body_ids"]["__base__"]])
        for obj in base_objects:
            obj.matrix_world = base @ rest[obj.name]
        for name in order:
            scale = np.linalg.norm(np.asarray(report["rest_matrices"][name])[:3, :3], axis=0)
            body_objects[name].matrix_world = pose_matrix(poses[report["body_ids"][name]]) @ Matrix.Diagonal((*scale, 1.0))
            bpy.context.view_layer.update()
        bpy.context.view_layer.update()
        inverse = control.matrix_world.inverted()
        for vertex, point in zip(control.data.vertices, points, strict=True):
            vertex.co = inverse @ Vector(point)
        control.data.update()
        bpy.context.view_layer.update()

    return apply, data, report, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--mode", choices=("preview", "render"), default="preview")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--aspect", type=float, default=1.0)
    parser.add_argument("--view-direction", type=float, nargs=3, default=(1.2, -1.5, 0.8),
                        help="Direction from the asset center toward the camera")
    parser.add_argument("--view-fill-light", action="store_true",
                        help="Add a soft light near the camera to reveal normally shaded mechanisms")
    parser.add_argument("--hide-fixtures", action="store_true",
                        help="Hide presentation fixtures when they obstruct an inspection viewpoint")
    parser.add_argument("--sample-frames", help="Additional cached frames to inspect, as a JSON integer list")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    folder = args.folder.resolve()
    output = args.output.resolve() if args.output else folder
    output.mkdir(parents=True, exist_ok=True)
    apply, data, report, manifest = load_player(folder)
    if args.hide_fixtures:
        for obj in bpy.context.scene.objects:
            if asset_value(obj, "real2sim_fixture") or matches_datablock(obj.name, "Real2Sim_Ground"):
                obj.hide_render = True
    end = len(data["times"]) if args.end_frame is None else args.end_frame
    if not 0 <= args.start_frame < end <= len(data["times"]):
        raise ValueError("Select a valid retained motion interval")
    selected = sorted(set(np.linspace(args.start_frame, end-1, 7).round().astype(int).tolist()))
    if args.sample_frames:
        extra = json.loads(args.sample_frames)
        if not isinstance(extra, list) or any(type(frame) is not int or not args.start_frame <= frame < end for frame in extra):
            raise ValueError("Additional preview frames must lie in the retained motion interval")
        selected = sorted(set(selected + extra))
    bounds = []
    for frame in selected:
        apply(frame)
        for obj in bpy.context.scene.objects:
            if obj.type == "MESH" and not obj.hide_render and not asset_value(obj, "real2sim_fixture") and not is_harness_datablock(obj.name):
                bounds.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
    low = Vector(tuple(min(point[i] for point in bounds) for i in range(3)))
    high = Vector(tuple(max(point[i] for point in bounds) for i in range(3)))
    center, radius = (low+high)/2, max((high-low).length/2, 0.05)
    scene = bpy.context.scene
    if scene.camera is None:
        camera = bpy.data.objects.new("Real2Sim_HybridCamera", bpy.data.cameras.new("Real2Sim_HybridCamera"))
        scene.collection.objects.link(camera)
        scene.camera = camera
    camera = scene.camera
    camera.data.type = "PERSP"
    camera.data.lens = 55
    camera.data.sensor_width = 36
    camera.data.clip_start = 0.001
    configure_cycles(args.resolution, args.samples)
    scene.render.resolution_y = round(args.resolution / args.aspect)
    direction = Vector(args.view_direction)
    if direction.length < 1e-6:
        raise ValueError("Camera view direction must be nonzero")
    fit_camera(scene, camera, center, direction, radius, bounds, fill=0.82)
    if args.view_fill_light:
        bpy.ops.object.light_add(type="AREA", location=camera.location)
        fill_light = bpy.context.object
        fill_light.name = "Real2Sim_ViewFill"
        fill_light.data.energy = 350 * radius * radius
        fill_light.data.size = radius * 2
        fill_light.rotation_euler = (center-fill_light.location).to_track_quat('-Z', 'Y').to_euler()
    if not any(obj.type == "LIGHT" for obj in scene.objects):
        bpy.ops.object.light_add(type="AREA", location=center+Vector((1, -1, 2))*radius)
        light = bpy.context.object
        light.data.energy = 1500*radius*radius
        light.data.shape = "DISK"
        light.data.size = radius*2
        light.rotation_euler = (center-light.location).to_track_quat('-Z', 'Y').to_euler()
    frames = selected if args.mode == "preview" else range(args.start_frame, end)
    images = []
    target_folder = output if args.mode == "preview" else output / "frames"
    target_folder.mkdir(exist_ok=True)
    for number, frame in enumerate(frames):
        apply(frame)
        target = target_folder / (f"preview_{number:02d}.png" if args.mode == "preview" else f"frame_{number:04d}.png")
        scene.render.filepath = str(target)
        bpy.ops.render.render(write_still=True)
        images.append(str(target.relative_to(output)))
        print("HYBRID_RENDER", number+1, len(frames), flush=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "simulation.blend"))
    if args.mode == "preview":
        evidence = {"ok": True, "fresh_process": True, "sample_frames": selected, "images": images,
                    "actual_newton_rigid_and_cloth_motion": True, "cache_frames": len(data["times"])}
        (output / "preview_report.json").write_text(json.dumps(evidence, indent=2))
    else:
        fps = report["fps"]
        evidence = {"ok": True, "frames": len(frames), "fps": fps, "duration_seconds": len(frames)/fps,
                    "preview_only": False, "motion": "Actual Newton rigid and cloth trajectory",
                    "source_cache_frame_interval": [args.start_frame, end-1],
                    "source_scene_sha256": manifest["source_blend_sha256"],
                    "simulation_directory": str(folder), "playback_speed": 1.0,
                    "view_direction": list(args.view_direction), "view_fill_light": args.view_fill_light,
                    "presentation_fixtures_hidden": args.hide_fixtures,
                    "resolution": args.resolution, "samples": args.samples}
        (output / "animation_report.json").write_text(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
