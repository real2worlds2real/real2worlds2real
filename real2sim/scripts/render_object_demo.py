"""Historical microwave animation fixture; framework demos use render_asset_motion.py."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import bpy
from mathutils import Vector

if not __package__:  # Blender also executes this file directly.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from real2sim_harness.asset_metadata import asset_value

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "real2sim_harness"))
from blender_final_harness import (  # noqa: E402
    axis_vector,
    bounds,
    configure_cycles,
    fit_camera,
    look_at,
    object_points,
)


def smooth(value):
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def door_angle(seconds):
    """Closed, open, orbit with door open, close, closed; angles in degrees."""
    if seconds < 3:
        return -105.0 * smooth((seconds - 1) / 2)
    if seconds < 9:
        return -105.0
    return -105.0 * (1.0 - smooth((seconds - 9) / 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frames = output / "frames"
    frames.mkdir(exist_ok=True)
    scene = bpy.context.scene
    source_scene = bpy.data.filepath
    for obj in scene.objects:
        if asset_value(obj, "real2sim_ground"):
            obj.scale *= 10
    joints = [obj for obj in scene.objects if obj.get("joint_type")]
    doors = [obj for obj in joints if "door" in obj.name.lower() and "hinge" in obj.name.lower()]
    if len(doors) != 1:
        raise ValueError("Expected one named door hinge joint")
    door = doors[0]
    if door.get("joint_type") != "revolute" or str(door.get("joint_axis")).upper() not in {"Z", "+Z"}:
        raise ValueError("This demo currently expects a revolute local +Z door hinge")
    if not float(door.get("lower_limit_degrees", 0)) <= -105 < 0 <= float(
        door.get("upper_limit_degrees", 0)
    ):
        raise ValueError("Requested motion exceeds declared door limits")
    renderable = [
        obj for obj in scene.objects
        if obj.type in {"MESH", "CURVE", "FONT"} and not asset_value(obj, "real2sim_ground")
    ]
    # Frame the union of closed, half-open, and open bounds, keeping a stable camera radius.
    points = []
    for angle in (0, -52.5, -105):
        door.rotation_euler.z = math.radians(angle)
        bpy.context.view_layer.update()
        points.extend(object_points(renderable))
    lo, hi = bounds(points)
    center = (lo + hi) / 2
    radius = max((hi - lo).length / 2, 0.1)
    front = axis_vector(asset_value(scene, "real2sim_front_axis", "+Y"))
    right = Vector((front.y, -front.x, 0))
    camera = scene.camera
    camera.data.lens = 52
    gpu_devices = configure_cycles(args.resolution, args.samples)
    scene.render.fps = args.fps
    scene.frame_start = 1
    scene.frame_end = 12 * args.fps
    scene.cycles.seed = 7
    scene.cycles.use_animated_seed = False
    # Choose one distance that fits all orbit directions, preventing scale pumping.
    distance = 0.0
    for index in range(24):
        angle = 2 * math.pi * index / 24
        direction = front * math.cos(angle) + right * math.sin(angle) + Vector((0, 0, 0.34))
        fit_camera(scene, camera, center, direction, radius, points, fill=0.85)
        distance = max(distance, (camera.location - center).length)
    distance *= 1.02
    times = (0.0, 3.0, 5.0, 7.0, 9.0, 11.0) if args.preview else (
        index / args.fps for index in range(scene.frame_end)
    )
    started = time.monotonic()
    count = 0
    for index, seconds in enumerate(times):
        scene.frame_set(index + 1)
        door.rotation_euler.z = math.radians(door_angle(seconds))
        orbit = smooth((seconds - 3) / 6)
        angle = math.radians(27) + 2 * math.pi * orbit
        direction = (
            front * math.cos(angle) + right * math.sin(angle) + Vector((0, 0, 0.34))
        ).normalized()
        camera.location = center + direction * distance
        look_at(camera, center)
        if not args.preview:
            door.keyframe_insert(data_path="rotation_euler", frame=index + 1)
            camera.keyframe_insert(data_path="location", frame=index + 1)
            camera.keyframe_insert(data_path="rotation_euler", frame=index + 1)
        bpy.context.view_layer.update()
        target = output / f"preview_{index:02d}.png" if args.preview else frames / f"frame_{index:04d}.png"
        scene.render.filepath = str(target)
        bpy.ops.render.render(write_still=True)
        count += 1
        print(f"DEMO_FRAME {count} time={seconds:.3f} door={door_angle(seconds):.2f}", flush=True)
    if not args.preview:
        scene.frame_set(1)
        bpy.ops.wm.save_as_mainfile(filepath=str(output / "animated_scene.blend"))
    report = {
        "ok": True,
        "source_scene": source_scene,
        "motion": "kinematic joint animation, not a dynamics simulation",
        "joint": door.name,
        "angle_range_degrees": [-105, 0],
        "duration_seconds": 12,
        "fps": args.fps,
        "frames": count,
        "resolution": args.resolution,
        "samples": args.samples,
        "gpu_devices": gpu_devices,
        "render_seconds": round(time.monotonic() - started, 3),
        "preview_only": args.preview,
    }
    name = "preview_report.json" if args.preview else "animation_report.json"
    (output / name).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
