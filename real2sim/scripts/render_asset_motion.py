"""Render a class-independent joint sweep and camera orbit from a saved Blender asset."""

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
from blender_harness import joint_axis, joint_limits, set_joint_value  # noqa: E402
from blender_surface import is_asset_object  # noqa: E402


def smooth(value):
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def current_value(joint, rotation):
    axis, sign, _ = joint_axis(joint)
    if joint.get("joint_type") == "revolute":
        return sign * math.degrees(rotation[axis])
    # Prismatic displacement is relative to the saved rest location.
    return 0.0


def sweep_value(phase, rest, lower, upper, units):
    span = upper - lower
    if units == "degrees" and span >= 350:
        return lower + (rest - lower + span * smooth(phase)) % span
    knots = (rest, lower, upper, rest)
    segment = min(int(phase * 3), 2)
    fraction = smooth(phase * 3 - segment)
    return knots[segment] * (1 - fraction) + knots[segment + 1] * fraction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--joint", action="append", default=[])
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--seconds-per-joint", type=float, default=2.0)
    parser.add_argument("--orbit-seconds", type=float, default=6.0)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    if args.fps <= 0 or args.seconds_per_joint <= 0 or args.orbit_seconds <= 0:
        raise ValueError("Frame rate and segment durations must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frames = output / "frames"
    frames.mkdir(exist_ok=True)
    scene = bpy.context.scene
    source_scene = bpy.data.filepath
    joints = sorted(
        (obj for obj in scene.objects if obj.get("joint_type")), key=lambda obj: obj.name
    )
    if args.joint:
        known = {joint.name for joint in joints}
        if set(args.joint) - known:
            raise ValueError(f"Unknown joint selection: {set(args.joint) - known}")
        joints = [joint for joint in joints if joint.name in args.joint]
    records = []
    for joint in joints:
        if joint.rotation_mode in {"QUATERNION", "AXIS_ANGLE"}:
            raise ValueError(
                f"Use an Euler variable joint under an oriented mount Empty: {joint.name}"
            )
        lower, upper, units = joint_limits(joint)
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError(f"Invalid joint range: {joint.name}")
        rotation, location = joint.rotation_euler.copy(), joint.location.copy()
        rest = current_value(joint, rotation)
        if not lower - 1e-4 <= rest <= upper + 1e-4:
            raise ValueError(f"Saved pose outside declared limits: {joint.name}")
        records.append((joint, rotation, location, lower, upper, units, rest))
    renderable = [
        obj for obj in scene.objects
        if is_asset_object(obj)
    ]
    if not renderable:
        raise ValueError("Scene contains no renderable asset objects")
    points = object_points(renderable)
    for joint, rotation, location, lower, upper, _, _ in records:
        for value in (lower, (lower + upper) / 2, upper):
            set_joint_value(joint, value, rotation, location)
            points.extend(object_points(renderable))
        joint.rotation_euler, joint.location = rotation, location
        bpy.context.view_layer.update()
    lo, hi = bounds(points)
    center, radius = (lo + hi) / 2, max((hi - lo).length / 2, 0.1)
    front = axis_vector(asset_value(scene, "real2sim_front_axis", "+Y"))
    right = Vector((front.y, -front.x, 0))
    # Base the presentation azimuth on the asset's horizontal aspect, without a category lookup.
    size = hi - lo
    base_angle = math.radians(65 if size.y > size.x * 2 else 27)
    camera = scene.camera
    if camera is None:
        data = bpy.data.cameras.new("Real2Sim_MotionCamera")
        camera = bpy.data.objects.new("Real2Sim_MotionCamera", data)
        bpy.context.collection.objects.link(camera)
        scene.camera = camera
    camera.data.lens = 52
    camera.data.sensor_width = 36
    camera.data.sensor_fit = "AUTO"
    camera.data.shift_x = camera.data.shift_y = 0
    devices = configure_cycles(args.resolution, args.samples)
    scene.cycles.seed = 7
    scene.cycles.use_animated_seed = False
    for obj in scene.objects:
        if asset_value(obj, "real2sim_ground"):
            obj.scale *= 10
    distance = 0.0
    for index in range(24):
        angle = 2 * math.pi * index / 24
        direction = front * math.cos(angle) + right * math.sin(angle) + Vector((0, 0, 0.30))
        fit_camera(scene, camera, center, direction, radius, points, fill=0.85)
        distance = max(distance, (camera.location - center).length)
    distance *= 1.02
    orbit_start = 1.0 + len(records) * args.seconds_per_joint
    duration = orbit_start + args.orbit_seconds + 1.0
    frame_count = round(duration * args.fps)
    scene.render.fps = args.fps
    scene.frame_start, scene.frame_end = 1, frame_count
    if args.preview:
        times = [0.0]
        times.extend(
            1 + (index + fraction) * args.seconds_per_joint
            for index in range(len(records)) for fraction in (1 / 3, 2 / 3)
        )
        times.extend(orbit_start + args.orbit_seconds * fraction for fraction in (0.25, 0.5, 0.75))
    else:
        times = (index / args.fps for index in range(frame_count))
    started = time.monotonic()
    count = 0
    for index, seconds in enumerate(times):
        scene.frame_set(index + 1)
        active = None
        for joint_index, record in enumerate(records):
            joint, rotation, location, lower, upper, units, rest = record
            joint.rotation_euler, joint.location = rotation, location
            phase = (seconds - 1.0 - joint_index * args.seconds_per_joint) / args.seconds_per_joint
            if 0 <= phase < 1:
                value = sweep_value(phase, rest, lower, upper, units)
                set_joint_value(joint, value, rotation, location)
                active = joint.name
            if not args.preview:
                joint.keyframe_insert(data_path="rotation_euler", frame=index + 1)
                joint.keyframe_insert(data_path="location", frame=index + 1)
        angle = base_angle + 2 * math.pi * smooth((seconds - orbit_start) / args.orbit_seconds)
        direction = (
            front * math.cos(angle) + right * math.sin(angle) + Vector((0, 0, 0.30))
        ).normalized()
        camera.location = center + direction * distance
        look_at(camera, center)
        if not args.preview:
            camera.keyframe_insert(data_path="location", frame=index + 1)
            camera.keyframe_insert(data_path="rotation_euler", frame=index + 1)
        bpy.context.view_layer.update()
        target = output / f"preview_{index:02d}.png" if args.preview else frames / f"frame_{index:04d}.png"
        scene.render.filepath = str(target)
        bpy.ops.render.render(write_still=True)
        count += 1
        print(f"MOTION_FRAME {count}/{frame_count} t={seconds:.3f} joint={active}", flush=True)
    if not args.preview:
        scene.frame_set(1)
        bpy.ops.wm.save_as_mainfile(filepath=str(output / "animated_scene.blend"))
    report = {
        "ok": True, "source_scene": source_scene,
        "motion": "kinematic joint sweeps and camera orbit; no dynamics claim",
        "joint_plan": [
            {
                "name": joint.name, "type": joint.get("joint_type"), "axis": joint.get("joint_axis"),
                "lower": lower, "upper": upper, "units": units, "rest": rest,
                "start_seconds": 1 + index * args.seconds_per_joint,
                "duration_seconds": args.seconds_per_joint,
            }
            for index, (joint, _, _, lower, upper, units, rest) in enumerate(records)
        ],
        "duration_seconds": duration, "fps": args.fps, "frames": count,
        "resolution": args.resolution, "samples": args.samples, "gpu_devices": devices,
        "render_seconds": round(time.monotonic() - started, 3), "preview_only": args.preview,
    }
    name = "preview_report.json" if args.preview else "animation_report.json"
    (output / name).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
