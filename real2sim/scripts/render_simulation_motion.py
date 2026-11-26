"""Replay retained native deformable physics, then orbit the final state."""

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
from blender_deformation_harness import file_hash, physics_objects, verify_reloaded  # noqa: E402
from blender_final_harness import (  # noqa: E402
    axis_vector,
    bounds,
    configure_cycles,
    fit_camera,
    look_at,
    object_points,
)
from blender_surface import is_asset_object  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--orbit-seconds", type=float, default=4)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    if args.fps <= 0 or args.orbit_seconds <= 0:
        raise ValueError("FPS and orbit duration must be positive")
    source = Path(bpy.data.filepath).resolve()
    evidence = json.loads((source.parent / "deformation.json").read_text())
    if not evidence.get("ok") or not evidence.get("cache_verification", {}).get("ok"):
        raise ValueError("Replay requires passed native simulation and cache verification")
    for relative, digest in evidence.get("cache_manifest", {}).items():
        if file_hash(source.parent.parent / relative) != digest:
            raise ValueError("Simulation scene/cache changed since verification")
    verify_reloaded(source.parent)
    scene = bpy.context.scene
    if not physics_objects(scene):
        raise ValueError("No native deformable physics found")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frames_dir = output / "frames"
    frames_dir.mkdir(exist_ok=True)
    if any(frames_dir.glob("frame_*.png")) and not args.preview:
        raise ValueError("Choose an empty output directory to avoid mixing frame sequences")
    start, end = evidence["frame_start"], evidence["frame_end"]
    original_fps = evidence["fps"]
    stride = max(1, round(original_fps / args.fps))
    fps = original_fps / stride
    if abs(fps - round(fps)) > 1e-6:
        raise ValueError("This renderer currently requires an integer output frame rate")
    fps = round(fps)
    objects = [obj for obj in scene.objects if is_asset_object(obj)]
    points = []
    for frame in evidence["sample_frames"]:
        scene.frame_set(frame)
        points.extend(object_points(objects))
    lo, hi = bounds(points)
    center, radius = (lo + hi) / 2, max((hi - lo).length / 2, 0.1)
    camera = scene.camera
    camera.data.lens, camera.data.sensor_width = 48, 36
    camera.data.shift_x = camera.data.shift_y = 0
    devices = configure_cycles(args.resolution, args.samples)
    front = axis_vector(asset_value(scene, "real2sim_front_axis", "+Y"))
    right = Vector((front.y, -front.x, 0))
    distance = 0.0
    for index in range(24):
        angle = 2 * math.pi * index / 24
        direction = front * math.cos(angle) + right * math.sin(angle) + Vector((0, 0, 0.85))
        fit_camera(scene, camera, center, direction, radius, points)
        distance = max(distance, (camera.location - center).length)
    plan = [(frame, .58, "native_cache_replay") for frame in range(start, end + 1, stride)]
    if plan[-1][0] != end:
        plan.append((end, .58, "native_cache_replay"))
    orbit_frames = round(args.orbit_seconds * fps)
    plan.extend((end, .58 + 2 * math.pi * index / max(orbit_frames - 1, 1), "final_state_orbit") for index in range(orbit_frames))
    if args.preview:
        plan = [(frame, .58, "native_cache_preview") for frame in evidence["sample_frames"]]
    started = time.monotonic()
    for index, (frame, angle, _) in enumerate(plan):
        scene.frame_set(frame)
        direction = (front * math.cos(angle) + right * math.sin(angle) + Vector((0, 0, .85))).normalized()
        camera.location = center + direction * distance
        look_at(camera, center)
        target = output / f"preview_{index:02d}.png" if args.preview else frames_dir / f"frame_{index:04d}.png"
        scene.render.filepath = str(target)
        bpy.ops.render.render(write_still=True)
        print(f"SIMULATION_FRAME {index + 1}/{len(plan)} source_frame={frame}", flush=True)
    report = {
        "ok": True, "source_scene": str(source), "source_scene_sha256": file_hash(source),
        "motion": "Native baked cloth/soft-body physics replay followed by a camera orbit; assumed parameters, no real-world calibration claim",
        "fps": fps, "frames": len(plan), "duration_seconds": len(plan) / fps,
        "source_fps": original_fps, "source_frame_start": start, "source_frame_end": end,
        "frame_plan": [{"source_frame": frame, "camera_angle": angle, "phase": phase} for frame, angle, phase in plan],
        "gpu_devices": devices, "resolution": args.resolution, "samples": args.samples,
        "render_seconds": round(time.monotonic() - started, 3), "preview_only": args.preview,
        "assumptions": evidence.get("assumptions"),
    }
    name = "preview_report.json" if args.preview else "animation_report.json"
    (output / name).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
