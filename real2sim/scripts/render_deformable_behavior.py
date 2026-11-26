"""Blender diagnostic replay of retained native shell/solid states, without resimulation."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

if not __package__:  # Blender also executes this file directly.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from real2sim_harness.asset_metadata import asset_value, is_harness_datablock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/real2sim_harness"))
from blender_final_harness import configure_cycles, fit_camera, object_points  # noqa: E402


def material(name, color, roughness=0.6):
    result = bpy.data.materials.new(name)
    result.diffuse_color = (*color, 1)
    result.use_nodes = True
    shader = result.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Roughness"].default_value = roughness
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--mode", choices=["preview", "render"], default="preview")
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--source-replay", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    folder = args.folder
    output = args.output or folder
    output.mkdir(parents=True, exist_ok=True)
    report = json.loads((folder / "validation.json").read_text())
    if not report.get("evidence_sha256"):
        raise ValueError("A source-matched motion manifest is required")
    for name, expected in report["evidence_sha256"].items():
        path = (folder / name).resolve()
        if (
            not path.is_relative_to(folder.resolve())
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError("Behavior request or retained native states changed")
    request = json.loads((folder / "request.json").read_text())
    cache = np.load(folder / "motion.npz", allow_pickle=False)
    points = cache["positions"]
    if not np.isfinite(points).all() or len(points) != report["frames"]:
        raise ValueError("Cannot render a nonfinite or incomplete motion cache")
    source_error = None
    source_envelope = []
    if args.source_replay:
        from blender_solid_harness import load_player

        apply, objects, source_error = load_player(folder)
        for obj in list(bpy.context.scene.objects):
            if obj not in objects and (is_harness_datablock(obj.name) or asset_value(obj, "real2sim_ground")):
                bpy.data.objects.remove(obj, do_unlink=True)
        for state in points:
            apply(state)
            source_envelope.extend(list(point) for point in object_points(objects))
    else:
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
        mesh = bpy.data.meshes.new("NativeControlBoundary")
        mesh.from_pydata(points[0], [], cache["triangles"])
        mesh.update()
        obj = bpy.data.objects.new("NativeDeformable", mesh)
        bpy.context.collection.objects.link(obj)
        obj.data.materials.append(
            material(
                "Diagnostic surface",
                (0.8, 0.22, 0.08) if request["representation"] == "volume" else (0.9, 0.88, 0.78),
            )
        )
        for face in mesh.polygons:
            face.use_smooth = True
        if request["representation"] == "surface":
            modifier = obj.modifiers.new("Visual physical thickness", "SOLIDIFY")
            modifier.thickness = request["material"]["thickness_m"]
            modifier.offset = 0.0

        def apply(state):
            mesh.vertices.foreach_set("co", state.ravel())
            mesh.update()

    probes = []
    envelope = [points.reshape(-1, 3)]
    if source_envelope:
        envelope.append(np.asarray(source_envelope))
    for i, probe in enumerate(request.get("probes", [])):
        if not probe.get("collision", True) or (
            args.source_replay and probe.get("source_visual_objects")
        ):
            continue
        if probe["shape"] == "sphere":
            bpy.ops.mesh.primitive_uv_sphere_add(
                segments=48, ring_count=24, radius=probe["radius_m"]
            )
            extent = np.full(3, probe["radius_m"])
        else:
            bpy.ops.mesh.primitive_cube_add(size=1)
            bpy.context.object.scale = probe["size_m"]
            extent = np.asarray(probe["size_m"]) / 2
        body = bpy.context.object
        body.name = probe["id"]
        body.data.materials.append(material("Probe", (0.045, 0.16, 0.23), 0.25))
        probes.append((i, body))
        translations = cache["body_poses"][:, i, :3]
        envelope.extend([translations - extent, translations + extent])
    all_points = np.concatenate(envelope)
    lo, hi = all_points.min(axis=0), all_points.max(axis=0)
    center = Vector((lo + hi) / 2)
    radius = max(float(np.linalg.norm(hi - lo)) / 2, 0.05)
    bpy.ops.mesh.primitive_plane_add(
        size=radius * 12,
        location=(center.x, center.y, request.get("environment", {}).get("floor_height", 0)),
    )
    bpy.context.object.data.materials.append(material("Ground", (0.12, 0.14, 0.17)))
    scene = bpy.context.scene
    scene.world.color = (0.3, 0.3, 0.3)
    for direction, power, size in [((1, -2, 3), 100, 1.0), ((-2, 1, 2), 65, 1.0)]:
        light = bpy.data.lights.new("Softbox", "AREA")
        light.energy = power * (radius / 0.2) ** 2
        light.shape = "DISK"
        light.size = radius * size * 3
        light_obj = bpy.data.objects.new("Softbox", light)
        scene.collection.objects.link(light_obj)
        light_obj.location = center + Vector(direction) * radius * 3
        light_obj.rotation_euler = (center - light_obj.location).to_track_quat("-Z", "Y").to_euler()
    camera = bpy.data.objects.new("BehaviorCamera", bpy.data.cameras.new("BehaviorCamera"))
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.data.lens = 55
    camera.data.clip_start = 0.001
    configure_cycles(args.resolution, args.samples)
    scene.render.resolution_y = round(args.resolution * 0.75)
    corners = [Vector(p) for p in itertools.product(*zip(lo, hi, strict=True))]
    fit_camera(scene, camera, center, Vector((1, -1.6, 1.1)), radius, corners)
    projected = [world_to_camera_view(scene, camera, p) for p in corners]
    if any(p.z <= 0 or not 0 <= p.x <= 1 or not 0 <= p.y <= 1 for p in projected):
        raise ValueError("Replay camera clips the retained motion envelope")
    selected = np.unique(np.linspace(0, len(points) - 1, 7).round().astype(int)).tolist()
    frames = selected if args.mode == "preview" else range(len(points))
    target = output / ("preview" if args.mode == "preview" else "frames")
    target.mkdir(exist_ok=True)
    images = []
    for ordinal, frame in enumerate(frames):
        apply(points[frame])
        for i, probe in probes:
            probe.location = cache["body_poses"][frame, i, :3]
        scene.render.filepath = str(target / f"frame_{ordinal:04d}.png")
        bpy.ops.render.render(write_still=True)
        images.append(str(Path(scene.render.filepath).relative_to(output)))
        print("BEHAVIOR_RENDER", ordinal + 1, len(frames), flush=True)
    evidence = {
        "ok": True,
        "native_simulation_ok": report["ok"],
        "images": images,
        "sample_frames": list(frames),
        "fps": report["fps"],
        "frames": len(frames),
        "duration_seconds": len(points) / report["fps"],
        "preview_only": args.mode == "preview",
        "cache_frames": len(points),
        "motion_sha256": report["evidence_sha256"]["motion.npz"],
        "all_motion_framing": True,
        "fresh_process": True,
        "source_material_replay": args.source_replay,
        "source_scene_sha256": request.get("source_scene_sha256"),
        "rest_binding_max_error_m": source_error,
        "scope": "Diagnostic control-boundary replay; no new physics or appearance certification.",
    }
    (
        output / ("preview_report.json" if args.mode == "preview" else "animation_report.json")
    ).write_text(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
