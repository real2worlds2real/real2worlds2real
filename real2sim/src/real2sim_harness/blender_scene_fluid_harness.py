"""Blender replay of a complete scene and actual MPM material surfaces."""

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blender_final_harness import configure_cycles, fit_camera  # noqa: E402
from blender_fluid_harness import apply_frame, transition_material  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--height", type=int)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--record-video", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    folder = args.scene.parent
    spec = json.loads((folder / "request.json").read_text())
    bindings = json.loads((args.output / "fluid_scene_bindings.json").read_text())
    configuration = json.loads((args.output / "fluid_configuration.json").read_text())
    report = json.loads((args.output / "physics_report.json").read_text())
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for background in spec["backgrounds"]:
        if background["format"] != "mesh":
            raise ValueError("This liquid replay adapter currently supports mesh backgrounds")
        native = folder / "backgrounds" / f"{background['id']}.blend"
        if background.get("editable_source"):
            # The composition inventory retains this source and its hash. Use
            # its procedural materials instead of lossy USD Preview Surface.
            before_images = set(bpy.data.images)
            with bpy.data.libraries.load(str(native), link=False) as (available, loaded):
                loaded.objects = available.objects
                loaded.worlds = available.worlds
            for image in set(bpy.data.images) - before_images:
                if image.source == "FILE" and not image.packed_file and not image.packed_files:
                    raise ValueError("Native background images must be packed for portable replay")
            imported = {obj for obj in loaded.objects if obj is not None and obj.type != "CAMERA"}
            for obj in imported:
                bpy.context.scene.collection.objects.link(obj)
                if obj.rigid_body or obj.rigid_body_constraint:
                    raise ValueError("Native background source must remain visual-only")
            if loaded.worlds:
                bpy.context.scene.world = loaded.worlds[0]
        else:
            before = set(bpy.data.objects)
            bpy.ops.wm.usd_import(filepath=str(folder / "backgrounds" / f"{background['id']}.usdz"))
            imported = set(bpy.data.objects) - before
        # Appended objects need dependency-graph evaluation before reading
        # matrix_world; otherwise freshly linked transforms can be identity.
        bpy.context.view_layer.update()
        placement = Matrix(background["transform"])
        for obj in imported:
            if obj.parent not in imported:
                obj.matrix_world = placement @ obj.matrix_world
    objects = [None] * len(configuration["parts"])
    for name, asset in bindings["assets"].items():
        source = folder / "assets" / name / asset["render_source"]
        names = [part["source_object"] for part in asset["parts"]]
        with bpy.data.libraries.load(str(source), link=False) as (available, loaded):
            if not set(names) <= set(available.objects):
                raise ValueError("Packed liquid source lost a material object")
            loaded.objects = names
        for part, source_object in zip(asset["parts"], loaded.objects, strict=True):
            index = part["global_index"]
            if source_object is None or not source_object.data.materials:
                raise ValueError("Liquid phase needs its retained source material")
            material = source_object.data.materials[0]
            transition = configuration["parts"][index].get("transition", {})
            if transition.get("appearance"):
                material = transition_material(material, transition)
            mesh = bpy.data.meshes.new(f"Liquid_{name}_{index}")
            mesh.materials.append(material)
            obj = bpy.data.objects.new(mesh.name, mesh)
            bpy.context.collection.objects.link(obj)
            objects[index] = obj
    files = sorted((args.output / "mpm/surfaces").glob("frame_*.npz"))
    if len(files) != report["frames"] or any(obj is None for obj in objects):
        raise ValueError("Incomplete scene fluid replay")
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    if not any(obj.type == "LIGHT" for obj in scene.objects):
        # Generic inspection lighting; an authored environment can supply its own.
        points = []
        for file in (files[0], files[-1]):
            with np.load(file, allow_pickle=False) as data:
                for i in range(len(objects)):
                    points.extend(data[f"part_{i:03d}_vertices"][::20])
        low, high = np.min(points, axis=0), np.max(points, axis=0)
        center = (low + high) / 2
        radius = max(float(np.linalg.norm(high - low)), 0.1)
        for offset, energy in [((1, -1, 2), 70.0), ((-1, 1, 1.5), 40.0)]:
            location = center + np.array(offset) * radius
            bpy.ops.object.light_add(type="AREA", location=location)
            light = bpy.context.object
            light.data.energy = energy
            light.data.size = radius * 2
            light.rotation_euler = (
                (Vector(center) - light.location).to_track_quat("-Z", "Y").to_euler()
            )
    bpy.ops.object.camera_add()
    scene.camera = bpy.context.object
    camera = scene.camera
    camera.data.clip_start = 0.001
    camera.data.clip_end = 1000
    view = spec.get("review_camera")
    if view:
        direction = Vector(view["eye_m"]) - Vector(view["target_m"])
        up = Vector(view["up"])
        right = up.cross(direction).normalized()
        corrected_up = direction.cross(right).normalized()
        rotation = Matrix((right, corrected_up, direction.normalized())).transposed()
        camera.matrix_world = rotation.to_4x4()
        camera.location = view["eye_m"]
        camera.data.lens = 18 / np.tan(np.radians(view["horizontal_fov_deg"]) / 2)
        camera.data.sensor_width = 36
    else:
        all_points = []
        for file in (files[0], files[-1]):
            with np.load(file, allow_pickle=False) as data:
                for i in range(len(objects)):
                    all_points.extend(Vector(p) for p in data[f"part_{i:03d}_vertices"][::20])
        for support in spec["supports"]:
            matrix = Matrix(support["transform"])
            half = np.array(support["size_m"]) / 2
            all_points.extend(
                matrix @ Vector((x, y, z))
                for x in (-half[0], half[0])
                for y in (-half[1], half[1])
                for z in (-half[2], half[2])
            )
        low = Vector(tuple(min(p[i] for p in all_points) for i in range(3)))
        high = Vector(tuple(max(p[i] for p in all_points) for i in range(3)))
        camera.data.lens = 45
        camera.data.sensor_width = 36
        fit_camera(
            scene,
            camera,
            (low + high) / 2,
            Vector((0.8, -1.3, 1.25)),
            max((high - low).length / 2, 0.1),
            all_points,
            fill=0.82,
        )
    configure_cycles(args.resolution, args.samples)
    if args.height is not None:
        if not 128 <= args.height <= 2048:
            raise ValueError("Use a bounded render height")
        scene.render.resolution_y = args.height
    scene.cycles.transmission_bounces = 12
    scene.render.fps = report["fps"]
    selected = set(np.linspace(0, len(files) - 1, 7, dtype=int))
    indices = range(len(files)) if args.record_video else sorted(selected)
    frame_paths = []
    if args.record_video:
        (args.output / "video_frames").mkdir(exist_ok=True)
    for frame in indices:
        apply_frame(objects, files[frame])
        path = (
            f"video_frames/frame_{frame:05d}.png" if args.record_video else f"scene_{frame:05d}.png"
        )
        scene.render.filepath = str(args.output / path)
        bpy.ops.render.render(write_still=True)
        if args.record_video:
            frame_paths.append(path)
        if frame in selected:
            report["images"].append(path)
        print("SCENE_MPM_RENDER", frame, len(files), flush=True)
    if args.record_video:
        report["frame_sequence"] = {
            "fps": report["fps"],
            "frames": frame_paths,
            "time_basis": "simulation_seconds",
        }
    report["render_replay"] = {
        "physics_advanced": False,
        "renderer": "Blender Cycles",
        "native_background_materials": [b["id"] for b in spec["backgrounds"] if b.get("editable_source")],
        "whole_scene": True,
        "materials": "Packed source Blender materials",
        "review_camera": view,
        "frames": len(list(indices)),
    }
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output / "scene_replay.blend"))
    (args.output / "runtime_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
