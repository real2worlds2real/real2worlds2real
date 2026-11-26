"""Render verified Newton liquid surfaces using the source Blender materials."""

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

if not __package__:  # Blender also executes this file directly.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real2sim_harness.asset_metadata import asset_value

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blender_final_harness import configure_cycles, fit_camera  # noqa: E402


def transition_material(source, specification):
    material = source.copy()
    material.name = source.name + " | simulated phase blend"
    nodes, links = material.node_tree.nodes, material.node_tree.links
    output = next(node for node in nodes if node.type == "OUTPUT_MATERIAL" and node.is_active_output)
    if not output.inputs["Surface"].links:
        raise ValueError("A transitioning material needs a source surface shader")
    original = output.inputs["Surface"].links[0].from_socket
    # An authored target can retain spatial procedural colour/relief without
    # baking unstable reconstructed UVs. Scalar values below remain defaults.
    target = asset_value(nodes, "real2sim_transition_target")
    if target is None:
        target = nodes.new("ShaderNodeBsdfPrincipled")
    elif target.type != "BSDF_PRINCIPLED":
        raise ValueError("real2sim_transition_target must be a Principled surface")
    names = {"base_color": "Base Color", "roughness": "Roughness", "transmission": "Transmission Weight",
             "subsurface_weight": "Subsurface Weight", "subsurface_scale": "Subsurface Scale",
             "ior": "IOR", "coat_weight": "Coat Weight", "coat_roughness": "Coat Roughness"}
    for key, value in specification.get("appearance", {}).items():
        target.inputs[names[key]].default_value = value
    # Retain source surface relief when blending to a new material state.
    # Reconstructed surfaces have no stable UV topology; procedural normals
    # using object/world coordinates remain meaningful across the transition.
    normal = original.node.inputs.get("Normal")
    if normal is not None and normal.is_linked and not target.inputs["Normal"].is_linked:
        links.new(normal.links[0].from_socket, target.inputs["Normal"])
    phase = nodes.new("ShaderNodeAttribute")
    phase.attribute_name = "real2sim_transition"
    blend = nodes.new("ShaderNodeMixShader")
    links.new(phase.outputs["Fac"], blend.inputs[0])
    links.new(original, blend.inputs[1])
    links.new(target.outputs["BSDF"], blend.inputs[2])
    links.new(blend.outputs[0], output.inputs["Surface"])
    return material


def apply_frame(objects, path):
    with np.load(path, allow_pickle=False) as data:
        for index, obj in enumerate(objects):
            key = f"part_{index:03d}"
            vertices, faces = data[key + "_vertices"], data[key + "_faces"]
            if not np.isfinite(vertices).all() or not len(faces):
                raise ValueError(f"Invalid cached fluid surface: {path}")
            if faces.min() < 0 or faces.max() >= len(vertices):
                raise ValueError("Cached surface indices exceed vertex count")
            previous = obj.data
            material = previous.materials[0] if previous.materials else None
            mesh = bpy.data.meshes.new(obj.name + "_frame")
            mesh.from_pydata(vertices.tolist(), [], faces.tolist())
            mesh.update()
            if f"{key}_transition" in data:
                phase = data[f"{key}_transition"]
                if len(phase) != len(vertices) or not np.isfinite(phase).all() or np.any((phase < 0)|(phase > 1)):
                    raise ValueError("Invalid reconstructed-surface phase state")
                attribute = mesh.attributes.new("real2sim_transition", "FLOAT", "POINT")
                attribute.data.foreach_set("value", phase)
            for polygon in mesh.polygons:
                polygon.use_smooth = True
            if material:
                mesh.materials.append(material)
            obj.data = mesh
            if previous.users == 0:
                bpy.data.meshes.remove(previous)
    bpy.context.view_layer.update()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("simulation", type=Path)
    parser.add_argument("--mode", choices=("preview", "render"), default="preview")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--aspect", type=float, default=1.0)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    simulation = args.simulation.resolve()
    configuration = json.loads((simulation / "fluid_configuration.json").read_text())
    report = json.loads((simulation / "fluid_report.json").read_text())
    if not report.get("ok"):
        raise ValueError("Fluid simulation must succeed before rendering")
    files = sorted((simulation / "surfaces").glob("frame_*.npz"))
    if len(files) != report["frames"]:
        raise ValueError("Incomplete cached surface sequence")
    objects = []
    for index, part in enumerate(configuration["parts"]):
        source = bpy.data.objects.get(part["source_object"])
        if source is None or not source.data.materials:
            raise ValueError(f"Source material volume or visual material missing: {part['source_object']}")
        source.hide_render = True
        mesh = bpy.data.meshes.new(f"Real2Sim_FluidSurface_{index:03d}")
        material = source.data.materials[0]
        if part.get("transition", {}).get("appearance"):
            material = transition_material(material, part["transition"])
        mesh.materials.append(material)
        obj = bpy.data.objects.new(mesh.name, mesh)
        bpy.context.collection.objects.link(obj)
        objects.append(obj)
    selected = sorted(set(np.linspace(0, len(files) - 1, 7).round().astype(int).tolist()))
    all_points = []
    for frame in selected:
        with np.load(files[frame], allow_pickle=False) as data:
            for index in range(len(objects)):
                vertices = data[f"part_{index:03d}_vertices"]
                if not np.isfinite(vertices).all():
                    raise ValueError("Non-finite retained surface")
                all_points.extend(Vector(point) for point in itertools.product(*zip(vertices.min(axis=0), vertices.max(axis=0), strict=True)))
    low = Vector(tuple(min(p[i] for p in all_points) for i in range(3)))
    high = Vector(tuple(max(p[i] for p in all_points) for i in range(3)))
    center, radius = (low + high) / 2, max((high - low).length / 2, 0.02)
    scene = bpy.context.scene
    camera = scene.camera
    if camera is None:
        camera = bpy.data.objects.new("Real2Sim_FluidCamera", bpy.data.cameras.new("Real2Sim_FluidCamera"))
        scene.collection.objects.link(camera)
        scene.camera = camera
    camera.data.type = "PERSP"
    camera.data.lens = 55
    camera.data.sensor_width = 36
    camera.data.clip_start = 0.0001
    camera.data.shift_x = camera.data.shift_y = 0
    configure_cycles(args.resolution, args.samples)
    scene.render.resolution_y = round(args.resolution / args.aspect)
    fit_camera(scene, camera, center, Vector((0.8, -1.3, 1.25)), radius, all_points, fill=0.82)
    scene.cycles.transmission_bounces = 12
    output = args.output.resolve() if args.output else simulation
    output.mkdir(parents=True, exist_ok=True)
    frames_dir = output / "frames"
    if args.mode == "render":
        frames_dir.mkdir(exist_ok=True)
    plan = selected if args.mode == "preview" else range(len(files))
    images = []
    for index, frame in enumerate(plan):
        apply_frame(objects, files[frame])
        name = f"preview_{index:02d}.png" if args.mode == "preview" else f"frame_{index:04d}.png"
        target = output / name if args.mode == "preview" else frames_dir / name
        scene.render.filepath = str(target)
        bpy.ops.render.render(write_still=True)
        images.append(str(target.relative_to(output)))
        print(f"FLUID_RENDER {index + 1}/{len(plan)} cache_frame={frame}", flush=True)
    # Keep the final visible state as an editable Blender snapshot; source seed
    # volumes and their material parameters stay in the original .blend.
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "simulation.blend"))
    if args.mode == "preview":
        result = {"ok": True, "fresh_process": True, "sample_frames": selected,
                  "images": images, "cache_frames": len(files), "rendered_from_actual_particle_surfaces": True}
        (output / "preview_report.json").write_text(json.dumps(result, indent=2))
    else:
        fps = args.fps or report["fps"]
        result = {"ok": True, "frames": len(files), "fps": fps,
                  "duration_seconds": len(files) / fps, "preview_only": False,
                  "source_physics_fps": report["fps"], "playback_speed": fps / report["fps"],
                  "motion": "Actual Newton MPM particle motion with reconstructed liquid surfaces",
                  "resolution": args.resolution, "samples": args.samples,
                  "source_scene_sha256": report.get("source_scene_sha256"),
                  "simulation_directory": str(simulation),
                  "configuration_sha256": hashlib.sha256((simulation / "fluid_configuration.json").read_bytes()).hexdigest()}
        (output / "animation_report.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
