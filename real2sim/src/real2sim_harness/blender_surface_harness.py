"""Render a requested world-space surface region as material, clay and grazing-light evidence."""

from __future__ import annotations

import itertools
import json
import math
import sys
import traceback
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blender_final_harness import configure_cycles, fit_camera  # noqa: E402


def main():
    request = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
    settings = json.loads(request.read_text())
    output = request.parent
    report = {"ok": False, "images": []}
    try:
        scene = bpy.context.scene
        if settings.get("native_replay"):
            replay = settings["native_replay"]
            if replay["kind"] == "strand":
                from blender_strand_harness import load_player
            elif replay["kind"] == "hybrid":
                from blender_hybrid_harness import load_player
            elif replay["kind"] == "rigid_contact":
                from blender_rigid_contact_harness import load_player
            elif replay["kind"] in {"solid", "native_deformable"}:
                from blender_solid_harness import load_player
            else:
                raise ValueError("Unknown native replay backend")
            apply, *_ = load_player(Path(replay["directory"]))
            if replay["kind"] in {"solid", "native_deformable"}:
                import numpy as np
                cache = np.load(Path(replay["directory"]) / "motion.npz", allow_pickle=False)
                apply(cache["positions"][replay["cache_frame"]])
            else:
                apply(replay["cache_frame"])
        elif settings.get("frame") is not None:
            scene.frame_set(settings["frame"])
        center, size = Vector(settings["center"]), Vector(settings["size_m"])
        direction = Vector(settings["view_direction"]).normalized()
        if any(not math.isfinite(value) or value <= 0 for value in size):
            raise ValueError("Region size must contain three positive finite values")
        points = [center + Vector(tuple(sign * length / 2 for sign, length in zip(signs, size, strict=True))) for signs in itertools.product((-1, 1), repeat=3)]
        camera = scene.camera
        camera.data.lens, camera.data.sensor_width = 58, 36
        camera.data.shift_x = camera.data.shift_y = 0
        devices = configure_cycles(1024, 96)
        fit_camera(scene, camera, center, direction, max(size.length / 2, .01), points, fill=.88)

        def render(name):
            scene.render.filepath = str(output / name)
            bpy.ops.render.render(write_still=True)
            report["images"].append(name)

        render("surface_material.png")
        clay = bpy.data.materials.new("REMO diagnostic neutral clay")
        clay.use_nodes = True
        shader = clay.node_tree.nodes.get("Principled BSDF")
        shader.inputs["Base Color"].default_value = (.4, .4, .4, 1)
        shader.inputs["Roughness"].default_value = .7
        for obj in scene.objects:
            if obj.type == "LIGHT":
                obj.hide_render = True
        if scene.world and scene.world.use_nodes:
            background = scene.world.node_tree.nodes.get("Background")
            if background:
                background.inputs["Strength"].default_value = .12
        tangent = direction.cross(Vector((0, 0, 1)))
        if tangent.length < 1e-4:
            tangent = direction.cross(Vector((0, 1, 0)))
        tangent.normalize()
        radius = max(size.length / 2, .1)
        for name, offset, power, lamp_size in (
            ("Real2Sim_GrazingKey", tangent + direction * .22, 450, .45),
            ("Real2Sim_GrazingFill", -tangent + direction, 40, 1.5),
        ):
            data = bpy.data.lights.new(name, "AREA")
            data.energy, data.shape, data.size = power * radius**2, "DISK", lamp_size * radius
            obj = bpy.data.objects.new(name, data)
            bpy.context.collection.objects.link(obj)
            obj.location = center + offset.normalized() * radius * 3
            obj.rotation_euler = (center - obj.location).to_track_quat("-Z", "Y").to_euler()
        render("surface_grazing.png")
        bpy.context.view_layer.material_override = clay
        render("surface_clay.png")
        bpy.context.view_layer.material_override = None
        report.update(ok=True, region=settings, gpu_devices=devices,
                      source_lighting_editable="Lights created in build_scene() are retained by semantic and final renders; the host adds default lights only when source creates none. Diagnostic grazing lights do not change the saved asset.",
                      meaning="Material uses source lighting. Grazing material and neutral clay use identical grazing lights and camera; clay removes shader and normal-map relief, exposing actual geometry. These diagnostic images do not change the saved asset.")
    except BaseException as exc:
        report.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-8000:])
        raise
    finally:
        (output / "surface_review.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
