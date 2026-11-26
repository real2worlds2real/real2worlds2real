"""Blender-side visual environment authoring; physics lives in the scene contract."""
from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import bpy

if not __package__:  # Blender also executes this file directly.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real2sim_harness.asset_metadata import asset_value

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    source, output = map(Path, sys.argv[sys.argv.index("--") + 1:])
    output.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    namespace = runpy.run_path(str(source))
    if not callable(namespace.get("build_scene")):
        raise ValueError("Define build_scene() for the visual environment")
    namespace["build_scene"]()
    if not any(obj.type == "MESH" for obj in bpy.context.scene.objects):
        raise ValueError("A mesh environment needs geometry")
    if any(obj.rigid_body or obj.rigid_body_constraint for obj in bpy.context.scene.objects):
        raise ValueError("Keep dynamics out of the visual environment; declare supports and complete assets separately")
    bpy.context.scene.unit_settings.system = "METRIC"
    bpy.context.scene.unit_settings.scale_length = 1.0
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "environment.blend"))
    bpy.ops.wm.usd_export(filepath=str(output / "environment.usdc"), export_animation=False,
                          export_materials=True, export_lights=True, export_cameras=False,
                          export_custom_properties=True, author_blender_name=True,
                          export_textures_mode="NEW", overwrite_textures=True, relative_paths=True,
                          convert_scene_units="METERS", meters_per_unit=1.0, root_prim_path="/Environment")
    from real2sim_harness.blender_glass import export_native_glass
    from real2sim_harness.blender_normal_transfer import export_normal_transfer

    export_native_glass(output / "environment.usdc", list(bpy.context.scene.objects))
    export_normal_transfer(output / "environment.usdc", list(bpy.context.scene.objects))
    previews = []
    scene = bpy.context.scene
    if asset_value(scene, "real2sim_preview_background", False):
        from real2sim_harness.blender_final_harness import configure_cycles

        configure_cycles(640, 24)
        cameras = [obj for obj in scene.objects if obj.type == "CAMERA"]
        if scene.camera in cameras:
            cameras.remove(scene.camera)
            cameras.insert(0, scene.camera)
        if not cameras:
            raise ValueError("Background preview requires an authored camera")
        for index, camera in enumerate(cameras[:4]):
            scene.camera = camera
            name = f"preview_{index:03d}.png"
            scene.render.filepath = str(output / name)
            bpy.ops.render.render(write_still=True)
            previews.append(name)
    (output / "export.json").write_text(json.dumps({"ok": True, "visual_only": True,
                                                    "source": str(source), "entrypoint": "environment.usdc",
                                                    "preview_images": previews}))


if __name__ == "__main__":
    main()
