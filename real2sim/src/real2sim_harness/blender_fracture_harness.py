"""Replay native PhysX shard poses using the Cell Fracture source surfaces."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

if not __package__:  # Blender also executes this file directly.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real2sim_harness.asset_metadata import matches_datablock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blender_final_harness import configure_cycles, fit_camera  # noqa: E402


def matrix(pose):
    m = Quaternion((pose[6], *pose[3:6])).to_matrix().to_4x4()
    m.translation = Vector(pose[:3])
    return m


def load_player(folder):
    folder = Path(folder).resolve()
    cfg = json.loads((folder / "fracture_configuration.json").read_text())
    report = json.loads((folder / "fracture_validation.json").read_text())
    if cfg["source_blend_sha256"] != report["source_scene_sha256"]:
        raise ValueError("Fracture source/cache mismatch")
    if (
        hashlib.sha256((folder / "fracture_motion.npz").read_bytes()).hexdigest()
        != report["motion_sha256"]
    ):
        raise ValueError("Fracture cache changed")
    data = np.load(folder / "fracture_motion.npz", allow_pickle=False)
    original = bpy.data.objects[cfg["source_object"]]
    pieces = [bpy.data.objects[r["object"]] for r in cfg["pieces"]]
    rest = Matrix.Translation(-Vector(cfg["center"])) @ Matrix(cfg["source_matrix"])
    if original.rigid_body:
        original.rigid_body.kinematic = True

    def apply(frame):
        broken = bool(data["broken"][frame])
        original.hide_render = broken
        original.matrix_world = matrix(data["intact"][frame]) @ rest
        for obj, pose in zip(pieces, data["pieces"][frame], strict=True):
            obj.hide_render = not broken
            obj.matrix_world = matrix(pose)
        bpy.context.view_layer.update()

    return apply, data, report, cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    p.add_argument("--mode", choices=["preview", "render"], default="preview")
    p.add_argument("--output", type=Path)
    p.add_argument("--resolution", type=int, default=768)
    p.add_argument("--samples", type=int, default=48)
    a = p.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    folder = a.folder.resolve()
    output = a.output.resolve() if a.output else folder
    output.mkdir(parents=True, exist_ok=True)
    apply, data, report, cfg = load_player(folder)
    bounds = []
    objects = [bpy.data.objects[cfg["source_object"]]] + [
        bpy.data.objects[r["object"]] for r in cfg["pieces"]
    ]
    # Every retained state contributes; the impact and any far-traveling shard stay in frame.
    for f in range(len(data["times"])):
        apply(f)
        for o in objects:
            if not o.hide_render:
                bounds.extend(o.matrix_world @ Vector(c) for c in o.bound_box)
    low = Vector(tuple(min(v[i] for v in bounds) for i in range(3)))
    high = Vector(tuple(max(v[i] for v in bounds) for i in range(3)))
    center = (low + high) / 2
    radius = max((high - low).length / 2, 0.05)
    scene = bpy.context.scene
    configure_cycles(a.resolution, a.samples)
    scene.render.resolution_y = round(a.resolution * 3 / 4)
    camera = scene.camera
    if camera is None:
        camera = bpy.data.objects.new(
            "Real2Sim_FractureCamera", bpy.data.cameras.new("Real2Sim_FractureCamera")
        )
        scene.collection.objects.link(camera)
        scene.camera = camera
    camera.data.type = "PERSP"
    camera.data.lens = 55
    camera.data.clip_start = 0.0005
    fit_camera(scene, camera, center, Vector((1.2, -1.7, 1.1)), radius, bounds, fill=0.82)
    for obj in list(scene.objects):
        if obj.type == "LIGHT" or matches_datablock(obj.name, "Real2Sim_Ground"):
            bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.mesh.primitive_plane_add(size=max(4.0, radius * 12), location=(center.x, center.y, 0))
    floor = bpy.context.object
    floor.name = "Real2Sim_Ground"
    mat = bpy.data.materials.new("Real2Sim_FractureGround")
    mat.diffuse_color = (0.18, 0.21, 0.25, 1)
    floor.data.materials.append(mat)
    for direction, power, size in [((1, -2, 3), 1000, 3), ((-2, 0, 1.5), 600, 2)]:
        bpy.ops.object.light_add(type="AREA", location=center + Vector(direction) * radius)
        light = bpy.context.object
        light.data.energy = power * radius * radius
        light.data.size = size * radius
        light.rotation_euler = (center - light.location).to_track_quat("-Z", "Y").to_euler()
    trigger = report["break_event"].get("time_seconds", 0)
    selected = sorted(
        set(
            [
                0,
                *[
                    int(np.clip(round(t * 24), 0, len(data["times"]) - 1))
                    for t in (trigger - 0.1, trigger, trigger + 0.05, trigger + 0.2, trigger + 0.7)
                ],
                len(data["times"]) - 1,
            ]
        )
    )
    sequence = selected if a.mode == "preview" else range(len(data["times"]))
    target = output if a.mode == "preview" else output / "frames"
    target.mkdir(exist_ok=True)
    images = []
    for i, f in enumerate(sequence):
        apply(f)
        dest = target / (f"preview_{i:02d}.png" if a.mode == "preview" else f"frame_{i:04d}.png")
        scene.render.filepath = str(dest)
        bpy.ops.render.render(write_still=True)
        images.append(str(dest.relative_to(output)))
        print("FRACTURE_RENDER", i + 1, len(sequence), flush=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "simulation.blend"))
    evidence = {
        "ok": True,
        "fresh_process": True,
        "actual_physx_fragment_poses": True,
        "source_scene_sha256": cfg["source_blend_sha256"],
        "images": images,
        "sample_frames": selected,
        "frames": len(sequence),
        "fps": report["fps"],
        "duration_seconds": len(sequence) / report["fps"],
        "preview_only": a.mode == "preview",
        "simulation_directory": str(folder),
        "motion_sha256": report["motion_sha256"],
        "camera_fit": "All retained native states and visible source/shard bounds",
    }
    (
        output / ("preview_report.json" if a.mode == "preview" else "animation_report.json")
    ).write_text(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
