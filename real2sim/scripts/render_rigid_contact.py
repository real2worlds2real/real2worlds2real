"""Fresh-process replay of exact SDF input meshes and retained rigid states."""

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Quaternion, Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/real2sim_harness"))
from blender_final_harness import configure_cycles, fit_camera  # noqa: E402


def mat(name, color):
    m = bpy.data.materials.new(name)
    m.diffuse_color = (*color, 1)
    m.use_nodes = True
    shader = m.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Roughness"].default_value = 0.3
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    p.add_argument("--mode", choices=["preview", "render"], default="preview")
    p.add_argument("--resolution", type=int, default=768)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--source-replay", action="store_true")
    a = p.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    folder = a.folder.resolve()
    report = json.loads((folder / "validation.json").read_text())
    for name, sha in report["evidence_sha256"].items():
        file = (folder / name).resolve()
        if not file.is_relative_to(folder) or hashlib.sha256(file.read_bytes()).hexdigest() != sha:
            raise ValueError("Request or motion changed after native simulation")
    req = json.loads((folder / "request.json").read_text())
    cache = np.load(folder / "motion.npz", allow_pickle=False)
    poses = cache["body_poses"]
    if not np.isfinite(poses).all() or len(poses) != report["frames"]:
        raise ValueError("Invalid native trajectory")
    objects = []
    envelope = []
    if a.source_replay:
        from blender_rigid_contact_harness import load_player

        apply, source_objects, bindings = load_player(folder)
        for i, obj, local, corners in bindings:
            objects.append((i, obj, local))
            for pose in poses[:, i]:
                rot = Quaternion((pose[6], *pose[3:6])).to_matrix()
                envelope.append(corners @ np.asarray(rot).T + pose[:3])
        for obj in list(bpy.context.scene.objects):
            if obj not in source_objects:
                bpy.data.objects.remove(obj, do_unlink=True)
    else:
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
        colors = [(0.62, 0.25, 0.075), (0.055, 0.24, 0.39), (0.55, 0.55, 0.55)]
        for i, b in enumerate(req["bodies"]):
            vertices = np.asarray(b["vertices"])
            tri = np.asarray(b["triangles"])
            mesh = bpy.data.meshes.new(b["id"])
            mesh.from_pydata(vertices.tolist(), [], tri.tolist())
            mesh.update()
            obj = bpy.data.objects.new(b["id"], mesh)
            bpy.context.collection.objects.link(obj)
            obj.data.materials.append(mat(b["id"], colors[i % len(colors)]))
            if not b.get("static", False):
                obj.data.materials.append(mat("Orientation witness", (0.88, 0.85, 0.65)))
                for face in mesh.polygons:
                    center = vertices[list(face.vertices)].mean(axis=0)
                    if center[0] > 0 and abs(np.arctan2(center[1], center[0])) < 0.06:
                        face.material_index = 1
            objects.append((i, obj, Matrix.Identity(4)))
            low, high = vertices.min(axis=0), vertices.max(axis=0)
            corners = np.asarray(list(itertools.product(*zip(low, high, strict=True))))
            for pose in poses[:, i]:
                rot = Quaternion((pose[6], *pose[3:6])).to_matrix()
                envelope.append(corners @ np.asarray(rot).T + pose[:3])
    pts = np.concatenate(envelope)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    center = Vector((lo + hi) / 2)
    radius = max(float(np.linalg.norm(hi - lo)) / 2, 0.015)
    bpy.ops.mesh.primitive_plane_add(
        size=radius * 12, location=(center.x, center.y, lo[2] - 0.0005)
    )
    bpy.context.object.data.materials.append(mat("Backdrop", (0.08, 0.095, 0.11)))
    scene = bpy.context.scene
    scene.world.color = (0.3, 0.3, 0.3)
    for direction, power in [((1, -2, 3), 120), ((-2, 1, 2), 85)]:
        light = bpy.data.lights.new("Softbox", "AREA")
        light.energy = power * (radius / 0.2) ** 2
        light.size = radius * 3
        obj = bpy.data.objects.new("Softbox", light)
        scene.collection.objects.link(obj)
        obj.location = center + Vector(direction) * radius * 3
        obj.rotation_euler = (center - obj.location).to_track_quat("-Z", "Y").to_euler()
    camera = bpy.data.objects.new("ReviewCamera", bpy.data.cameras.new("ReviewCamera"))
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.data.lens = 65
    camera.data.clip_start = 0.0001
    configure_cycles(a.resolution, a.samples)
    scene.render.resolution_y = round(a.resolution * 0.75)
    box = [Vector(p) for p in itertools.product(*zip(lo, hi, strict=True))]
    fit_camera(scene, camera, center, Vector((1, -1.6, 1.3)), radius, box)
    projections = [world_to_camera_view(scene, camera, p) for p in box]
    if any(p.z <= 0 or not 0 <= p.x <= 1 or not 0 <= p.y <= 1 for p in projections):
        raise ValueError("Camera clips motion envelope")
    frames = (
        np.unique(np.linspace(0, len(poses) - 1, 7).round().astype(int)).tolist()
        if a.mode == "preview"
        else list(range(len(poses)))
    )
    directory = folder / ("preview" if a.mode == "preview" else "frames")
    directory.mkdir(exist_ok=True)
    images = []
    for index, frame in enumerate(frames):
        for i, obj, local in objects:
            pose = poses[frame, i]
            obj.matrix_world = (
                Matrix.Translation(Vector(pose[:3]))
                @ Quaternion((pose[6], *pose[3:6])).to_matrix().to_4x4()
                @ local
            )
        scene.render.filepath = str(directory / f"frame_{index:04d}.png")
        bpy.ops.render.render(write_still=True)
        images.append(str(Path(scene.render.filepath).relative_to(folder)))
    data = {
        "ok": True,
        "images": images,
        "frames": len(frames),
        "fps": report["fps"],
        "duration_seconds": len(poses) / report["fps"],
        "simulated_span_seconds": (len(poses) - 1) / report["fps"],
        "sample_frames": frames,
        "motion_sha256": report["evidence_sha256"]["motion.npz"],
        "native_simulation_ok": report["ok"],
        "all_motion_framing": True,
        "fresh_process": True,
        "source_material_replay": a.source_replay,
        "source_scene_sha256": req.get("source_scene_sha256"),
        "scope": "Source material replay of retained native states."
        if a.source_replay
        else "Exact input collision mesh replay with diagnostic materials and an orientation stripe; not the final source appearance.",
    }
    (
        folder / ("preview_report.json" if a.mode == "preview" else "animation_report.json")
    ).write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
