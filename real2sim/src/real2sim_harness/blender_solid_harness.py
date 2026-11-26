"""Fresh-process source-material playback of a native tetrahedral solid."""

import hashlib
import json
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real2sim_harness.surface_binding import deform_surface  # noqa: E402


def load_player(folder):
    folder = Path(folder)
    meta = json.loads((folder / "visual_binding.json").read_text())
    if (
        hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest()
        != meta["source_scene_sha256"]
    ):
        raise ValueError("Solid replay source changed")
    for name, expected in (
        ("visual_binding.npz", meta["binding_sha256"]),
        ("behavior_request.json", meta["configuration_sha256"]),
    ):
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Solid construction or visual binding changed")
    configuration = json.loads((folder / "behavior_request.json").read_text())
    bindings = np.load(folder / "visual_binding.npz", allow_pickle=False)
    rest = np.asarray(configuration["geometry"]["vertices"])
    triangles = np.asarray(configuration["geometry"]["triangles"])
    objects, rows, maximum = [], [], 0.0
    for entry in meta["visuals"]:
        obj = bpy.data.objects[entry["name"]]
        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated = obj.evaluated_get(depsgraph)
        mesh = bpy.data.meshes.new_from_object(
            evaluated, preserve_all_data_layers=True, depsgraph=depsgraph
        )
        obj.modifiers.clear()
        obj.data = mesh
        if obj.type != "MESH" or len(obj.data.vertices) != entry["vertices"]:
            raise ValueError("Source visual topology changed")
        binding = {
            key: bindings[f"visual_{entry['index']:03d}_{key}"]
            for key in ("triangle_ids", "weights", "offsets")
        }
        world = np.array([obj.matrix_world @ vertex.co for vertex in obj.data.vertices])
        error = np.linalg.norm(deform_surface(rest, triangles, binding) - world, axis=1).max()
        maximum = max(maximum, float(error))
        rows.append((obj, binding))
        objects.append(obj)
    if maximum > 1e-6:
        raise ValueError("Native rest binding does not reconstruct the source visuals")

    def apply(points):
        for obj, binding in rows:
            world = deform_surface(points, triangles, binding)
            inverse = np.asarray(obj.matrix_world.inverted())
            local = world @ inverse[:3, :3].T + inverse[:3, 3]
            obj.data.vertices.foreach_set("co", local.ravel())
            obj.data.update()
        bpy.context.view_layer.update()

    for name in meta.get("static_visual_objects", []):
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != "MESH":
            raise ValueError("A fixed source visual is missing")
        objects.append(obj)
    return apply, objects, maximum
