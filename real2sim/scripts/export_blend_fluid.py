"""Sample closed Blender material volumes into a generic Newton MPM configuration."""

import hashlib
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

if not __package__:  # Blender also executes this file directly.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from real2sim_harness.asset_metadata import asset_key, asset_value

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from real2sim_harness.fluid_environment import static_colliders  # noqa: E402
from real2sim_harness.material_transition import validate_transition  # noqa: E402


def inside(tree, point, epsilon):
    origin = Vector(point)
    direction = Vector((1.0, 0.000371, 0.000791)).normalized()
    intersections = 0
    for _ in range(128):
        location, _, _, _ = tree.ray_cast(origin, direction)
        if location is None:
            return intersections % 2 == 1
        intersections += 1
        origin = location + direction * epsilon
    raise ValueError("Material volume ray test exceeded the intersection limit")


def main():
    output = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.frame_set(scene.frame_start)
    source = Path(bpy.data.filepath)
    objects = sorted((obj for obj in scene.objects if obj.type == "MESH" and asset_value(obj, "real2sim_mpm_material")),
                     key=lambda obj: (-int(asset_value(obj, "real2sim_mpm_priority", 0)), obj.name))
    if not objects:
        raise ValueError("No closed material volumes tagged real2sim_mpm_material")
    spacing_default = float(asset_value(scene, "real2sim_mpm_particle_spacing", 0.0015))
    voxel = float(asset_value(scene, "real2sim_mpm_voxel_size", spacing_default * 2))
    tree_records = []
    graph = bpy.context.evaluated_depsgraph_get()
    unit = scene.unit_settings.scale_length
    for obj in objects:
        evaluated = obj.evaluated_get(graph)
        mesh = evaluated.to_mesh()
        try:
            mesh.calc_loop_triangles()
            vertices = [obj.matrix_world @ vertex.co * unit for vertex in mesh.vertices]
            faces = [list(face.vertices) for face in mesh.loop_triangles]
            if not vertices or not faces:
                raise ValueError(f"Empty fluid volume: {obj.name}")
            tree_records.append((obj, BVHTree.FromPolygons(vertices, faces, all_triangles=True), np.asarray(vertices)))
        finally:
            evaluated.to_mesh_clear()
    parts = []
    previous = []
    for obj, tree, vertices in tree_records:
        spacing = float(asset_value(obj, "real2sim_mpm_particle_spacing", spacing_default))
        if spacing <= 0:
            raise ValueError("Particle spacing must be positive")
        low, high = vertices.min(axis=0), vertices.max(axis=0)
        dimensions = np.ceil((high - low) / spacing).astype(int)
        if np.prod(dimensions) > 600_000:
            raise ValueError(f"Sampling budget exceeded for {obj.name}; increase particle spacing")
        axes = [np.arange(low[i] + spacing * 0.5, high[i], spacing) for i in range(3)]
        candidates = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        points = [point.tolist() for point in candidates
                  if inside(tree, point, spacing * 1e-4)
                  and not any(inside(other, point, spacing * 1e-4) for other in previous)]
        if len(points) < 50:
            raise ValueError(f"Too few interior particles for {obj.name}: {len(points)}")
        previous.append(tree)
        raw = obj[asset_key(obj, "real2sim_mpm_material")]
        material = json.loads(raw) if isinstance(raw, str) else dict(raw)
        defaults = {"density": 1000.0, "young_modulus": 1e15, "poisson_ratio": 0.3,
                    "friction": 0.0, "tensile_yield_ratio": 1.0, "yield_pressure": 1e12,
                    "yield_stress": 0.0, "viscosity": 0.5}
        defaults.update(material)
        color = list(obj.active_material.diffuse_color[:3]) if obj.active_material else [0.7] * 3
        parts.append({"source_object": obj.name, "positions": points, "spacing": spacing,
                      "material": defaults, "color": color})
        if asset_value(obj, "real2sim_mpm_transition"):
            raw = obj[asset_key(obj, "real2sim_mpm_transition")]
            parts[-1]["transition"] = validate_transition(json.loads(raw) if isinstance(raw, str) else dict(raw))
        print(f"MATERIAL_VOLUME {obj.name} particles={len(points)}", flush=True)
    if sum(len(part["positions"]) for part in parts) > 180_000:
        raise ValueError("Combined material particle count exceeds the review budget")
    with source.open("rb") as stream:
        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    configuration = {
        "schema_version": 1, "source_blend": str(source), "source_scene_sha256": source_hash,
        "floor_z": float(asset_value(scene, "real2sim_mpm_floor_z", 0.0)),
        "solver": {"voxel_size": voxel, "substeps": int(asset_value(scene, "real2sim_mpm_substeps", 2)),
                   "surface_voxel_size": float(asset_value(scene, "real2sim_mpm_surface_voxel_size", spacing_default * 0.55)),
                   "max_iterations": int(asset_value(scene, "real2sim_mpm_iterations", 60))},
        "parts": parts,
    }
    raw = asset_value(scene, 'real2sim_mpm_static_colliders')
    if raw is not None:
        configuration['static_colliders'] = json.loads(raw) if isinstance(raw,str) else list(raw)
        configuration['use_ground_plane'] = bool(asset_value(scene, 'real2sim_mpm_use_ground_plane',True))
        static_colliders(configuration)
    configuration['source_meters_per_blender_unit'] = float(unit)
    # Keep an interchange seed preview and packed material source. These are
    # construction data; scene motion is recomputed in the target environment.
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects:
        obj.select_set(True)
    bpy.ops.wm.usd_export(filepath=str(output.parent/'visual.usdc'), selected_objects_only=True,
        export_animation=False, export_materials=True, export_custom_properties=True,
        author_blender_name=True, export_lights=False, export_cameras=False,
        export_textures_mode='NEW', overwrite_textures=True, relative_paths=True,
        convert_scene_units='METERS', meters_per_unit=1.0, root_prim_path='/Asset')
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(output.parent/'render_source.blend'),copy=True)
    output.write_text(json.dumps(configuration, separators=(",", ":")))
    print(f"FLUID_CONFIGURATION_OK parts={len(parts)} output={output}", flush=True)


if __name__ == "__main__":
    main()
