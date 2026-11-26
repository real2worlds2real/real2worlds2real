"""Fracture a closed material volume using the official Blender Cell Fracture add-on."""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Matrix

if not __package__:  # Blender also executes this file directly.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from real2sim_harness.asset_metadata import asset_key, asset_value

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor/cell-fracture"))
import object_fracture_cell  # noqa: E402


def properties(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    loose = [e for e in bm.edges if not e.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context="EDGES")
    bad = sum(not e.is_manifold for e in bm.edges)
    histogram = {n: sum(len(e.link_faces) == n for e in bm.edges) for n in (0, 1, 2, 3, 4)}
    boundary = [
        float((obj.matrix_world.to_3x3() @ (e.verts[0].co - e.verts[1].co)).length)
        for e in bm.edges
        if len(e.link_faces) == 1
    ]
    bmesh.ops.triangulate(bm, faces=list(bm.faces))
    triangles = np.asarray([[obj.matrix_world @ v.co for v in f.verts] for f in bm.faces])
    bm.free()
    signed = np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])) / 6
    volume = signed.sum()
    if bad or not np.isfinite(volume) or volume <= 0:
        raise ValueError(
            f"{obj.name}: requires a closed, consistently outward material volume; nonmanifold={bad}, volume={volume}, edge_face_counts={histogram}, boundary_edge_lengths_m={boundary[:40]}"
        )
    center = ((triangles.sum(axis=1) / 4) * signed[:, None]).sum(axis=0) / volume
    return float(volume), center


def main():
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    a = p.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = Path(bpy.data.filepath)
    tagged = [o for o in bpy.context.scene.objects if asset_value(o, "real2sim_fracture")]
    if len(tagged) != 1 or tagged[0].type != "MESH":
        raise ValueError("Tag exactly one closed MESH with JSON real2sim_fracture metadata")
    obj = tagged[0]
    spec = json.loads(obj[asset_key(obj, "real2sim_fracture")])
    count = int(spec.get("pieces", 48))
    if not 8 <= count <= 128:
        raise ValueError("Choose 8–128 fracture sites")
    # Evaluate modifiers once; fracture and intact replay share this exact surface.
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    obj.data = bpy.data.meshes.new_from_object(
        evaluated, preserve_all_data_layers=True, depsgraph=bpy.context.evaluated_depsgraph_get()
    )
    obj.modifiers.clear()
    source_volume, source_center = properties(obj)
    interior_name = spec.get("interior_material")
    interior = bpy.data.materials.get(interior_name) if interior_name else None
    if interior is None:
        raise ValueError("Name an existing interior_material for exposed fracture surfaces")
    if interior.name not in [m.name for m in obj.data.materials if m]:
        obj.data.materials.append(interior)
    index = list(obj.data.materials).index(interior)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    before = set(bpy.data.objects)
    random.seed(int(spec.get("seed", 0)))
    from mathutils import noise

    noise.seed_set(int(spec.get("seed", 0)))
    # Legacy Boolean/Voronoi code assumes ordinary scene units. Normalize around
    # the source COM, then restore exact world scale before physical export.
    original_matrix = obj.matrix_world.copy()
    normalizing = Matrix.Scale(10.0 / max(obj.dimensions.length, 1e-9), 4) @ Matrix.Translation(
        -source_center
    )
    obj.matrix_world = normalizing @ original_matrix
    bpy.context.view_layer.update()
    object_fracture_cell.register()
    bpy.ops.object.add_fracture_cell_objects(
        source={"VERT_OWN"},
        source_limit=count,
        source_noise=float(spec.get("site_noise", 0.15)),
        margin=0.0,
        material_index=index,
        use_data_match=True,
        use_island_split=True,
        use_smooth_faces=False,
        use_sharp_edges=False,
        use_sharp_edges_apply=False,
        recursion=0,
    )
    pieces = sorted(
        [o for o in set(bpy.data.objects) - before if o.type == "MESH"], key=lambda o: o.name
    )
    obj.matrix_world = original_matrix
    for shard in pieces:
        shard.matrix_world = normalizing.inverted() @ shard.matrix_world
    bpy.context.view_layer.update()
    rows = []
    for i, shard in enumerate(pieces):
        volume, center = properties(shard)
        if volume < source_volume * 1e-8:
            bpy.data.objects.remove(shard, do_unlink=True)
            continue
        world = np.asarray([shard.matrix_world @ v.co for v in shard.data.vertices])
        for vertex, value in zip(shard.data.vertices, world - center, strict=True):
            vertex.co = value
        shard.parent = None
        shard.matrix_world = Matrix.Translation(center)
        shard.name = f"Real2Sim_Shard_{i:03d}"
        shard.hide_render = True
        if shard.rigid_body:
            bpy.context.view_layer.objects.active = shard
            bpy.ops.rigidbody.object_remove()
        rows.append(
            {
                "object": shard.name,
                "volume": volume,
                "center": center.tolist(),
                "vertices": (world - center).tolist(),
                "faces": [list(p.vertices) for p in shard.data.polygons],
            }
        )
    total = sum(r["volume"] for r in rows)
    error = abs(total - source_volume) / source_volume
    weighted = sum(np.asarray(r["center"]) * r["volume"] for r in rows) / total
    if (
        not 8 <= len(rows) <= 200
        or error > 0.015
        or np.linalg.norm(weighted - source_center) > 0.0002
    ):
        raise ValueError(
            f"Fracture coverage failed: pieces={len(rows)}, volume_error={error}, center_error={np.linalg.norm(weighted - source_center)}"
        )
    mass = float(spec.get("mass_kg", source_volume * float(spec.get("density", 2300))))
    for row in rows:
        row["mass_kg"] = mass * row["volume"] / total
    motion = dict(spec.get("motion", {}))
    cfg = {
        "schema_version": 1,
        "source_blend_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_object": obj.name,
        "source_matrix": np.asarray(obj.matrix_world).tolist(),
        "center": weighted.tolist(),
        "pieces": rows,
        "mass_kg": mass,
        "impulse_threshold_ns": float(spec.get("impulse_threshold_ns", 0.08)),
        "friction": float(spec.get("friction", 0.55)),
        "restitution": float(spec.get("restitution", 0.08)),
        "motion": motion,
        "coverage": {
            "relative_volume_error": error,
            "closed_shards": len(rows),
            "source_volume_m3": source_volume,
        },
        "model": "Prefractured closed volumes; contact-impulse threshold releases all shards. No crack propagation or calibrated failure law.",
    }
    for key in ("mass_kg", "impulse_threshold_ns", "friction", "restitution"):
        if not np.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f"Invalid {key}")
    if mass <= 0 or cfg["impulse_threshold_ns"] <= 0:
        raise ValueError("Positive mass and threshold required")
    obj.hide_render = False
    (output / "fracture_configuration.json").write_text(json.dumps(cfg))
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "fractured.blend"))
    print("FRACTURE_EXPORT", len(rows), "volume_error", error, flush=True)


if __name__ == "__main__":
    main()
