"""Bake native deformable physics and retain evidence; executed inside Blender."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
import traceback
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

if not __package__:  # Blender also executes this file directly.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real2sim_harness.asset_metadata import asset_value

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blender_final_harness import (  # noqa: E402
    axis_vector,
    bounds,
    configure_cycles,
    fit_camera,
    object_points,
)
from blender_surface import is_asset_object, proxy_bindings, sampled_world_points  # noqa: E402
from deformation_metrics import cloth_stretch_check, trajectory_metrics  # noqa: E402


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def physics_objects(scene):
    return [
        (obj, modifier)
        for obj in scene.objects
        if obj.type == "MESH" and not asset_value(obj, "real2sim_ground") and not asset_value(obj, "real2sim_fixture")
        for modifier in obj.modifiers
        if modifier.type in {"CLOTH", "SOFT_BODY"} and modifier.show_viewport and modifier.show_render
    ]


def control_mesh(obj, modifier):
    """Read physics vertices without downstream subdivision/solidification."""
    downstream = list(obj.modifiers)[list(obj.modifiers).index(modifier) + 1:]
    saved = [(item, item.show_viewport) for item in downstream]
    try:
        for item, _ in saved:
            item.show_viewport = False
        bpy.context.view_layer.update()
        evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        mesh = evaluated.to_mesh()
        try:
            local = np.empty((len(mesh.vertices), 3), dtype=np.float64)
            mesh.vertices.foreach_get("co", local.ravel())
            transform = np.asarray(evaluated.matrix_world, dtype=np.float64)
            points = local @ transform[:3, :3].T + transform[:3, 3]
            edges = np.empty((len(mesh.edges), 2), dtype=np.int32)
            mesh.edges.foreach_get("vertices", edges.ravel())
            return points, edges
        finally:
            evaluated.to_mesh_clear()
    finally:
        for item, value in saved:
            item.show_viewport = value
        bpy.context.view_layer.update()


def verify_reloaded(output):
    report = json.loads((output / "deformation.json").read_text())
    saved = np.load(output / "control_samples.npz", allow_pickle=False)
    errors = []
    for index, row in enumerate(report["objects"]):
        obj = bpy.context.scene.objects[row["name"]]
        modifier = obj.modifiers[row["modifier"]]
        if not modifier.point_cache.is_baked:
            errors.append(f"{obj.name}: native cache no longer baked")
        for sample, frame in enumerate(report["sample_frames"]):
            bpy.context.scene.frame_set(frame)
            points, _ = control_mesh(obj, modifier)
            expected = saved[f"object_{index}"][sample]
            if points.shape != expected.shape or not np.allclose(points, expected, atol=1e-5, rtol=1e-6):
                errors.append(f"{obj.name}: cache replay mismatch at frame {frame}")
    for index, row in enumerate(report.get("visual_motion", [])):
        obj = bpy.context.scene.objects[row["name"]]
        for sample, frame in enumerate(report["sample_frames"]):
            bpy.context.scene.frame_set(frame)
            points, _, count = sampled_world_points(obj, saved[f"visual_indices_{index}"])
            expected = saved[f"visual_{index}"][sample]
            if count != row["evaluated_vertices"] or not np.allclose(points, expected, atol=1e-5, rtol=1e-6):
                errors.append(f"{obj.name}: detailed surface replay mismatch at frame {frame}")
    verification = {"ok": not errors, "errors": errors, "fresh_process": True}
    (output / "cache_verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    if errors:
        raise RuntimeError(str(errors))


def simulate(output, frames):
    started = time.monotonic()
    report = {"ok": False, "backend": "Blender native cloth/soft-body", "images": []}
    try:
        scene = bpy.context.scene
        source_scene = Path(bpy.data.filepath).resolve()
        report["source_scene_sha256"] = file_hash(source_scene)
        physical = physics_objects(scene)
        if not physical:
            raise RuntimeError("No enabled native CLOTH or SOFT_BODY modifier on asset geometry")
        if len({obj.name for obj, _ in physical}) != len(physical):
            raise RuntimeError("Use at most one native deformable physics modifier per object")
        links = proxy_bindings(scene, physical)
        report["proxy_bindings"] = links
        if any(not row["ok"] for row in links):
            raise RuntimeError("Every real2sim_sim_proxy needs enabled native physics and a bound, enabled visible surface follower; inspect proxy_bindings")
        start = scene.frame_start
        end = start + frames - 1
        scene.frame_end = end
        scene.frame_set(start)
        report.update(frame_start=start, frame_end=end, fps=scene.render.fps / scene.render.fps_base)
        # Save first: Blender disk caches are relative to this retained .blend.
        target = output / "simulation.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(target))
        for index, (obj, modifier) in enumerate(physical):
            cache = modifier.point_cache
            with bpy.context.temp_override(object=obj, active_object=obj, point_cache=cache):
                if cache.is_baked:
                    bpy.ops.ptcache.free_bake()
                cache.frame_start, cache.frame_end = start, end
                cache.frame_step = 1
                cache.name = f"real2sim_{index:03d}_{re.sub('[^a-zA-Z0-9_]', '_', obj.name)[:32]}"
                cache.use_disk_cache = True
                bpy.ops.ptcache.bake(bake=True)
            if not cache.is_baked:
                raise RuntimeError(f"Native point-cache bake failed: {obj.name}")
        samples = sorted(set(np.linspace(start, end, 7).round().astype(int).tolist()))
        records, arrays = [], {}
        report["sample_frames"] = samples
        for index, (obj, modifier) in enumerate(physical):
            trajectory, initial_edges = [], None
            for frame in samples:
                scene.frame_set(frame)
                points, edges = control_mesh(obj, modifier)
                if len(points) > 250_000:
                    raise RuntimeError("Physics control mesh exceeds 250k vertices; move detail after physics")
                if initial_edges is None:
                    initial_edges = edges
                if not np.array_equal(edges, initial_edges):
                    raise RuntimeError(f"Physics control topology changes over time: {obj.name}")
                trajectory.append(points)
            arrays[f"object_{index}"] = np.stack(trajectory)
            metrics = trajectory_metrics(trajectory, initial_edges)
            row = {
                "name": obj.name, "modifier": modifier.name, "type": modifier.type,
                "vertices": len(trajectory[0]), "cache_baked": modifier.point_cache.is_baked,
                "metrics": metrics,
            }
            if modifier.type == "CLOTH":
                row["stretch_check"] = cloth_stretch_check(metrics, asset_value(obj, "real2sim_cloth_stretch_p95_limit", 2.0))
                row["parameters"] = {
                    key: getattr(modifier.settings, key)
                    for key in ("quality", "mass", "tension_stiffness", "compression_stiffness", "shear_stiffness", "bending_stiffness", "vertex_group_mass")
                }
                row["collision_settings"] = {
                    key: getattr(modifier.collision_settings, key)
                    for key in ("use_collision", "use_self_collision", "distance_min", "self_distance_min", "collision_quality")
                }
            records.append(row)
        report["objects"] = records
        followers = sorted({driver["object"] for row in links for driver in row["drivers"]})
        visual_records = []
        for index, name in enumerate(followers):
            obj = scene.objects[name]
            trajectory, indices, expected_count = [], None, None
            for frame in samples:
                scene.frame_set(frame)
                positions, indices, count = sampled_world_points(obj, indices)
                if expected_count is None:
                    expected_count = count
                if count != expected_count:
                    raise RuntimeError(f"Visual follower topology changed: {name}")
                trajectory.append(positions)
            arrays[f"visual_{index}"] = np.stack(trajectory)
            arrays[f"visual_indices_{index}"] = indices
            metrics = trajectory_metrics(trajectory, [])
            visual_records.append({"name": name, "evaluated_vertices": expected_count, "sampled_vertices": len(indices), "metrics": metrics})
        report["visual_motion"] = visual_records
        for link in links:
            relevant = [row for row in visual_records if row["name"] in {driver["object"] for driver in link["drivers"]}]
            link["visible_motion"] = any(row["metrics"].get("finite") and row["metrics"].get("bounded") and row["metrics"].get("max_displacement_m", 0) > 1e-5 for row in relevant)
            link["ok"] = link["ok"] and link["visible_motion"]
        np.savez_compressed(output / "control_samples.npz", **arrays)
        # Fit one fixed camera to the union of all sampled asset bounds.
        asset = [
            obj for obj in scene.objects
            if is_asset_object(obj)
        ]
        points = []
        for frame in samples:
            scene.frame_set(frame)
            points.extend(object_points(asset))
        lo, hi = bounds(points)
        center, radius = (lo + hi) / 2, max((hi - lo).length / 2, 0.1)
        camera = scene.camera
        camera.data.lens, camera.data.sensor_width = 48, 36
        camera.data.shift_x = camera.data.shift_y = 0
        front = axis_vector(asset_value(scene, "real2sim_front_axis", "+Y"))
        right = Vector((front.y, -front.x, 0))
        report["gpu_devices"] = configure_cycles(640, 24)
        fit_camera(scene, camera, center, front + right * 0.65 + Vector((0, 0, 0.85)), radius, points)
        for index, frame in enumerate(samples):
            scene.frame_set(frame)
            name = f"sample_{index:02d}_frame_{frame:04d}.png"
            scene.render.filepath = str(output / name)
            bpy.ops.render.render(write_still=True)
            report["images"].append(name)
        scene.frame_set(start)
        bpy.ops.wm.save_as_mainfile(filepath=str(target))
        caches = sorted(output.rglob("*.bphys"))
        report.update(
            ok=all(row["metrics"]["ok"] and row.get("stretch_check", {"ok": True})["ok"] for row in records) and all(row["ok"] for row in links) and bool(caches),
            cache_files=[str(path.relative_to(output)) for path in caches],
            simulation_blend="simulation.blend",
            limitations=[
                "Parameters and hidden structure are hypotheses unless measured independently.",
                "Finite nonrigid motion and cache replay do not prove physical accuracy.",
                "Self-intersections and contact penetration are not certified by this diagnostic; inspect sampled renders.",
                "GLB is a static visual snapshot; replay native deformation using simulation.blend and its cache directory.",
            ],
        )
        if not report["ok"]:
            report["error"] = "Deformation gate failed: inspect per-object metrics and disk-cache presence"
    except BaseException as exc:
        report.update(ok=False, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-8000:])
        raise
    finally:
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        (output / "deformation.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    arguments = sys.argv[sys.argv.index("--") + 1:]
    output = Path(arguments[1]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if arguments[0] == "verify":
        verify_reloaded(output)
    else:
        simulate(output, int(arguments[2]))


if __name__ == "__main__":
    main()
