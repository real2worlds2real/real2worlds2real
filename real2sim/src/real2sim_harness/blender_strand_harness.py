"""Replay native rod material frames while retaining editable surface topology/materials."""

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Quaternion, Vector

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(Path(__file__).resolve().parent)]
from blender_final_harness import bounds, configure_cycles, fit_camera, object_points  # noqa: E402
from blender_surface import is_asset_object  # noqa: E402

from real2sim_harness.strand_geometry import bind_vertices, deform_vertices, quaternion_matrices  # noqa: E402


def pose_matrix(p):
    m = Quaternion((p[6], *p[3:6])).to_matrix().to_4x4()
    m.translation = Vector(p[:3])
    return m


def load_player(folder):
    folder = Path(folder)
    c = json.loads((folder / "strand_configuration.json").read_text())
    report = json.loads((folder / "strand_validation.json").read_text())
    if hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest() != c["source_scene_sha256"]:
        raise ValueError("Strand source does not match the simulated configuration")
    data = np.load(folder / "strand_motion.npz", allow_pickle=False)
    for fixture in c["environment"].get("capsule_obstacles", []):
        center = Vector(fixture["center_m"])
        axis = Vector(fixture["axis"])
        bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=fixture["radius_m"],
                                           depth=2 * fixture["half_height_m"], location=center)
        pieces = [bpy.context.object]
        pieces[0].rotation_euler = axis.to_track_quat("Z", "Y").to_euler()
        for sign in (-1, 1):
            bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16,
                radius=fixture["radius_m"], location=center + sign * fixture["half_height_m"] * axis)
            pieces.append(bpy.context.object)
        material = bpy.data.materials.new("Diagnostic contact fixture")
        material.diffuse_color = (.035, .14, .24, 1)
        for i, piece in enumerate(pieces):
            piece.name = f"{fixture['id']}_contact_fixture_{i}"
            piece["real2sim_fixture"] = True
            piece["real2sim_strand_contact_fixture"] = True
            piece.data.materials.append(material)
            for polygon in piece.data.polygons:
                polygon.use_smooth = True
    bindings = []
    maximum_rest_error = 0.0
    for i, s in enumerate(c["strands"]):
        rest_q = np.asarray(report["strands"][i]["rest_quaternions"])
        for name in s["visual_objects"]:
            obj = bpy.data.objects[name]
            if obj.type != "MESH" or obj.data.shape_keys:
                raise ValueError("Use an editable mesh without shape keys for a rod visual")
            points = np.array([obj.matrix_world @ v.co for v in obj.data.vertices])
            binding = bind_vertices(points, s["points"], rest_q)
            rest = deform_vertices(binding, np.asarray(s["points"]), rest_q)
            maximum_rest_error = max(
                maximum_rest_error, float(np.max(np.linalg.norm(rest - points, axis=1)))
            )
            bindings.append((i, obj, binding))
    if maximum_rest_error > 1e-6:
        raise ValueError("Surface binding changed the authored reference geometry")

    def apply(frame):
        for name, index in report["body_ids"].items():
            bpy.data.objects[name].matrix_world = pose_matrix(data["bodies"][frame, index])
        bpy.context.view_layer.update()
        for i, obj, binding in bindings:
            points = deform_vertices(
                binding,
                data[f"strand_{i:03d}_nodes"][frame],
                data[f"strand_{i:03d}_quaternions"][frame],
            )
            if not np.isfinite(points).all():
                raise ValueError("Nonfinite strand replay surface")
            inverse = np.asarray(obj.matrix_world.inverted())
            local = points @ inverse[:3, :3].T + inverse[:3, 3]
            obj.data.vertices.foreach_set("co", local.ravel())
            obj.data.update()
        bpy.context.view_layer.update()

    return apply, data, report, maximum_rest_error


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    p.add_argument("--mode", choices=["preview", "render"], default="preview")
    p.add_argument("--output", type=Path)
    p.add_argument("--resolution", type=int, default=640)
    p.add_argument("--samples", type=int, default=48)
    p.add_argument("--aspect", type=float, default=4 / 3)
    p.add_argument("--view-direction", type=float, nargs=3, default=(.9, -1.4, 1.4))
    args = p.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    output = args.output or args.folder
    output.mkdir(parents=True, exist_ok=True)
    apply, data, report, rest_error = load_player(args.folder)
    selected = np.unique(np.linspace(0, len(data["times"]) - 1, 7).round().astype(int)).tolist()
    objects = [o for o in bpy.context.scene.objects if is_asset_object(o) and not o.hide_render]
    all_points = []
    for f in selected:
        apply(f)
        all_points.extend(object_points(objects))
    # Include the complete physical trajectory, rather than only preview timestamps.
    configuration = report["configuration"]
    for i, strand in enumerate(configuration["strands"]):
        nodes = data[f"strand_{i:03d}_nodes"].reshape(-1, 3)
        padding = strand["radius"] * 2
        lower, upper = nodes.min(axis=0) - padding, nodes.max(axis=0) + padding
        all_points.extend(Vector(p) for p in itertools.product(*zip(lower, upper, strict=True)))
    for body in configuration["bodies"]:
        parent = bpy.data.objects[body["name"]]
        inverse = parent.matrix_world.inverted()
        local = np.array(
            [
                inverse @ (bpy.data.objects[name].matrix_world @ Vector(corner))
                for name in body["visual_objects"]
                for corner in bpy.data.objects[name].bound_box
            ]
        )
        poses = data["bodies"][:, report["body_ids"][body["name"]]]
        world = (
            np.einsum("fij,pj->fpi", quaternion_matrices(poses[:, 3:]), local) + poses[:, None, :3]
        )
        all_points.extend(Vector(p) for p in world.reshape(-1, 3))
    lo, hi = bounds(all_points)
    center = (lo + hi) / 2
    radius = max((hi - lo).length / 2, 0.05)
    scene = bpy.context.scene
    if scene.camera is None:
        camera = bpy.data.objects.new(
            "Real2Sim_StrandCamera", bpy.data.cameras.new("Real2Sim_StrandCamera")
        )
        scene.collection.objects.link(camera)
        scene.camera = camera
    camera = scene.camera
    camera.data.type = "PERSP"
    camera.data.lens = 55
    camera.data.clip_start = 0.001
    configure_cycles(args.resolution, args.samples)
    scene.render.resolution_y = round(args.resolution / args.aspect)
    direction = Vector(args.view_direction)
    if not np.isfinite(args.view_direction).all() or direction.length < 1e-9:
        raise ValueError("Use a finite nonzero replay view direction")
    fit_camera(scene, camera, center, direction, radius, all_points, fill=0.82)
    projected = [world_to_camera_view(scene, camera, p) for p in all_points]
    screen_bounds = [
        min(p.x for p in projected),
        max(p.x for p in projected),
        min(p.y for p in projected),
        max(p.y for p in projected),
    ]
    if (
        min(p.z for p in projected) <= 0
        or screen_bounds[0] < 0
        or screen_bounds[1] > 1
        or screen_bounds[2] < 0
        or screen_bounds[3] > 1
    ):
        raise ValueError("Motion camera does not contain the complete retained asset envelope")
    frames = selected if args.mode == "preview" else range(len(data["times"]))
    target = output if args.mode == "preview" else output / "frames"
    target.mkdir(exist_ok=True)
    images = []
    for number, frame in enumerate(frames):
        apply(frame)
        path = target / (
            f"preview_{number:02d}.png" if args.mode == "preview" else f"frame_{number:04d}.png"
        )
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        images.append(str(path.relative_to(output)))
        print("STRAND_RENDER", number + 1, len(frames), flush=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "simulation.blend"))
    evidence = {
        "ok": True,
        "fresh_process": True,
        "actual_native_rod_motion": True,
        "rest_binding_max_error_m": rest_error,
        "images": images,
        "sample_frames": selected,
        "frames": len(frames),
        "fps": report["fps"],
        "duration_seconds": len(data["times"]) / report["fps"],
        "cache_frames": len(data["times"]),
        "preview_only": args.mode == "preview",
        "motion": "Actual Newton native rod and endpoint-body trajectory",
        "source_scene_sha256": report["source_scene_sha256"],
        "motion_sha256": hashlib.sha256(
            (args.folder / "strand_motion.npz").read_bytes()
        ).hexdigest(),
        "framing": {
            "all_cache_frames": len(data["times"]),
            "projected_bounds": screen_bounds,
            "body_visuals_and_centerline_envelope": True,
        },
    }
    (
        output / ("preview_report.json" if args.mode == "preview" else "animation_report.json")
    ).write_text(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
