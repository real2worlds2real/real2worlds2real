"""Native Newton implementation of the retained particle-material review protocol.

Uses the same source volumes, phase progression, surface reconstruction and
reports as the Lab launcher, without importing or altering Isaac Lab/Sim.
"""

# ruff: noqa: E402 -- Select isolated packages before importing the solver.
import argparse
import json
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("configuration", type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--frames", type=int, default=72)
parser.add_argument("--fps", type=int, default=24)
parser.add_argument("--gravity", type=float, nargs=3)
parser.add_argument("--native-runtime", type=Path)
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()
if args.native_runtime:
    sys.path.insert(0, str(args.native_runtime.resolve()))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import newton
import numpy as np
import warp as wp
from newton.geometry import ParticleSurface
from newton.solvers import SolverImplicitMPM
from scipy.spatial import ConvexHull, cKDTree

from real2sim_harness.artifacts import atomic_json, sha256_file
from real2sim_harness.fluid_environment import audit_static_boxes, static_colliders
from real2sim_harness.material_transition import transition_progress, validate_transition
from real2sim_harness.scene_fluid import validate_configuration


def statistics(points):
    if not np.isfinite(points).all():
        raise RuntimeError("Non-finite particle positions after MPM stepping")
    area = float(ConvexHull(points[:, :2]).volume) if len(points) > 3 else 0.0
    return {"footprint_m2": area, "min_z_m": float(points[:, 2].min()),
            "height_p99_m": float(np.quantile(points[:, 2], 0.99)),
            "center_m": points.mean(axis=0).tolist(), "finite": True}


def main():
    if not 2 <= args.frames <= 1800 or args.fps <= 0 or 120 % args.fps:
        raise ValueError("Use 2..1800 frames and a positive FPS dividing 120")
    started = time.monotonic()
    configuration = validate_configuration(json.loads(args.configuration.read_text()))
    boxes = static_colliders(configuration)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    surfaces = output / "surfaces"
    surfaces.mkdir(exist_ok=True)
    atomic_json(output / "fluid_configuration.json", configuration)
    numerical = configuration["solver"]
    gravity = tuple(args.gravity) if args.gravity else (0, 0, -9.81)
    if not np.isfinite(gravity).all():
        raise ValueError("Gravity must be finite")
    wp.config.kernel_cache_dir = str(output.parents[0] / ".warp-cache")
    wp.init()
    wp.set_device(args.device)
    builder = newton.ModelBuilder(gravity=gravity)
    SolverImplicitMPM.register_custom_attributes(builder)
    if configuration.get("use_ground_plane", True):
        builder.add_ground_plane(height=float(configuration.get("floor_z", 0)))
    for box in boxes:
        builder.add_shape_box(body=-1,
            xform=wp.transform(wp.vec3(*box["center_m"]), wp.quat_identity()),
            hx=box["size_m"][0]/2, hy=box["size_m"][1]/2, hz=box["size_m"][2]/2)
    particle_ids = []
    for part in configuration["parts"]:
        positions = np.asarray(part["positions"], dtype=np.float32)
        spacing = float(part["spacing"])
        density = part["material"].get("density", 1000.0)
        if density <= 0:
            raise ValueError("Fluid density must be positive")
        particle_ids.append(np.arange(builder.particle_count, builder.particle_count + len(positions)))
        builder.add_particles(pos=positions.tolist(), vel=np.zeros_like(positions).tolist(),
            mass=[density * spacing**3] * len(positions), radius=[spacing*.5] * len(positions))
    model = builder.finalize()
    fields = set().union(*(set(p["material"]) for p in configuration["parts"])) - {"density"}
    for name in fields:
        target = getattr(model.mpm, name, None)
        if target is None or not hasattr(target, "numpy"):
            raise ValueError(f"Unsupported native MPM material field: {name}")
        values = target.numpy().copy()
        for part, ids in zip(configuration["parts"], particle_ids, strict=True):
            if name in part["material"]:
                values[ids] = part["material"][name]
        target.assign(values)
    options = SolverImplicitMPM.Config()
    options.voxel_size = numerical["voxel_size"]
    options.max_iterations = numerical.get("max_iterations", 60)
    options.tolerance = 0.0001
    options.grid_type = "sparse"
    options.grid_padding = 0
    options.warmstart_mode = "particles"
    options.collider_basis = "pic"
    solver = SolverImplicitMPM(model, config=options)
    state, next_state = model.state(), model.state()
    state.mpm.particle_Jp.fill_(1.0)
    next_state.mpm.particle_Jp.fill_(1.0)
    transitions = [validate_transition(p["transition"]) if p.get("transition") else None for p in configuration["parts"]]
    progress = [np.zeros(len(p["positions"]), dtype=np.float32) for p in configuration["parts"]]
    phase_history = [[] for _ in particle_ids]
    changing_fields = set().union(*(set(t.get("material", {})) for t in transitions if t))
    original_materials = {name: getattr(model.mpm, name).numpy().copy() for name in changing_fields}
    reconstructors = []
    trajectories = [[] for _ in particle_ids]
    history = [[] for _ in particle_ids]
    for part in configuration["parts"]:
        spacing = float(part["spacing"])
        reconstructors.append(ParticleSurface(
            voxel_size=numerical.get("surface_voxel_size", spacing * 0.55),
            kernel_radius=spacing * 3.5, threshold=0.25,
            anisotropic=True, anisotropy_ratio=8.0, anisotropy_min_neighbors=8,
            anisotropy_scale=1.4, kernel_scale=0.6, field_smooth_iterations=1,
            mesh_smooth_iterations=2, device=args.device,
        ))
    if 120 % args.fps:
        raise ValueError("Output FPS must divide the 120 Hz base physics rate")
    substeps = (120 // args.fps) * numerical.get("substeps", 2)
    dt = 1 / (args.fps * substeps)
    frame_records = []
    for frame in range(args.frames):
        if any(transitions):
            material_arrays = {name: values.copy() for name, values in original_materials.items()}
            for i, specification in enumerate(transitions):
                if not specification:
                    continue
                positions = state.particle_q.numpy()[particle_ids[i]]
                progress[i] = transition_progress(positions, frame/args.fps, specification, progress[i]).astype(np.float32)
                for name, target in specification.get("material", {}).items():
                    ids = particle_ids[i]
                    material_arrays[name][ids] = original_materials[name][ids]*(1-progress[i])+target*progress[i]
            for name, values in material_arrays.items():
                getattr(model.mpm, name).assign(values)
            if material_arrays:
                from newton import ModelFlags
                solver.notify_model_changed(ModelFlags.MODEL_PROPERTIES)
        if frame:
            for _ in range(substeps):
                state.clear_forces()
                solver.step(state, next_state, None, None, dt)
                solver.project_outside(next_state, next_state, dt)
                state, next_state = next_state, state
        arrays = {}
        counts = []
        for index, surface in enumerate(reconstructors):
            positions = state.particle_q.numpy()[particle_ids[index]]
            points = wp.array(positions, dtype=wp.vec3, device=args.device)
            stats = statistics(positions)
            if not stats["finite"]:
                raise RuntimeError("Non-finite particle state")
            radii = wp.full(len(positions), value=configuration["parts"][index]["spacing"] * 0.5,
                            dtype=wp.float32, device=args.device)
            mesh = surface.extract(points, radii)
            vertices, indices, normals = mesh.to_arrays()
            if vertices is None:
                raise RuntimeError(f"Empty extracted surface for material {index}")
            key = f"part_{index:03d}"
            arrays[f"{key}_vertices"] = vertices.numpy()
            arrays[f"{key}_faces"] = indices.numpy().reshape(-1, 3)
            arrays[f"{key}_normals"] = normals.numpy()
            if transitions[index]:
                # Transfer the same irreversible particle state to the reconstructed surface.
                distance, ids = cKDTree(positions).query(arrays[f"{key}_vertices"], k=min(4, len(positions)))
                weights = 1/np.maximum(distance, 1e-7)
                arrays[f"{key}_transition"] = (np.sum(progress[index][ids]*weights, axis=1)/weights.sum(axis=1)).astype(np.float32)
            phase_history[index].append(progress[index].copy())
            counts.append(len(arrays[f"{key}_faces"]))
            trajectories[index].append(positions.copy())
            history[index].append(stats)
        np.savez_compressed(surfaces / f"frame_{frame:04d}.npz", **arrays)
        frame_records.append({"frame": frame, "time_seconds": frame / args.fps, "triangles": counts})
        print(f"MATERIAL_FRAME {frame + 1}/{args.frames} triangles={counts}", flush=True)
    np.savez_compressed(output / "particle_motion.npz", **{
        f"part_{i:03d}": np.stack(values) for i, values in enumerate(trajectories)
    }, **{f"part_{i:03d}_transition": np.stack(phase_history[i]) for i, t in enumerate(transitions) if t})
    parts = []
    for i, part in enumerate(configuration["parts"]):
        first, last = history[i][0], history[i][-1]
        parts.append({"source_object": part["source_object"], "particle_count": len(part["positions"]),
                      "initial": first, "final": last,
                      "footprint_ratio": last["footprint_m2"] / max(first["footprint_m2"], 1e-12),
                      "history": history[i]})
        if transitions[i]:
            parts[-1]["transition"] = {"model": "prescribed irreversible phase progression; no heat PDE or chemical kinetics",
                                       "specification": transitions[i], "final_fraction_mean": float(progress[i].mean()),
                                       "final_native_material_ranges": {name: [float(getattr(model.mpm, name).numpy()[particle_ids[i]].min()), float(getattr(model.mpm, name).numpy()[particle_ids[i]].max())] for name in transitions[i].get("material", {})}}
    static_check=audit_static_boxes(np.concatenate([np.asarray(values).reshape(-1,3) for values in trajectories]),boxes,
                                   tolerance_m=numerical['voxel_size']*.5)
    static_check['native_shape_count']=int(model.shape_count)
    static_check['ok'] = bool(static_check['ok'] and model.shape_count>=len(boxes))
    result = {
        "ok": all(row["final"]["finite"] for row in parts) and static_check['ok'], "backend": "Newton implicit MPM",
        "runtime": {'name':'Newton implicit MPM','version':newton.__version__},
        "configuration_sha256": sha256_file(output / "fluid_configuration.json"),
        "source_scene_sha256": configuration.get("source_scene_sha256"),
        "frames": args.frames, "fps": args.fps, "duration_seconds": args.frames / args.fps,
        "gravity_m_s2": list(gravity),
        "static_environment": {'boxes':boxes,'ground_plane':configuration.get('use_ground_plane',True),
            'scope':'Finite fixed axis-aligned boxes; no dynamic rigid-body two-way coupling is claimed'},
        "static_environment_check":static_check,
        "parts": parts, "frame_records": frame_records,
        "compute_seconds": round(time.monotonic() - started, 3),
        "launcher": "native Newton; no Isaac Lab",
        "physics_dt_seconds": dt,
        "device": str(model.device),
        "warp_version": wp.__version__,
        "scope": "visual and operability demonstration using assumed editable material parameters",
    }
    atomic_json(output / "fluid_report.json", result)
    status = 'MATERIAL_SIMULATION_OK' if result['ok'] else 'MATERIAL_SIMULATION_FAILED'
    print(status + ' ' + json.dumps({"parts": len(parts), "seconds": result["compute_seconds"]}), flush=True)
    if not result['ok']:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
