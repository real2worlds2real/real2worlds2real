"""Advance generic Blender-authored particle materials using Newton implicit MPM."""

# ruff: noqa: E402 -- The launcher resolves the Newton runtime first.
import argparse
import json
import time
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("configuration", type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--frames", type=int, default=72)
parser.add_argument("--fps", type=int, default=24)
parser.add_argument("--gravity", type=float, nargs=3, help="Optional explicit gravity vector for a matched diagnostic")
add_launcher_args(parser)
args = parser.parse_args()

import isaaclab.sim as sim_utils
import newton
import numpy as np
import warp as wp
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass, instantiate
from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.physics import MPMSolverCfg, NewtonCfg, NewtonManager
from isaaclab_newton.sim.spawners.mpm import MPMParticleMaterialCfg, MPMPointsCfg
from newton.geometry import ParticleSurface
from scipy.spatial import ConvexHull, cKDTree

from real2sim_harness.artifacts import atomic_json, sha256_file
from real2sim_harness.fluid_environment import audit_static_boxes, static_colliders
from real2sim_harness.material_transition import transition_progress, validate_transition


@configclass
class MaterialsSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())


def statistics(points):
    if not np.isfinite(points).all():
        raise RuntimeError("Non-finite particle positions after MPM stepping")
    area = float(ConvexHull(points[:, :2]).volume) if len(points) > 3 else 0.0
    return {"footprint_m2": area, "min_z_m": float(points[:, 2].min()),
            "height_p99_m": float(np.quantile(points[:, 2], 0.99)),
            "center_m": points.mean(axis=0).tolist(), "finite": bool(np.isfinite(points).all())}


def main():
    started = time.monotonic()
    configuration = json.loads(args.configuration.read_text())
    boxes = static_colliders(configuration)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    surfaces = output / "surfaces"
    surfaces.mkdir(exist_ok=True)
    atomic_json(output / "fluid_configuration.json", configuration)
    numerical = configuration["solver"]
    physics = NewtonCfg(
        solver_cfg=MPMSolverCfg(
            voxel_size=numerical["voxel_size"], max_iterations=numerical.get("max_iterations", 60),
            tolerance=0.0001, grid_type="sparse", grid_padding=0, warmstart_mode="particles",
            project_outside_colliders=True, collider_basis="pic",
        ),
        num_substeps=numerical.get("substeps", 2),
    )
    with launch_simulation(physics, args) as physics_cfg:
        sim_cfg = sim_utils.SimulationCfg(dt=1 / 120, device=args.device, physics=physics_cfg,
                                          gravity=tuple(args.gravity) if args.gravity else (0, 0, -9.81))
        sim = sim_utils.SimulationContext(sim_cfg)
        scene_cfg = MaterialsSceneCfg(num_envs=1, env_spacing=0.0)
        if configuration.get('use_ground_plane',True):
            scene_cfg.ground.init_state.pos = (0, 0, float(configuration.get("floor_z", 0.0)))
        else:
            scene_cfg.ground = None
        if boxes:
            from isaaclab_newton.sim.schemas import NewtonCollisionPropertiesCfg
            for index,box in enumerate(boxes):
                entry=AssetBaseCfg(prim_path=f'/World/StaticBox_{index:03d}',spawn=sim_utils.CuboidCfg(
                    size=tuple(box['size_m']), collision_props=NewtonCollisionPropertiesCfg(collision_enabled=True)))
                entry.init_state.pos=tuple(box['center_m'])
                setattr(scene_cfg,f'static_box_{index:03d}',entry)
        for index, part in enumerate(configuration["parts"]):
            spacing = float(part["spacing"])
            material = MPMParticleMaterialCfg(**part["material"])
            setattr(scene_cfg, f"medium_{index:03d}", MPMObjectCfg(
                prim_path=f"/World/Medium_{index:03d}",
                spawn=MPMPointsCfg(
                    positions=part["positions"], mass=material.density * spacing**3,
                    radius=spacing * 0.5, material=material,
                    visual_color=part.get("color", [0.7, 0.7, 0.7]),
                ),
            ))
        scene = instantiate(scene_cfg)
        sim.reset()
        state = NewtonManager.get_state_0()
        print("MPM_INITIAL_JP", state.mpm.particle_Jp.numpy().min(), state.mpm.particle_Jp.numpy().max(), flush=True)
        state.mpm.particle_Jp.fill_(1.0)
        model = NewtonManager.get_model()
        print("MPM_MATERIALS", {name: [float(getattr(model.mpm, name).numpy().min()), float(getattr(model.mpm, name).numpy().max())]
              for name in ("young_modulus", "viscosity", "friction", "tensile_yield_ratio")}, flush=True)
        assets = [scene[f"medium_{i:03d}"] for i in range(len(configuration["parts"]))]
        transitions = [validate_transition(p["transition"]) if p.get("transition") else None for p in configuration["parts"]]
        progress = [np.zeros(len(p["positions"]), dtype=np.float32) for p in configuration["parts"]]
        phase_history = [[] for _ in assets]
        global_positions = state.particle_q.numpy()
        lookup = cKDTree(global_positions)
        particle_ids = []
        for asset in assets:
            distance, ids = lookup.query(asset.data.particle_pos_w.torch[0].cpu().numpy())
            if distance.max() > 1e-6 or len(np.unique(ids)) != len(ids):
                raise ValueError("Cannot map material-phase particles to native Newton storage")
            particle_ids.append(ids)
        if len(np.unique(np.concatenate(particle_ids))) != sum(len(ids) for ids in particle_ids):
            raise ValueError("Material phases map to overlapping particle ownership")
        changing_fields = set().union(*(set(t.get("material", {})) for t in transitions if t))
        original_materials = {name: getattr(model.mpm, name).numpy().copy() for name in changing_fields}
        reconstructors = []
        trajectories = [[] for _ in assets]
        history = [[] for _ in assets]
        for part in configuration["parts"]:
            spacing = float(part["spacing"])
            reconstructors.append(ParticleSurface(
                voxel_size=numerical.get("surface_voxel_size", spacing * 0.55),
                kernel_radius=spacing * 3.5, threshold=0.25,
                anisotropic=True, anisotropy_ratio=8.0, anisotropy_min_neighbors=8,
                anisotropy_scale=1.4, kernel_scale=0.6, field_smooth_iterations=1,
                mesh_smooth_iterations=2, device=args.device,
            ))
        substeps = round(1 / (args.fps * sim_cfg.dt))
        if abs(substeps * sim_cfg.dt * args.fps - 1) > 1e-6:
            raise ValueError("Output FPS must divide the 120 Hz physics rate")
        frame_records = []
        for frame in range(args.frames):
            if any(transitions):
                material_arrays = {name: values.copy() for name, values in original_materials.items()}
                for i, specification in enumerate(transitions):
                    if not specification:
                        continue
                    positions = assets[i].data.particle_pos_w.torch[0].cpu().numpy()
                    progress[i] = transition_progress(positions, frame/args.fps, specification, progress[i]).astype(np.float32)
                    for name, target in specification.get("material", {}).items():
                        ids = particle_ids[i]
                        material_arrays[name][ids] = original_materials[name][ids]*(1-progress[i])+target*progress[i]
                for name, values in material_arrays.items():
                    getattr(model.mpm, name).assign(values)
                if material_arrays:
                    from newton import ModelFlags
                    NewtonManager.add_model_change(ModelFlags.MODEL_PROPERTIES)
            if frame:
                for _ in range(substeps):
                    scene.write_data_to_sim()
                    sim.step()
                    scene.update(sim_cfg.dt)
            arrays = {}
            counts = []
            for index, (asset, surface) in enumerate(zip(assets, reconstructors, strict=True)):
                points = asset.data.particle_pos_w.torch[0]
                positions = points.cpu().numpy()
                stats = statistics(positions)
                if not stats["finite"]:
                    raise RuntimeError("Non-finite particle state")
                radii = wp.full(len(positions), value=configuration["parts"][index]["spacing"] * 0.5,
                                dtype=wp.float32, device=args.device)
                mesh = surface.extract(wp.from_torch(points.contiguous(), dtype=wp.vec3), radii)
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
            "gravity_m_s2": list(sim_cfg.gravity),
            "static_environment": {'boxes':boxes,'ground_plane':configuration.get('use_ground_plane',True),
                'scope':'Finite fixed axis-aligned boxes; no dynamic rigid-body two-way coupling is claimed'},
            "static_environment_check":static_check,
            "parts": parts, "frame_records": frame_records,
            "compute_seconds": round(time.monotonic() - started, 3),
            "scope": "visual and operability demonstration using assumed editable material parameters",
        }
        atomic_json(output / "fluid_report.json", result)
        status = 'MATERIAL_SIMULATION_OK' if result['ok'] else 'MATERIAL_SIMULATION_FAILED'
        print(status + ' ' + json.dumps({"parts": len(parts), "seconds": result["compute_seconds"]}), flush=True)
        if not result['ok']:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
