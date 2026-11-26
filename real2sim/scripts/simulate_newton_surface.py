"""Load an accepted Blender cloth mesh into Newton VBD and step it."""

# ruff: noqa: E402 -- Isaac runtime imports intentionally follow launcher arguments.

import argparse
import json
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("usd", help="Path to authored surface USD")
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--pin-mode", choices=("source", "none"))
parser.add_argument("--lift-distance", type=float, default=0.0)
parser.add_argument("--lift-seconds", type=float, help="Optional lift duration; remaining time holds the grasp still")
parser.add_argument("--lift-delay-seconds", type=float, default=0.0)
parser.add_argument("--quiet-start-seconds", type=float)
parser.add_argument("--quiet-speed-tolerance", type=float)
parser.add_argument("--report", type=Path)
parser.add_argument("--trajectory", type=Path)
parser.add_argument("--substeps", type=int)
parser.add_argument("--iterations", type=int)
parser.add_argument("--self-contact", action=argparse.BooleanOptionalAction, default=None)
parser.add_argument("--snapshot", type=Path)
add_launcher_args(parser)
args = parser.parse_args()
args.physics = "newton_vbd"

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.assets import AssetBaseCfg, DeformableObjectCfg
from isaaclab.physics import PhysicsCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.utils import get_current_stage
from isaaclab.utils import configclass, instantiate
from isaaclab_newton.physics import NewtonSoftContactCfg
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import NewtonSurfaceDeformableBodyMaterialCfg
from pxr import Usd, UsdGeom

from real2sim_harness.artifacts import atomic_json, sha256_file
from real2sim_harness.motion_quality import surface_trajectory_metrics
from real2sim_harness.sim_asset_loader import (
    asset_configs,
    bind_surface_visuals,
    configure_surface_environment,
)
from real2sim_harness.sim_parameters import load_parameters, resolve_surface_contact
from real2sim_harness.simulation import check_surface_motion


@configclass
class SurfaceSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())
    visuals = AssetBaseCfg(prim_path="/World/Visuals", spawn=sim_utils.UsdFileCfg(usd_path=""))
    cloth = DeformableObjectCfg(
        prim_path="/World/Asset",
        spawn=sim_utils.UsdFileCfg(
            usd_path="",
            deformable_props=NewtonDeformableBodyPropertiesCfg(),
            physics_material=NewtonSurfaceDeformableBodyMaterialCfg(),
        ),
    )


def main():
    if args.lift_delay_seconds < 0 or (args.lift_seconds is not None and args.lift_seconds <= 0):
        raise ValueError("Lift duration must be positive and lift delay nonnegative")
    if args.quiet_start_seconds is not None and args.lift_distance:
        if args.lift_seconds is None or args.quiet_start_seconds < args.lift_delay_seconds + args.lift_seconds:
            raise ValueError("A quiet-hold check must begin after the prescribed lift has stopped")
    parameters = load_parameters(Path(args.usd).parent, create=True)
    args.substeps = args.substeps if args.substeps is not None else parameters["solver"]["substeps"]
    args.iterations = args.iterations if args.iterations is not None else parameters["solver"]["iterations"]
    args.self_contact = args.self_contact if args.self_contact is not None else parameters["solver"]["self_contact"]
    args.pin_mode = args.pin_mode or ("source" if parameters["constraints"]["use_source_pins"] else "none")
    source = Usd.Stage.Open(args.usd)
    mesh = UsdGeom.Mesh.Get(source, "/Asset/SimMesh")
    pins = list(mesh.GetPrim().GetAttribute("remo:pinIndices").Get() or [])
    faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    source_points = np.asarray(mesh.GetPointsAttr().Get())
    mesh_data = json.loads((Path(args.usd).parent / "sim_mesh.json").read_text())
    edge_lengths = np.linalg.norm(source_points[edges[:, 0]]-source_points[edges[:, 1]], axis=-1)
    contact = resolve_surface_contact(parameters, source_physics=mesh_data.get("source_physics", []),
                                      median_edge_length=float(np.median(edge_lengths)))
    if args.pin_mode == "none":
        pins = []
    if args.lift_distance and not pins:
        raise ValueError("Lifting requires source pin indices")
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args) as physics_cfg:
        physics_cfg.num_substeps = args.substeps
        physics_cfg.solver_cfg.iterations = args.iterations
        physics_cfg.solver_cfg.particle_enable_self_contact = args.self_contact
        contact_radius = contact["interaction_distance_m"]
        contact_margin = contact["query_distance_m"]
        if contact_margin < contact_radius:
            raise ValueError("Self-contact detection margin must cover its interaction radius")
        physics_cfg.solver_cfg.particle_self_contact_radius = contact_radius
        physics_cfg.solver_cfg.particle_self_contact_margin = contact_margin
        physics_cfg.solver_cfg.particle_topological_contact_filter_threshold = parameters["solver"]["self_contact_topology_distance"]
        physics_cfg.solver_cfg.particle_vertex_contact_buffer_size = parameters["solver"]["self_contact_vertex_capacity"]
        physics_cfg.solver_cfg.particle_edge_contact_buffer_size = parameters["solver"]["self_contact_edge_capacity"]
        if physics_cfg.soft_contact_cfg is None:
            physics_cfg.soft_contact_cfg = NewtonSoftContactCfg()
        physics_cfg.soft_contact_cfg.soft_contact_mu = parameters["material"]["friction"]
        sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args.device, physics=physics_cfg,
                                          gravity=tuple(parameters["environment"]["gravity"]))
        sim = sim_utils.SimulationContext(sim_cfg)
        scene_cfg = SurfaceSceneCfg(num_envs=1, env_spacing=0.0)
        folder = Path(args.usd).parent
        entries = asset_configs(folder, physics_path="/World/Asset", visual_path="/World/Visuals")
        scene_cfg.cloth = entries["asset"]
        has_binding = (folder / "surface_binding.json").is_file()
        if has_binding:
            scene_cfg.visuals = entries["visuals"]
        else:
            scene_cfg.visuals = None
        scene = instantiate(scene_cfg)
        sim.reset()
        air_drag = configure_surface_environment(folder)
        cloth = scene["cloth"]
        visual_binding = None
        if has_binding:
            stage = get_current_stage()
            visual_binding = bind_surface_visuals(folder, stage, physics_path="/World/Asset", visual_path="/World/Visuals")
        initial = cloth.data.nodal_pos_w.torch.clone()
        source_points = np.asarray(mesh.GetPointsAttr().Get())
        if initial.shape[-2] != len(source_points):
            raise ValueError("Imported Newton node count differs from the source control mesh")
        correspondence_error = float(np.linalg.norm(initial[0].cpu().numpy() - source_points, axis=-1).max())
        if correspondence_error > 1e-5:
            raise ValueError("Newton rest-node ordering does not match exported pins and visual binding")
        if any(index < 0 or index >= initial.shape[-2] for index in pins):
            raise ValueError("Source pin indices do not match imported Newton topology")
        moving_pins = pins
        lift_selection = parameters["constraints"]["lift_selection"]
        if pins and lift_selection != "all":
            side, axis_name = lift_selection.split("_")
            axis = "xyz".index(axis_name)
            values = initial[0, pins, axis].cpu().numpy()
            midpoint = (values.min()+values.max())/2
            selected = values > midpoint if side == "positive" else values < midpoint
            moving_pins = [index for index, active in zip(pins, selected, strict=True) if active]
            if not moving_pins:
                raise ValueError("The selected source-pin side has no moving vertices")
        edge_ids = torch.tensor(edges, dtype=torch.long, device=args.device)
        rest_lengths = torch.linalg.vector_norm(initial[:, edge_ids[:, 0]] - initial[:, edge_ids[:, 1]], dim=-1)
        valid_edges = rest_lengths > 1e-7
        kinematic = torch.ones((*initial.shape[:-1], 4), dtype=initial.dtype, device=args.device)
        kinematic[..., :3] = initial
        kinematic[:, pins, 3] = 0.0
        frames = [initial[0].cpu().numpy()]
        times = [0.0]
        max_pin_error = 0.0
        min_z = float(initial[..., 2].min())
        finite = True
        print(f"NEWTON_ASSET_LOADED instances={cloth.num_instances} nodes={initial.shape[-2]}")
        for step in range(args.steps):
            duration = args.lift_seconds if args.lift_seconds is not None else args.steps * sim_cfg.dt
            progress = min(1.0, max(0.0, ((step + 1) * sim_cfg.dt - args.lift_delay_seconds) / duration))
            lift = args.lift_distance * (3 * progress**2 - 2 * progress**3)
            kinematic[:, moving_pins, 2] = initial[:, moving_pins, 2] + lift
            cloth.write_nodal_kinematic_target_to_sim_index(kinematic)
            cloth.write_data_to_sim()
            sim.step()
            cloth.update(sim_cfg.dt)
            positions = cloth.data.nodal_pos_w.torch
            finite = finite and bool(torch.isfinite(positions).all())
            min_z = min(min_z, float(positions[..., 2].min()))
            if pins:
                error = torch.linalg.vector_norm(positions[:, pins] - kinematic[:, pins, :3], dim=-1).max()
                max_pin_error = max(max_pin_error, float(error))
            if step % 4 == 3 or step == args.steps - 1:
                frames.append(positions[0].cpu().numpy())
                times.append((step + 1) * sim_cfg.dt)
                if visual_binding:
                    visual_binding.update(frames[-1])
            if step % 60 == 59:
                print(f"NEWTON_PROGRESS step={step+1}/{args.steps}", flush=True)
        final = cloth.data.nodal_pos_w.torch.clone()
        final_lengths = torch.linalg.vector_norm(final[:, edge_ids[:, 0]] - final[:, edge_ids[:, 1]], dim=-1)
        ratios = final_lengths[valid_edges] / rest_lengths[valid_edges]
        result = check_surface_motion(
            finite=finite, min_z=min_z, ground_z=0.0,
            pin_count=len(pins), pin_error_max=max_pin_error,
            displacement_max=float(torch.linalg.vector_norm(final - initial, dim=-1).max()),
            edge_ratio_p95=float(torch.quantile(ratios, 0.95)), edge_ratio_min=float(ratios.min()),
        )
        temporal = (surface_trajectory_metrics(frames, times, quiet_start=args.quiet_start_seconds,
                                              speed_tolerance=args.quiet_speed_tolerance)
                    if finite else {"ok": False, "failure": "nonfinite_trajectory"})
        if temporal.get("quiet_hold", {}).get("ok") is False:
            result["ok"] = False
            result["failures"].append("unforced_hold_did_not_settle")
        report = {
            "schema_version": 1, "backend": "newton_vbd", "asset_kind": "deformable",
            "ok": result["ok"], "usd": args.usd,
            "usd_sha256": sha256_file(Path(args.usd)), "nodes": initial.shape[-2],
            "steps": args.steps, "dt": sim_cfg.dt, "lift_distance_m": args.lift_distance,
            "lift_seconds": args.lift_seconds,
            "lift_delay_seconds": args.lift_delay_seconds,
            "temporal_quality": temporal,
            "substeps": args.substeps, "solver_iterations": args.iterations,
            "self_contact": args.self_contact, "robot_interaction_tested": False,
            "self_contact_interaction_distance_m": contact_radius,
            "self_contact_query_distance_m": contact_margin,
            "self_contact_initialization": contact,
            "gravity_m_s2": parameters["environment"]["gravity"],
            "air_drag": air_drag,
            "physics_parameters_sha256": sha256_file(Path(args.usd).parent / "physics_parameters.json"),
            "pin_mode": args.pin_mode,
            "lift_selection": lift_selection, "moving_pin_count": len(moving_pins),
            "rest_node_correspondence_error_m": correspondence_error,
            "visual_binding": {"ok": has_binding, "mapped_vertices": visual_binding.report["mapped_vertices"] if visual_binding else 0},
            "checks": result,
        }
        output = args.report or Path(args.usd).with_name("newton_validation.json")
        atomic_json(output, report)
        if args.trajectory:
            args.trajectory.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.trajectory, positions=np.stack(frames), faces=faces,
                                source_usd_sha256=report["usd_sha256"], times=np.asarray(times))
        if args.snapshot:
            from isaac_export_evidence import save_stage
            save_stage(get_current_stage(), args.snapshot)
        print(
            f"NEWTON_ASSET_SIM_OK steps={args.steps} nodes={initial.shape[-2]} "
            f"mean_z_before={initial[..., 2].mean().item():.6f} "
            f"mean_z_after={final[..., 2].mean().item():.6f}"
        )
        print("NEWTON_ASSET_VALIDATION " + json.dumps(report), flush=True)
        if not report["ok"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
