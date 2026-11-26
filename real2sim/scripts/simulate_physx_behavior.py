"""Record real PhysX joint or rigid-body behavior for material-preserving replay."""

# ruff: noqa: E402 -- runtime imports follow the Isaac launcher setup.

import argparse
import json
from pathlib import Path

from isaac_physx_compat import runtime

parser = argparse.ArgumentParser()
parser.add_argument("folder", type=Path)
parser.add_argument("--plan", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
runtime.add_launcher_args(parser)
args = parser.parse_args()
args.physics = "isaacsim_physx"

runtime.prepare(args)

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sim.utils import get_current_stage

from real2sim_harness.artifacts import atomic_json, sha256_file
from real2sim_harness.behavior_plan import resolve_controls, sample_keyframes
from real2sim_harness.sim_asset_loader import asset_configs


def main():
    manifest = json.loads((args.folder / "manifest.json").read_text())
    if manifest["asset_kind"] not in {"rigid", "articulated"}:
        raise ValueError("This behavior recorder supports rigid and articulated PhysX assets")
    plan = json.loads(args.plan.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    articulated = manifest["asset_kind"] == "articulated"
    fps, duration = int(plan.get("fps", 24)), float(plan.get("duration", 7))
    if fps not in {24, 30, 60} or not 0 < duration <= 30:
        raise ValueError("Choose a bounded duration and 24/30/60 FPS")
    frames, times, coordinates = [], [], []
    with runtime.launch(args) as physics_cfg:
        from isaac_physx_settings import enable_requested_ccd
        sim_cfg = runtime.simulation_cfg(physics_cfg, dt=1/120, device=args.device)
        sim = sim_utils.SimulationContext(sim_cfg)
        runtime.ground("/World/Ground", float(plan.get("ground_z", 0)))
        cfg = asset_configs(args.folder, physics_path="/World/Asset")["asset"]
        cfg.init_state.pos = (0, 0, float(plan.get("root_height", 0)))
        if articulated:
            cfg.actuators = {"behavior_test": runtime.actuator_cfg(joint_names_expr=[".*"],
                stiffness=float(plan.get("stiffness", 100)), damping=float(plan.get("damping", 10)), joint_effort_limit=1000.)}
        asset = Articulation(cfg) if articulated else RigidObject(cfg)
        if articulated:
            runtime.prepare_articulation(asset)
        enable_requested_ccd(get_current_stage())
        sim.reset()
        asset.update(sim_cfg.dt)
        body_ids = {"__base__": 0}
        rest = {"__base__": np.eye(4).tolist()}
        rest.update({row["name"]: row["matrix_world"] for row in manifest["joints"]})
        if articulated:
            stage = get_current_stage()
            paths = runtime.body_paths(asset)
            for index, path in enumerate(paths):
                prim = stage.GetPrimAtPath(path)
                attr = prim.GetAttribute("userProperties:blender:object_name")
                if attr and attr.Get() in rest:
                    body_ids[str(attr.Get())] = index
                elif path.endswith("/BaseLink"):
                    body_ids["__base__"] = index
            if set(rest) != set(body_ids):
                raise ValueError("Simulator body order could not be mapped back to source links")
            controls = resolve_controls(plan, asset.joint_names, runtime.tensor(asset.data.joint_pos_limits)[0].cpu().numpy())
            if not controls:
                raise ValueError("An articulation behavior needs a joint control schedule")
            initial = runtime.tensor(asset.data.joint_pos).clone()

            def command(time):
                target = initial.clone()
                for record in controls:
                    target[0, record["index"]] = sample_keyframes(record["keyframes"], time)
                runtime.set_position(asset, target)
                return target

            for _ in range(round(float(plan.get("warmup_seconds", 2))*120)):
                command(0)
                asset.write_data_to_sim()
                sim.step()
                asset.update(sim_cfg.dt)
        else:
            controls = []
        finite = True
        for step in range(round(duration*120)):
            time = step*sim_cfg.dt
            if step % (120//fps) == 0:
                poses = runtime.pose_xyzw(asset.data.body_link_pose_w)[0] if articulated else runtime.pose_xyzw(asset.data.root_link_pose_w)
                frames.append(poses.cpu().numpy().copy())
                times.append(time)
                if articulated:
                    coordinates.append(runtime.tensor(asset.data.joint_pos)[0].cpu().numpy().copy())
            if articulated:
                command(time)
            asset.write_data_to_sim()
            sim.step()
            asset.update(sim_cfg.dt)
            finite = finite and bool(torch.isfinite(runtime.pose_xyzw(asset.data.root_link_pose_w)).all())
        failures = []
        if not finite or not np.isfinite(frames).all():
            failures.append("nonfinite_state")
        joint_checks = []
        if articulated:
            for record in controls:
                values = np.asarray(coordinates)[:, record["index"]]
                requested = np.ptp(np.asarray(record["keyframes"])[:, 1])
                error = abs(float(values[-1])-sample_keyframes(record["keyframes"], times[-1]))
                ok = np.ptp(values) > min(0.01, requested*.2) and error < .05
                joint_checks.append({"joint": record["joint"], "ok": bool(ok), "actual_span_rad": float(np.ptp(values)), "final_tracking_error_rad": error})
                if not ok:
                    failures.append(f"joint_control_failed:{record['joint']}")
        else:
            displacement = float(np.linalg.norm(frames[-1][0, :3]-frames[0][0, :3]))
            if displacement < 0.02:
                failures.append("rigid_body_did_not_move")
        np.savez_compressed(args.output / "behavior_motion.npz", bodies=np.asarray(frames), times=np.asarray(times), joints=np.asarray(coordinates))
        report = {"ok": not failures, "failures": failures, "backend": "physx", "asset_kind": manifest["asset_kind"],
            "fps": fps, "frames": len(frames), "duration_seconds": len(frames)/fps,
            "requested_duration_seconds": duration, "body_ids": body_ids,
            "rest_matrices": rest, "source_scene_sha256": manifest["source_blend_sha256"],
            "source_blend": manifest["source_blend"], "joint_checks": joint_checks, "controls": controls,
            "physics_usd_sha256": sha256_file(args.folder / "physx.usda"), "control_plan": plan,
            "quaternion_order": "xyzw", "actuation": "temporary physical test drives; delivered passive asset unchanged",
            "real_physics_calibrated": False, "robot_task_implemented": False}
        atomic_json(args.output / "behavior_report.json", report)
        print("PHYSX_BEHAVIOR", report["ok"], report["joint_checks"], flush=True)
        raise SystemExit(0 if report["ok"] else 2)


if __name__ == "__main__":
    main()
