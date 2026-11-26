"""Verify actual joint control, isolation, and finite states in PhysX."""

# ruff: noqa: E402 -- Isaac runtime imports intentionally follow launcher arguments.

import argparse
import json
from pathlib import Path

from isaac_physx_compat import runtime

parser = argparse.ArgumentParser()
parser.add_argument("usd")
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--joint")
parser.add_argument("--all-joints", action="store_true")
parser.add_argument("--target-rad", type=float, default=0.5)
parser.add_argument("--torque-pd", action="store_true")
parser.add_argument("--position-drive", action="store_true", help="Native position drive (default)")
parser.add_argument("--report", type=Path)
parser.add_argument("--snapshot", type=Path)
parser.add_argument("--kp", type=float, default=20.0)
parser.add_argument("--kd", type=float, default=2.0)
parser.add_argument("--max-torque", type=float, default=10.0)
runtime.add_launcher_args(parser)
args = parser.parse_args()
args.physics = "isaacsim_physx"

runtime.prepare(args)

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation

from real2sim_harness.artifacts import atomic_json, sha256_file
from real2sim_harness.sim_asset_loader import asset_configs
from real2sim_harness.simulation import check_joint_motion


def main():
    if args.steps < 1 or args.target_rad == 0:
        raise ValueError("Positive steps and a nonzero target are required")
    results = []
    with runtime.launch(args) as physics_cfg:
        from isaac_physx_settings import enable_requested_ccd
        from isaaclab.sim.utils import get_current_stage
        print("PHYSX_REVIEW_STAGE runtime_launched", flush=True)
        sim_cfg = runtime.simulation_cfg(physics_cfg, dt=1 / 120, device=args.device)
        sim = sim_utils.SimulationContext(sim_cfg)
        print("PHYSX_REVIEW_STAGE context_created", flush=True)
        cfg = asset_configs(Path(args.usd).parent, physics_path="/World/Asset")["asset"]
        cfg.actuators = {"all_joints": runtime.actuator_cfg(
                joint_names_expr=[".*"], stiffness=0.0 if args.torque_pd else 400.0,
                damping=0.0 if args.torque_pd else 20.0, joint_effort_limit=1000.0,
            )}
        asset = Articulation(cfg=cfg)
        runtime.prepare_articulation(asset)
        enable_requested_ccd(get_current_stage())
        print("PHYSX_REVIEW_STAGE articulation_created", flush=True)
        sim.reset()
        print("PHYSX_REVIEW_STAGE reset_completed", flush=True)
        asset.update(sim_cfg.dt)
        expected_joints = len(json.loads((Path(args.usd).parent / "manifest.json").read_text())["joints"])
        if asset.num_joints != expected_joints:
            raise RuntimeError(f"Expected {expected_joints} source joints, loaded {asset.num_joints}")
        if not args.torque_pd:
            # Let the installed position drives hold the authored pose before
            # measuring drift. Initial gravity/drive settling is not a commanded
            # response of one joint and can otherwise contaminate other joints.
            rest = runtime.tensor(asset.data.joint_pos).clone()
            for _ in range(120):
                runtime.set_position(asset, rest)
                asset.write_data_to_sim()
                sim.step()
                asset.update(sim_cfg.dt)
        initial = runtime.tensor(asset.data.joint_pos).clone()
        limits = runtime.tensor(asset.data.joint_pos_limits).clone()
        print(f"PHYSX_ASSET_LOADED joints={asset.num_joints} names={asset.joint_names}")
        selected = range(asset.num_joints) if args.all_joints else [
            asset.joint_names.index(args.joint) if args.joint else asset.num_joints - 1
        ]
        for index in selected:
            asset.reset()
            runtime.write_joint_state(asset, initial, torch.zeros_like(initial))
            asset.update(sim_cfg.dt)
            target = initial.clone()
            positive = float(limits[0, index, 1] - initial[0, index])
            negative = float(initial[0, index] - limits[0, index, 0])
            direction = 1.0 if args.target_rad > 0 else -1.0
            room = positive if direction > 0 else negative
            if room < 0.01:
                direction *= -1
                room = positive if direction > 0 else negative
            delta = direction * min(abs(args.target_rad), max(0.0, room) * 0.7)
            target[0, index] += delta
            peak_drift = 0.0
            other = [j for j in range(asset.num_joints) if j != index]
            for _ in range(args.steps):
                with torch.no_grad():
                    if args.torque_pd:
                        torque = (args.kp * (target - runtime.tensor(asset.data.joint_pos))
                                  - args.kd * runtime.tensor(asset.data.joint_vel)).clamp(
                                      -args.max_torque, args.max_torque)
                        runtime.set_effort(asset, torque)
                    else:
                        runtime.set_position(asset, target)
                    asset.write_data_to_sim()
                    sim.step()
                    asset.update(sim_cfg.dt)
                    if other:
                        drift = (runtime.tensor(asset.data.joint_pos)[:, other] - initial[:, other]).abs().max()
                        peak_drift = max(peak_drift, float(drift))
            final = runtime.tensor(asset.data.joint_pos).clone()
            result = check_joint_motion(
                name=asset.joint_names[index], initial=initial[0].tolist(),
                target=target[0].tolist(), final=final[0].tolist(), joint_index=index,
                peak_other_drift=peak_drift,
            )
            if abs(delta) < 0.005:
                result["ok"] = False
                result["failures"].append("no_testable_joint_range")
            results.append(result)
            print("PHYSX_JOINT_CHECK " + json.dumps(result), flush=True)
        report = {
            "schema_version": 1, "backend": "physx", "asset_kind": "articulated",
            "ok": bool(results) and all(row["ok"] for row in results),
            "usd": args.usd, "usd_sha256": sha256_file(Path(args.usd)),
            "control": "torque_pd" if args.torque_pd else "native_position_drive",
            "review_drive": {"stiffness": args.kp if args.torque_pd else 400.0,
                             "damping": args.kd if args.torque_pd else 20.0},
            "steps_per_joint": args.steps, "checks": results,
            "robot_interaction_tested": False,
        }
        if args.snapshot:
            from isaac_export_evidence import bake_body_poses, save_stage
            from isaaclab.sim.utils import get_current_stage
            stage = get_current_stage()
            bake_body_poses(stage, runtime.body_paths(asset), runtime.pose_xyzw(asset.data.body_link_pose_w)[0].cpu().numpy())
            save_stage(stage, args.snapshot)
        output = args.report or Path(args.usd).with_name("physx_validation.json")
        atomic_json(output, report)
        print(f"PHYSX_ASSET_VALIDATION ok={report['ok']} report={output}", flush=True)
        if not report["ok"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
