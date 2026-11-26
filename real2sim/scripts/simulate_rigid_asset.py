"""Verify a generated rigid asset falls, contacts a support, and stays finite."""

# ruff: noqa: E402 -- Isaac runtime imports intentionally follow launcher arguments.

import argparse
import itertools
import json
from pathlib import Path

from isaac_physx_compat import runtime

parser = argparse.ArgumentParser()
parser.add_argument("usd")
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--snapshot", type=Path)
parser.add_argument("--report", type=Path)
runtime.add_launcher_args(parser)
args = parser.parse_args()
args.physics = "isaacsim_physx"

runtime.prepare(args)

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.assets import RigidObject
from isaaclab.sim.utils import get_current_stage
from scipy.spatial.transform import Rotation

from real2sim_harness.artifacts import atomic_json, sha256_file
from real2sim_harness.rigid_support import collider_support
from real2sim_harness.sim_asset_loader import asset_configs
from real2sim_harness.sim_parameters import load_parameters


def main():
    folder = Path(args.usd).parent
    manifest = json.loads((folder / "manifest.json").read_text())
    bounds = manifest["base_bounds"]
    corners = np.array(list(itertools.product(*zip(bounds["min"], bounds["max"], strict=True))), dtype=np.float32)
    with runtime.launch(args) as physics_cfg:
        from isaac_export_evidence import bake_body_poses, save_stage
        from isaac_physx_settings import enable_requested_ccd
        sim_cfg = runtime.simulation_cfg(physics_cfg, dt=1 / 120, device=args.device)
        sim = sim_utils.SimulationContext(sim_cfg)
        runtime.ground("/World/Ground")
        cfg = asset_configs(folder, physics_path="/World/Asset")["asset"]
        cfg.init_state.pos = (0.0, 0.0, 0.2 - bounds["min"][2])
        asset = RigidObject(cfg)
        ccd = enable_requested_ccd(get_current_stage())
        sim.reset()
        mass_properties = {
            name: runtime.tensor(getattr(asset.root_physx_view, "get_" + name)()).cpu().numpy()
            for name in ("masses", "coms", "inertias")
        }
        expected_mass = load_parameters(folder)["physx"]["mass_properties"].get("base")
        initial = runtime.pose_xyzw(asset.data.root_link_pose_w).clone()
        finite = True
        for _ in range(args.steps):
            asset.write_data_to_sim()
            sim.step()
            asset.update(sim_cfg.dt)
            finite = finite and bool(torch.isfinite(runtime.pose_xyzw(asset.data.root_link_pose_w)).all())
        final = runtime.pose_xyzw(asset.data.root_link_pose_w)[0].cpu().numpy()
        points = Rotation.from_quat(final[3:]).apply(corners) + final[:3]
        aabb_min_z = float(points[:, 2].min())
        support = collider_support(get_current_stage(), "/World/Asset", final) if finite else None
        min_z = support["min_z_m"] if support else float("nan")
        displacement = float(torch.linalg.vector_norm(runtime.pose_xyzw(asset.data.root_link_pose_w)[:, :3] - initial[:, :3]))
        speed = float(torch.linalg.vector_norm(runtime.tensor(asset.data.root_link_vel_w)[:, :3]))
        failures = []
        if expected_mass and not (
            np.allclose(mass_properties["masses"].reshape(-1), expected_mass["mass_kg"], rtol=1e-5, atol=1e-8)
            and np.allclose(mass_properties["coms"].reshape(-1, 7)[0, :3], expected_mass["center_of_mass_m"], rtol=1e-5, atol=1e-8)
            and np.allclose(mass_properties["inertias"].reshape(-1, 3, 3)[0], expected_mass["inertia_kg_m2"], rtol=1e-5, atol=1e-8)
        ):
            failures.append("explicit_mass_properties_readback_mismatch")
        if not finite:
            failures.append("non_finite_state")
        if not np.isfinite(min_z) or min_z < -0.02 or min_z > 0.04:
            failures.append("ground_contact_failed")
        if displacement < 0.08:
            failures.append("rigid_asset_did_not_fall")
        if speed > 0.1:
            failures.append("rigid_asset_did_not_settle")
        if args.snapshot:
            stage = get_current_stage()
            bake_body_poses(stage, ["/World/Asset"], [final])
            save_stage(stage, args.snapshot)
        report = {
            "schema_version": 1, "ok": not failures, "backend": "physx", "asset_kind": "rigid",
            "usd": args.usd, "usd_sha256": sha256_file(Path(args.usd)), "steps": args.steps,
            "continuous_collision_detection": ccd,
            "collision_support": support,
            "mass_properties_readback": {name: value.tolist() for name, value in mass_properties.items()},
            "checks": {"finite": finite, "min_z_m": min_z, "displacement_m": displacement,
                       "visual_aabb_min_z_m": aabb_min_z,
                       "final_speed_m_s": speed, "failures": failures},
        }
        output = args.report or folder / "rigid_validation.json"
        atomic_json(output, report)
        print("RIGID_ASSET_VALIDATION " + json.dumps(report), flush=True)
        if not report["ok"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
