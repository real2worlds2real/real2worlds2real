"""Render retained simulator geometry and materials inside Isaac Sim RTX."""

# ruff: noqa: E402 -- Isaac runtime imports intentionally follow launcher arguments.

import argparse
from pathlib import Path

from isaac_physx_compat import runtime

parser = argparse.ArgumentParser()
parser.add_argument("usd")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--resolution", type=int, default=640)
runtime.add_launcher_args(parser)
args = parser.parse_args()
args.physics = "isaacsim_physx"
args.enable_cameras = True

runtime.prepare(args)

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.sim.utils import get_current_stage
from PIL import Image

from real2sim_harness.artifacts import atomic_json, sha256_file
from real2sim_harness.usd_review import inspection_focus


def main():
    args.output.mkdir(parents=True, exist_ok=True)
    with runtime.launch(args) as physics_cfg:
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics
        print("ISAAC_RTX_RUNTIME_READY", flush=True)
        sim = sim_utils.SimulationContext(runtime.simulation_cfg(physics_cfg, device=args.device))
        stage = get_current_stage()
        source = Usd.Stage.Open(args.usd)
        source_root = source.GetDefaultPrim()
        source_path = source_root.GetPath() if source_root else Sdf.Path("/World")
        root = UsdGeom.Xform.Define(stage, "/World/Imported").GetPrim()
        root.GetReferences().AddReference(args.usd, source_path)
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdPhysics.Joint):
                UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        focus = inspection_focus(stage, "/World/Imported")
        box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]).ComputeWorldBound(focus).ComputeAlignedRange()
        low, high = np.asarray(box.GetMin()), np.asarray(box.GetMax())
        center = (low + high) * 0.5
        radius = max(float(np.linalg.norm(high - low)) * 0.5, 0.05)
        if not np.isfinite(center).all() or radius > 100:
            raise ValueError("Cannot fit a camera to the visible asset bounds")
        ground = sim_utils.CuboidCfg(size=(radius * 30, radius * 30, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.20, 0.23), roughness=0.85))
        ground.func("/World/InspectionFloor", ground, translation=(float(center[0]), float(center[1]), float(low[2] - 0.012)))
        light = sim_utils.DomeLightCfg(intensity=350.0, color=(0.9, 0.94, 1.0))
        light.func("/World/InspectionLight", light)
        key = sim_utils.DistantLightCfg(intensity=1800.0, angle=0.5)
        key.func("/World/InspectionKey", key, orientation=runtime.camera_quaternion((0.31, -0.25, -0.2, 0.894)))
        camera = runtime.camera(
            prim_path="/World/InspectionCamera", height=args.resolution, width=args.resolution,
            data_types=["rgb"], carb_settings={
                    "/rtx/rendermode": "PathTracing", "/rtx/pathtracing/spp": 16,
                    "/rtx/pathtracing/totalSpp": 128,
                    "/rtx/pathtracing/optixDenoiser/enabled": True,
                },
            spawn=sim_utils.PinholeCameraCfg(focal_length=38.0, horizontal_aperture=36.0,
                                           clipping_range=(0.001, 1000.0)),
        )
        sim.reset()
        print("ISAAC_RTX_SCENE_READY", flush=True)
        records = []
        for name, direction in (("front", (1.1, 1.6, 0.7)),
                                ("side", (-1.5, 0.6, 0.6)),
                                ("top", (0.15, 0.3, 2.0))):
            direction = np.asarray(direction)
            position = center + direction / np.linalg.norm(direction) * radius * 3.3
            camera.set_world_poses_from_view(
                torch.tensor(position[None], dtype=torch.float32, device=args.device),
                torch.tensor(center[None], dtype=torch.float32, device=args.device),
            )
            for _ in range(16):
                sim.render()
                camera.update(1 / 60, force_recompute=True)
            raw = camera.data.output["rgb"]
            raw = getattr(raw, "torch", raw)
            pixels = raw[0].cpu().numpy()[..., :3]
            if pixels.dtype != np.uint8:
                pixels = (np.clip(pixels, 0, 1) * 255).astype(np.uint8)
            variation = float(pixels.std())
            if variation < 2:
                raise RuntimeError(f"RTX camera produced a flat image for {name}")
            output = args.output / f"{name}.png"
            Image.fromarray(pixels).save(output)
            records.append({"view": name, "file": output.name, "pixel_std": variation})
            print(f"ISAAC_RTX_VIEW {name} {output}", flush=True)
        atomic_json(args.output / "render_report.json", {
            "ok": True, "renderer": "Isaac Sim RTX", "usd_sha256": sha256_file(Path(args.usd)),
            "source_usd": args.usd, "focus_bounds": {"min": low.tolist(), "max": high.tolist()},
            "focus_prim": str(focus.GetPath()),
            "views": records,
        })


if __name__ == "__main__":
    main()
