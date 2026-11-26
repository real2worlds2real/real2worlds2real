"""Render saved Newton states in RTX without advancing another physics engine."""

# ruff: noqa: E402
import argparse
import json
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scene", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--review-request", type=Path)
parser.add_argument("--resolution", type=int, default=640)
parser.add_argument("--height", type=int)
parser.add_argument("--light-exposure", type=float)
add_launcher_args(parser)
args = parser.parse_args()
args.physics = "isaacsim_physx"
args.enable_cameras = True

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.physics import PhysicsCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.utils import get_current_stage
from isaaclab.utils import instantiate
from isaaclab_physx.renderers import IsaacRtxRendererCfg, IsaacRtxRendererGlobalSettingsCfg
from PIL import Image

from real2sim_harness.artifacts import atomic_json
from real2sim_harness.scene_motion import camera_at


def main():
    if not 128 <= args.resolution <= 2048:
        raise ValueError("Use a bounded review resolution")
    height = args.height or args.resolution
    if not 128 <= height <= 2048:
        raise ValueError("Use a bounded review image height")
    spec = json.loads(args.manifest.read_text())
    if any(row["format"] != "mesh" for row in spec["backgrounds"]):
        raise ValueError("This Newton RTX replay adapter currently accepts mesh backgrounds only")
    request = (
        json.loads(args.review_request.read_text())
        if args.review_request
        else {"record_video": False}
    )
    bindings = json.loads((args.output / "trajectory_bindings.json").read_text())
    report = json.loads((args.output / "physics_report.json").read_text())
    with np.load(args.output / "scene_motion.npz", allow_pickle=False) as data:
        bodies, particles, times = data["bodies"], data["particles"], data["times"]
    record_video = request.get("record_video", False)
    selected = (
        list(range(len(times)))
        if record_video
        else sorted(set(np.linspace(0, len(times) - 1, 7, dtype=int)))
    )
    with launch_simulation(PhysicsCfg(), args) as physics_cfg:
        from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics

        from real2sim_harness.scene_newton_visuals import SceneNewtonVisuals

        sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(device=args.device, physics=physics_cfg)
        )
        stage = get_current_stage()
        root = UsdGeom.Xform.Define(stage, "/World/Scene").GetPrim()
        root.GetReferences().AddReference(str(args.scene.resolve()))
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        visual = SceneNewtonVisuals(args.scene.parent, stage, bindings)
        visual.update(bodies[0], particles[0])
        render = spec.get("review_render", {})
        exposure = args.light_exposure if args.light_exposure is not None else render.get("authored_light_exposure", 0)
        if not np.isfinite(exposure) or not -16 <= exposure <= 16:
            raise ValueError("Authored light exposure must be within -16..16 stops")
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdLux.LightAPI):
                light_api = UsdLux.LightAPI(prim)
                light_api.GetExposureAttr().Set(float(light_api.GetExposureAttr().Get() or 0)+exposure)
        if render.get("default_lighting", True):
            light = sim_utils.DomeLightCfg(intensity=700.0)
            light.func("/World/ReviewLight", light)
            light = sim_utils.DistantLightCfg(intensity=1500.0)
            light.func("/World/ReviewKey", light, orientation=(0.31, -0.25, -0.2, 0.894))
        view = camera_at(spec, 0)
        bound = (
            UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
            .ComputeWorldBound(stage.GetPrimAtPath("/World/Scene/Objects"))
            .ComputeAlignedRange()
        )
        low, high = np.asarray(bound.GetMin()), np.asarray(bound.GetMax())
        center = (low + high) / 2
        radius = max(float(np.linalg.norm(high - low) / 2), 0.1)
        eye = center + np.array([1.2, 1.8, 1.0]) / np.linalg.norm([1.2, 1.8, 1.0]) * radius * 3.8
        if not np.isfinite(center).all() or not np.isfinite(radius) or radius > 100:
            raise ValueError("Cannot fit a scene review camera")
        if view:
            center, eye = np.array(view["target_m"]), np.array(view["eye_m"])
        up = np.array(view["up"]) if view else np.array([0, 0, 1])
        orientation = (
            Gf.Matrix4d()
            .SetLookAt(
                Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, center)), Gf.Vec3d(*map(float, up))
            )
            .GetInverse()
            .ExtractRotationQuat()
        )
        quat = [*map(float, orientation.GetImaginary()), float(orientation.GetReal())]
        focal = 18 / np.tan(np.radians(view["horizontal_fov_deg"]) / 2) if view else 38.0
        camera = instantiate(
            CameraCfg(
                prim_path="/World/ReviewCamera",
                height=height,
                width=args.resolution,
                data_types=["rgb"],
                renderer_cfg=IsaacRtxRendererCfg(
                    global_settings=IsaacRtxRendererGlobalSettingsCfg(
                        carb_settings={
                            "/rtx/rendermode": "PathTracing",
                            "/rtx/pathtracing/spp": 8,
                            "/rtx/pathtracing/totalSpp": 64,
                            "/rtx/pathtracing/optixDenoiser/enabled": True,
                        }
                    )
                ),
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=float(focal),
                    horizontal_aperture=36.0,
                    clipping_range=(0.001, 1000.0),
                ),
            )
        )
        sim.reset()
        frame_paths = []
        review_indices = set(np.linspace(0, len(times) - 1, 7, dtype=int))
        if record_video:
            (args.output / "video_frames").mkdir(exist_ok=True)
        for frame in selected:
            visual.update(bodies[frame], particles[frame], time=float(times[frame]))
            moving_view = camera_at(spec, float(times[frame]))
            if moving_view:
                eye = np.asarray(moving_view["eye_m"])
                orientation = (Gf.Matrix4d().SetLookAt(
                    Gf.Vec3d(*map(float, eye)),
                    Gf.Vec3d(*map(float, moving_view["target_m"])),
                    Gf.Vec3d(*map(float, moving_view["up"]))
                ).GetInverse().ExtractRotationQuat())
                quat = [*map(float, orientation.GetImaginary()), float(orientation.GetReal())]
            camera.set_world_poses(
                torch.tensor(eye[None], device=args.device, dtype=torch.float32),
                torch.tensor([quat], device=args.device, dtype=torch.float32),
                convention="opengl",
            )
            # Eight updates at 8 spp reach the configured 64-spp accumulation.
            # Read back once after convergence instead of transferring every
            # intermediate render. Give the initial scene extra warm-up.
            for _ in range(24 if frame == selected[0] else 8):
                sim.render()
            camera.update(1 / 30, force_recompute=True)
            raw = (
                getattr(camera.data.output["rgb"], "torch", camera.data.output["rgb"])[0]
                .cpu()
                .numpy()[..., :3]
            )
            if raw.dtype != np.uint8:
                raw = (np.clip(raw, 0, 1) * 255).astype(np.uint8)
            if raw.std() < 2:
                raise RuntimeError("Flat RTX image")
            if record_video:
                path = f"video_frames/frame_{frame:05d}.png"
                Image.fromarray(raw).save(args.output / path)
                frame_paths.append(path)
            if frame in review_indices:
                path = f"scene_{frame:05d}.png"
                Image.fromarray(raw).save(args.output / path)
                report["images"].append(path)
            if frame % 30 == 0:
                print("NEWTON_SCENE_RENDER", frame, len(times), flush=True)
        if record_video:
            report["frame_sequence"] = {
                "fps": report["fps"],
                "frames": frame_paths,
                "time_basis": "simulation_seconds",
            }
        report["render_replay"] = {
            "physics_advanced": False,
            "trajectory": "scene_motion.npz",
            "camera": {"eye_m": eye.tolist(), "target_m": center.tolist()},
            "frames": len(selected),
            "authored_light_exposure":exposure,
            "default_lighting":render.get("default_lighting", True),
        }
        atomic_json(args.output / "runtime_report.json", report)


if __name__ == "__main__":
    main()
