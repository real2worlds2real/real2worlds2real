"""Example adapter for the currently validated Isaac Lab API.

The scene/runner protocol is independent of this implementation. Other simulator
versions may replace this script without changing the scene orchestration.
"""
# ruff: noqa: E402
import argparse
import json
import math
import sys
from pathlib import Path

from isaac_physx_compat import runtime

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scene", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--physics-hz", type=int, default=120,
                    help="Physics steps per second (default 120); timestep is 1 / physics-hz")
parser.add_argument("--resolution", type=int, default=640)
parser.add_argument("--width", dest="review_width", type=int, help="Optional review width; defaults to --resolution")
parser.add_argument("--height", dest="review_height", type=int, help="Optional review height; defaults to --resolution")
parser.add_argument("--capture-fps", type=int, default=0, help="Capture rate, zero or an integer divisor of --physics-hz")
parser.add_argument("--joint-excursion", type=float, default=0.35, help="Bounded joint excursion for the operation test")
parser.add_argument("--rigid-review-drop-m", type=float, default=0.0,
                    help="Optional explicit runtime-only upward displacement before a rigid support-response test")
parser.add_argument("--review-request", type=Path, help="Host review request, including optional continuous video capture")
parser.add_argument("--render-mode", choices=("PathTracing", "RaytracedLighting"), help="Optional target renderer override")
parser.add_argument("--render-samples", type=int, help="Same-physics-state display samples per captured frame; default 64 for Gaussian scenes, otherwise 1")
parser.add_argument("--render-warmup", type=int, help="Optional renderer updates before each captured state")
parser.add_argument("--disable-environment-collision", action="store_true", help="Negative control for support-response validation")
parser.add_argument("--disable-cloth-rigid-collision", action="store_true", help="Negative control for particle-cloth interaction")
parser.add_argument("--cloth-max-stretch", type=float, default=2.0, help="Maximum allowed cloth control-edge length ratio")
parser.add_argument("--cloth-penetration-tolerance", type=float, default=0.002, help="Retained triangle/support-box overlap tolerance in metres")
parser.add_argument("--rigid-kick", nargs=8, action="append", default=[],
                    metavar=("OBJECT", "SECONDS", "VX", "VY", "VZ", "WX", "WY", "WZ"),
                    help="Apply a one-time world-space velocity increment (m/s, rad/s) to a rigid body")
runtime.add_launcher_args(parser)
args = parser.parse_args()
args.explicit_physics_hz = any(v == "--physics-hz" or v.startswith("--physics-hz=") for v in sys.argv)
if any(value is not None and (value < 16 or value % 2) for value in (args.review_width, args.review_height)):
    parser.error("Review width and height must be even integers of at least 16 pixels")
if args.render_warmup is not None and not 0 <= args.render_warmup <= 256:
    parser.error("Render warmup must be from 0 to 256 updates")
if args.review_request:
    request = json.loads(args.review_request.read_text())
    if request.get("schema_version") != 1 or not isinstance(request.get("record_video"), bool):
        raise ValueError("Invalid scene review request")
    args.capture_fps = request.get("fps", 30) if request["record_video"] else 0
    if request.get("physics_hz") is not None:
        args.explicit_physics_hz = True
        from real2sim_harness.scene_timing import override_frequency
        args.physics_hz, args.steps = override_frequency(args.physics_hz, args.steps, request["physics_hz"])
args.physics = "isaacsim_physx"
args.enable_cameras = True

# Lab 2 starts Kit in prepare(), so geometry settings must precede this call.
from real2sim_harness.gaussian_runtime import (
    configure_gaussian_rendering,
    requested_formats,
    startup_arguments,
)

scene_spec = json.loads(args.manifest.read_text())
args.kit_args = startup_arguments(scene_spec, getattr(args, "kit_args", ""))
if scene_spec.get('review_camera', {}).get('calibration'):
    args.kit_args += ' --enable omni.usd.schema.omni_lens_distortion'
runtime.prepare(args)

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sim.utils import get_current_stage
from PIL import Image

from real2sim_harness.artifacts import atomic_json, sha256_file
from real2sim_harness.rigid_support import collider_support
from real2sim_harness.scene_motion import camera_at
from real2sim_harness.scene_physics import check_support_response
from real2sim_harness.scene_timing import SceneTiming
from real2sim_harness.sim_asset_loader import asset_configs

LAST_REPORT = {}


def tensor(value):
    return getattr(value, "torch", value)


def main():
    if args.steps < 1 or not 128 <= args.resolution <= 2048:
        raise ValueError("Use a positive physics step count and a bounded image resolution")
    spec = scene_spec
    from real2sim_harness.scene_physx_cloth import ScenePhysxCloth, read_configuration
    cloth_inputs = {}
    for item in spec["objects"]:
        folder = args.scene.parent / "assets" / item["id"]
        bundle = json.loads((folder / "asset_bundle.json").read_text())
        if bundle.get("asset_kind") == "deformable":
            cloth_inputs[item["id"]] = read_configuration(folder, bundle)
    if cloth_inputs:
        if spec.get("actions"):
            raise ValueError("This cloth scene adapter does not implement manifest actions; supply an action-capable adapter")
        if not runtime.legacy or not args.device.startswith("cuda"):
            raise ValueError("This particle-cloth scene route requires Sim 5.1 and CUDA physics")
        if len(cloth_inputs) > 1:
            raise ValueError("This adapter currently supports one cloth participant; cloth-cloth coupling needs a shared particle-system adapter")
        if not np.isfinite(args.cloth_max_stretch) or args.cloth_max_stretch <= 1:
            raise ValueError("Cloth stretch tolerance must be finite and greater than one")
        if not np.isfinite(args.cloth_penetration_tolerance) or args.cloth_penetration_tolerance < 0:
            raise ValueError("Cloth penetration tolerance must be finite and nonnegative")
        if not args.explicit_physics_hz:
            from real2sim_harness.physx_cloth import resolve_cloth_settings
            from real2sim_harness.scene_timing import override_frequency
            hz = math.lcm(args.physics_hz, *(round(1 / resolve_cloth_settings(c)["dt_s"]) for c in cloth_inputs.values()))
            args.physics_hz, args.steps = override_frequency(args.physics_hz, args.steps, hz)
    timing = SceneTiming(args.physics_hz, args.steps, args.capture_fps)
    if not 0 < args.joint_excursion <= 3.14:
        raise ValueError("Use a positive bounded joint excursion")
    if not 0 <= args.rigid_review_drop_m <= 0.2:
        raise ValueError("Rigid review displacement must be finite and between zero and 0.2 metres")
    if args.render_samples is not None and not 1 <= args.render_samples <= 512:
        raise ValueError("Use from 1 to 512 renderer samples per captured state")
    args.output.mkdir(parents=True, exist_ok=True)
    kicks = []
    for name, *values in args.rigid_kick:
        values = np.asarray(values, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Rigid kick values must be finite and occur within the simulation")
        if np.linalg.norm(values[1:4]) > 5 or np.linalg.norm(values[4:]) > 20:
            raise ValueError("Review kicks are limited to 5 m/s and 20 rad/s")
        kicks.append({"id": name, "seconds": float(values[0]), "step": timing.action_step(float(values[0])),
                      "delta_velocity": values[1:].tolist()})
    if spec.get("review_camera_path") and spec["review_camera_path"][-1]["seconds"] > timing.duration_s:
        raise ValueError("Simulation duration must cover the requested review camera path")
    gaussian_background = bool(requested_formats(spec))
    report = LAST_REPORT
    report.update({"schema_version": 1, "ok": False, "scene_sha256": sha256_file(args.scene),
              "runtime": {"name": "Isaac Sim / Isaac Lab PhysX", "version": "pending-runtime-start"},
              "checked_asset_ids": [], "images": [], "videos": [], "checks": [],
              "real_physics_calibrated": False, "robot_task_implemented": False})
    contact_pairs = set()
    assets, joint_tests, rigid_traces = [], [], {}
    cloth_assets = []
    atomic_json(args.output / "runtime_report.json", report)
    with runtime.launch(args) as physics_cfg:
        from isaac_physx_settings import enable_requested_ccd
        from isaacsim.core.version import get_version
        from omni.physx import get_physx_simulation_interface
        from pxr import (
            Gf,
            PhysicsSchemaTools,
            PhysxSchema,
            Usd,
            UsdGeom,
            UsdLux,
            UsdPhysics,
        )

        report["runtime"]["version"] = get_version()[0]

        sim_cfg = runtime.simulation_cfg(physics_cfg, dt=timing.dt, device=args.device)
        if cloth_inputs:
            sim_cfg.use_fabric = False
            sim_cfg.enable_scene_query_support = True
        report["timing"] = {"physics_hz": timing.physics_hz, "physics_dt_s": float(sim_cfg.dt),
                            "steps": args.steps, "duration_s": timing.duration_s,
                            "capture_fps": args.capture_fps,
                            "first_capture_time_s": timing.dt if args.capture_fps else None}
        if not np.isclose(sim_cfg.dt, timing.dt, rtol=0, atol=1e-12):
            raise ValueError("Runtime configuration did not preserve the requested physics timestep")
        sim = sim_utils.SimulationContext(sim_cfg)
        stage = get_current_stage()
        imported = UsdGeom.Xform.Define(stage, "/World/Scene").GetPrim()
        imported.GetReferences().AddReference(str(args.scene.resolve()))
        if gaussian_background:
            import carb
            report["gaussian_setup"] = configure_gaussian_rendering(
                stage, imported, legacy=runtime.legacy, settings=carb.settings.get_settings())
        if args.disable_environment_collision:
            for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Scene/Environment")):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        for item in spec["objects"]:
            folder = args.scene.parent / "assets" / item["id"]
            kind = json.loads((folder / "asset_bundle.json").read_text())["asset_kind"]
            path = f"/World/Scene/Objects/{item['id']}/Asset"
            if kind == "deformable":
                cloth_assets.append(ScenePhysxCloth(
                    folder, stage, item, physics_hz=args.physics_hz,
                    simulation_owner=sim_cfg.physics_prim_path, max_stretch=args.cloth_max_stretch,
                    disable_rigid_collision=args.disable_cloth_rigid_collision))
                continue
            cfg = asset_configs(folder, physics_path=path)["asset"]
            cfg.spawn = None
            if kind == "articulated":
                cfg.actuators = {"joints": runtime.actuator_cfg(joint_names_expr=[".*"], stiffness=400.0,
                                                              damping=20.0, joint_effort_limit=1000.0)}
                asset = Articulation(cfg=cfg)
                runtime.prepare_articulation(asset)
            else:
                asset = RigidObject(cfg=cfg)
            assets.append((item["id"], kind, asset))
            for prim in Usd.PrimRange(stage.GetPrimAtPath(path)):
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)

        def on_contact(headers, data):
            del data
            for header in headers:
                if header.num_contact_data:
                    paths = [str(PhysicsSchemaTools.intToSdfPath(header.collider0)),
                             str(PhysicsSchemaTools.intToSdfPath(header.collider1))]
                    if any("/Scene/Environment/" in v for v in paths) and any("/Scene/Objects/" in v for v in paths):
                        contact_pairs.add(tuple(sorted(paths)))

        rigid_assets = {name: asset for name, kind, asset in assets if kind == "rigid"}
        if any(kick["id"] not in rigid_assets for kick in kicks):
            raise ValueError("Rigid kicks must name an existing rigid scene participant")
        report["rigid_kicks"] = kicks
        applied_kicks = set()

        subscription = get_physx_simulation_interface().subscribe_contact_report_events(on_contact)
        render = spec.get("review_render", {})
        exposure = render.get("authored_light_exposure", 0)
        if not np.isfinite(exposure) or not -16 <= exposure <= 16:
            raise ValueError("Authored light exposure must be within -16..16 stops")
        authored_lights = []
        for prim in Usd.PrimRange(imported):
            if prim.HasAPI(UsdLux.LightAPI):
                light_api = UsdLux.LightAPI(prim)
                light_api.GetExposureAttr().Set(float(light_api.GetExposureAttr().Get() or 0) + exposure)
                authored_lights.append(str(prim.GetPath()))
        if render.get("default_lighting", True):
            light = sim_utils.DomeLightCfg(intensity=700.0, color=(0.94, 0.96, 1.0))
            light.func("/World/ReviewLight", light)
            key = sim_utils.DistantLightCfg(intensity=1800.0, angle=0.5)
            key.func("/World/ReviewKey", key, orientation=runtime.camera_quaternion((0.31, -0.25, -0.2, 0.894)))
        report["review_render"] = {"authored_light_exposure": exposure,
                                   "default_lighting": render.get("default_lighting", True),
                                   "authored_light_paths": authored_lights}
        review_view = camera_at(spec, 0)
        render_mode = args.render_mode or ("RaytracedLighting" if gaussian_background else "PathTracing")
        report["renderer_mode"] = render_mode
        render_samples = args.render_samples or (64 if gaussian_background else 1)
        report["render_capture"] = {"same_state_samples": render_samples,
            "aggregation": "display_sample_mean" if render_samples > 1 else "renderer_output",
            "physics_advances_during_capture": False}
        focal = 18 / np.tan(np.radians(review_view["horizontal_fov_deg"]) / 2) if review_view else 38.0
        calibrated_plan = None
        camera_width, camera_height = args.review_width or args.resolution, args.review_height or args.resolution
        if review_view and review_view.get('calibration'):
            from real2sim_harness.scene_camera import render_plan
            calibrated_plan = render_plan(review_view['calibration'], args.review_width, args.review_height)
            camera_width, camera_height = calibrated_plan['sensor']['image_size']
            report['camera_projection'] = calibrated_plan
            (args.output / 'raw_camera').mkdir()
        camera = runtime.camera(
            prim_path="/World/ReviewCamera", height=camera_height,
            width=camera_width, data_types=["rgb"],
            carb_settings={
                "/rtx/rendermode": render_mode, "/rtx/pathtracing/spp": 8,
                "/rtx/pathtracing/totalSpp": 64, "/rtx/pathtracing/optixDenoiser/enabled": True},
            spawn=sim_utils.PinholeCameraCfg(focal_length=float(focal), horizontal_aperture=36.0,
                                           clipping_range=(0.001, 1000.0)))
        if review_view and review_view.get('calibration'):
            from real2sim_harness.scene_camera import configure_rtx
            report['camera_calibration'] = configure_rtx(
                UsdGeom.Camera(stage.GetPrimAtPath('/World/ReviewCamera')),
                calibrated_plan['sensor'], camera_width, camera_height)
        report["continuous_collision_detection"] = enable_requested_ccd(stage)
        sim.reset()
        for cloth in cloth_assets:
            cloth.initialize()
            report["checked_asset_ids"].append(cloth.name)
        actual_dt = float(sim.get_physics_dt())
        report["timing"]["runtime_physics_dt_s"] = actual_dt
        if not np.isclose(actual_dt, timing.dt, rtol=0, atol=1e-9):
            raise ValueError("Runtime physics timestep differs from the requested timestep")
        support_requirements = spec.get("support_requirements", [])
        releases = {r["asset_id"]: r["release_height_m"] for r in support_requirements}
        required_pairs = {(r["asset_id"], r["support_id"]): r for r in support_requirements}
        for name, kind, asset in assets:
            asset.update(sim_cfg.dt)
            report["checked_asset_ids"].append(name)
            if kind == "rigid":
                release = releases.get(name, args.rigid_review_drop_m)
                if release:
                    pose = tensor(asset.data.root_link_pose_w).clone()
                    pose[:, 2] += release
                    asset.write_root_pose_to_sim(pose)
                    asset.update(sim_cfg.dt)
                path = runtime.rigid_body_path(asset)
                rigid_traces[name] = {"path": path, "poses": [runtime.pose_xyzw(asset.data.body_link_pose_w)[0, 0].cpu().numpy().copy()]}
            if kind == "articulated":
                initial = tensor(asset.data.joint_pos).clone()
                limits = tensor(asset.data.joint_pos_limits)
                room_up, room_down = limits[0, :, 1] - initial[0], initial[0] - limits[0, :, 0]
                delta = torch.where(room_up >= room_down, room_up, -room_down)
                delta = delta.clamp(-args.joint_excursion, args.joint_excursion) * 0.8
                joint_tests.append((name, asset, initial, delta))
        bound = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(
            stage.GetPrimAtPath("/World/Scene/Objects")).ComputeAlignedRange()
        low, high = np.array(bound.GetMin()), np.array(bound.GetMax())
        for support in spec["supports"]:
            extents = np.asarray(support["size_m"]) / 2
            points = np.array([[x, y, z, 1] for x in (-extents[0], extents[0])
                               for y in (-extents[1], extents[1]) for z in (-extents[2], extents[2])])
            world = points @ np.asarray(support["transform"]).T
            low, high = np.minimum(low, world[:, :3].min(0)), np.maximum(high, world[:, :3].max(0))
        center, radius = (low + high) / 2, max(np.linalg.norm(high - low) / 2, 0.1)
        if not np.isfinite(center).all() or not np.isfinite(radius) or radius > 100:
            raise ValueError("Cannot fit scene review camera")
        direction = np.array([1.2, 1.8, 0.9])
        position = center + direction / np.linalg.norm(direction) * radius * 3.8
        if review_view:
            center, position = np.array(review_view["target_m"]), np.array(review_view["eye_m"])
        up = np.array(review_view["up"]) if review_view else np.array([0, 0, 1])
        orientation = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, position)), Gf.Vec3d(*map(float, center)), Gf.Vec3d(*map(float, up))).GetInverse().ExtractRotationQuat()
        # This Lab camera API consumes XYZW; USD/Gf exposes the scalar separately.
        # Supplying WXYZ silently points the camera in a different direction.
        orientation_xyzw = [*map(float, orientation.GetImaginary()), float(orientation.GetReal())]
        orientation_tensor = torch.tensor([orientation_xyzw], dtype=torch.float32, device=args.device)
        report["camera_fit"] = {"center": center.tolist(), "radius": float(radius), "position": position.tolist(),
                                "bounds_min": low.tolist(), "bounds_max": high.tolist(),
                                "orientation_xyzw": orientation_xyzw, "convention": "opengl"}
        camera.set_world_poses(torch.tensor(position[None], dtype=torch.float32, device=args.device), runtime.camera_quaternion(orientation_tensor), convention="opengl")
        finite = True
        first_sample = min(args.steps - 1, round(16 / 120 * timing.physics_hz))
        samples = set(np.linspace(first_sample, args.steps - 1, 7, dtype=int).tolist())
        if args.capture_fps:
            (args.output / "video_frames").mkdir()
            report["frame_sequence"] = {"fps": args.capture_fps, "frames": [], "time_basis": "simulation_seconds"}
        for step in range(args.steps):
            for index, kick in enumerate(kicks):
                if index not in applied_kicks and step == kick["step"]:
                    asset = rigid_assets[kick["id"]]
                    velocity = tensor(asset.data.root_com_vel_w).clone()
                    velocity += torch.tensor([kick["delta_velocity"]], device=args.device)
                    asset.write_root_com_velocity_to_sim(velocity)
                    applied_kicks.add(index)
                    kick["applied_seconds"] = step * sim_cfg.dt
            padding_s = 20 / 120
            fraction = min(1.0, max(0.0, (step * timing.dt - padding_s) /
                                   max(timing.dt, timing.duration_s - 2 * padding_s)))
            for _, asset, initial, delta in joint_tests:
                runtime.set_position(asset, initial + fraction * delta[None])
                asset.write_data_to_sim()
            sim.step(render=False)
            for cloth in cloth_assets:
                cloth.sample((step + 1) * timing.dt)
                if step % max(1, timing.physics_hz // 30) == 0 or step == args.steps - 1:
                    cloth.inspect_contacts()
            for name, kind, asset in assets:
                asset.update(sim_cfg.dt)
                finite &= bool(torch.isfinite(runtime.pose_xyzw(asset.data.body_link_pose_w)).all())
                if kind == "rigid":
                    rigid_traces[name]["poses"].append(runtime.pose_xyzw(asset.data.body_link_pose_w)[0, 0].cpu().numpy().copy())
            capture = timing.capture_after_step(step)
            if step in samples or capture:
                for cloth in cloth_assets:
                    cloth.update_visuals()
                if spec.get("review_camera_path"):
                    view = camera_at(spec, (step + 1) * sim_cfg.dt)
                    position, center = np.array(view["eye_m"]), np.array(view["target_m"])
                    up = np.array(view["up"])
                    orientation = Gf.Matrix4d().SetLookAt(
                        Gf.Vec3d(*map(float, position)), Gf.Vec3d(*map(float, center)),
                        Gf.Vec3d(*map(float, up))).GetInverse().ExtractRotationQuat()
                    orientation_xyzw = [*map(float, orientation.GetImaginary()), float(orientation.GetReal())]
                    orientation_tensor = torch.tensor([orientation_xyzw], dtype=torch.float32, device=args.device)
                    report.setdefault("camera_samples", []).append({"seconds": (step + 1) * sim_cfg.dt, **view})
                # PhysX/Fabric frame synchronization may restore registered frame
                # transforms during stepping. Author the inspection view after it.
                camera.set_world_poses(torch.tensor(position[None], dtype=torch.float32, device=args.device), runtime.camera_quaternion(orientation_tensor), convention="opengl")
                warmup = args.render_warmup if args.render_warmup is not None else (8 if gaussian_background else 23)
                accumulated = None
                for render_tick in range(warmup + render_samples):
                    sim.render()
                    camera.update(sim_cfg.dt, force_recompute=True)
                    if render_tick < warmup:
                        continue
                    current = tensor(camera.data.output["rgb"])[0].cpu().numpy()[..., :3]
                    if current.dtype != np.uint8:
                        current = np.clip(current, 0, 1) * 255
                    if accumulated is None:
                        accumulated = current.astype(np.float32)
                    else:
                        accumulated += current
                raw = np.clip(np.rint(accumulated / render_samples), 0, 255).astype(np.uint8)
                if calibrated_plan:
                    from real2sim_harness.scene_camera import resample_rgb
                    raw_name = f'raw_camera/step_{step:06d}.png'
                    Image.fromarray(raw).save(args.output / raw_name)
                    report.setdefault('raw_camera_images', []).append(raw_name)
                    raw = resample_rgb(raw, calibrated_plan)
                if step in samples:
                    name = f"scene_{step:04d}.png"
                    Image.fromarray(raw).save(args.output / name)
                    report["images"].append(name)
                    report.setdefault("image_variation", {})[name] = float(raw.std())
                if capture:
                    name = f"video_frames/frame_{len(report['frame_sequence']['frames']):05d}.png"
                    Image.fromarray(raw).save(args.output / name)
                    report["frame_sequence"]["frames"].append(name)
                    report.setdefault("capture_times_s", []).append((step + 1) * sim_cfg.dt)
                report["observed_camera_position"] = tensor(camera.data.pos_w)[0].cpu().tolist()
                atomic_json(args.output / "runtime_report.json", report)
                if raw.std() < 2:
                    raise RuntimeError("Scene renderer returned a flat image")
        joint_records = []
        for name, asset, initial, delta in joint_tests:
            final = tensor(asset.data.joint_pos)
            movement = (final - initial).abs()[0]
            expected = delta.abs()
            ok = bool(torch.isfinite(final).all() and torch.all(expected > 0.005) and torch.all(movement >= expected * 0.5))
            joint_records.append({"id": name, "ok": ok, "displacement_rad": movement.tolist(), "requested_rad": expected.tolist()})
        support_records = []
        np.savez_compressed(args.output / "rigid_trajectories.npz",
                            **{name: np.array(trace["poses"]) for name, trace in rigid_traces.items()})
        report["rigid_trajectories"] = "rigid_trajectories.npz"
        for name, trace in rigid_traces.items():
            first, last = trace["poses"][0], trace["poses"][-1]
            initial_support = collider_support(stage, trace["path"], first)
            final_support = collider_support(stage, trace["path"], last)
            bottom_initial, bottom_final = initial_support["min_z_m"], final_support["min_z_m"]
            report.setdefault("collider_support", {})[name] = {"initial": initial_support, "final": final_support}
            history = np.array(trace["poses"][-max(2, round(20 / 120 * timing.physics_hz)):])[:, :3]
            speed = float(np.max(np.linalg.norm(np.diff(history, axis=0), axis=1) / sim_cfg.dt))
            for support in spec["supports"]:
                transform = np.array(support["transform"])
                if not np.allclose(transform[:3, 2], [0, 0, 1], atol=1e-5):
                    continue
                local = np.linalg.inv(transform) @ np.r_[last[:3], 1]
                inside = bool(np.all(np.abs(local[:2]) <= np.array(support["size_m"][:2]) / 2))
                top = float(transform[2, 3] + support["size_m"][2] / 2)
                requirement = required_pairs.get((name, support["id"]))
                bounds = ({"minimum_drop_m": requirement["minimum_drop_m"],
                           "separation_tolerance_m": requirement["separation_tolerance_m"],
                           "settled_speed_tolerance_m_s": requirement["max_late_speed_m_s"]}
                          if requirement else {})
                support_records.append({"id": name, "support": support["id"], **check_support_response(
                    initial_bottom_m=bottom_initial, final_bottom_m=bottom_final,
                    support_top_m=top, late_speed_m_s=speed, footprint_inside=inside, **bounds)})
        report["checks"] = [
            {"name": "scene_loaded", "ok": len(assets) + len(cloth_assets) == len(spec["objects"])},
            {"name": "finite_state", "ok": finite},
            {"name": "environment_contacts", "ok": bool(contact_pairs) or any(v["ok"] for v in support_records),
             "direct_report_pairs": sorted(contact_pairs), "support_response_tests": support_records,
             "scope": "Native contact reports or an observed free-drop-and-settle response; not contact-force calibration."},
        ]
        if support_requirements:
            from real2sim_harness.scene_support import audit_support_requirements
            report["checks"].append({"name": "required_support_relations",
                                     **audit_support_requirements(report, support_requirements)})
        cloth_reports = [cloth.finish(args.output, supports=spec["supports"],
                        penetration_tolerance=args.cloth_penetration_tolerance) for cloth in cloth_assets]
        if cloth_reports:
            report["cloth"] = cloth_reports
            report["checks"].append({"name": "deformation_integrity",
                "ok": all(c["finite"] and c["deformation_ok"] for c in cloth_reports)})
            cloth_support = all(c["rigid_collision_enabled"] and
                (c["environment_proximity_late"] or c["rigid_proximity_late_ids"]) for c in cloth_reports)
            environment_check = next(c for c in report["checks"] if c["name"] == "environment_contacts")
            environment_check["ok"] = bool((environment_check["ok"] or
                all(c["environment_proximity_late"] for c in cloth_reports)) and cloth_support)
            environment_check["cloth_scope"] = "Late proximity to native environment colliders at actual cloth nodes; no force measurement."
            if assets:
                report["checks"].append({"name": "rigid_deformable_contact",
                    "ok": all(c["rigid_collision_enabled"] and c["rigid_proximity_ids"] for c in cloth_reports),
                    "scope": "Sampled live collider proximity during native particle-cloth dynamics; inspect actual motion, not a calibrated contact-force check."})
        if joint_records:
            report["checks"].append({"name": "joint_motion", "ok": all(v["ok"] for v in joint_records), "assets": joint_records})
        if rigid_traces:
            names = sorted(rigid_traces)
            state_path = args.output / "rigid_states.npz"
            np.savez_compressed(
                state_path, asset_ids=np.asarray(names),
                times_s=np.arange(args.steps + 1, dtype=np.float64) * sim_cfg.dt,
                body_poses_wxyz=np.stack([np.asarray(rigid_traces[name]["poses"]) for name in names], axis=1)[..., [0, 1, 2, 6, 3, 4, 5]],
            )
            report["state_files"] = [{"path": state_path.name, "sha256": sha256_file(state_path)}]
            report["rigid_state_convention"] = "World metres; arrays [time,asset,xyz+wxyz]; actual physics states including the explicit initial probe when enabled."
        report.setdefault("state_files", []).extend(c["state_file"] for c in cloth_reports)
        report["rigid_review_probe"] = {
            "kind": "upward_displacement_then_free_release" if args.rigid_review_drop_m or any(releases.values()) else "none",
            "displacement_m": args.rigid_review_drop_m,
            "asset_ids": sorted(name for name in rigid_traces if releases.get(name, args.rigid_review_drop_m)),
            "displacement_by_asset_m": {name: releases.get(name, args.rigid_review_drop_m) for name in rigid_traces},
            "passive_scene_modified": False,
            "scope": "Explicit runtime test only; saved source assets and authored scene placement remain unchanged.",
        }
        report["ok"] = all(v["ok"] for v in report["checks"])
        report["steps"] = args.steps
        report["seconds"] = args.steps * sim_cfg.dt
        # Some launcher builds terminate the process while leaving the context;
        # retain the complete report before application shutdown.
        atomic_json(args.output / "runtime_report.json", report)
        print(json.dumps(report, indent=2), flush=True)
        # Keep subscription alive until after all simulation steps.
        del subscription
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as exc:
        if isinstance(exc, SystemExit) and exc.code in (None, 0):
            raise
        LAST_REPORT.update(ok=False, error=f"{type(exc).__name__}: {exc}")
        LAST_REPORT.setdefault("checks", []).append({"name": "adapter_execution", "ok": False, "error": str(exc)})
        atomic_json(args.output / "runtime_report.json", LAST_REPORT)
        raise
