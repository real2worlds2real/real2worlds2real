"""Native PhysX impact-triggered prefracture, with an explicit pre-contact handoff."""

# ruff: noqa: E402
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from isaac_physx_compat import runtime

p = argparse.ArgumentParser()
p.add_argument("folder", type=Path)
p.add_argument("--frames", type=int, default=192)
runtime.add_launcher_args(p)
args = p.parse_args()
args.physics = "isaacsim_physx"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
runtime.prepare(args)

import isaaclab.sim as sim_utils
from isaaclab.sim.utils import get_current_stage

from real2sim_harness.fracture_math import audit_motion, release_states


def main():
    folder = args.folder.resolve()
    cfg = json.loads((folder / "fracture_configuration.json").read_text())
    rows = cfg["pieces"]
    center = np.asarray(cfg["center"])
    masses = np.asarray([r["mass_kg"] for r in rows])
    offsets = np.asarray([r["center"] for r in rows]) - center
    dt, fps = 1 / 480, 24
    frames, intact_frames, masks, times, impulses, speeds = [], [], [], [], [], []
    fail = []
    with runtime.launch(args) as physics_cfg:
        from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade

        sim = sim_utils.SimulationContext(
            runtime.simulation_cfg(physics_cfg, dt=dt, device="cpu")
        )
        sim.set_setting("/physics/disableContactProcessing", False)
        stage = get_current_stage()
        # Analytic plane avoids any remote content dependency.
        plane = UsdGeom.Plane.Define(stage, "/World/Ground")
        plane.CreateAxisAttr("Z")
        plane.CreateWidthAttr(200.0)
        plane.CreateLengthAttr(200.0)
        UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
        material = UsdShade.Material.Define(stage, "/World/Material")
        mat = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        mat.CreateStaticFrictionAttr(cfg["friction"])
        mat.CreateDynamicFrictionAttr(cfg["friction"])
        mat.CreateRestitutionAttr(cfg["restitution"])
        UsdShade.MaterialBindingAPI.Apply(plane.GetPrim()).Bind(material, materialPurpose="physics")
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                PhysxSchema.PhysxSceneAPI.Apply(prim).CreateEnableCCDAttr(True)

        def rigid(path, mass, position, dormant):
            x = UsdGeom.Xform.Define(stage, path)
            x.AddTranslateOp().Set(Gf.Vec3d(*map(float, position)))
            rb = UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
            rb.CreateRigidBodyEnabledAttr(True)
            # Keep all actors dynamic: Lab caches its initial kinematic set.
            # Dormant fragments have neither gravity nor enabled collision shapes.
            rb.CreateKinematicEnabledAttr(False)
            m = UsdPhysics.MassAPI.Apply(x.GetPrim())
            m.CreateMassAttr(float(mass))
            m.CreateCenterOfMassAttr(Gf.Vec3f(0.0))
            native = PhysxSchema.PhysxRigidBodyAPI.Apply(x.GetPrim())
            native.CreateSolverPositionIterationCountAttr(32)
            native.CreateSolverVelocityIterationCountAttr(8)
            native.CreateMaxDepenetrationVelocityAttr(2.0)
            native.CreateEnableCCDAttr(True)
            native.CreateDisableGravityAttr(dormant)
            return rb, x

        def collider(path, vertices, faces, enabled):
            mesh = UsdGeom.Mesh.Define(stage, path)
            mesh.CreatePointsAttr([Gf.Vec3f(*map(float, v)) for v in vertices])
            mesh.CreateFaceVertexCountsAttr([len(f) for f in faces])
            mesh.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
            mesh.CreateSubdivisionSchemeAttr("none")
            col = UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            col.CreateCollisionEnabledAttr(enabled)
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(
                "convexDecomposition"
            )
            native = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
            native.CreateContactOffsetAttr(0.00005)
            native.CreateRestOffsetAttr(0.0)
            decomp = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(mesh.GetPrim())
            decomp.CreateMaxConvexHullsAttr(16)
            decomp.CreateHullVertexLimitAttr(64)
            decomp.CreateMinThicknessAttr(0.0)
            decomp.CreateShrinkWrapAttr(True)
            decomp.CreateErrorPercentageAttr(1.0)
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
                material, materialPurpose="physics"
            )
            return col

        motion = cfg.get("motion", {})
        height = float(motion.get("drop_height", 0.45))
        if not 0.05 <= height <= 2:
            raise ValueError("Use a drop height from .05 to 2 metres")
        rb, root = rigid("/World/Intact", cfg["mass_kg"], center + [0, 0, height], False)
        angles = motion.get("rotation_degrees", [20.0, 35.0, 10.0])
        rot = (
            Gf.Rotation(Gf.Vec3d(1, 0, 0), float(angles[0]))
            * Gf.Rotation(Gf.Vec3d(0, 1, 0), float(angles[1]))
            * Gf.Rotation(Gf.Vec3d(0, 0, 1), float(angles[2]))
        )
        root.AddOrientOp().Set(Gf.Quatf(rot.GetQuat()))
        aggregate_colliders = [
            collider(
                f"/World/Intact/Piece_{i:03d}",
                np.asarray(r["vertices"]) + offsets[i],
                r["faces"],
                True,
            )
            for i, r in enumerate(rows)
        ]
        refs = []
        for i, r in enumerate(rows):
            body, _ = rigid(f"/World/Shard_{i:03d}", r["mass_kg"], [3 + i * 0.2, 0, 1], True)
            col = collider(f"/World/Shard_{i:03d}/Shape", r["vertices"], r["faces"], False)
            refs.append((body, col))
            UsdGeom.Imageable(body.GetPrim()).MakeInvisible()
        PhysxSchema.PhysxContactReportAPI.Apply(root.GetPrim()).CreateThresholdAttr(0.0)
        stage.GetRootLayer().Export(str(folder / "initial_physics.usdc"))
        sim.reset()
        view = runtime.numpy_physics_view()
        view.set_subspace_roots("/")
        whole = view.create_rigid_body_view("/World/Intact")
        shards = view.create_rigid_body_view("/World/Shard_*")
        contacts = view.create_rigid_contact_view("/World/Intact", max_contact_data_count=512)
        ids = np.arange(len(rows), dtype=np.int32)
        broken, event, peak_speed, peak_omega, peak_tip_speed = False, {}, 0.0, 0.0, 0.0
        handoff_poses = None
        shard_radii = np.asarray([np.linalg.norm(r["vertices"], axis=1).max() for r in rows])
        stride = round(1 / dt / fps)
        for step in range(args.frames * stride):
            previous_pose = whole.get_transforms()[0].copy()
            previous_velocity = whole.get_velocities()[0].copy()
            if step % stride == 0:
                pose = previous_pose.copy()
                states = (
                    shards.get_transforms().copy()
                    if broken
                    else release_states(pose, previous_velocity, offsets)[0]
                )
                frames.append(states)
                intact_frames.append(pose)
                masks.append(broken)
                times.append(step * dt)
                speeds.append(
                    shards.get_velocities().copy()
                    if broken
                    else release_states(pose, previous_velocity, offsets)[1]
                )
            sim.step(render=False)
            if broken and "first_dynamic_step_center_displacement_m" not in event:
                assert handoff_poses is not None
                jump = float(
                    np.linalg.norm(
                        shards.get_transforms()[:, :3] - handoff_poses[:, :3], axis=1
                    ).max()
                )
                event["first_dynamic_step_center_displacement_m"] = jump
            impulse = (
                float(np.linalg.norm(runtime.contact_impulses(contacts)))
                if not broken
                else 0.0
            )
            impulses.append(impulse)
            if not broken and impulse >= cfg["impulse_threshold_ns"]:
                # Re-simulate the detected impact using fragments from the last unbroken state.
                # Copying the post-impact stopped velocity would discard the impact energy.
                poses, velocities = release_states(previous_pose, previous_velocity, offsets)
                handoff_poses = poses.copy()
                with Sdf.ChangeBlock():
                    PhysxSchema.PhysxRigidBodyAPI(rb.GetPrim()).CreateDisableGravityAttr(True)
                    for col in aggregate_colliders:
                        col.CreateCollisionEnabledAttr(False)
                    UsdGeom.Imageable(root.GetPrim()).MakeInvisible()
                    for i, (body, col) in enumerate(refs):
                        # CCD/body-flag updates can recreate the native actor from USD.
                        # Author the same handoff state there so it cannot reset to parking.
                        xform = UsdGeom.Xform(body.GetPrim())
                        xform.GetOrderedXformOps()[0].Set(Gf.Vec3d(*map(float, poses[i, :3])))
                        xform.AddOrientOp().Set(
                            Gf.Quatf(float(poses[i, 6]), Gf.Vec3f(*map(float, poses[i, 3:6])))
                        )
                        body.CreateVelocityAttr(Gf.Vec3f(*map(float, velocities[i, :3])))
                        body.CreateAngularVelocityAttr(
                            Gf.Vec3f(*map(float, np.degrees(velocities[i, 3:])))
                        )
                        body.CreateKinematicEnabledAttr(False)
                        PhysxSchema.PhysxRigidBodyAPI(body.GetPrim()).CreateEnableCCDAttr(True)
                        PhysxSchema.PhysxRigidBodyAPI(body.GetPrim()).CreateDisableGravityAttr(False)
                        col.CreateCollisionEnabledAttr(True)
                        UsdGeom.Imageable(body.GetPrim()).MakeVisible()
                shards.set_transforms(poses, ids)
                shards.set_velocities(velocities, ids)
                whole.set_velocities(
                    np.zeros((1, 6), dtype=np.float32), np.array([0], dtype=np.int32)
                )
                initial_momentum = cfg["mass_kg"] * previous_velocity[:3]
                released_momentum = (velocities[:, :3] * masses[:, None]).sum(axis=0)
                event = {
                    "step": step,
                    "time_seconds": (step + 1) * dt,
                    "impulse_ns": impulse,
                    "pre_contact_pose": previous_pose.tolist(),
                    "pre_contact_velocity": previous_velocity.tolist(),
                    "handoff_time_rewind_seconds": dt,
                    "linear_momentum_error_kg_m_s": float(
                        np.linalg.norm(initial_momentum - released_momentum)
                    ),
                    "added_radial_impulse": False,
                }
                broken = True
                print("FRACTURE_TRIGGER", json.dumps(event), flush=True)
            if broken:
                velocities = shards.get_velocities()
                peak_speed = max(peak_speed, float(np.linalg.norm(velocities[:, :3], axis=1).max()))
                peak_omega = max(peak_omega, float(np.linalg.norm(velocities[:, 3:], axis=1).max()))
                peak_tip_speed = max(
                    peak_tip_speed,
                    float(
                        (
                            np.linalg.norm(velocities[:, :3], axis=1)
                            + np.linalg.norm(velocities[:, 3:], axis=1) * shard_radii
                        ).max()
                    ),
                )
        final_vel = shards.get_velocities().copy()
        final_poses = shards.get_transforms().copy()
        inertias = shards.get_inertias().copy()
        runtime.stop(sim)
        # Kit launcher terminates on context exit; persist evidence before leaving it.
        finite = np.isfinite(frames).all() and np.isfinite(final_vel).all()
        settled_speed = float(np.linalg.norm(final_vel[:, :3], axis=1).max())
        if not broken:
            fail.append("impact_threshold_not_reached")
        if not finite:
            fail.append("nonfinite_state")
        if peak_speed > 15 or peak_tip_speed > 15:
            fail.append("unbounded_fragment_motion")
        if settled_speed > 0.1:
            fail.append("fragments_not_settled")
        if event and event["linear_momentum_error_kg_m_s"] > 1e-5:
            fail.append("handoff_momentum_error")
        if event.get("first_dynamic_step_center_displacement_m", 0.0) > 0.03:
            fail.append("handoff_native_actor_reset")
        audit = audit_motion(cfg, np.asarray(frames), np.asarray(speeds), inertias, masks)
        if audit["max_visible_vertex_floor_penetration_m"] > 0.0015:
            fail.append("visible_shard_floor_penetration")
        if audit["max_energy_ratio_to_initial"] > 1.75:
            fail.append("numerical_energy_injection")
        if audit["relative_pair_distance_change_p95_m"] < 0.01:
            fail.append("fragments_did_not_separate")
        np.savez_compressed(
            folder / "fracture_motion.npz",
            pieces=np.asarray(frames),
            intact=np.asarray(intact_frames),
            broken=np.asarray(masks),
            times=np.asarray(times),
            velocities=np.asarray(speeds),
            inertias=inertias,
        )
        report = {
            "ok": not fail,
            "backend": "Isaac Sim / Isaac Lab / native PhysX CPU",
            "source_scene_sha256": cfg["source_blend_sha256"],
            "frames": args.frames,
            "fps": fps,
            "pieces": len(rows),
            "break_event": event,
            "failures": fail,
            "mass_kg": cfg["mass_kg"],
            "mass_sum_error_kg": float(abs(masses.sum() - cfg["mass_kg"])),
            "peak_fragment_speed_m_s": peak_speed,
            "peak_fragment_angular_speed_rad_s": peak_omega,
            "peak_fragment_surface_speed_bound_m_s": peak_tip_speed,
            "final_max_speed_m_s": settled_speed,
            "final_poses": final_poses.tolist(),
            "collision_model": "Native PhysX convex decomposition per closed shard; CCD and friction enabled",
            "fracture_model": cfg["model"],
            "coverage": cfg["coverage"],
            "retained_state_audit": audit,
            "motion_sha256": hashlib.sha256(
                (folder / "fracture_motion.npz").read_bytes()
            ).hexdigest(),
        }
        (folder / "fracture_validation.json").write_text(json.dumps(report, indent=2))
        print(
            "FRACTURE_METRICS",
            json.dumps(
                {k: v for k, v in report.items() if k not in {"final_poses", "break_event"}}
            ),
            flush=True,
        )
        print("FRACTURE_RESULT", report["ok"], fail, flush=True)
        if fail:
            raise RuntimeError(str(fail))


if __name__ == "__main__":
    main()
