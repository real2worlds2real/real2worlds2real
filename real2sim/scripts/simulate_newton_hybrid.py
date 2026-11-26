"""Advance an authored rigid/cloth model and retain inspectable real trajectories."""

import argparse
import json
from pathlib import Path

import newton
import numpy as np
import warp as wp

from real2sim_harness.artifacts import atomic_json
from real2sim_harness.hybrid_runtime import load_hybrid_asset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    wp.init()
    context = load_hybrid_asset(args.folder, device=args.device)
    model, solver, control = [context[key] for key in ("model", "solver", "control")]
    state, next_state = model.state(), model.state()
    substeps = context["parameters"]["solver"]["substeps"]
    dt = 1.0 / args.fps / substeps
    bodies, particles, joints, times, errors = [], [], [], [], []
    original_targets = control.joint_target_q.numpy().copy()
    edges = np.unique(np.sort(np.asarray(context["control_mesh"]["faces"])[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0)
    initial = state.particle_q.numpy().copy()
    rest_lengths = np.linalg.norm(initial[edges[:, 0]] - initial[edges[:, 1]], axis=1)
    valid_edges = rest_lengths > 1e-8
    finite, max_gap, max_edge_ratio, max_loop_gap = True, 0.0, 1.0, 0.0
    contact_overflow_max = 0
    for frame in range(args.frames):
        time = frame / args.fps
        target = original_targets.copy()
        for record in context["configuration"]["motion"]:
            keys = np.asarray(record["keyframes"])
            target[context["joint_dofs"][record["joint"]]] = np.interp(time, keys[:, 0], keys[:, 1])
        control.joint_target_q.assign(target)
        for _ in range(substeps):
            state.clear_forces()
            solver.step(state, next_state, control, None, dt)
            contact_overflow_max = max(contact_overflow_max, int(solver.body_particle_contact_overflow_max.numpy()[0]))
            state, next_state = next_state, state
        newton.eval_ik(model, state, state.joint_q, state.joint_qd)
        points, poses = state.particle_q.numpy(), state.body_q.numpy()
        finite = finite and bool(np.isfinite(points).all() and np.isfinite(poses).all())
        if not finite:
            break
        gap = 0.0
        for particle, body, anchor in context["attachments"]:
            pose = poses[body]
            transform = wp.transform(wp.vec3(*pose[:3]), wp.quat(*pose[3:]))
            world = np.asarray(wp.transform_point(transform, wp.vec3(*anchor)))
            gap = max(gap, float(np.linalg.norm(points[particle] - world)))
        max_gap = max(max_gap, gap)
        for first, second, anchor_a, anchor_b in context["loop_anchors"]:
            endpoints = []
            for body, anchor in ((first, anchor_a), (second, anchor_b)):
                pose = poses[body]
                transform = wp.transform(wp.vec3(*pose[:3]), wp.quat(*pose[3:]))
                endpoints.append(np.asarray(wp.transform_point(transform, wp.vec3(*anchor))))
            max_loop_gap = max(max_loop_gap, float(np.linalg.norm(endpoints[0]-endpoints[1])))
        ratios = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)[valid_edges] / rest_lengths[valid_edges]
        max_edge_ratio = max(max_edge_ratio, float(np.quantile(ratios, 0.95)))
        particles.append(points)
        bodies.append(poses)
        joints.append(state.joint_q.numpy())
        times.append(time)
        errors.append(gap)
        if frame % 12 == 0:
            print(f"HYBRID_FRAME {frame}/{args.frames} attachment_gap={gap:.6f} stretch_p95={max_edge_ratio:.4f}", flush=True)
    if particles:
        np.savez_compressed(args.folder / "hybrid_motion.npz", particles=np.asarray(particles), bodies=np.asarray(bodies),
                            joints=np.asarray(joints), times=np.asarray(times), attachment_gaps=np.asarray(errors))
    motion = float(np.linalg.norm(np.asarray(particles) - initial, axis=-1).max()) if particles else 0.0
    gap_limit = max(0.01, 4 * context["parameters"]["surface"]["particle_radius"])
    failures = []
    if contact_overflow_max > solver.body_particle_contact_buffer_pre_alloc:
        failures.append("integration_contact_buffer_overflow")
    if not finite or len(particles) != args.frames:
        failures.append("nonfinite_or_incomplete_motion")
    if motion < 0.001:
        failures.append("cloth_did_not_move")
    if max_gap > gap_limit:
        failures.append("cloth_body_attachment_drift")
    if max_edge_ratio > 2:
        failures.append("cloth_excessive_stretch")
    if max_loop_gap > 0.01:
        failures.append("mechanical_loop_anchor_drift")
    joint_checks = []
    q_starts = model.joint_q_start.numpy()
    for record in context["configuration"]["motion"]:
        name = record["joint"]
        index = q_starts[context["joint_ids"][name]]
        values = np.asarray(joints)[:, index] if joints else np.array([0.])
        keys = np.asarray(record["keyframes"])
        requested_span = float(np.ptp(keys[:, 1]))
        actual_span = float(np.ptp(values))
        target = float(np.interp(times[-1], keys[:, 0], keys[:, 1])) if times else 0.
        error = abs(float(values[-1])-target)
        spec = next(item for item in json.loads((args.folder / "manifest.json").read_text())["joints"] if item["name"] == name)
        tolerance = 0.02 if spec["joint_type"] == "prismatic" else 0.04
        ok = actual_span > max(1e-4, requested_span * .2) and error < tolerance
        joint_checks.append({"name": name, "ok": ok, "actual_span": actual_span, "requested_span": requested_span,
                             "final_tracking_error": error, "tolerance": tolerance})
        if not ok:
            failures.append(f"controlled_joint_motion_failed:{name}")
    report = {"ok": not failures, "backend": "newton_vbd_native_attachments", "newton_version": newton.__version__, "failures": failures,
              "frames": len(particles), "fps": args.fps, "finite": finite,
              "cloth_motion_max_m": motion, "attachment_gap_max_m": max_gap, "attachment_gap_limit_m": gap_limit,
              "edge_ratio_p95_max": max_edge_ratio, "body_ids": context["body_ids"],
              "joint_checks": joint_checks,
              "contact_buffer_capacity": solver.body_particle_contact_buffer_pre_alloc,
              "contact_buffer_overflow_max": contact_overflow_max,
              "loop_anchor_gap_max_m": max_loop_gap,
              "rest_matrices": {key: value.tolist() for key, value in context["rest_matrices"].items()},
              "rigid_frame_self_collision": context["rigid_self_collision"], "two_way_attachments": True,
              "real_physics_calibrated": False, "robot_task_implemented": False}
    atomic_json(args.folder / "hybrid_validation.json", report)
    print("HYBRID_VALIDATION " + json.dumps(report), flush=True)
    raise SystemExit(0 if report["ok"] else 2)


if __name__ == "__main__":
    main()
