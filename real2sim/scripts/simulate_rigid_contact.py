"""Retain native concave rigid mesh contact with explicit external test wrenches."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import newton
import numpy as np
import warp as wp

from real2sim_harness.artifacts import atomic_json, sha256_file
from real2sim_harness.native_controls import apply_wrench, track_contact_loss
from real2sim_harness.rigid_contact import thread_motion
from real2sim_harness.rigid_contact_runtime import load_rigid_contact


def relative_poses(reference, moving):
    from scipy.spatial.transform import Rotation

    r = Rotation.from_quat(reference[:, 3:])
    xyz = r.inv().apply(moving[:, :3] - reference[:, :3])
    quat = (r.inv() * Rotation.from_quat(moving[:, 3:])).as_quat()
    return np.column_stack([xyz, quat])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("request", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    wp.config.kernel_cache_dir = os.environ.get("WARP_CACHE_PATH", str(a.output / ".warp-cache"))
    newton.use_coord_layout_targets = True
    request = json.loads(a.request.read_text())
    builder, body_ids, c = load_rigid_contact(
        request, cache_dir=a.output.parent / "sdf-cache", device=a.device
    )
    atomic_json(a.output / "request.json", c)
    model = builder.finalize(device=a.device)
    model.rigid_contact_max = 20000
    solver = newton.solvers.SolverXPBD(
        model, iterations=c["iterations"], rigid_contact_relaxation=0.8
    )
    pipeline = newton.CollisionPipeline(
        model, reduce_contacts=True, rigid_contact_max=20000, broad_phase="nxn"
    )
    contacts = pipeline.contacts()
    s0, s1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)

    def snapshot():
        q = s0.body_q.numpy()
        return np.asarray(
            [
                b.get("pose", [0, 0, 0, 0, 0, 0, 1])
                if body_ids[b["id"]] == -1
                else q[body_ids[b["id"]]]
                for b in c["bodies"]
            ]
        )

    states = [snapshot()]
    counts = [0]
    failures = []
    dt = 1 / c["fps"] / c["substeps"]
    shape_ids = {label: i for i, label in enumerate(model.shape_label)}
    controllers = []
    for drive in c.get("controls", []):
        controllers.append(
            {
                "drive": drive,
                "state": wp.zeros(2, dtype=wp.int32, device=a.device),
                "timer": wp.zeros(1, dtype=float, device=a.device),
                "event": wp.full(1, -1.0, dtype=float, device=a.device),
            }
        )
    for frame in range(1, c["frames"]):
        for sub in range(c["substeps"]):
            time = (frame - 1) / c["fps"] + sub * dt
            s0.clear_forces()
            for controller in controllers:
                d = controller["drive"]
                if d.get("start_s", 0) <= time < d.get("end_s", (c["frames"] - 1) / c["fps"]):

                    def v(key, default, drive=d):
                        return wp.vec3(*drive.get(key, default))

                    stop = d.get("stop_rotation_after_contact_loss")
                    if stop is not None:
                        wp.launch(
                            track_contact_loss,
                            dim=1,
                            inputs=[
                                contacts.rigid_contact_count,
                                contacts.rigid_contact_shape0,
                                contacts.rigid_contact_shape1,
                                shape_ids[d["body"]],
                                shape_ids[stop["other_body"]],
                                dt,
                                stop.get("dwell_seconds", 0.05),
                                time,
                                controller["state"],
                                controller["timer"],
                                controller["event"],
                            ],
                        )
                    wp.launch(
                        apply_wrench,
                        dim=1,
                        inputs=[
                            s0.body_q,
                            s0.body_qd,
                            s0.body_f,
                            body_ids[d["body"]],
                            controller["state"],
                            v("position_target_m", [0, 0, 0]),
                            wp.vec3(*[float(x) for x in d.get("position_axes", [False] * 3)]),
                            v("linear_velocity_m_s", [0, 0, 0]),
                            v("force_n", [0, 0, 0]),
                            v("angular_velocity_rad_s", [0, 0, 0]),
                            v("torque_nm", [0, 0, 0]),
                            d.get("position_stiffness", 0.0),
                            d.get("linear_damping", 0.0),
                            d.get("angular_gain", 0.0),
                            d.get("alignment_stiffness", 0.0),
                            v("alignment_axis_world", [0, 0, 1]),
                            v("force_limit_n", [0, 0, 0]),
                            v("torque_limit_nm", [0, 0, 0]),
                        ],
                    )
            pipeline.collide(s0, contacts)
            solver.step(s0, s1, control, contacts, dt)
            s0, s1 = s1, s0
        states.append(snapshot())
        counts.append(int(contacts.rigid_contact_count.numpy()[0]))
        if not np.isfinite(states[-1]).all() or np.max(np.abs(states[-1][:, :3])) > 100:
            failures.append("nonfinite_or_unbounded_state")
            break
        if counts[-1] >= 20000:
            failures.append("contact_buffer_capacity_reached")
        if frame % c["fps"] == 0:
            print("RIGID_CONTACT_FRAME", frame, flush=True)
    states = np.asarray(states)
    if len(states) != c["frames"]:
        failures.append("incomplete_trajectory")
    checks = []
    ids = [b["id"] for b in c["bodies"]]
    for t in c.get("thread_checks", []):
        end = t.get("end_frame", len(states) - 1)
        if end >= len(states):
            checks.append({"ok": False, "error": "incomplete_check_window"})
            continue
        poses = relative_poses(
            states[:, ids.index(t["reference_body"])], states[:, ids.index(t["moving_body"])]
        )
        start = t.get("start_frame", 0)
        measured = thread_motion(
            poses[start : end + 1],
            lead_m_per_turn=t["lead_m_per_turn"],
            axis=t.get("axis_reference", [0, 0, 1]),
            tolerance_m=t["tolerance_m"],
            minimum_turns=t.get("minimum_turns", 0.25),
        )
        measured.update(start_frame=start, end_frame=end)
        checks.append(measured)
    np.savez_compressed(
        a.output / "motion.npz",
        body_poses=states,
        times=np.arange(len(states)) / c["fps"],
        contact_counts=counts,
    )
    from scipy.spatial.transform import Rotation

    bounds = []
    for i, body in enumerate(c["bodies"]):
        pose = states[-1, i]
        vertices = Rotation.from_quat(pose[3:]).apply(np.asarray(body["vertices"])) + pose[:3]
        bounds.append((vertices.min(axis=0), vertices.max(axis=0)))
    gaps = []
    for i in range(len(bounds)):
        for j in range(i + 1, len(bounds)):
            lo0, hi0 = bounds[i]
            lo1, hi1 = bounds[j]
            gap = np.maximum(np.maximum(lo0 - hi1, lo1 - hi0), 0)
            gaps.append(
                {
                    "body0": ids[i],
                    "body1": ids[j],
                    "final_aabb_gap_m": float(np.linalg.norm(gap)),
                    "scope": "Positive gap proves final collision-mesh separation; overlapping AABBs do not imply penetration.",
                }
            )
    separation_checks = []
    for check in c.get("separation_checks", []):
        gap = next(
            x["final_aabb_gap_m"]
            for x in gaps
            if {x["body0"], x["body1"]} == {check["body0"], check["body1"]}
        )
        separation_checks.append(
            {**check, "final_aabb_gap_m": gap, "ok": gap >= check["minimum_gap_m"]}
        )
    masses, centers = model.body_mass.numpy(), model.body_com.numpy()
    body_properties = {
        name: {"static": True}
        if index < 0
        else {
            "static": False,
            "mass_kg": float(masses[index]),
            "com_local_m": centers[index].tolist(),
        }
        for name, index in body_ids.items()
    }
    rotation_stop_events = [
        {
            "body": x["drive"]["body"],
            "other_body": x["drive"]["stop_rotation_after_contact_loss"]["other_body"],
            "ever_contacted": bool(x["state"].numpy()[0]),
            "stop_triggered": bool(x["state"].numpy()[1]),
            "time_s": float(x["event"].numpy()[0]),
        }
        for x in controllers
        if "stop_rotation_after_contact_loss" in x["drive"]
    ]
    report = {
        "rotation_stop_events": rotation_stop_events,
        "body_properties": body_properties,
        "separation_checks": separation_checks,
        "initial_poses": {name: states[0, i].tolist() for i, name in enumerate(ids)},
        "final_poses": {name: states[-1, i].tolist() for i, name in enumerate(ids)},
        "final_pair_bounds": gaps,
        "ok": not failures
        and all(t["ok"] for t in checks)
        and all(t["ok"] for t in separation_checks),
        "failures": sorted(set(failures)),
        "frames": len(states),
        "fps": c["fps"],
        "body_ids": body_ids,
        "body_order": ids,
        "thread_checks": checks,
        "maximum_contacts": max(counts),
        "solver": "Newton SolverXPBD with native SDF mesh contacts",
        "prescribed_helix": False,
        "external_test_controls": c.get("controls", []),
        "real_physics_calibrated": False,
        "asset_acceptance_updated": False,
        "evidence_sha256": {n: sha256_file(a.output / n) for n in ["request.json", "motion.npz"]},
        "scope": "Diagnostic native mesh contact and declared measured checks; not a complete asset delivery or general collision certificate.",
    }
    atomic_json(a.output / "validation.json", report)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ["external_test_controls", "body_ids"]}
        ),
        flush=True,
    )
    raise SystemExit(0 if report["ok"] else 2)


if __name__ == "__main__":
    main()
