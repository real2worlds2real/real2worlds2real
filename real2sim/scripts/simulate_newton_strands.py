"""Simulate generic native Newton rods and endpoint bodies; retain full motion."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / ".tools/newton-hybrid"), str(ROOT / "src")]
import warp as wp  # noqa: E402

from real2sim_harness.motion_quality import surface_trajectory_metrics  # noqa: E402
from real2sim_harness.strand_contacts import audit_capsules  # noqa: E402
from real2sim_harness.strand_contract import translation_at  # noqa: E402
from real2sim_harness.strand_geometry import quaternion_matrices  # noqa: E402
from real2sim_harness.strand_runtime import apply_strand_environment, load_strand_asset  # noqa: E402


@wp.kernel
def prescribed_bodies(
    ids: wp.array(dtype=wp.int32),
    target: wp.array(dtype=wp.transform),
    dt: float,
    q0: wp.array(dtype=wp.transform),
    q1: wp.array(dtype=wp.transform),
    v0: wp.array(dtype=wp.spatial_vector),
    v1: wp.array(dtype=wp.spatial_vector),
):
    i = wp.tid()
    b = ids[i]
    velocity = (wp.transform_get_translation(target[i]) - wp.transform_get_translation(q0[b])) / dt
    q0[b] = target[i]
    q1[b] = target[i]
    v0[b] = wp.spatial_vector(wp.vec3(0.0), velocity)
    v1[b] = wp.spatial_vector(wp.vec3(0.0), velocity)


def transform_points(poses, points):
    return poses[:, :3] + np.einsum("bij,bj->bi", quaternion_matrices(poses[:, 3:]), points)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    p.add_argument("--frames", type=int, default=192)
    p.add_argument("--fps", type=int, default=24)
    args = p.parse_args()
    c = json.loads((args.folder / "strand_configuration.json").read_text())
    runtime = load_strand_asset(c, prescribed_motion=True)
    wp.init()
    model, solver = runtime["model"], runtime["solver"]
    states = [runtime["state0"], runtime["state1"]]
    dt = 1 / (args.fps * c["solver"]["substeps"])
    last_time = (args.frames - 1) / args.fps
    if any(m["keyframes"][-1][0] > last_time + 1e-6 for m in c["motion"]):
        raise ValueError("Motion keyframes exceed the simulated duration")
    driven = wp.array([runtime["bodies"][m["body"]]["body"] for m in c["motion"]], dtype=wp.int32)
    initial = model.body_q.numpy()
    targets = initial[driven.numpy()].copy()
    target_array = wp.array(targets, dtype=wp.transform)
    samples = []
    started = time.monotonic()
    for frame in range(args.frames):
        if frame:
            for sub in range(c["solver"]["substeps"]):
                now = ((frame - 1) + (sub + 1) / c["solver"]["substeps"]) / args.fps
                states[0].clear_forces()
                if len(c["motion"]):
                    for i, m in enumerate(c["motion"]):
                        targets[i, :3] = initial[driven.numpy()[i], :3] + translation_at(
                            m["keyframes"], now
                        )
                    target_array.assign(targets)
                    wp.launch(
                        prescribed_bodies,
                        dim=len(targets),
                        inputs=[driven, target_array, dt],
                        outputs=[
                            states[0].body_q,
                            states[1].body_q,
                            states[0].body_qd,
                            states[1].body_qd,
                        ],
                    )
                apply_strand_environment(runtime, states[0])
                solver.step(states[0], states[1], runtime["control"], None, dt)
                states.reverse()
        q = states[0].body_q.numpy()
        if not np.isfinite(q).all():
            raise RuntimeError("Nonfinite rod state at frame " + str(frame))
        samples.append(q)
        if frame % 24 == 0:
            print("STRAND_FRAME", frame, args.frames, flush=True)
    poses = np.asarray(samples)
    arrays = {"bodies": poses, "times": np.arange(args.frames) / args.fps}
    reports = []
    for i, rod in enumerate(runtime["rods"]):
        q = poses[:, rod["body_ids"]]
        rotation = quaternion_matrices(q[:, :, 3:])
        offset = rotation[:, :, :, 2] * rod["lengths"][None, :, None] / 2
        starts, ends = q[:, :, :3] - offset, q[:, :, :3] + offset
        nodes = np.concatenate(
            [starts[:, :1], (ends[:, :-1] + starts[:, 1:]) / 2, ends[:, -1:]], axis=1
        )
        gaps = np.linalg.norm(ends[:, :-1] - starts[:, 1:], axis=2)
        stretch = np.linalg.norm(np.diff(nodes, axis=1), axis=2) / rod["lengths"]
        arrays[f"strand_{i:03d}_nodes"] = nodes
        arrays[f"strand_{i:03d}_quaternions"] = q[:, :, 3:]
        floor = float(
            np.min(np.minimum(starts[:, :, 2], ends[:, :, 2]))
            - rod["specification"]["radius"]
            - c["environment"]["floor_height"]
        )
        reports.append(
            {
                "name": rod["specification"]["name"],
                "segments": len(rod["body_ids"]),
                "body_ids": rod["body_ids"],
                "rest_quaternions": rod["rest_quaternions"].tolist(),
                "length_m": float(np.sum(rod["lengths"])),
                "maximum_joint_gap_m": float(gaps.max()),
                "maximum_stretch_ratio": float(stretch.max()),
                "minimum_floor_clearance_m": floor,
                "max_node_displacement_m": float(np.linalg.norm(nodes - nodes[:1], axis=2).max()),
                "temporal_metrics": surface_trajectory_metrics(
                    nodes,
                    arrays["times"],
                    quiet_start=c.get("quality", {}).get("quiet_start_seconds"),
                    speed_tolerance=c.get("quality", {}).get("quiet_speed_tolerance_m_s"),
                ),
            }
        )
    attachment_gap = 0.0
    for a in runtime["attachments"]:
        left = transform_points(
            poses[:, a["body"]], np.broadcast_to(a["body_anchor"], (len(poses), 3))
        )
        right = transform_points(
            poses[:, a["rod_body"]], np.broadcast_to(a["rod_anchor"], (len(poses), 3))
        )
        attachment_gap = max(attachment_gap, float(np.linalg.norm(left - right, axis=1).max()))
    failures = []
    for r in reports:
        if r["maximum_joint_gap_m"] > 0.003:
            failures.append("rod_joint_gap")
        if r["maximum_stretch_ratio"] > 1.15:
            failures.append("excessive_stretch")
        if r["minimum_floor_clearance_m"] < -0.002:
            failures.append("floor_penetration")
    if attachment_gap > 0.003:
        failures.append("endpoint_attachment_gap")
    contacts = audit_capsules(poses, runtime)
    if not contacts["ok"]:
        failures.append("capsule_contact_audit")
    if any(r["temporal_metrics"].get("quiet_hold", {}).get("ok") is False for r in reports):
        failures.append("unforced_hold_did_not_settle")
    report = {
        "ok": not failures,
        "backend": "Newton 1.7 native Rod / VBD",
        "frames": args.frames,
        "fps": args.fps,
        "source_scene_sha256": c["source_scene_sha256"],
        "body_ids": {name: row["body"] for name, row in runtime["bodies"].items()},
        "strands": reports,
        "maximum_attachment_gap_m": attachment_gap,
        "contact_audit": contacts,
        "failures": failures,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "configuration": c,
        "scope": "Actual capsule rod dynamics, native stretch/shear/bend/twist, frictional ground and nonneighbor self-contact; prescribed rigid endpoint translations; no measured calibration or knot certification.",
    }
    np.savez_compressed(args.folder / "strand_motion.npz", **arrays)
    (args.folder / "strand_validation.json").write_text(json.dumps(report, indent=2))
    compact = {k: v for k, v in report.items() if k not in {"configuration", "body_ids", "strands"}}
    compact["strands"] = [
        {k: v for k, v in row.items() if k not in {"body_ids", "rest_quaternions"}}
        for row in reports
    ]
    print(json.dumps(compact), flush=True)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
