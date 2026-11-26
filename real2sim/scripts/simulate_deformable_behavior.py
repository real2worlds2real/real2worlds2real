"""Run native shell/solid diagnostic interactions and retain solver states."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / ".tools/newton-hybrid"), str(ROOT / "src")]
import warp as wp  # noqa: E402

from real2sim_harness.deformable_behavior import measure_response  # noqa: E402
from real2sim_harness.deformable_geometry import (  # noqa: E402
    audit_box_contacts,
    audit_sphere_contacts,
    shell_metrics,
    solid_metrics,
)
from real2sim_harness.deformable_runtime import apply_deformable_environment, load_deformable  # noqa: E402
from real2sim_harness.native_controls import move_grasp, move_probe  # noqa: E402
from real2sim_harness.strand_contract import translation_at  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    c = json.loads(args.request.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        wp.config.kernel_cache_dir = str(args.cache_dir.resolve())
    wp.init()
    runtime = load_deformable(c, device=args.device, prescribed_motion=True, include_probes=True)
    model, solver = runtime["model"], runtime["solver"]
    a, b = runtime["state0"], runtime["state1"]
    fps, frames = c.get("fps", 24), c.get("frames", 120)
    substeps = c.get("solver", {}).get("substeps", 12)
    if type(frames) is not int or not 2 <= frames <= 1440 or not 1 <= substeps <= 128:
        raise ValueError("Choose a bounded frame and substep count")
    dt = 1 / (fps * substeps)
    grasps = [
        (spec, wp.array(spec["indices"], dtype=wp.int32, device=args.device))
        for spec in c.get("grasps", [])
    ]
    rest = wp.array(runtime["points"], dtype=wp.vec3, device=args.device)
    positions, velocities, bodies, contact_counts = [], [], [], []
    start = time.monotonic()
    for frame in range(frames):
        positions.append(a.particle_q.numpy().copy())
        velocities.append(a.particle_qd.numpy().copy())
        bodies.append(
            a.body_q.numpy().copy() if a.body_q is not None else np.empty((0, 7), dtype=np.float32)
        )
        if frame % fps == 0:
            print("DEFORMABLE_FRAME", frame, "OF", frames, flush=True)
        if frame == frames - 1:
            break
        for step in range(substeps):
            t = (frame + (step + 1) / substeps) / fps
            a.clear_forces()
            apply_deformable_environment(runtime, a)
            for probe in runtime["probes"]:
                spec = probe["specification"]
                offset = translation_at(spec["keyframes"], t)
                target = np.asarray(spec["position_m"]) + offset
                wp.launch(
                    move_probe,
                    dim=1,
                    inputs=[
                        probe["body"],
                        wp.vec3(*target),
                        dt,
                        a.body_q,
                        b.body_q,
                        a.body_qd,
                        b.body_qd,
                    ],
                    device=args.device,
                )
            for spec, ids in grasps:
                offset = translation_at(spec["keyframes"], t)
                wp.launch(
                    move_grasp,
                    dim=len(spec["indices"]),
                    inputs=[
                        ids,
                        rest,
                        wp.vec3(*offset),
                        dt,
                        t >= spec.get("release_seconds", np.inf),
                        model.particle_flags,
                        a.particle_q,
                        b.particle_q,
                        a.particle_qd,
                        b.particle_qd,
                    ],
                    device=args.device,
                )
            owned = runtime.get("solver_owns_pipeline", False)
            if not owned:
                runtime["pipeline"].collide(a, runtime["contacts"])
            solver.step(a, b, runtime["control"], None if owned else runtime["contacts"], dt)
            if step == 0:
                contact_counts.append(int(runtime["contacts"].soft_contact_count.numpy()[0]))
            a, b = b, a
    positions, velocities, bodies = map(np.asarray, (positions, velocities, bodies))
    finite = bool(np.isfinite(positions).all() and np.isfinite(velocities).all())
    metrics = (
        (
            solid_metrics(runtime["points"], runtime["tetrahedra"], positions)
            if c["representation"] == "volume"
            else shell_metrics(runtime["points"], runtime["triangles"], positions)
        )
        if finite
        else {}
    )
    floor = float(c.get("environment", {}).get("floor_height", 0.0))
    half_thickness = (
        c["material"].get("thickness_m", 0.0) / 2 if c["representation"] == "surface" else 0.0
    )
    failures = []
    if not finite:
        failures.append("nonfinite_state")
    if finite and positions[:, :, 2].min() - half_thickness < floor - c.get("review", {}).get(
        "floor_tolerance_m", 0.003
    ):
        failures.append("floor_penetration")
    if metrics.get("inverted_element_samples", 0):
        failures.append("inverted_volume_element")
    if metrics.get("minimum_area_ratio", 1.0) < 0.05:
        failures.append("collapsed_surface_element")
    stretch = metrics.get("maximum_principal_stretch", metrics.get("maximum_edge_stretch"))
    if stretch is not None and stretch > c.get("review", {}).get("maximum_stretch", 1.5):
        failures.append("excessive_stretch")
    measurements = (
        measure_response(positions, np.arange(frames) / fps, c.get("measurements", []))
        if finite
        else []
    )
    failures.extend(
        f"measurement:{row['id']}:{reason}" for row in measurements for reason in row["failures"]
    )
    contact_audit = (
        audit_sphere_contacts(
            positions,
            runtime["triangles"],
            bodies,
            c.get("probes", []),
            closed=c["representation"] == "volume",
            tolerance_m=c.get("review", {}).get("contact_tolerance_m", 0.001),
            surface_thickness_m=2 * half_thickness,
        )
        if finite
        else {"ok": False}
    )
    if finite:
        box_audit = audit_box_contacts(
            positions,
            runtime["triangles"],
            bodies,
            c.get("probes", []),
            closed=c["representation"] == "volume",
            tolerance_m=c.get("review", {}).get("contact_tolerance_m", 0.001),
            surface_thickness_m=2 * half_thickness,
        )
        contact_audit["box_contacts"] = box_audit
        contact_audit["ok"] = contact_audit["ok"] and box_audit["ok"]
        contact_audit["unchecked_non_sphere_probes"] = []
    if not contact_audit["ok"]:
        failures.append("triangle_probe_penetration")
    displacement = np.linalg.norm(positions - positions[0], axis=2)
    report = {
        "ok": not failures,
        "failures": failures,
        "representation": c["representation"],
        "backend": "Newton native VBD",
        "frames": frames,
        "fps": fps,
        "metrics": metrics,
        "minimum_floor_clearance_m": float(positions[:, :, 2].min() - floor - half_thickness)
        if finite
        else None,
        "maximum_displacement_m": float(displacement.max()) if finite else None,
        "candidate_contact_frames": int(np.count_nonzero(contact_counts)),
        "contact_scope": "Broad-phase candidates, not measured contact force or all-pair coverage.",
        "elapsed_seconds": time.monotonic() - start,
        "material_initialization": runtime["material_initialization"],
        "measurements": measurements,
        "total_mass_kg": float(model.particle_mass.numpy().sum()),
        "real_physics_calibrated": False,
        "contact_audit": contact_audit,
        "delivery_scope": "Behavior diagnostic; does not replace source-bound asset acceptance.",
    }
    if "source_scene_sha256" in c:
        report["source_scene_sha256"] = c["source_scene_sha256"]
    np.savez_compressed(
        args.output / "motion.npz",
        positions=positions,
        velocities=velocities,
        body_poses=bodies,
        times=np.arange(frames) / fps,
        triangles=runtime["triangles"],
        tetrahedra=(
            runtime["tetrahedra"]
            if runtime["tetrahedra"] is not None
            else np.empty((0, 4), dtype=np.int32)
        ),
    )
    (args.output / "request.json").write_text(json.dumps(c, indent=2))
    report["evidence_sha256"] = {
        name: hashlib.sha256((args.output / name).read_bytes()).hexdigest()
        for name in ("motion.npz", "request.json")
    }
    (args.output / "validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
