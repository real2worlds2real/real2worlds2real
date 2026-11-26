"""Measure declared rigid-body intrinsic tests through the native assay tool.

This adapter neither identifies real materials nor approves object semantics.
The agent must inspect source evidence and declare scenarios, bounds, uncertainty
and setup assumptions first. Flexible/uncertain-to-rigid approximation needs a
separate compliance test and is deliberately unsupported here.
"""
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from .object_understanding import declaration, digest


UNITS = {
    "maximum_translation_m": "m", "final_translation_m": "m",
    "settled_translation_m": "m", "maximum_rotation_rad": "rad",
    "maximum_linear_speed_m_s": "m/s", "maximum_penetration_m": "m",
    "recorded_contact_points": "1", "peak_force_N": "N", "peak_torque_Nm": "N*m",
    "native_mass_kg": "kg", "native_minimum_inertia_kg_m2": "kg*m^2",
}
ASSUMPTIONS = {"initial_state", "attachments", "fixture", "material_uncertainty",
               "contact_uncertainty", "source_robot_placement"}


def measurements(native, poses, times, *, body_indices, settled_window_s, masses, inertias):
    """Compute quantities from recorded native trajectories, without a verdict."""
    poses, times = np.asarray(poses, float), np.asarray(times, float)
    if (poses.ndim != 3 or poses.shape[2] != 7 or times.shape != (len(poses),)
            or len(times) < 2 or not np.isfinite(poses).all() or not np.isfinite(times).all()
            or (np.diff(times) <= 0).any() or not body_indices
            or not 0 < settled_window_s <= times[-1] - times[0]):
        raise ValueError("Finite complete native trajectories and a bounded settling window required")
    selected = poses[:, body_indices, :]
    p, q = selected[:, :, :3], selected[:, :, 3:]
    norms = np.linalg.norm(q, axis=2)
    if not np.allclose(norms, 1., atol=1e-3, rtol=0):
        raise ValueError("Native orientations must be unit quaternions")
    q = q / norms[:, :, None]
    displacement = np.linalg.norm(p - p[:1], axis=2)
    speed = np.linalg.norm(np.diff(p, axis=0) / np.diff(times)[:, None, None], axis=2)
    tail = p[times >= times[-1] - settled_window_s]
    if len(tail) < 2:
        raise ValueError("Settling window must include at least two recorded states")
    tested_paths = {native["body_paths"][index] for index in body_indices}
    values = {
        "maximum_translation_m": float(displacement.max()),
        "final_translation_m": float(displacement[-1].max()),
        "settled_translation_m": float(np.linalg.norm(tail - tail[:1], axis=2).max()),
        "maximum_rotation_rad": float((2 * np.arccos(np.clip(np.abs((q * q[:1]).sum(2)), 0, 1))).max()),
        "maximum_linear_speed_m_s": float(speed.max()),
        "maximum_penetration_m": float(native["maximum_native_penetration_m"]),
        "recorded_contact_points": int(sum(row["recorded_contact_points"] for row in native["pairs"]
                                           if tested_paths.intersection(row["paths"]))),
        "peak_force_N": float(native.get("peak_force_N", 0.)),
        "peak_torque_Nm": float(native.get("peak_torque_Nm", 0.)),
        "native_mass_kg": float(np.sum(masses)),
        "native_minimum_inertia_kg_m2": float(min(np.linalg.eigvalsh(i).min() for i in inertias)),
    }
    if not np.isfinite(list(values.values())).all() or min(masses) <= 0 or values["native_minimum_inertia_kg_m2"] <= 0:
        raise ValueError("Native measurements, mass and inertia must be finite and physically valid")
    return {name: {"value": value, "unit": UNITS[name]} for name, value in values.items()}


def _snapshot(session):
    """Bind configuration, active geometry and source calibration, not old QA."""
    files = {}
    for file in sorted(session.config_dir.rglob("*")):
        if file.is_file() and "__pycache__" not in file.parts and file.suffix != ".pyc":
            files[str(file.resolve())] = digest(file)
    for group, directory in (("dynamic_objects", "objects"), ("furniture", "support")):
        for name, spec in session.scene_graph.get(group, {}).items():
            fix = spec.get("fix") or {}
            if not fix.get("model"):
                continue
            for leaf in ("model.py", "mesh.npz"):
                path = session.dir / directory / name / f"round_{fix['round']}" / leaf
                files[str(path.resolve())] = digest(path)
    for leaf in ("parsing/world_frame.json", "splat/robot_visual_fit.json", "splat/robot_joint_fit.json",
                 "splat/robot_fit.json", "splat/align.json"):
        path = session.dir / leaf
        if path.is_file():
            files[str(path.resolve())] = digest(path)
    return files


def run(ctx, obj, part_id, test_id, output, *, native_run, seconds, settled_window_s,
        assumptions, probe_plan=None, observed_bodies=None, maximum_penetration_m=None):
    """Run one declared test; return a hash-bound proof for an independent review.

    Call inside a native task-check context before policy/task execution. The
    caller sets up or resets the scene and discloses that setup in assumptions.
    The application injects the native assay executor; the scene layer owns
    declarations, measurements and evidence rather than importing task runners.
    Contact and material-response tests require an independent force-and-release
    probe with gravity compensation disabled. No pose commands occur here.
    """
    if not callable(native_run):
        raise ValueError("The composition application must supply a native assay executor")
    version = importlib.metadata.version("isaacsim")
    if not version.startswith("5.1."):
        raise ValueError("Intrinsic qualification requires the primary Isaac Sim 5.1 backend")
    session = ctx.s
    item = declaration(session, obj)
    part = next((p for p in item["parts"] if p["id"] == part_id), None)
    if part is None or part["representation"] != "rigid" or part["expected_response"] != "rigid":
        raise ValueError("This tool only tests source-justified rigid parts; no flexible approximation approval")
    test = next((t for t in part["physics_tests"] if t["id"] == test_id), None)
    if test is None or test["purpose"] not in {"passive_state", "contact_response", "material_response"}:
        raise ValueError("Choose one declared independent rigid test")
    if not isinstance(assumptions, dict) or not ASSUMPTIONS <= assumptions.keys() or any(
            not isinstance(assumptions[k], str) or not assumptions[k].strip() for k in ASSUMPTIONS):
        raise ValueError("Disclose initial state, attachments, fixture, robot placement and material/contact uncertainty")
    for metric in test["metrics"]:
        if UNITS.get(metric["name"]) != metric["unit"]:
            raise ValueError("Unsupported intrinsic metric or units: " + metric["name"])
    active_probe = test["purpose"] != "passive_state"
    if (not active_probe and probe_plan is not None) or (active_probe and (
            not isinstance(probe_plan, dict) or probe_plan.get("gravity_compensation") is not False)):
        raise ValueError("Passive tests are unforced; contact/material probes must be bounded and not compensate gravity")
    names = [obj + "/" + link for link in part["links"]]
    observed = list(observed_bodies) if observed_bodies is not None else names
    if len(set(observed)) != len(observed) or not set(names) <= set(observed):
        raise ValueError("Observe every physical link of the tested part exactly once")
    if active_probe and probe_plan.get("link") not in names:
        raise ValueError("The independent probe must load a physical link owned by the tested part")
    output = Path(output).resolve()
    if not output.is_relative_to((session.dir / "qa").resolve()) or output.exists():
        raise ValueError("Use a new output directory under this session's QA")
    if not np.isfinite([seconds, settled_window_s]).all() or not 0 < settled_window_s <= seconds <= 30:
        raise ValueError("Declare a positive settling window within a bounded assay duration")
    before = _snapshot(session)
    candidate_dir = session.dir / "objects" / obj / f"round_{session.scene_graph['dynamic_objects'][obj]['fix']['round']}"
    candidate = {name: digest(candidate_dir / name) for name in ("model.py", "mesh.npz")}
    masses, inertias = zip(*(ctx.rigid_mass_properties(name) for name in names))
    binding = {"candidate_sha256": candidate, "understanding": item, "part_id": part_id,
               "test_id": test_id, "scenario": test["scenario"], "source_files_sha256": before,
               "source_robot_placement": assumptions["source_robot_placement"], "assumptions": assumptions,
               "observed_bodies": observed, "tested_bodies": names, "settled_window_s": settled_window_s,
               "seconds": seconds, "probe_plan": probe_plan, "maximum_penetration_m": maximum_penetration_m,
               "native_masses_kg": list(map(float, masses)),
               "native_inertias_kg_m2": [np.asarray(i).tolist() for i in inertias]}
    # Save intent before any step. Failed native trials retain the same request,
    # even when the simulator cannot produce a completed response report.
    request = output.with_name(output.name + ".intrinsic_request.json")
    request.parent.mkdir(parents=True, exist_ok=True)
    with request.open("x") as stream:
        json.dump(binding, stream, indent=2)
        stream.write("\n")
    native, poses, times = native_run(
        ctx, observed, seconds, output, probe_plan=probe_plan,
        maximum_penetration_m=maximum_penetration_m)
    measured = measurements(native, poses, times, body_indices=[observed.index(name) for name in names],
                            settled_window_s=settled_window_s, masses=masses, inertias=inertias)
    valid = (before == _snapshot(session) and native["finite_poses"] and native["implementation_unchanged"]
             and native["native_physics"]["pass"] and native["precision"]["pass"])
    if active_probe:
        valid = valid and native["release_forces_zero"] and native["release_torques_zero"] and measured["peak_force_N"]["value"] > 0
    if test["purpose"] == "contact_response":
        valid = valid and measured["recorded_contact_points"]["value"] > 0
    selected = {metric["name"]: measured[metric["name"]] for metric in test["metrics"]}
    passed = valid and all(metric["bounds"][0] <= selected[metric["name"]]["value"] <= metric["bounds"][1]
                           for metric in test["metrics"])
    raw = {str(file.relative_to(session.dir)): digest(file) for file in sorted(output.rglob("*")) if file.is_file()}
    raw[str(request.relative_to(session.dir))] = digest(request)
    producer = Path(__file__).resolve()
    report = {"version": 1, "simulator": "Isaac Sim 5.1", "simulator_version": version,
              "test_id": test_id, "candidate_sha256": candidate, "understanding": item,
              "task_actuation": False, "pass": bool(passed), "measurements": selected,
              "all_measurements": measured, "raw_evidence": raw,
              "producer": {"path": str(producer.relative_to(producer.parents[4])), "sha256": digest(producer)},
              "source_and_configuration_unchanged": before == _snapshot(session),
              "material_identification": False, "semantic_approval": False, "robot_execution_approval": False,
              "scope": "Native rigid-body passive/contact/force-release response against declared priors; "
                       "no real stiffness identification, no flexible approximation proof, no task success judgment. "
                       "Robot placement and any initial attachment/fixture remain explicit setup assumptions."}
    target = output / "intrinsic_report.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    relative = str(target.relative_to(session.dir))
    return {"pass": bool(passed), "report": relative, "evidence": {relative: digest(target)}}
