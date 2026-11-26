"""Adopt a restored native world through the real outer controller.

``draft`` creates a proposal only. ``check`` recomputes recorded evidence without
running physics. ``advance`` is explicit: it starts the wall budget, executes six
read-only verification packets, accepts their reports and exports a binding.
It never collects data, trains a policy or substitutes hashes for semantic review.
"""
import argparse
import copy
import json
from pathlib import Path
import secrets
import sys

import numpy as np

from astra_e2e.layout import first_party_sources, pythonpath, unified
from astra_e2e.oversight.contracts import MODULES, intake
from astra_e2e.oversight.evidence import compare, digest, file_hash, fingerprint, native_twin, unchanged
from astra_e2e.oversight.planning import build_plan, review
from astra_e2e.oversight.policy_bridge import IO_FIELDS


FRONTEND = ("research", "capture", "model", "physics", "environment", "twin")
AUTO = {"model": {"geometry_error_m"}, "physics": {"contact_error_m", "native_robot_release_probe"},
        "environment": {"qualified_cousins"}, "twin": {"native_gates", "independent_freeze"}}


def read(path):
    def invalid(value):
        raise ValueError("Non-finite JSON: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def draft(native_config, context_path, output):
    """Context supplies physical/semantic choices; code supplies proposed budgets."""
    config_path, context_path = (Path(p).resolve(strict=True) for p in (native_config, context_path))
    config, context = read(config_path), read(context_path)
    if config.get("simulation_only") is not True or not config.get("success_rule"):
        raise ValueError("A native simulation proposal with exact success predicates is required")
    io = config.get("policy_io")
    if not isinstance(io, dict) or set(io) != set(IO_FIELDS):
        raise ValueError("Export actual adapter policy_io in native configuration before proposing intake")
    required = {"name", "objective", "source", "task", "fidelity", "variation", "allowed_workdirs",
                "gpu_ids", "memory_gb", "assumptions"}
    if set(context) != required:
        raise ValueError("Context keys must be exactly: " + ", ".join(sorted(required)))
    # These are new, disjoint simulation initialization draws, not new captures.
    # The split labels below are scenario identities, not independent source scenes.
    seeds = set()
    while len(seeds) < 56:
        seeds.add(secrets.randbelow(2**31))
    seeds = sorted(seeds)
    splits = {name: {"groups": [name + "_initial_conditions"], "seeds": seeds[left:right]}
              for name, left, right in (("train", 0, 32), ("validation", 32, 40), ("holdout", 40, 56))}
    spec = {"version": 1, "name": context["name"], "validation_scope": "simulation_verification",
            "objective": context["objective"], "source": copy.deepcopy(context["source"]),
            "task": {**context["task"], "success_predicates": config["success_rule"]},
            "policy": {"family": "custom", "selection_reason": "Astra selects a registered compatible policy from development evidence.", **io},
            "fidelity": context["fidelity"], "variation": context["variation"],
            "data": {"minimum_episodes": 24, "expert": "Astra selects a qualified registered simulation collector.",
                     "failure_policy": "success_only", "coverage": ["declared_cases", "initial_offsets"],
                     "split_unit": "scene", "splits": splits},
            "training": {"recipe": "Astra selects pinned compatible tools and bounded training parameters.",
                         "initialization": "From scratch unless an explicitly registered checkpoint is selected.",
                         "feedback_methods": ["targeted_data", "retrain", "research"],
                         "dagger_expert": "Unavailable unless separately registered and qualified; no implicit relabeling.",
                         "checkpoint_selection": "Development cases only; sealed holdout is used once."},
            "evaluation": {"success_rate_min": .75, "minimum_trials": 8, "statistic": "rate", "strata": ["nominal"],
                           "deployment_checks": ["io_parity", "action_bounds", "released_success", "simulation_only"]},
            "budget": {"max_cycles": 4, "max_attempts_per_module": 4, "max_jobs": 30,
                       "wall_seconds": 14400, "command_timeout_seconds": 3600,
                       "gpu_ids": context["gpu_ids"], "memory_gb": context["memory_gb"]},
            "authority": {"execute_local_jobs": True, "network_research": False, "physical_robot": False,
                          "publish": False, "allowed_workdirs": context["allowed_workdirs"]},
            "novelty": [], "requirements": {name: [] for name in MODULES}, "unresolved": [],
            "assumptions": context["assumptions"] + [
                "This proposal does not start execution or establish native acceptance.",
                "Split groups denote disjoint initial-condition draws of one restored scene; no independent-scene generalization claim.",
                "Geometry error measures restoration displacement from approved parent meshes, not real reconstruction accuracy.",
                "Privileged observations and initialized pregrasp restrict this run to simulation verification."]}
    spec["requirements"]["physics"].append(review("native_robot_release_probe"))
    assessment = intake(spec)
    if not assessment["ready"]:
        raise ValueError("Invalid candidate intake: " + "; ".join(assessment["questions"]))
    plan = build_plan(spec)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    spec_path = output / "intake.candidate.json"
    write(spec_path, spec)
    checks = {module: {item["id"]: {"value": False, "reason": "PENDING: independent inspection required.", "evidence": {}}
                       for item in plan["modules"][module]["checks"] if item["id"] not in AUTO.get(module, set())}
              for module in FRONTEND}
    write(output / "semantic_review.template.json", {
        "schema": "astra_frontend_semantic_review_v1", "reviewer": "PENDING",
        "scope": "Restored-world adoption; simulation verification only; no new reconstruction or real-robot claim.",
        "native_config_sha256": file_hash(config_path), "checks": checks})
    write(output / "evidence.template.json", {"schema": "astra_frontend_evidence_v1",
        "native_config": str(config_path), "native_config_sha256": file_hash(config_path),
        "geometry_pairs": [], "intrinsic_reports": [], "robot_probe": None,
        "qualified_cousins": []})
    package = {"schema": "astra_outer_candidate_v1", "status": "proposal_only",
               "native_config": str(config_path), "native_config_sha256": file_hash(config_path),
               "intake": str(spec_path), "intake_sha256": file_hash(spec_path),
               "context": str(context_path), "context_sha256": file_hash(context_path),
               "max_model_calls": 40, "frontend_jobs": 6,
               "scope": "No Controller initialization, simulation, collection, training or model call occurred."}
    write(output / "candidate.json", package)
    return package


def geometry_displacement(pairs):
    """Require identical vertex correspondence/topology; never invent correspondence."""
    if not pairs:
        raise ValueError("Supply approved parent/current mesh pairs for restoration preservation")
    values, files = [], []
    for pair in pairs:
        if set(pair) != {"label", "reference", "current", "reference_sha256", "current_sha256", "basis"}:
            raise ValueError("Geometry pair needs identity, both hashes and an explicit approved-reference basis")
        if not pair["basis"].strip():
            raise ValueError("A mesh hash is not approval of reference semantics")
        for key in ("reference", "current"):
            if file_hash(pair[key]) != pair[key + "_sha256"]:
                raise ValueError("Restoration mesh changed")
        with np.load(pair["reference"], allow_pickle=False) as a, np.load(pair["current"], allow_pickle=False) as b:
            va, vb = np.asarray(a["V"], float), np.asarray(b["V"], float)
            if va.shape != vb.shape or va.ndim != 2 or va.shape[1] != 3 or not len(va) or not np.isfinite([va, vb]).all():
                raise ValueError("Restoration comparison requires finite corresponding 3D vertices")
            if "F" not in a or "F" not in b or not np.array_equal(a["F"], b["F"]):
                raise ValueError("Restoration topology/correspondence changed; supply a separately reviewed measurement")
            values.append(float(np.linalg.norm(va - vb, axis=1).max()))
        files += [pair["reference"], pair["current"]]
    return max(values), files


def current_scene_inputs(session_path):
    """Inventory active geometry and the intrinsic proofs named by current reviews."""
    from calibrated_scene_reconstruction.scene.session import Session
    from calibrated_scene_reconstruction.core.paths import resolve
    session = Session(session_path)
    graph = session.scene_graph
    meshes, reports, reviews = set(), set(), []
    for kind in ("dynamic_objects", "furniture"):
        for name, obj in graph.get(kind, {}).items():
            fix = obj.get("fix", {})
            if fix.get("mesh"):
                meshes.add(str(resolve(fix["mesh"]).resolve(strict=True)))
            path = session.dir / "qa/object_understanding" / (name + ".json")
            if not path.is_file():
                continue  # Native gates independently decide whether this object requires it.
            reviews.append(str(path.resolve()))
            for part in read(path).get("parts", {}).values():
                for proof in part.get("physics", {}).values():
                    reports.add(str((session.dir / proof["report"]).resolve(strict=True)))
    return meshes, reports, reviews


def intrinsic_measurements(report_path, session):
    """Replay numerical intrinsic measurements from genuine recorded arrays.

    Native gate verification additionally binds declarations and current candidates.
    The replay is not an independent identification of real material parameters.
    """
    from calibrated_scene_reconstruction.scene.objects.rigid_intrinsic import measurements
    report_path, session = Path(report_path).resolve(strict=True), Path(session).resolve(strict=True)
    row = read(report_path)
    if (row.get("simulator") != "Isaac Sim 5.1" or row.get("task_actuation") is not False
            or row.get("pass") is not True or row.get("source_and_configuration_unchanged") is not True):
        raise ValueError("Intrinsic report is not a completed native independent assay")
    repository = Path(__file__).resolve().parents[2]
    producer = (repository / row["producer"]["path"]).resolve(strict=True)
    if not producer.is_relative_to(repository) or file_hash(producer) != row["producer"]["sha256"]:
        raise ValueError("Intrinsic numerical measurement producer changed")
    raw = {str((session / rel).resolve()): sha for rel, sha in row["raw_evidence"].items()}
    if not raw or any(not Path(p).is_relative_to(session) for p in raw) or not unchanged(raw):
        raise ValueError("Intrinsic raw evidence is missing or changed")
    directory = report_path.parent
    request_path = directory.with_name(directory.name + ".intrinsic_request.json")
    needed = [directory / leaf for leaf in ("native_result.json", "physics_poses.npz", "forces.npz", "native_contacts.npz")]
    if any(str(p) not in raw for p in [request_path, *needed]):
        raise ValueError("Completed native arrays/request must be members of the recorded evidence")
    request, native = read(request_path), read(needed[0])
    if (not unchanged(request["source_files_sha256"]) or not unchanged(native["implementation_sha256"])
            or native.get("implementation_unchanged") is not True
            or native.get("native_physics", {}).get("pass") is not True
            or native.get("precision", {}).get("pass") is not True):
        raise ValueError("Native implementation, source, physics or precision qualification is stale/failed")
    with np.load(needed[1], allow_pickle=False) as data:
        poses, times, names = data["poses"], data["times_s"], list(data["names"])
    with np.load(needed[3], allow_pickle=False) as data:
        contacts, contact_times = np.asarray(data["maximum_penetration_m"], float), data["times_s"]
        if (contacts.shape != (len(times) - 1,) or not np.array_equal(contact_times, times[1:])
                or not np.isfinite(contacts).all() or (contacts < 0).any()):
            raise ValueError("Native contact recording is incomplete or non-finite")
        native["maximum_native_penetration_m"] = float(contacts.max())
        if "contact_point_counts" not in data:
            raise ValueError("Native contact counts must be recorded, not inferred from a report")
        counts, bodies, filters = data["contact_point_counts"], list(data["body_paths"]), list(data["filter_paths"])
        if (counts.shape != (len(times) - 1, len(bodies), len(filters)) or not np.isfinite(counts).all()
                or (counts < 0).any() or not np.equal(counts, np.floor(counts)).all()):
            raise ValueError("Incomplete native contact-point count recording")
        pair_counts = {}
        for i, body in enumerate(bodies):
            for j, target in enumerate(filters):
                key = tuple(sorted((body, target)))
                pair_counts[key] = pair_counts.get(key, 0) + int(counts[:, i, j].sum())
        for pair in native["pairs"]:
            actual = pair_counts.pop(tuple(sorted(pair["paths"])), 0)
            if pair["recorded_contact_points"] != actual:
                raise ValueError("Reported contact-point count differs from native array")
        if any(pair_counts.values()):
            raise ValueError("Native contact pairs omitted measured contacts")
    with np.load(needed[2], allow_pickle=False) as data:
        forces, torques, force_times = data["forces_N"], data["torques_Nm"], data["times_s"]
        if not np.array_equal(force_times, times[:-1]) or not np.isfinite(forces).all() or not np.isfinite(torques).all():
            raise ValueError("Native force recording is incomplete or non-finite")
        native["peak_force_N"] = float(np.linalg.norm(forces, axis=-1).max())
        native["peak_torque_Nm"] = float(np.linalg.norm(torques, axis=-1).max())
        probe = request["probe_plan"]
        if probe is not None:
            release = force_times >= probe["start_s"] + probe["duration_s"]
            if (probe.get("gravity_compensation") is not False or not release.any()
                    or not np.all(forces[release] == 0) or not np.all(torques[release] == 0)
                    or native["peak_force_N"] <= 0):
                raise ValueError("Active intrinsic test needs real force application followed by zero-force release")
    computed = measurements(native, poses, times,
        body_indices=[names.index(name) for name in request["tested_bodies"]],
        settled_window_s=request["settled_window_s"], masses=request["native_masses_kg"],
        inertias=request["native_inertias_kg_m2"])
    part = next(p for p in request["understanding"]["parts"] if p["id"] == request["part_id"])
    test = next(t for t in part["physics_tests"] if t["id"] == request["test_id"])
    if test["purpose"] == "contact_response" and computed["recorded_contact_points"]["value"] <= 0:
        raise ValueError("Contact-response assay recorded no tested-body contact")
    for key, value in computed.items():
        recorded = row["all_measurements"][key]
        if recorded["unit"] != value["unit"] or not np.isclose(recorded["value"], value["value"], atol=1e-10, rtol=1e-7):
            raise ValueError("Intrinsic report disagrees with numerical replay: " + key)
    for metric in test["metrics"]:
        value = computed[metric["name"]]
        if value["unit"] != metric["unit"] or not metric["bounds"][0] <= value["value"] <= metric["bounds"][1]:
            raise ValueError("Intrinsic replay violates declared physical prior: " + metric["name"])
    return computed["maximum_penetration_m"]["value"], [str(report_path), str(producer), *raw, *request["source_files_sha256"], *native["implementation_sha256"]]


def robot_probe(report_path, config_path, session):
    """Re-score unbound qualification traces; they never become training data."""
    from bound_native_review import success_from_measurement
    report, config = read(report_path), read(config_path)
    if (report.get("schema") != "astra_unbound_native_probe_v1" or report.get("is_policy_receipt") is not False
            or report.get("config_sha256") != file_hash(config_path)
            or report.get("factory_sha256") != file_hash(config["factory_file"])):
        raise ValueError("Robot probe is not bound to this exact native configuration/factory")
    runtime = report["metadata"]
    if (runtime.get("engine") != "isaacsim-physx" or Path(runtime["native_session"]).resolve() != Path(session).resolve()
            or runtime.get("success_rule") != config["success_rule"]
            or runtime.get("policy_io_sha256") != digest(config["policy_io"])
            or not runtime.get("loaded_scene_inputs") or not unchanged(runtime["loaded_scene_inputs"])
            or runtime.get("final_diagnostics", {}).get("native_physics_errors", {}).get("pass") is not True
            or runtime.get("final_diagnostics", {}).get("collider_precision", {}).get("pass") is not True):
        raise ValueError("Robot probe has stale inputs or failed native runtime/precision evidence")
    implementation = report.get("implementation_sha256", runtime.get("implementation_sha256", {}))
    if not implementation or not unchanged(implementation) or report.get("implementation_unchanged") is not True:
        raise ValueError("Robot qualification must pin actual executed implementation files")
    if runtime.get("robot_usd") and file_hash(runtime["robot_usd"]) != runtime.get("robot_usd_sha256"):
        raise ValueError("Qualified robot articulation asset changed")
    external = runtime.get("external_runtime_inputs", {})
    expected_assets = {str(Path(path).resolve()) for path in config.get("runtime_asset_files", [])}
    if (not external or not set(external) <= expected_assets or not unchanged(external)
            or str(Path(runtime["robot_usd"]).resolve()) not in external):
        raise ValueError("Robot specification and USD dependency closure need exact current qualification pins")
    files, successes, seeds = [str(report_path), config["factory_file"], *runtime["loaded_scene_inputs"]], 0, []
    files += [*implementation, *external]
    if runtime.get("robot_usd"):
        files.append(runtime["robot_usd"])
    for row in report["episodes"]:
        if file_hash(row["trace"]) != row["sha256"]:
            raise ValueError("Robot probe trace changed")
        trace = read(row["trace"])
        initial = trace["initial_measurement"]
        if (not initial or initial.get("attachment_active") is not True or initial.get("success") is not False
                or success_from_measurement(initial, config["success_rule"])):
            raise ValueError("Qualification must include a non-successful attached initial state")
        streak, previous = 0, None
        for index, step in enumerate(trace["steps"]):
            action = np.asarray(step["executed_action"], float)
            obs_dim = sum(int(np.prod(channel["shape"])) for channel in config["policy_io"]["observations"])
            observation, nxt = (np.asarray(step[key], float) for key in ("observation", "next_observation"))
            if (observation.shape != (obs_dim,) or nxt.shape != (obs_dim,)
                    or not np.isfinite(observation).all() or not np.isfinite(nxt).all()):
                raise ValueError("Probe observations are non-finite or differ from actual adapter I/O")
            if (step["step"] != index or action.shape != np.asarray(config["action_low"]).shape
                    or not np.isfinite(action).all() or (action < config["action_low"]).any()
                    or (action > config["action_high"]).any()
                    or step["transition"].get("teleport_used") is not False
                    or step["transition"].get("physics_steps", 0) < 1
                    or not np.array_equal(action, step["transition"].get("command"))):
                raise ValueError("Probe lacks bounded exact native command execution")
            if previous is not None and not np.array_equal(previous, step["observation"]):
                raise ValueError("Qualification observation timing discontinuity")
            previous = step["next_observation"]
            scored = success_from_measurement(step["measurement"], config["success_rule"])
            if step["measurement"].get("success") is not scored:
                raise ValueError("Qualification success differs from frozen predicates")
            streak = streak + 1 if scored else 0
        success = trace["failure"] is None and streak >= config["success_ticks"]
        if trace["success"] is not success or row["success"] is not success or len(trace["steps"]) > config["max_steps"]:
            raise ValueError("Qualification outcome/counting is inconsistent")
        if success and trace["steps"][-1]["measurement"].get("attachment_active") is not False:
            raise ValueError("Successful native robot probe must remain successful after fixture release")
        successes += int(success)
        seeds.append(trace["case"]["seed"])
        files.append(row["trace"])
    if successes < 2 or len(seeds) != len(set(seeds)):
        raise ValueError("At least two distinct released successes in one probe run are required to qualify sequential resets")
    return {"successes": successes, "trials": len(seeds), "seeds": seeds}, files


def checks(intake_path, semantic_path, evidence_path, module):
    """Return truthful current checks; missing/failed evidence raises, never passes."""
    spec, semantic, evidence = read(intake_path), read(semantic_path), read(evidence_path)
    if module not in FRONTEND:
        raise ValueError("Frontend adoption cannot execute a learning module")
    config_path = evidence["native_config"]
    config = read(config_path)
    if (evidence.get("schema") != "astra_frontend_evidence_v1"
            or semantic.get("schema") != "astra_frontend_semantic_review_v1"
            or semantic.get("native_config_sha256") != file_hash(config_path)
            or evidence.get("native_config_sha256") != file_hash(config_path)
            or semantic.get("reviewer") in (None, "", "PENDING") or not semantic.get("scope")
            or config["success_rule"] != spec["task"]["success_predicates"]
            or config["policy_io"] != {k: spec["policy"][k] for k in IO_FIELDS}):
        raise ValueError("Semantic evidence, native config and intake do not share the frozen operating contract")
    expected = build_plan(spec)["modules"][module]["checks"]
    rows, files = {}, [str(Path(p).resolve()) for p in (intake_path, semantic_path, evidence_path, config_path)]
    semantic_rows = semantic["checks"].get(module, {})
    semantic_ids = {c["id"] for c in expected} - AUTO.get(module, set())
    if set(semantic_rows) != semantic_ids:
        raise ValueError("Semantic review IDs differ from exact planned checks: " + module)
    for key, row in semantic_rows.items():
        if (set(row) != {"value", "reason", "evidence"} or type(row["value"]) is not bool
                or not row["reason"].strip() or row["reason"].startswith("PENDING")
                or not row["evidence"] or not unchanged(row["evidence"])):
            raise ValueError("Missing substantive current semantic review: " + module + "/" + key)
        fingerprint(row["evidence"])  # Absolute actual file paths, not inferred labels.
        rows[key] = {"value": row["value"], "reason": row["reason"], "evidence": list(row["evidence"]) + [str(Path(semantic_path).resolve())]}
        files += list(row["evidence"])
    def measured(key, value, reason, paths):
        rows[key] = {"value": value, "reason": reason, "evidence": list(paths)}
        files.extend(paths)
    if module == "model":
        meshes, _, _ = current_scene_inputs(spec["source"]["session"])
        selected = [str(Path(pair["current"]).resolve()) for pair in evidence["geometry_pairs"]]
        if not meshes or set(selected) != meshes or len(selected) != len(set(selected)):
            raise ValueError("Restoration measurement must cover every active reconstructed scene mesh exactly once")
        value, paths = geometry_displacement(evidence["geometry_pairs"])
        measured("geometry_error_m", value, "Maximum corresponding vertex displacement from approved parent geometry; restoration preservation only, not real reconstruction accuracy.", paths)
    if module == "physics":
        if not evidence["intrinsic_reports"]:
            raise ValueError("Completed native intrinsic reports are required")
        _, expected_reports, current_reviews = current_scene_inputs(spec["source"]["session"])
        supplied = [str(Path(path).resolve()) for path in evidence["intrinsic_reports"]]
        if not expected_reports or set(supplied) != expected_reports or len(supplied) != len(set(supplied)):
            raise ValueError("Numerical replay must include every intrinsic report cited by current object/support reviews")
        files.extend(current_reviews)
        replay = [intrinsic_measurements(path, spec["source"]["session"]) for path in evidence["intrinsic_reports"]]
        measured("contact_error_m", max(v for v, _ in replay), "Maximum penetration replayed from current native intrinsic contact arrays; semantic native checks remain separate.", [p for _, paths in replay for p in paths])
        result, paths = robot_probe(evidence["robot_probe"], config_path, spec["source"]["session"])
        if set(result["seeds"]) & {seed for split in spec["data"]["splits"].values() for seed in split["seeds"]}:
            raise ValueError("Qualification starts overlap the new policy experiment")
        measured("native_robot_release_probe", True, f"Replayed {result['successes']}/{result['trials']} released native qualification successes; initial attached states were negative. This is not policy training/evaluation.", paths)
    if module == "environment":
        if evidence.get("qualified_cousins"):
            raise ValueError("Cousin adoption needs a separately implemented per-cousin semantic verifier")
        measured("qualified_cousins", 0, "No cousins are qualified or claimed; this contract evaluates one restored scene.", [str(Path(evidence_path).resolve())])
    if module == "twin":
        native_twin(spec["source"]["session"])
        freeze_path = Path(spec["source"]["session"]) / "qa/independent_freeze.json"
        from calibrated_scene_reconstruction.scene.session import Session
        session = Session(spec["source"]["session"])
        roots = {"session": session.dir, "config": session.config_dir}
        paths = [str(freeze_path), *(str((roots[row["base"]] / row["path"]).resolve()) for row in read(freeze_path)["files"])]
        measured("native_gates", True, "Recomputed every applicable native stage/object/support gate; no top-level override.", paths)
        measured("independent_freeze", True, "Verified the existing native freeze and every frozen member; semantics are established by separate inspected reviews.", paths)
    failed = [c["id"] for c in expected if not compare(c, rows[c["id"]]["value"])]
    return {"module": module, "checks": rows, "failed_checks": failed, "inputs": fingerprint(sorted(set(files))),
            "scope": "Adoption and re-verification of existing artifacts, not re-running capture/reconstruction or policy learning."}


def advance(candidate_path, semantic_path, evidence_path, run_path, binding_path, *, task_id, python=sys.executable):
    """Explicitly accept the front half, never start learning as a side effect."""
    from astra_e2e.oversight.controller import Controller
    from astra_e2e.oversight.policy_bridge import export_binding
    package = read(candidate_path)
    if (package.get("schema") != "astra_outer_candidate_v1"
            or file_hash(package["intake"]) != package["intake_sha256"]
            or file_hash(package["native_config"]) != package["native_config_sha256"]
            or file_hash(package["context"]) != package["context_sha256"]):
        raise ValueError("Candidate changed; regenerate the proposal before activation")
    selected_evidence = read(evidence_path)
    if (selected_evidence.get("native_config_sha256") != package["native_config_sha256"]
            or file_hash(selected_evidence["native_config"]) != package["native_config_sha256"]):
        raise ValueError("Qualification evidence refers to a different native configuration than this candidate")
    # All dependencies must already be inspectable before starting the wall budget.
    reports = [checks(package["intake"], semantic_path, evidence_path, module) for module in FRONTEND]
    if any(r["failed_checks"] for r in reports):
        raise ValueError("Frontend adoption has failed checks; no controller was initialized")
    repository = Path(__file__).resolve().parents[2]
    code = [*first_party_sources(repository), *unified(repository).glob("*.py")]
    pinned = fingerprint([str(p.resolve()) for p in code] + [str(Path(candidate_path).resolve())])
    for report in reports:
        pinned.update(report["inputs"])
    run_path = Path(run_path).resolve()
    if run_path.exists():
        raise ValueError("Use a new outer run; never overwrite an experiment")
    ctrl = Controller.create(run_path, read(package["intake"]))
    input_path = run_path / "frontend_inputs.json"
    write(input_path, pinned)
    for module in FRONTEND:
        output = run_path / (module + ".verification.json")
        result = ctrl.run({"module": module, "purpose": "Independently reverify restored-world frontend evidence for " + module,
            "commands": [{"argv": [str(python), "-B", str(Path(__file__).resolve()), "check", "--intake", package["intake"],
                "--semantic", str(Path(semantic_path).resolve()), "--evidence", str(Path(evidence_path).resolve()),
                "--module", module, "--pins", str(input_path), "--output", str(output)],
                "cwd": str(repository), "env": {"PYTHONPATH": pythonpath(repository, unified(repository))},
                "inputs": [*pinned, str(input_path)], "timeout_seconds": min(3600, read(package["intake"])["budget"]["command_timeout_seconds"])}]})
        verified = read(output)
        report = {"module": module, "attempt": result["attempt"], "plan_sha256": ctrl.status()["plan_sha256"],
                  "summary": verified["scope"], "artifacts": [str(output)], "checks": verified["checks"], "failures": [],
                  "native_session": read(package["intake"])["source"]["session"] if module == "twin" else None,
                  "policy_io": None, "dataset_manifest": None, "checkpoint": None, "evaluation": None}
        verdict = ctrl.accept(report)
        if not verdict["accepted"]:
            raise ValueError("Frontend check rejected; retained actual failed attempt: " + module)
    config = read(package["native_config"])
    return export_binding(run_path, binding_path, task_id=task_id, backend_id=config["backend_id"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    p = commands.add_parser("draft")
    for key in ("native-config", "context", "output"):
        p.add_argument("--" + key, required=True)
    p = commands.add_parser("check")
    for key in ("intake", "semantic", "evidence", "module", "output"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--pins")
    p = commands.add_parser("advance")
    for key in ("candidate", "semantic", "evidence", "run", "binding", "task-id"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    if args.operation == "draft":
        print(json.dumps(draft(args.native_config, args.context, args.output)))
    elif args.operation == "check":
        if args.pins and not unchanged(read(args.pins)):
            raise ValueError("Pinned frontend evidence changed")
        result = checks(args.intake, args.semantic, args.evidence, args.module)
        write(args.output, result)
        if result["failed_checks"]:
            raise SystemExit(2)
    else:
        result = advance(args.candidate, args.semantic, args.evidence, args.run, args.binding,
                         task_id=args.task_id, python=args.python)
        print(json.dumps({"binding": str(Path(args.binding).resolve()), "binding_sha256": result["binding_sha256"],
                          "learning_started": False, "real_validation": "pending"}))


if __name__ == "__main__":
    main()
