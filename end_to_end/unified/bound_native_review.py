"""Independent artifact review for bound vector tools; unknown obligations fail.

The reviewer replays finite success predicates over saved physical measurements,
checks labels against executed-command traces, checks held-out accounting and
recomputes checkpoint lineage. This is not a substitute for upstream independent
reconstruction/contact review or real-robot validation.
"""
import json
from pathlib import Path

import numpy as np

from bound_native import cases
from bound_training import FORMAT, _dimensions, predict, read, write_new
from astra_e2e.oversight.evidence import compare, dataset_manifest, digest, evaluate_episodes, file_hash, unchanged
from astra_e2e.oversight.policy_bridge import _resolve_artifacts, verify_binding


def success_from_measurement(measurement, rules):
    """A small closed predicate language; no executable policy-selected scoring."""
    if not rules:
        return False
    for rule in rules:
        value = measurement.get(rule["field"])
        target, operation = rule["target"], rule["operator"]
        if operation == "eq" and type(target) is bool:
            passed = value is target
        else:
            if type(value) not in (int, float) or not np.isfinite(value):
                return False
            if operation == "le":
                passed = value <= target
            elif operation == "ge":
                passed = value >= target
            elif operation == "abs_le":
                passed = abs(value) <= target
            elif operation == "eq":
                passed = value == target
            else:
                raise ValueError("Unknown frozen success predicate")
        if not passed:
            return False
    return True


def _trace_check(trace, binding, request, config, *, checkpoint_sha=None):
    obs_dim, act_dim = (_dimensions(binding["policy_io"][key]) for key in ("observations", "actions"))
    if (trace.get("schema") != "astra_bound_trace_v1" or trace.get("binding_sha256") != binding["binding_sha256"]
            or trace.get("world_sha256") != binding.get("world_sha256")
            or trace.get("request_sha256") != digest(request)
            or trace.get("policy_io_sha256") != digest(binding["policy_io"])
            or trace.get("checkpoint_sha256") != checkpoint_sha):
        raise ValueError("Trace lineage differs from this executed job")
    if trace.get("label_timing") != "observation_t -> executed_command_t -> physics_step -> measurement_t_plus_1":
        raise ValueError("Unknown command-label alignment")
    if trace["teacher_used"] is not (request["operation"] == "collect"):
        raise ValueError("Teacher executed outside collection, or collection omitted teacher lineage")
    steps, streak = trace["steps"], 0
    if len(steps) > config["max_steps"]:
        raise ValueError("Trace exceeds frozen horizon")
    previous = None
    for index, step in enumerate(steps):
        if step["step"] != index:
            raise ValueError("Non-contiguous rollout trace")
        obs, action, nxt = (np.asarray(step[key], float) for key in ("observation", "executed_action", "next_observation"))
        if (obs.shape != (obs_dim,) or action.shape != (act_dim,) or nxt.shape != (obs_dim,)
                or not all(np.isfinite(v).all() for v in (obs, action, nxt))):
            raise ValueError("Trace vector I/O mismatch")
        if previous is not None and not np.array_equal(previous, obs):
            raise ValueError("Next observation does not equal following policy input")
        previous = nxt
        if not (np.asarray(config["action_low"]) <= action).all() or not (action <= np.asarray(config["action_high"])).all():
            raise ValueError("Executed command exceeds pinned actuator bounds")
        transition = step["transition"]
        if (not isinstance(transition, dict) or transition.get("physics_steps", 0) < 1
                or transition.get("teleport_used") is not False
                or not np.array_equal(np.asarray(transition.get("command")), action)):
            raise ValueError("Transition lacks exact physical command execution without teleportation")
        measured = success_from_measurement(step["measurement"], config["success_rule"])
        if step["measurement"].get("success") is not measured:
            raise ValueError("Worker success disagrees with independent frozen predicate")
        streak = streak + 1 if measured else 0
        latency = step["latency_ms"]
        if type(latency) not in (float, int) or not np.isfinite(latency) or latency < 0:
            raise ValueError("Invalid measured policy latency")
    expected = bool(trace["failure"] is None and streak >= config["success_ticks"])
    if trace["success"] is not expected:
        raise ValueError("Episode success differs from independent consecutive-step accounting")
    if trace["initial_measurement"] and success_from_measurement(trace["initial_measurement"], config["success_rule"]):
        raise ValueError("Trivial already-successful reset")
    return True


def review(binding_path, result, *, output_dir):
    binding = verify_binding(binding_path)
    receipt = read(result["receipt"])
    paths, refs = _resolve_artifacts(receipt, result["artifact_paths"])
    info = receipt["facts"]["oversight"]
    module = info["module"]
    from astra_e2e.oversight.controller import Controller
    controller = Controller(binding["oversight_run"])
    plan, state = controller._load()
    row, spec = state["modules"][module], plan["contract"]
    if row["status"] != "awaiting_review":
        raise ValueError("Only completed current attempts can be independently reviewed")
    attempt = row["attempts"][-1]
    if not unchanged(attempt["inputs"]):
        raise ValueError("Attempt inputs changed before independent review")
    requests = [read(path) for path in attempt["inputs"] if Path(path).name == "request.json"]
    matches = [r for r in requests if r.get("schema") == "astra_bound_job_v1" and digest(r) == info["request_sha256"]]
    if len(matches) != 1:
        raise ValueError("Reviewer cannot identify the unique pinned job")
    request = matches[0]
    if read(paths[info["policy_io_ref"]]) != binding["policy_io"]:
        raise ValueError("Artifact policy I/O differs from binding")
    evidence = sorted({str(Path(result["receipt"]).resolve()), *paths.values()})
    artifact_files = {str(Path(path).resolve()) for path in paths.values()}
    diagnostics, values, reasons = {}, {}, {}

    def record(name, value, reason):
        values[name], reasons[name] = value, reason

    if module == "train":
        checkpoint = paths[info["checkpoint_ref"]]
        with np.load(checkpoint, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            finite = all(np.isfinite(archive[key]).all() for key in archive.files if key != "metadata_json")
            weights = {key: archive[key].copy() for key in archive.files if key != "metadata_json"}
        accepted = state["modules"]["data"]["accepted"]
        manifest = read(metadata["dataset_manifest"])
        expected_ids = [ep["id"] for ep in manifest["episodes"] if ep["split"] == "train"]
        squared_error, frame_count, decoder_finite = 0., 0, True
        obs_dim, act_dim = (_dimensions(binding["policy_io"][key]) for key in ("observations", "actions"))
        for episode in manifest["episodes"]:
            if episode["split"] != "train":
                continue
            episode_path = str(Path(episode["artifact"]).resolve(strict=True))
            if accepted["files"].get(episode_path) != file_hash(episode_path):
                raise ValueError("Training review encountered an episode outside current accepted data")
            with np.load(episode_path, allow_pickle=False) as archive:
                x, y = np.asarray(archive["observations"], float), np.asarray(archive["actions"], float)
            if x.ndim != 2 or y.ndim != 2 or x.shape[1] != obs_dim or y.shape != (len(x), act_dim):
                raise ValueError("Accepted training arrays no longer match bound vector I/O")
            prediction = predict(weights, x)
            decoder_finite &= prediction.shape == y.shape and np.isfinite(prediction).all()
            squared_error += float(np.sum((prediction - y)**2))
            frame_count += len(x)
        metrics = read(paths["metrics"])
        recomputed_mse = squared_error / (frame_count * act_dim) if frame_count else float("inf")
        mse_match = bool(np.isclose(recomputed_mse, metrics["training_mse"], rtol=1e-8, atol=1e-12)
                         and metrics["frames"] == frame_count)
        record("io_parity", metadata.get("schema") == FORMAT and metadata.get("policy_io") == binding["policy_io"]
               and finite and bool(decoder_finite) and frame_count > 0 and mse_match,
               "Decoded the saved checkpoint on accepted training observations, checked output shape/finite values and independently recomputed training MSE.")
        provenance = (metadata.get("binding_sha256") == binding["binding_sha256"]
                      and metadata.get("request_sha256") == digest(request)
                      and metadata.get("data_acceptance_id") == accepted["id"]
                      and metadata.get("training_episode_ids") == expected_ids
                      and accepted["files"].get(metadata["dataset_manifest"]) == file_hash(metadata["dataset_manifest"])
                      and metadata.get("dataset_sha256") == file_hash(metadata["dataset_manifest"]))
        record("training_provenance", provenance, "Matched current accepted data, exact training episode IDs and request hash.")
        record("checkpoint_selection", provenance and metrics.get("validation_used_for_fit") is False
               and metadata.get("method") == request["method"],
               "This job exposes one terminal checkpoint, fitted on training rows; subsequent selection uses development only.")
        diagnostics["training"] = {**metrics, "independent_decoded_frames": frame_count,
                                    "independent_training_mse": recomputed_mse if np.isfinite(recomputed_mse) else None}
    else:
        configurations = []
        for path in attempt["inputs"]:
            if Path(path).suffix != ".json" or Path(path).name == "request.json":
                continue
            value = read(path)
            if isinstance(value, dict) and value.get("schema") == "astra_bound_native_v1":
                configurations.append(value)
        if len(configurations) != 1:
            raise ValueError("Reviewer needs the unique pinned native configuration")
        config = configurations[0]
        if not binding["success_contract"].get("predicates") or config.get("success_rule") != binding["success_contract"]["predicates"]:
            raise ValueError("Reviewer refuses a backend-specific substitute for the frozen task success predicate")
        if "policy_io" in config and config["policy_io"] != binding["policy_io"]:
            raise ValueError("Native adapter I/O differs from the frozen policy contract")
        runtime = read(paths["runtime"])
        external = runtime.get("external_runtime_inputs", {})
        if config.get("engine") == "isaacsim-physx" and (not external or not unchanged(external)
                or not set(external) <= set(config.get("runtime_asset_files", []))
                or any(attempt["inputs"].get(path) != sha for path, sha in external.items())):
            raise ValueError("Reviewer cannot bind the actual robot specification/USD dependency closure")
        if (runtime.get("success_rule") != config["success_rule"]
                or runtime.get("engine") != config["engine"]
                or runtime.get("world_sha256") != binding["world_sha256"]
                or runtime.get("policy_io_sha256") != digest(binding["policy_io"])
                or not runtime.get("loaded_scene_inputs")
                or not unchanged(runtime.get("loaded_scene_inputs", {}))):
            raise ValueError("Runtime physics/scoring/I/O source attestation does not match pinned configuration")
        attempts = read(paths["attempts"])["episodes"]
        if any(str(Path(ep["trace"]).resolve()) not in artifact_files for ep in attempts):
            raise ValueError("Attempt trace is not one of the receipt's verified immutable artifacts")
        traces = [(ep, read(ep["trace"])) for ep in attempts]
        if len({ep["id"] for ep in attempts}) != len(attempts):
            raise ValueError("Duplicate execution identity")
        checkpoint_sha = file_hash(paths[info["checkpoint_ref"]]) if module != "data" else None
        for episode, trace in traces:
            if episode["success"] is not trace["success"] or episode["case"] != trace["case"]:
                raise ValueError("Attempt ledger disagrees with trace")
            _trace_check(trace, binding, request, config, checkpoint_sha=checkpoint_sha)
        repetitions = request["parameters"].get("episodes_per_case", 1) if module == "data" else 1
        expected = list(cases(binding, "train" if module == "data" else module, repetitions=repetitions))
        exact_cases = [ep["case"] for ep in attempts] == expected
        observations = [step["observation"] for _, trace in traces for step in trace["steps"]]
        latency = [step["latency_ms"] for _, trace in traces for step in trace["steps"]]
        successful = [ep["id"] for ep in attempts if ep["success"]]
        diagnostics.update(episodes=len(attempts), successes=len(successful), failures=[ep for ep in attempts if not ep["success"]],
                           engine=runtime.get("engine"), verification_scope=runtime.get("verification_scope"),
                           privileged_observations=any(not f["available_on_robot"] for f in binding["policy_io"]["observations"]))
        terminals = [trace["steps"][-1]["measurement"] for _, trace in traces if trace["steps"]]
        diagnostics["terminal_predicates"] = []
        for rule in config["success_rule"]:
            finite_values = [float(row[rule["field"]]) for row in terminals
                             if type(row.get(rule["field"])) in (float, int) and np.isfinite(row[rule["field"]])]
            summary = {**rule, "measured_episodes": len(terminals),
                       "failed_episodes": sum(not success_from_measurement(row, [rule]) for row in terminals)}
            if finite_values:
                summary.update(minimum=min(finite_values), maximum=max(finite_values), mean=float(np.mean(finite_values)))
            diagnostics["terminal_predicates"].append(summary)
        if runtime.get("engine") == "isaacsim-physx":
            final_native = runtime.get("final_diagnostics", {})
            diagnostics["native_runtime_valid"] = (final_native.get("native_physics_errors", {}).get("pass") is True
                                                    and final_native.get("collider_precision", {}).get("pass") is True)
        robot_obs = all(f["available_on_robot"] for f in binding["policy_io"]["observations"])
        # A separate explicit contract profile may permit simulator state only
        # for verification. It must never be relabelled robot-deployable.
        verification = spec.get("validation_scope") == "simulation_verification"
        scope_ok = robot_obs or (verification and runtime.get("robot_deployable") is False
                                and runtime.get("privileged_state") is True)
        if module == "data":
            manifest_path = paths[info["dataset_manifest_ref"]]
            count, _ = dataset_manifest(spec, manifest_path)
            manifest = read(manifest_path)
            by_id = {ep["id"]: trace for ep, trace in traces}
            alignment = True
            for episode in manifest["episodes"]:
                if str(Path(episode["artifact"]).resolve()) not in artifact_files:
                    raise ValueError("Dataset episode is not one of the receipt's verified immutable artifacts")
                trace = by_id.get(episode["id"])
                if not trace or not trace["success"] or episode["label"] != "expert":
                    alignment = False
                    continue
                with np.load(episode["artifact"], allow_pickle=False) as archive:
                    for key, trace_key in (("observations", "observation"), ("actions", "executed_action"),
                                           ("next_observations", "next_observation")):
                        expected_array = np.asarray([s[trace_key] for s in trace["steps"]], np.float32)
                        if not np.array_equal(archive[key], expected_array):
                            alignment = False
            record("label_alignment", alignment, "Each label equals the recorded command executed after its own observation.")
            record("split_integrity", exact_cases and all(ep["split"] == "train" for ep in manifest["episodes"]),
                   "All requested cases are from frozen training groups/seeds; validation and holdout were not collected.")
            coverage = exact_cases and set(ep["id"] for ep in manifest["episodes"]) == set(successful)
            record("coverage", coverage, "Every frozen training case was attempted; all successes are included and all failures retained.")
            record("coverage:declared_cases", coverage, "Exact frozen group/seed/stratum/repeat product is recorded, including failures.")
            record("coverage:initial_offsets", len(traces) > 1 and len({json.dumps(t["steps"][0]["observation"])
                   for _, t in traces if t["steps"]}) > 1, "Verified distinct finite initial policy observations across collection cases.")
            record("deployment_observations", scope_ok, "Observation availability is explicit; privileged state only qualifies simulation verification.")
            record("declared_observation_scope", scope_ok and bool(observations),
                   "Actual finite ordered policy inputs match the declared simulation-only privileged observation scope.")
            record("usable_episodes", count, "Counted successful expert-labelled training episodes in the verified manifest.")
        else:
            evaluation = read(paths[info["evaluation_ref"]])
            scored = evaluate_episodes(spec, module, evaluation)
            evaluations = {(ep["group"], ep["seed"], ep["stratum"]): ep for ep in evaluation["episodes"]}
            parity = all(evaluations[(ep["case"]["group"], ep["case"]["seed"], ep["case"]["stratum"])] ==
                         {"id": ep["id"], "group": ep["case"]["group"], "seed": ep["case"]["seed"],
                          "stratum": ep["case"]["stratum"], "success": ep["success"], "trace": ep["trace"]}
                         for ep in attempts)
            current_policy = state["modules"]["train"]["accepted"]["checkpoint_sha256"]
            record("closed_loop", exact_cases and parity and checkpoint_sha == current_policy and bool(observations),
                   "Decoded the accepted student at every step, checked physical transitions and independently rescored measurements.")
            record("failure_analysis", all(trace["success"] or trace["failure"] for _, trace in traces),
                   "Retained a diagnosis and full trace for every unsuccessful rollout, including reset/runtime errors.")
            record("deployment_gap", scope_ok and runtime.get("robot_deployable") is False,
                   "Scope states simulation-only with unresolved real transfer; privileged observation gap is not hidden.")
            record("latency_ms", float(max(latency)) if latency else 1e12,
                   "Maximum measured student inference latency; physics/reset wall time is excluded.")
            record("sealed_protocol", module == "holdout" and state.get("holdout_consumed") is True and exact_cases,
                   "Current holdout attempt was consumed by the controller; no caller-selected cases or teacher actions.")
            record("deployment:io_parity", bool(observations) and scope_ok, "Exact ordered bound I/O was checked on every recorded physical transition.")
            record("deployment:action_bounds", bool(observations), "All executed commands satisfy pinned actuator bounds.")
            record("deployment:released_success", bool(observations) and all(
                   not trace["success"] or trace["steps"][-1]["measurement"].get("attachment_active") is False
                   for _, trace in traces), "Every successful final outcome was measured after the disclosed initial fixture was released.")
            record("deployment:simulation_only", runtime.get("robot_deployable") is False,
                   "Native adapter is explicitly simulation-only and exposes no robot actuator connection.")
            diagnostics["evaluation"] = scored
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = output_dir / "native_review_audit.json"
    write_new(audit, {"schema": "astra_bound_review_audit_v1", "binding_sha256": binding["binding_sha256"],
                      "request_sha256": digest(request), "module": module, "values": values,
                      "diagnostics": diagnostics, "reviewer_sha256": file_hash(__file__),
                      "verified_artifact_evidence": evidence})
    # The receipt already binds all artifacts and the translator pins them.
    # Repeating thousands of paths for each obligation can exceed transport
    # limits without adding provenance; the audit retains the complete list.
    evidence = [str(audit), str(Path(result["receipt"]).resolve())]
    checks, failures = {}, []
    for obligation in plan["modules"][module]["checks"]:
        name = obligation["id"]
        known = name in values
        checks[name] = {"value": values.get(name, False), "reason": reasons.get(name,
                        "Unsupported independent obligation; supply a reviewed implementation instead of assuming it passes."),
                        "evidence": evidence}
        if not known or not compare(obligation, checks[name]["value"]):
            category = "novel" if not known else ("labels" if name == "label_alignment" else "policy")
            if name in ("usable_episodes", "coverage") or name.startswith("coverage:"):
                category = "coverage"
            if name == "latency_ms":
                category = "latency"
            failures.append({"category": category,
                             "diagnosis": name + ": " + checks[name]["reason"], "evidence": [str(audit)]})
    if diagnostics.get("evaluation", {}).get("pass") is False:
        failures.append({"category": "policy", "diagnosis": "Frozen success-rate requirement was not met; retained failed trajectories identify development evidence.",
                         "evidence": [str(audit)]})
    if diagnostics.get("native_runtime_valid") is False:
        failures.append({"category": "physics", "diagnosis": "Native physics reported an error, rejected actuation/contact, or a silently reduced collider profile; inspect preserved native diagnostics.",
                         "evidence": [str(audit)]})
    return {"checks": checks, "failures": failures, "diagnostics": diagnostics}
