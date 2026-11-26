"""Execute vector policy tools in an immutable, reconstructed-world binding.

This worker owns cases, split accounting, label timing and artifact provenance.
The injected environment owns physics and task semantics. A numeric test factory
is not native physics evidence; its engine and verification scope remain visible.
"""
import argparse
import importlib
import itertools
import json
from pathlib import Path
import time
import traceback

import numpy as np

from bound_training import FORMAT, _dimensions, predict, read, write_new
from astra_e2e.oversight.evidence import digest, file_hash, unchanged
from astra_e2e.oversight.policy_bridge import verify_binding


OPERATIONS = {"collect": "data", "develop": "validation", "confirm": "holdout"}


def context(binding_path, request_path, config_path):
    binding = verify_binding(binding_path)
    request_path, config_path = (Path(p).resolve(strict=True) for p in (request_path, config_path))
    request, config = read(request_path), read(config_path)
    operation = request.get("operation")
    module = OPERATIONS.get(operation)
    if (not module or request.get("schema") != "astra_bound_job_v1"
            or request.get("module") != module or request.get("simulation_only") is not True
            or request.get("binding_sha256") != binding["binding_sha256"] or not request.get("job_id")):
        raise ValueError("Expected a current world-bound collection or evaluation request")
    from astra_e2e.oversight.controller import Controller
    controller = Controller(binding["oversight_run"])
    plan, state = controller._load()  # Parent owns writer lock during execution.
    row = state["modules"][module]
    if row["status"] != "running" or not row["attempts"]:
        raise ValueError("Native execution requires the current running attempt")
    attempt = row["attempts"][-1]
    if bool(state.get("holdout_consumed")) != (module == "holdout"):
        raise ValueError("Native execution conflicts with the sealed holdout lifecycle")
    for path in (request_path, config_path, Path(__file__).resolve(), Path(predict.__code__.co_filename).resolve()):
        if attempt["inputs"].get(str(path)) != file_hash(path):
            raise ValueError("Current attempt must pin request, configuration and worker: " + str(path))
    if not unchanged(attempt["inputs"]):
        raise ValueError("A native execution input changed")
    dependencies = {name: state["modules"][name]["accepted"]
                    for name in plan["modules"][module]["depends_on"]}
    if request.get("dependencies") != dependencies or any(
            not controller._current(name, plan, state) or attempt["dependencies"].get(name) != item["id"]
            for name, item in dependencies.items()):
        raise ValueError("Native execution has stale accepted dependencies")
    if (config.get("schema") != "astra_bound_native_v1" or config.get("backend_id") != binding["backend_id"]
            or config.get("simulation_only") is not True):
        raise ValueError("Native configuration backend does not match binding")
    if not binding["success_contract"].get("predicates") or config.get("success_rule") != binding["success_contract"]["predicates"]:
        raise ValueError("Native backend must execute the exact machine-frozen task success predicates")
    if "policy_io" in config and config["policy_io"] != binding["policy_io"]:
        raise ValueError("Native adapter I/O differs from the frozen policy contract")
    source = Path(config["factory_file"]).resolve(strict=True)
    for path in (source, *(Path(raw).resolve(strict=True) for raw in [*config.get("implementation_files", []), *config.get("runtime_asset_files", [])])):
        if attempt["inputs"].get(str(path)) != file_hash(path):
            raise ValueError("Environment factory and implementation files must be pinned by the owned attempt")
    return binding, request, config, plan, state


def cases(binding, split, *, repetitions=1):
    """Case identity comes exclusively from the frozen contract, never a request."""
    declared = binding["splits"][split]
    strata = binding["success_contract"]["evaluation"]["strata"]
    for group, seed, stratum, repeat in itertools.product(
            declared["groups"], declared["seeds"], strata, range(repetitions)):
        yield {"split": split, "group": group, "seed": seed, "stratum": stratum, "repeat": repeat}


def _parameters(request, config):
    params = request["parameters"]
    if not isinstance(params, dict):
        raise ValueError("Tool parameters must be an object")
    if request["operation"] == "collect":
        if request["method"] not in config["collectors"] or set(params) - {"episodes_per_case"}:
            raise ValueError("Unknown collection method or unsupported parameters")
        count = params.get("episodes_per_case", 1)
        if type(count) is not int or not 1 <= count <= 16:
            raise ValueError("episodes_per_case must be an integer in [1,16]")
        return count
    if params:
        raise ValueError("Evaluation accepts no case, checkpoint or controller overrides")
    return 1


def _limits(binding, config):
    obs_dim, action_dim = (_dimensions(binding["policy_io"][key]) for key in ("observations", "actions"))
    low, high = (np.asarray(config[key], dtype=float) for key in ("action_low", "action_high"))
    if (low.shape != (action_dim,) or high.shape != (action_dim,) or not np.isfinite([low, high]).all()
            or not (low < high).all()):
        raise ValueError("Invalid finite actuator bounds")
    for key in ("max_steps", "success_ticks"):
        if type(config[key]) is not int or not 1 <= config[key] <= 10000:
            raise ValueError("Invalid bounded rollout horizon: " + key)
    return obs_dim, action_dim, low, high


def _checkpoint(binding, state):
    accepted = state["modules"]["train"]["accepted"]
    if not accepted:
        raise ValueError("Evaluation requires an accepted training checkpoint")
    files = accepted["files"]
    matches = [Path(path) for path, sha in files.items() if sha == accepted["checkpoint_sha256"]]
    if len(matches) != 1 or file_hash(matches[0]) != accepted["checkpoint_sha256"]:
        raise ValueError("Cannot identify the unique unchanged accepted checkpoint")
    path = matches[0]
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        policy = {key: archive[key].copy() for key in archive.files if key != "metadata_json"}
    if (metadata.get("schema") != FORMAT or metadata.get("world_sha256") != binding["world_sha256"]
            or metadata.get("binding_sha256") != binding["binding_sha256"]
            or metadata.get("policy_io") != binding["policy_io"]):
        raise ValueError("Accepted checkpoint decoder/world/I/O mismatch")
    return path, metadata, policy


def _vector(value, dim, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (dim,) or not np.isfinite(array).all():
        raise ValueError("Non-finite or incorrectly shaped " + name)
    return array


def run_episode(env, case, policy, *, obs_dim, action_dim, low, high, max_steps, success_ticks):
    """Store observation BEFORE its executed command; outcome from independent body measurement."""
    steps, streak, failure = [], 0, None
    initial, failure_traceback = None, None
    try:
        env.reset(case)
        initial = env.measure()
        if initial.get("success") is True:
            raise ValueError("Reset already satisfies success; trivial starts are forbidden")
        for index in range(max_steps):
            observation = _vector(env.observe(), obs_dim, "observation")
            started = time.perf_counter()
            proposed = _vector(policy(observation), action_dim, "policy command")
            latency = (time.perf_counter() - started) * 1000
            command = np.clip(proposed, low, high)
            transition = env.step(command)
            measurement = env.measure()
            if type(measurement.get("success")) is not bool:
                raise ValueError("Environment must supply an independently measured boolean outcome")
            next_observation = _vector(env.observe(), obs_dim, "next observation")
            steps.append({"step": index, "observation": observation.tolist(),
                          "proposed_action": proposed.tolist(), "executed_action": command.tolist(),
                          "next_observation": next_observation.tolist(), "latency_ms": latency,
                          "measurement": measurement, "transition": transition})
            streak = streak + 1 if measurement["success"] else 0
            if streak >= success_ticks:
                break
            if measurement.get("terminal_failure"):
                failure = str(measurement["terminal_failure"])
                break
    except Exception as error:
        failure = type(error).__name__ + ": " + str(error)
        failure_traceback = traceback.format_exc()
    success = bool(failure is None and streak >= success_ticks)
    return {"schema": "astra_bound_trace_v1", "case": case, "initial_measurement": initial,
            "steps": steps, "success": success, "failure": failure or (None if success else "horizon_exhausted"),
            "failure_traceback": failure_traceback,
            "label_timing": "observation_t -> executed_command_t -> physics_step -> measurement_t_plus_1",
            "teacher_used": False}  # Caller sets teacher use only on actual collection.


def run_bound(binding_path, request_path, result_path, config_path):
    validated = context(binding_path, request_path, config_path)
    binding, request, config, _, _ = validated
    result_path = Path(result_path).resolve()
    if result_path.exists() or result_path.parent != Path(request_path).resolve().parent:
        raise ValueError("Write a new native result beside its owned request")
    source_identity = {"config": str(Path(config_path).resolve()), "config_sha256": file_hash(config_path),
                       "request": str(Path(request_path).resolve()), "request_sha256": file_hash(request_path)}
    resources = {"environment": None, "paths": {}, "traces": []}
    try:
        result = _execute_bound(binding_path, request_path, result_path, config_path, validated, resources)
    except BaseException as error:
        # A native app can terminate Python in close(). Save failure provenance
        # first, only after ownership/context and the output directory passed.
        artifacts = {}
        for identity, raw in resources["paths"].items():
            path = Path(raw).resolve()
            if path.is_relative_to(result_path.parent) and path.is_file():
                artifacts[identity] = {"path": str(path), "sha256": file_hash(path)}
        failure = {"schema": "astra_bound_native_failure_v1", "is_policy_receipt": False,
                   "job_id": request.get("job_id"), "operation": request.get("operation"),
                   "binding_sha256": binding.get("binding_sha256"), "world_sha256": binding.get("world_sha256"),
                   **source_identity,
                   "error": type(error).__name__ + ": " + str(error), "traceback": traceback.format_exc(),
                   "completed_traces": resources["traces"], "partial_artifacts": artifacts}
        write_new(result_path.parent / "native_failure.json", failure)
        print(failure["traceback"], flush=True)
        if resources["environment"] is not None:
            resources["environment"].close()
        raise
    # Successful native shutdown remains outside failure handling: some native
    # runtimes deliberately terminate the interpreter after a saved result.
    resources["environment"].close()
    return result


def _execute_bound(binding_path, request_path, result_path, config_path, validated, resources):
    binding, request, config, plan, state = validated
    repetitions = _parameters(request, config)
    obs_dim, action_dim, low, high = _limits(binding, config)
    module_name, factory_name = config["factory"].split(":")
    module = importlib.import_module(module_name)
    if Path(module.__file__).resolve() != Path(config["factory_file"]).resolve():
        raise ValueError("Imported environment differs from pinned factory source")
    env = getattr(module, factory_name)(binding, request, config)
    resources["environment"] = env
    paths, dataset, evaluations, traces = {}, [], [], []
    resources.update(paths=paths, traces=traces)
    checkpoint_path = None
    try:
        metadata = env.metadata()
        expected = {"world_sha256": binding["world_sha256"], "task_id": binding["task_id"],
                    "backend_id": binding["backend_id"], "native_session": binding["native_session"],
                    "policy_io_sha256": digest(binding["policy_io"]), "engine": config["engine"]}
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("Live environment failed world/task/engine/I/O attestation")
        if not metadata.get("loaded_scene_inputs") or not unchanged(metadata["loaded_scene_inputs"]):
            raise ValueError("Environment must fingerprint the actual loaded scene inputs")
        external = metadata.get("external_runtime_inputs", {})
        if config.get("engine") == "isaacsim-physx" and (not external or not unchanged(external)
                or not set(external) <= set(config.get("runtime_asset_files", []))
                or any(state["modules"][request["module"]]["attempts"][-1]["inputs"].get(path) != sha for path, sha in external.items())):
            raise ValueError("Actual robot specification/USD dependency closure is not fully pinned")
        # Native freeze members are transitively pinned by its hash and verified
        # by verify_binding; expose exactly which of those the environment used.
        from calibrated_scene_reconstruction.scene.session import Session
        session = Session(binding["native_session"])
        freeze = read(session.dir / "qa" / "independent_freeze.json")
        roots = {"session": session.dir, "config": session.config_dir}
        frozen = {str((roots[row["base"]] / row["path"]).resolve()): row["sha256"] for row in freeze["files"]}
        if any(frozen.get(path) != sha for path, sha in metadata["loaded_scene_inputs"].items()):
            raise ValueError("Live environment loaded a scene input outside the immutable native freeze")
        metadata["native_freeze_sha256"] = binding["native_freeze_sha256"]
        metadata["scene_attestation_scope"] = "Loaded file hashes; physical adequacy requires upstream native review."
        split = "train" if request["operation"] == "collect" else request["module"]
        if split == "train":
            policy = env.controller(request["method"], config["collectors"][request["method"]])
        else:
            checkpoint_path, checkpoint_meta, weights = _checkpoint(binding, state)
            if request["method"] not in (checkpoint_meta["method"], "dense_bc"):
                raise ValueError("Evaluation method does not decode accepted checkpoint family")
            policy = lambda observation: predict(weights, observation)
        for number, case in enumerate(cases(binding, split, repetitions=repetitions)):
            identity = f"{request['job_id']}-{number:05d}"
            trace = run_episode(env, case, policy, obs_dim=obs_dim, action_dim=action_dim,
                                low=low, high=high, max_steps=config["max_steps"], success_ticks=config["success_ticks"])
            trace.update({"teacher_used": split == "train", "world_sha256": binding["world_sha256"],
                          "binding_sha256": binding["binding_sha256"], "policy_io_sha256": digest(binding["policy_io"]),
                          "request_sha256": digest(request), "checkpoint_sha256": file_hash(checkpoint_path) if checkpoint_path else None})
            trace_path = result_path.parent / (identity + ".trace.json")
            write_new(trace_path, trace)
            paths[identity + "-trace"] = str(trace_path)
            traces.append({"id": identity, "trace": str(trace_path), "case": case, "success": trace["success"], "failure": trace["failure"]})
            if split == "train" and trace["success"]:
                episode_path = result_path.parent / (identity + ".npz")
                with episode_path.open("xb") as stream:
                    np.savez(stream, observations=np.asarray([s["observation"] for s in trace["steps"]], np.float32),
                             actions=np.asarray([s["executed_action"] for s in trace["steps"]], np.float32),
                             next_observations=np.asarray([s["next_observation"] for s in trace["steps"]], np.float32),
                             steps=np.arange(len(trace["steps"]), dtype=np.int64))
                paths[identity + "-episode"] = str(episode_path)
                dataset.append({"id": identity, "group": case["group"], "seed": case["seed"],
                                "source": digest(case), "split": "train", "success": True,
                                "label": "expert", "artifact": str(episode_path)})
            elif split != "train":
                evaluations.append({"id": identity, "group": case["group"], "seed": case["seed"],
                                    "stratum": case["stratum"], "success": trace["success"], "trace": str(trace_path)})
        metadata["final_loaded_scene_inputs_unchanged"] = unchanged(metadata["loaded_scene_inputs"])
        if hasattr(env, "diagnostics"):
            metadata["final_diagnostics"] = env.diagnostics()
        if not metadata["final_loaded_scene_inputs_unchanged"]:
            raise ValueError("Loaded scene changed during execution")
    except BaseException:
        raise
    # Parent rechecks at review too; no result claim survives a changed binding.
    current = context(binding_path, request_path, config_path)
    if current[:3] != (binding, request, config):
        raise ValueError("Bound native job changed during execution")
    io_path = result_path.parent / "policy_io.json"
    write_new(io_path, binding["policy_io"])
    paths["policy_io"] = str(io_path)
    runtime_path, traces_path = result_path.parent / "runtime.json", result_path.parent / "attempts.json"
    write_new(runtime_path, metadata)
    write_new(traces_path, {"schema": "astra_bound_attempts_v1", "episodes": traces})
    paths.update(runtime=str(runtime_path), attempts=str(traces_path))
    data_ref = evaluation_ref = None
    if request["operation"] == "collect":
        data_ref = "dataset"
        manifest = {"version": 1, "format": FORMAT, "task_id": binding["task_id"],
                    "world_sha256": binding["world_sha256"], "policy_io_sha256": digest(binding["policy_io"]),
                    "observation_fields": [f["name"] for f in binding["policy_io"]["observations"]],
                    "action_fields": [f["name"] for f in binding["policy_io"]["actions"]],
                    "action_alignment": binding["policy_io"]["action_alignment"], "episodes": dataset}
        manifest_path = result_path.parent / "dataset.json"
        write_new(manifest_path, manifest)
        paths[data_ref] = str(manifest_path)
    else:
        evaluation_ref = "evaluation"
        evaluation_path = result_path.parent / "evaluation.json"
        write_new(evaluation_path, {"split": request["module"], "episodes": evaluations})
        paths.update(evaluation=str(evaluation_path), policy=str(checkpoint_path))
    facts = {"episodes": len(traces), "successes": sum(t["success"] for t in traces),
             "method": request["method"], "simulation_only": True,
             "failure_counts": {name: sum(t["failure"] == name for t in traces)
                                for name in sorted({t["failure"] for t in traces if t["failure"]})},
             "verification_scope": metadata.get("verification_scope"),
             "oversight": {"binding_sha256": binding["binding_sha256"], "task_id": binding["task_id"],
                           "backend_id": binding["backend_id"], "module": request["module"], "request_sha256": digest(request),
                           "policy_io_ref": "policy_io", "dataset_manifest_ref": data_ref,
                           "checkpoint_ref": "policy" if checkpoint_path else None, "evaluation_ref": evaluation_ref}}
    receipt = {"schema": "astra_bound_policy_receipt_v1", "status": "completed", "operation": request["operation"],
               "facts": facts, "artifacts": [{"id": key, "kind": key, "sha256": file_hash(path),
                    "revision_sha256": binding["world_sha256"], "contract_sha256": binding["contract_sha256"],
                    "scope": "holdout" if request["module"] == "holdout" else "development"}
                    for key, path in paths.items()]}
    receipt_path = result_path.parent / "receipt.json"
    write_new(receipt_path, receipt)
    result = {"receipt": str(receipt_path), "artifact_paths": paths}
    write_new(result_path, result)
    # Isaac shutdown may terminate the interpreter: all artifacts must already
    # be durable before closing the native application.
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("binding", "request", "result", "config"):
        parser.add_argument("--" + argument, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run_bound(args.binding, args.request, args.result, args.config), allow_nan=False))


if __name__ == "__main__":
    main()
