"""Register world-bound dense tools and freeze a real-Astra runner configuration.

Preparation executes no collection, training, evaluation or semantic approval.
It accepts a previously qualified binding and a reviewed native configuration;
the task identity, geometry, I/O and cases come from those inputs, not this file.
"""
import argparse
import json
import os
from pathlib import Path
import sys

from bound_training import METHODS, _dimensions
from astra_e2e.harness.backends import read, write_new
from astra_e2e.layout import first_party_sources, pythonpath
from astra_e2e.harness.service import Harness
from astra_e2e.oversight.evidence import digest, fingerprint
from astra_e2e.oversight.policy_bridge import verify_binding


def object_schema(fields):
    return {"type": "object", "properties": fields, "additionalProperties": False}


PARAMETERS = {
    "collect": object_schema({"episodes_per_case": {"type": "integer", "minimum": 1, "maximum": 16}}),
    "ridge_bc": object_schema({"ridge_alpha": {"type": "number", "minimum": 1e-8, "maximum": 100},
        "seed": {"type": "integer", "minimum": 0, "maximum": 2**32 - 1}}),
    "mlp_bc": object_schema({"hidden_dim": {"type": "integer", "minimum": 1, "maximum": 512},
        "updates": {"type": "integer", "minimum": 1, "maximum": 5000},
        "learning_rate": {"type": "number", "minimum": 1e-6, "maximum": .1},
        "seed": {"type": "integer", "minimum": 0, "maximum": 2**32 - 1}}),
}


def _save(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError("Preparation is immutable; use a new campaign directory for changed inputs")
    else:
        write_new(path, value)


def prepare(binding_path, native_config, harness_root, output, *, campaign,
            native_python=sys.executable, training_python=sys.executable, environment=None,
            decision_transport="authenticated_relay", max_model_calls=40, timeout_seconds=3600):
    binding_path, native_config = (Path(path).resolve(strict=True) for path in (binding_path, native_config))
    binding, config = verify_binding(binding_path), read(native_config)
    if (config.get("schema") != "astra_bound_native_v1" or config.get("backend_id") != binding["backend_id"]
            or config.get("simulation_only") is not True):
        raise ValueError("Native tool configuration does not match the qualified binding")
    if not isinstance(config.get("collectors"), dict) or not config["collectors"]:
        raise ValueError("No executable collectors have been configured")
    if not binding["success_contract"].get("predicates") or config.get("success_rule") != binding["success_contract"]["predicates"]:
        raise ValueError("Native success predicates differ from the frozen task contract")
    assets = config.get("runtime_asset_files", [])
    if not isinstance(assets, list) or any(not isinstance(path, str) or not Path(path).is_absolute() for path in assets):
        raise ValueError("Runtime asset dependencies must be explicit absolute file paths")
    if config.get("engine") == "isaacsim-physx" and not assets:
        raise ValueError("Native robot tools require a pinned robot specification and USD dependency closure")
    for side in ("observations", "actions"):
        _dimensions(binding["policy_io"][side])  # Reject image inputs; this adapter is explicitly dense BC.
    if type(max_model_calls) is not int or not 1 <= max_model_calls <= 100:
        raise ValueError("A finite model-call budget from 1 to 100 is required")
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 86400:
        raise ValueError("A finite per-job timeout is required")
    if decision_transport not in {"local_codex", "authenticated_relay"}:
        raise ValueError("Select a supported actual-Astra transport")
    if environment is not None and (not isinstance(environment, dict) or
            not all(isinstance(k, str) and isinstance(v, str) for k, v in environment.items())):
        raise ValueError("Worker environment must map strings to strings")
    learning = Path(__file__).resolve().parent
    repository = learning.parents[1]
    files = set(first_party_sources(repository))
    files.update(learning / name for name in ("bound_native.py", "bound_training.py", "bound_native_review.py",
        "astra_unified_decide.py", "astra_unified_relay.py", "astra_decide.py", "campaign_ledger.py",
        "prepare_unified_campaign.py"))
    files.update(Path(path).resolve(strict=True) for path in [config["factory_file"], *config.get("implementation_files", [])])
    files.update(Path(path).resolve(strict=True) for path in assets)
    files.add(native_config)
    pinned = fingerprint(sorted(str(path.resolve()) for path in files))
    worker_environment = {**(environment or {}), "PYTHONPATH": pythonpath(repository, learning)}
    # The controller additionally enforces the contracted GPU allocation. No
    # authentication environment is added to worker commands or model prompts.
    api = Harness(Path(harness_root).resolve(), execute=False, learning_root=learning)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    base_id = "native_" + digest({"campaign": campaign, "binding": binding["binding_sha256"]})[:20]

    def command(method, worker, python):
        argv = [str(python), "-B", str(learning / worker), "--binding", "{binding}",
                "--request", "{request}", "--result", "{result}"]
        if worker == "bound_native.py":
            argv.extend(["--config", str(native_config)])
        return {"method": method, "argv": argv, "cwd": str(repository), "env": worker_environment,
                "inputs": sorted(pinned), "timeout_seconds": timeout_seconds}

    backends = []
    def register(identity, operations, schemas):
        package = api.dispatch("backend_register", {"manifest": {
            "schema": "astra_bound_backend_v1", "id": base_id + "_" + identity,
            "task_id": binding["task_id"], "backend_id": binding["backend_id"],
            "simulation_only": True, "operations": operations}})
        backends.append({"path": package["path"], "parameter_schemas": schemas})

    for method in sorted(config["collectors"]):
        if not isinstance(method, str) or not method.strip():
            raise ValueError("A collector needs a concrete implemented method name")
        register("collect_" + digest(method)[:12],
                 {"collect": command(method, "bound_native.py", native_python)}, {"collect": PARAMETERS["collect"]})
    for method in METHODS:
        register("train_" + method, {"train": command(method, "bound_training.py", training_python)},
                 {"train": PARAMETERS[method]})
    register("evaluate", {operation: command("dense_bc", "bound_native.py", native_python)
                          for operation in ("develop", "confirm")}, {})
    settings = {"schema": "astra_unified_agent_v1", "campaign": campaign, "binding": str(binding_path),
                "state": str(output / "agent"), "harness_state": str(api.root), "learning_root": str(learning),
                "python": str(training_python), "execution_mode": "production",
                "decision_transport": decision_transport, "max_model_calls": max_model_calls,
                "runtime_preflight": {"directory": str(output / "runtime_preflight")},
                "backends": backends, "reviewer": {"factory": "bound_native_review:review", "inputs": pinned}}
    from astra_e2e.oversight.controller import Controller
    plan, _ = Controller(binding["oversight_run"])._load()
    prepared = {"schema": "astra_unified_preparation_v1", "agent_config": str(output / "agent.json"),
                "binding_sha256": binding["binding_sha256"], "native_configuration": str(native_config),
                "inputs": pinned, "methods": {"collect": sorted(config["collectors"]), "train": list(METHODS),
                                                "evaluation": "accepted_dense_bc_checkpoint"},
                "validation_scope": plan["contract"].get("validation_scope", "deployment"),
                "runtime_readiness": "pending_exact_runtime_preflight",
                "claim": "Executable tool registration only; no new semantic, policy or physics acceptance.",
                "real_validation": "pending"}
    _save(output / "agent.json", settings)
    _save(output / "preparation.json", prepared)
    return prepared


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binding", "native-config", "harness-root", "output", "campaign"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--native-python", default=sys.executable)
    parser.add_argument("--training-python", default=sys.executable)
    parser.add_argument("--environment-json", default="{}")
    parser.add_argument("--decision-transport", choices=("local_codex", "authenticated_relay"), default="authenticated_relay")
    parser.add_argument("--max-model-calls", type=int, default=40)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    args = parser.parse_args(argv)
    result = prepare(args.binding, args.native_config, args.harness_root, args.output, campaign=args.campaign,
        native_python=args.native_python, training_python=args.training_python,
        environment=json.loads(args.environment_json), decision_transport=args.decision_transport,
        max_model_calls=args.max_model_calls, timeout_seconds=args.timeout_seconds)
    print(json.dumps({key: result[key] for key in ("agent_config", "methods", "validation_scope", "claim")}))


if __name__ == "__main__":
    main()
