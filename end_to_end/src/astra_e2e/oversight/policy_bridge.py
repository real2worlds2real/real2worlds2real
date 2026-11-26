"""Bind native reconstruction evidence to policy-development tool receipts.

This module transports evidence; it does not qualify a backend, infer expert
labels, or approve semantic reviews. A backend must actually load the bound
session and emit the binding in its execution receipt. Adding the binding to an
old result is not a valid implementation of that backend contract.
"""
import argparse
import json
from pathlib import Path

from .contracts import REPORT_SCHEMA, validate
from .evidence import (dataset_manifest, digest, evaluate_episodes, file_hash,
                       fingerprint, policy_io, unchanged)


BINDING_SCHEMA = "astra_native_policy_binding_v1"
IO_FIELDS = ("observations", "actions", "control_hz", "max_latency_ms",
             "action_alignment", "normalization", "deployment_controller")
OPERATIONS = {"data": {"collect", "import_data", "register_data"}, "train": {"train"},
              "validation": {"develop"}, "holdout": {"confirm"}}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Non-finite JSON")))


def _controller(root):
    # Keep read-only schema imports usable on platforms without the POSIX worker.
    from .controller import Controller
    return Controller(root)


def _binding(controller, plan, state, task_id, backend_id):
    if not isinstance(task_id, str) or not task_id.strip() or not isinstance(backend_id, str) or not backend_id.strip():
        raise ValueError("A concrete learning task and backend identity are required")
    if not controller._current("twin", plan, state):
        raise ValueError("A current accepted native twin is required; a file path alone is insufficient")
    ancestors = set()

    def visit(module):
        if module in ancestors:
            return
        ancestors.add(module)
        for dependency in plan["modules"][module]["depends_on"]:
            visit(dependency)

    visit("twin")
    accepted = {name: state["modules"][name]["accepted"] for name in sorted(ancestors)}
    files = {}
    for acceptance in accepted.values():
        files.update(acceptance["files"])
    spec = plan["contract"]
    native_session = Path(spec["source"]["session"]).resolve()
    freeze_path = native_session / "qa" / "independent_freeze.json"
    # Native verification checks the manifest's listed files. Pin the manifest
    # itself as well so replacing its file list cannot preserve an old binding.
    files.update(fingerprint([str(freeze_path)]))
    payload = {
        "schema": BINDING_SCHEMA,
        "oversight_run": str(controller.root.resolve()),
        "plan_sha256": state["plan_sha256"],
        "contract_sha256": digest(spec),
        "task_id": task_id, "backend_id": backend_id,
        "native_session": str(native_session),
        "native_freeze_sha256": files[str(freeze_path)],
        "upstream_acceptances": {name: item["id"] for name, item in accepted.items()},
        "evidence_files": files,
        "source_files": state["sources"],
        "policy_io": {key: spec["policy"][key] for key in IO_FIELDS},
        "success_contract": {"definition": spec["task"]["success_definition"],
                             "failure_modes": spec["task"]["failure_modes"],
                             "evaluation": spec["evaluation"]},
        "splits": spec["data"]["splits"],
        "simulation_only": True,
        "real_validation": "pending",
    }
    if "success_predicates" in spec["task"]:
        payload["success_contract"]["predicates"] = spec["task"]["success_predicates"]
    payload["world_sha256"] = digest({key: payload[key] for key in
        ("native_session", "upstream_acceptances", "source_files", "evidence_files")})
    return {**payload, "binding_sha256": digest(payload)}


def export_binding(oversight_run, output, *, task_id, backend_id):
    """Write a new immutable scene/task handoff after native twin acceptance.

    This is a qualification of the input scene. Backend compatibility and actual
    runtime use of the scene require separate executable adapter evidence.
    """
    controller = _controller(oversight_run)
    with controller.lock():
        plan, state = controller._load()
        binding = _binding(controller, plan, state, task_id, backend_id)
        target = Path(output).resolve()
        with target.open("x", encoding="utf-8") as stream:
            json.dump(binding, stream, indent=2, allow_nan=False)
            stream.write("\n")
        return binding


def verify_binding(binding):
    """Recompute current native gates and upstream identities, not only hashes."""
    value = _read(binding) if isinstance(binding, (str, Path)) else binding
    if not isinstance(value, dict) or value.get("schema") != BINDING_SCHEMA:
        raise ValueError("Expected a native policy binding")
    payload = {key: item for key, item in value.items() if key != "binding_sha256"}
    if digest(payload) != value.get("binding_sha256"):
        raise ValueError("Binding integrity failure")
    controller = _controller(value["oversight_run"])
    # Controller.run owns the writer lock while executing a child worker. Read
    # immutable history snapshots here; acquiring that lock in the child would
    # deadlock every correctly supervised backend. The parent rechecks evidence
    # on review, and mutating controller operations still acquire the lock.
    plan, state = controller._load()
    current = _binding(controller, plan, state, value["task_id"], value["backend_id"])
    if current != value:
        raise ValueError("Scene, contract or upstream acceptance changed; create a new binding and experiment")
    return value


def _resolve_artifacts(receipt, artifact_paths):
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("Receipt must include immutable artifact references")
    required = {"id", "kind", "sha256", "revision_sha256", "contract_sha256", "scope"}
    references = {}
    for reference in artifacts:
        if not isinstance(reference, dict) or set(reference) != required:
            raise ValueError("Malformed learning artifact reference")
        if reference["id"] in references:
            raise ValueError("Duplicate learning artifact identity")
        references[reference["id"]] = reference
    if set(artifact_paths) != set(references):
        raise ValueError("Resolve every receipt artifact exactly once; do not inject unbound artifacts")
    paths = {}
    for identity, raw in artifact_paths.items():
        path = Path(raw)
        if not path.is_absolute() or not path.is_file() or file_hash(path) != references[identity]["sha256"]:
            raise ValueError("Learning artifact is missing, changed or not an absolute file: " + identity)
        paths[identity] = str(path.resolve())
    return paths, references


def translate_receipt(binding, receipt_path, *, artifact_paths, checks, failures=None):
    """Build a review report from an executed, scene-bound learning operation.

    ``facts.oversight`` must contain binding_sha256, task_id, backend_id, module,
    request_sha256, policy_io_ref, dataset_manifest_ref, checkpoint_ref and evaluation_ref. Null
    denotes an inapplicable reference; all non-null references must appear in the
    receipt's artifact list. The dataset/evaluation artifacts use the native
    oversight schemas. Labels and scenario identities are never inferred from
    an aggregate success count. ``checks`` is the independently authored review,
    including its evidence; this function never creates passing review values.

    The corresponding oversight work packet must already be awaiting review.
    In particular, final confirmation executes inside a holdout packet after the
    controller consumes that split, not inside a development/data command.
    """
    value = verify_binding(binding)
    receipt_path = Path(receipt_path).resolve(strict=True)
    receipt = _read(receipt_path)
    if (not isinstance(receipt, dict) or set(receipt) != {"schema", "status", "operation", "facts", "artifacts"}
            or receipt.get("schema") != "astra_bound_policy_receipt_v1" or receipt.get("status") != "completed"
            or not isinstance(receipt.get("facts"), dict)):
        raise ValueError("Only completed worker receipts can be translated; retain execution failures separately")
    info = receipt.get("facts", {}).get("oversight")
    expected = {"binding_sha256", "task_id", "backend_id", "module", "request_sha256", "policy_io_ref",
                "dataset_manifest_ref", "checkpoint_ref", "evaluation_ref"}
    if not isinstance(info, dict) or set(info) != expected:
        raise ValueError("Worker receipt lacks its exact executed-world handoff")
    for key in ("binding_sha256", "task_id", "backend_id"):
        if info[key] != value[key]:
            raise ValueError("Worker receipt belongs to a different " + key)
    module = info["module"]
    if module not in OPERATIONS or receipt.get("operation") not in OPERATIONS[module]:
        raise ValueError("Learning operation cannot be translated into this oversight module")
    paths, references = _resolve_artifacts(receipt, artifact_paths)
    for reference in references.values():
        if reference["contract_sha256"] != value["contract_sha256"]:
            raise ValueError("Learning artifact belongs to a different task contract")
        if module != "holdout" and reference["scope"] == "holdout":
            raise ValueError("Sealed holdout artifacts cannot enter a development module")

    def resolve(field, required):
        identity = info[field]
        if identity is None:
            if required:
                raise ValueError("Missing required artifact: " + field)
            return None
        if identity not in paths:
            raise ValueError("Handoff references an artifact absent from the worker receipt: " + field)
        if not required:
            raise ValueError("Inapplicable artifact must be null: " + field)
        return paths[identity]

    io_path = resolve("policy_io_ref", True)
    dataset_path = resolve("dataset_manifest_ref", module == "data")
    checkpoint_path = resolve("checkpoint_ref", module != "data")
    evaluation_path = resolve("evaluation_ref", module in ("validation", "holdout"))
    controller = _controller(value["oversight_run"])
    with controller.lock():
        plan, state = controller._load()
        # Recheck after receipt reading to detect a concurrent feedback revision.
        if _binding(controller, plan, state, value["task_id"], value["backend_id"]) != value:
            raise ValueError("Binding changed while reading the receipt")
        row = state["modules"][module]
        if row["status"] != "awaiting_review" or not row["attempts"]:
            raise ValueError("Execute the corresponding oversight packet before translating its receipt")
        attempt = row["attempts"][-1]
        if not unchanged(attempt["inputs"]):
            raise ValueError("The packet's implementation or inputs changed")
        requests = [_read(path) for path in attempt["inputs"] if Path(path).name == "request.json"]
        matching = [request for request in requests if isinstance(request, dict)
                    and request.get("schema") == "astra_bound_job_v1"
                    and digest(request) == info["request_sha256"]]
        if len(matching) != 1:
            raise ValueError("Receipt does not bind the unique job request pinned by the current attempt")
        request = matching[0]
        if (not isinstance(request.get("job_id"), str) or not request["job_id"]
                or request.get("binding_sha256") != value["binding_sha256"]
                or request.get("module") != module or request.get("operation") != receipt["operation"]
                or request.get("simulation_only") is not True):
            raise ValueError("Executed job request differs from the world-bound module")
        current_dependencies = {name: state["modules"][name]["accepted"]
                                for name in attempt["dependencies"]}
        if request.get("dependencies") != current_dependencies:
            raise ValueError("Executed job used a stale dependency snapshot; prepare a new attempt")
        for dependency, acceptance_id in attempt["dependencies"].items():
            if (not controller._current(dependency, plan, state) or
                    state["modules"][dependency]["accepted"]["id"] != acceptance_id):
                raise ValueError("Learning dependencies changed during execution")
        if module == "holdout" and not state["holdout_consumed"]:
            raise ValueError("Final confirmation must execute in a consumed oversight holdout attempt")
        if module != "holdout" and state["holdout_consumed"]:
            raise ValueError("Sealed holdout results cannot feed development reports")
        spec = plan["contract"]
        policy_io(spec, io_path)
        if set(checks) != {check["id"] for check in plan["modules"][module]["checks"]}:
            raise ValueError("Independent review must include exactly the contracted module checks")
        evaluation = _read(evaluation_path) if evaluation_path else None
        report = {
            "module": module, "attempt": len(row["attempts"]),
            "plan_sha256": value["plan_sha256"],
            "summary": "Executed learning receipt; independent module checks remain authoritative.",
            "artifacts": sorted({str(receipt_path), *paths.values()}),
            "checks": checks, "failures": [] if failures is None else failures,
            "native_session": None, "policy_io": io_path,
            "dataset_manifest": dataset_path, "checkpoint": checkpoint_path,
            "evaluation": evaluation,
        }
        validate(report, REPORT_SCHEMA)
        evidence = [p for check in checks.values() for p in check["evidence"]]
        evidence += [p for failure in report["failures"] for p in failure["evidence"]]
        fingerprint(evidence)
        if dataset_path:
            count, _ = dataset_manifest(spec, dataset_path)
            if checks["usable_episodes"]["value"] != count:
                raise ValueError("Usable episode review count differs from the bound dataset")
        if evaluation:
            evaluate_episodes(spec, module, evaluation)
            fingerprint([episode["trace"] for episode in evaluation["episodes"]])
            facts = receipt["facts"]
            for field, actual in (("episodes", len(evaluation["episodes"])),
                                  ("successes", sum(ep["success"] for ep in evaluation["episodes"]))):
                if field in facts and facts[field] != actual:
                    raise ValueError("Worker aggregate disagrees with saved evaluation: " + field)
            trained = state["modules"]["train"]["accepted"]
            if not trained or file_hash(checkpoint_path) != trained["checkpoint_sha256"]:
                raise ValueError("Evaluated checkpoint differs from the accepted training checkpoint")
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    bind = commands.add_parser("bind")
    bind.add_argument("run")
    bind.add_argument("output")
    bind.add_argument("--task-id", required=True)
    bind.add_argument("--backend-id", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("binding")
    report = commands.add_parser("report")
    report.add_argument("binding")
    report.add_argument("receipt")
    report.add_argument("artifact_paths", help="JSON object mapping artifact IDs to absolute files")
    report.add_argument("review", help="JSON object with independent checks and optional failures")
    args = parser.parse_args(argv)
    if args.command == "bind":
        result = export_binding(args.run, args.output, task_id=args.task_id, backend_id=args.backend_id)
    elif args.command == "verify":
        result = verify_binding(args.binding)
    else:
        review = _read(args.review)
        result = translate_receipt(args.binding, args.receipt,
            artifact_paths=_read(args.artifact_paths), checks=review["checks"], failures=review.get("failures"))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
