"""Task-independent execution adapters with an explicit, current world binding.

An adapter is maintained executable code, not an assertion that an old backend
can consume any scene. Registration pins that code; native qualification and
independent review remain separate. No algorithm or task name is hardcoded here.
"""
import json
from pathlib import Path
import re
import time
import uuid

from ..oversight.evidence import digest, fingerprint, unchanged
from ..oversight.policy_bridge import verify_binding


SCHEMA = "astra_bound_backend_v1"
OPERATIONS = {"collect": "data", "import_data": "data", "train": "train",
              "develop": "validation", "confirm": "holdout"}
WRITER_RETRY_BUDGET = {"max_retries": 40, "wall_seconds": 10., "interval_seconds": .25}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def register_backend(root, manifest):
    """Pin a task/backend adapter. Registration alone never certifies a task."""
    required = {"schema", "id", "task_id", "backend_id", "simulation_only", "operations"}
    if set(manifest) != required or manifest["schema"] != SCHEMA or manifest["simulation_only"] is not True:
        raise ValueError("Expected a simulation-only bound-backend manifest")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", manifest["id"]):
        raise ValueError("Invalid adapter identity")
    if not all(isinstance(manifest[k], str) and manifest[k].strip() for k in ("task_id", "backend_id")):
        raise ValueError("Task and backend identities are required")
    if not manifest["operations"] or not set(manifest["operations"]) <= set(OPERATIONS):
        raise ValueError("Unknown or empty adapter operations")
    files = {}
    for operation, command in manifest["operations"].items():
        required_command = {"method", "argv", "cwd", "env", "inputs", "timeout_seconds"}
        if set(command) != required_command:
            raise ValueError("Adapter command fields do not match the execution contract")
        if not isinstance(command["method"], str) or not command["method"].strip():
            raise ValueError("Name the implemented method; names do not confer qualification")
        if not isinstance(command["argv"], list) or not all(isinstance(x, str) for x in command["argv"]):
            raise ValueError("Commands require an argv array")
        if not {"{binding}", "{request}", "{result}"} <= set(command["argv"]):
            raise ValueError("Every adapter must consume binding/request and write a structured result")
        cwd = Path(command["cwd"])
        if not cwd.is_absolute() or not cwd.is_dir():
            raise ValueError("Adapter cwd must be an existing absolute directory")
        if type(command["timeout_seconds"]) is not int or command["timeout_seconds"] < 1:
            raise ValueError("Adapter timeout must be a positive integer")
        if not isinstance(command["env"], dict) or not all(isinstance(k, str) and isinstance(v, str)
                                                           for k, v in command["env"].items()):
            raise ValueError("Adapter environment must map strings to strings")
        if not command["inputs"]:
            raise ValueError("Pin the adapter implementation and its configuration")
        paths = list(command["inputs"])
        paths += [str((cwd / arg).resolve()) for arg in command["argv"] if (cwd / arg).is_file()]
        files.update(fingerprint(paths))
    package = {"manifest": manifest, "files": files, "status": "registered_requires_native_qualification"}
    package["sha256"] = digest(package)
    target = Path(root).resolve() / manifest["id"] / (package["sha256"] + ".json")
    if target.exists():
        if read(target) != package:
            raise ValueError("Immutable adapter package changed")
    else:
        write_new(target, package)
    return {"path": str(target), **package}


def load_backend(path):
    package = read(path)
    payload = {k: v for k, v in package.items() if k != "sha256"}
    if digest(payload) != package.get("sha256") or not unchanged(package["files"]):
        raise ValueError("Adapter package or pinned code changed; register a new revision")
    return package


def catalog(root, *, task_id=None, backend_id=None):
    results = []
    for path in sorted(Path(root).glob("*/*.json")):
        package = read(path)
        manifest = package["manifest"]
        if task_id is not None and manifest["task_id"] != task_id:
            continue
        if backend_id is not None and manifest["backend_id"] != backend_id:
            continue
        try:
            load_backend(path)
            current, reason = True, None
        except (OSError, ValueError) as exc:
            current, reason = False, str(exc)
        results.append({"path": str(path), "id": manifest["id"], "task_id": manifest["task_id"],
                        "backend_id": manifest["backend_id"], "sha256": package["sha256"],
                        "operations": {k: v["method"] for k, v in manifest["operations"].items()},
                        "source_current": current, "reason": reason,
                        "qualification": "Native evidence is required per task and world."})
    return results


def prepare_execution(binding_path, backend_path, operation, parameters):
    """Prepare a supervised oversight packet; no command runs in this function."""
    binding = verify_binding(binding_path)
    package = load_backend(backend_path)
    manifest = package["manifest"]
    if any(manifest[k] != binding[k] for k in ("task_id", "backend_id")):
        raise ValueError("Adapter does not match the bound task/backend")
    if operation not in manifest["operations"] or operation not in OPERATIONS:
        raise ValueError("Requested operation is not implemented by this adapter")
    if not isinstance(parameters, dict):
        raise ValueError("Parameters must be an object")
    # Parameters are opaque method inputs, never shell fragments or authority.
    if {"binding", "task_id", "backend_id", "split", "success_contract", "simulation_only"} & set(parameters):
        raise ValueError("Method parameters cannot override the frozen experiment")
    from ..oversight.controller import Controller
    ctrl = Controller(binding["oversight_run"])
    module = OPERATIONS[operation]
    with ctrl.lock():
        plan, state = ctrl._load()
        ctrl._ensure_ready(module, plan, state)
        directory = ctrl.root / "policy_jobs" / uuid.uuid4().hex
        directory.mkdir(parents=True)
        dependencies = {name: state["modules"][name]["accepted"]
                        for name in plan["modules"][module]["depends_on"]}
        request = {"schema": "astra_bound_job_v1", "job_id": directory.name, "operation": operation, "module": module,
                   "binding_sha256": binding["binding_sha256"], "backend_sha256": package["sha256"],
                   "method": manifest["operations"][operation]["method"], "parameters": parameters,
                   "dependencies": dependencies, "simulation_only": True}
        request_path, result_path = directory / "request.json", directory / "result.json"
        write_new(request_path, request)
        replacements = {"{binding}": str(Path(binding_path).resolve()),
                        "{request}": str(request_path), "{result}": str(result_path)}
        command = manifest["operations"][operation]
        argv = [replacements.get(arg, arg) for arg in command["argv"]]
        packet_command = {k: command[k] for k in ("cwd", "env", "timeout_seconds")}
        packet_command["argv"] = argv
        packet_command["inputs"] = sorted(set(package["files"]) | {
            str(Path(backend_path).resolve()), str(Path(binding_path).resolve()), str(request_path)})
        packet = {"module": module, "purpose": "Execute the selected world-bound " + operation + " adapter",
                  "commands": [packet_command]}
        write_new(directory / "packet.json", packet)
    return {"packet": packet, "request": str(request_path), "result": str(result_path),
            "backend_sha256": package["sha256"], "binding_sha256": binding["binding_sha256"]}


def execute(binding_path, backend_path, operation, parameters):
    from ..oversight.controller import Controller
    binding = verify_binding(binding_path)
    ctrl = Controller(binding["oversight_run"])
    retry = _WriterRetry(ctrl, operation)
    prepared = retry.call("prepare", lambda: prepare_execution(binding_path, backend_path, operation, parameters))
    # Reuse this exact packet if acquiring the run lock is briefly blocked. A
    # retry cannot silently create another policy job or reserve another attempt.
    execution = retry.call("run", lambda: ctrl.run(prepared["packet"]))
    # A failed/malformed adapter remains awaiting review; never rerun it silently.
    output = read(prepared["result"])
    if set(output) != {"receipt", "artifact_paths"}:
        raise ValueError("Adapter result must resolve a receipt and its immutable artifact paths")
    if not Path(output["receipt"]).is_absolute() or not Path(output["receipt"]).is_file():
        raise ValueError("Adapter did not return a receipt file")
    receipt = read(output["receipt"])
    info = receipt.get("facts", {}).get("oversight", {})
    if any(info.get(k) != binding[k] for k in ("binding_sha256", "task_id", "backend_id")):
        raise ValueError("Adapter did not execute the bound world/task")
    if receipt.get("operation") != operation or info.get("module") != OPERATIONS[operation]:
        raise ValueError("Adapter receipt is from a different operation")
    if receipt.get("status") != "completed" or info.get("request_sha256") != digest(read(prepared["request"])):
        raise ValueError("Adapter receipt is incomplete or belongs to a different job attempt")
    load_backend(backend_path)
    verify_binding(binding_path)
    return {**prepared, **output, "execution": execution, "infrastructure": retry.summary(), "accepted": False,
            "next": "Inspect artifacts, translate the receipt with evidence-backed checks, then review."}


class _WriterRetry:
    """Bound pre-reservation lock retries inside the same selected tool call.

    Only the typed writer-lock exception is eligible. Adapter failures, native
    exceptions, malformed receipts and any newly reserved module attempt are
    never retried here. Concurrent active writers remain protected by flock.
    """
    def __init__(self, controller, operation):
        if operation not in OPERATIONS:
            raise ValueError("Requested operation is not implemented by this adapter")
        self.controller, self.operation = controller, operation
        self.module = OPERATIONS[operation]
        self.baseline = controller.progress_snapshot()["modules"][self.module]["attempts"]
        self.budget = dict(WRITER_RETRY_BUDGET)
        self.directory = controller.root / "infrastructure_retries" / uuid.uuid4().hex
        self.directory.mkdir(parents=True)
        self.previous, self.events, self.retries, self.started = None, 0, 0, None
        self._record("created", None)

    def _record(self, event, stage):
        row = {"schema": "oversight_writer_retry_v1", "previous": self.previous,
               "event": event, "stage": stage, "operation": self.operation,
               "budget": self.budget, "retries": self.retries, "time": time.time(),
               "elapsed_seconds": 0. if self.started is None else time.monotonic() - self.started}
        self.previous = digest(row)
        write_new(self.directory / f"{self.events:04d}.json", {**row, "sha256": self.previous})
        self.events += 1

    def call(self, stage, function):
        from ..oversight.controller import OversightBusyError
        while True:
            try:
                result = function()
            except OversightBusyError:
                if self.started is None:
                    self.started = time.monotonic()
                current = self.controller.progress_snapshot()["modules"][self.module]
                if current["attempts"] != self.baseline or current["state"] in {"running", "awaiting_review", "accepted"}:
                    self._record("retry_forbidden_after_reservation", stage)
                    raise
                elapsed = time.monotonic() - self.started
                if self.retries >= self.budget["max_retries"] or elapsed >= self.budget["wall_seconds"]:
                    self._record("retry_budget_exhausted", stage)
                    raise
                self.retries += 1
                self._record("writer_busy", stage)
                time.sleep(min(self.budget["interval_seconds"], self.budget["wall_seconds"] - elapsed))
            except BaseException:
                self._record("operation_failed_without_retry", stage)
                raise
            else:
                self._record("operation_completed", stage)
                return result

    def summary(self):
        return {"audit_directory": str(self.directory), "budget": self.budget,
                "retries": self.retries, "last_sha256": self.previous,
                "scope": "Infrastructure lock acquisition only; no additional model decision or policy attempt."}
