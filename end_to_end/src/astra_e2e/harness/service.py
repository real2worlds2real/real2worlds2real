"""One JSON tool surface for native reconstruction and Astra policy development.

The caller is the reasoning agent. This service schedules and verifies work; it
does not pretend to be an LLM or silently substitute an old task backend.
"""
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

from ..oversight.contracts import PACKET_SCHEMA, REPORT_SCHEMA, SPEC_SCHEMA, intake
from ..oversight.evidence import digest
from ..oversight.planning import build_plan
from ..oversight.policy_bridge import export_binding, translate_receipt, verify_binding
from . import backends


READ_ONLY_LEARNING = {"task_inspect", "plan_validate", "tool_status", "job_status",
                      "library_inspect", "library_search"}


def obj(properties, required=None):
    return {"type": "object", "properties": properties,
            "required": list(properties) if required is None else required, "additionalProperties": False}


STRING = {"type": "string", "minLength": 1}
OBJECT = {"type": "object"}


def public_binding(binding):
    """The worker receives the full binding; the planning agent never needs test seeds."""
    keys = ("schema", "oversight_run", "plan_sha256", "contract_sha256", "task_id", "backend_id",
            "native_session", "policy_io", "world_sha256", "binding_sha256", "simulation_only", "real_validation")
    return {**{k: binding[k] for k in keys},
            "development_splits": {k: v for k, v in binding["splits"].items() if k != "holdout"},
            "holdout": "sealed_worker_only"}


def redact_holdout(value):
    if isinstance(value, dict):
        return {k: ("sealed_worker_only" if k == "holdout" else redact_holdout(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_holdout(v) for v in value]
    return value


def _sealed_evidence(path):
    path = Path(path).resolve(strict=True)
    if any(part.startswith("holdout-") or part == "sealed" for part in path.parts) or path.name in {"heldout.json", "sealed_evaluation.json", "real_preparation.json"}:
        return True
    candidates = [path] if path.suffix == ".json" else []
    # Bound job outputs are deliberately opaque UUID paths, not split-named paths.
    candidates.extend(parent / "request.json" for parent in path.parents)
    for candidate in candidates:
        if not candidate.is_file() or candidate.stat().st_size > 4 * 1024 * 1024:
            continue
        try:
            value = backends.read(candidate)
        except (ValueError, UnicodeError):
            continue
        if not isinstance(value, dict):
            continue
        if value.get("module") == "holdout" or value.get("split") == "holdout" or value.get("operation") == "confirm":
            return True
        if value.get("facts", {}).get("oversight", {}).get("module") == "holdout":
            return True
        if isinstance(value.get("evaluation"), dict) and value["evaluation"].get("split") == "holdout":
            return True
    return False


def tool_schemas():
    """Strict tool contracts, independent of model provider and manipulation task."""
    definitions = {
        "chain_init": ("Create an observations-to-delivery or policy handover run.", obj({"run": STRING, "request": OBJECT})),
        "chain_next": ("Inspect the current chain stage and next native module.", obj({"run": STRING})),
        "chain_verify": ("Recheck the accepted delivery and native source evidence.", obj({"run": STRING})),
        "chain_catalog": ("Discover native reconstruction operations.", obj({})),
        "chain_plan": ("Select or revise native reconstruction steps.", obj({"run": STRING, "plan": OBJECT})),
        "chain_step": ("Execute the next selected operation under its native gates.",
                       obj({"run": STRING, "id": STRING, "arguments": OBJECT, "evidence": {"type": "array", "items": STRING}},
                           ["run", "id", "arguments"])),
        "chain_campaign_step": ("Run one decision of the bound station campaign.", obj({"run": STRING})),
        "chain_advance": ("Verify the current delivery or advance the station campaign.",
                          obj({"run": STRING, "delivery": STRING, "execute": {"type": "boolean"}}, ["run"])),
        "chain_recall": ("Recall scoped methods and skills for this chain.", obj({"run": STRING, "request": OBJECT})),
        "chain_learn": ("Record a scoped experience from development evidence.", obj({"run": STRING, "request": OBJECT})),
        "intake": ("Validate task, source, I/O, success, splits and execution authority.", obj({"spec": SPEC_SCHEMA})),
        "plan": ("Compile obligations and dependency graph without execution.", obj({"spec": SPEC_SCHEMA})),
        "init": ("Create a new immutable experiment contract.", obj({"run": STRING, "spec": SPEC_SCHEMA})),
        "status": ("Inspect current and stale module evidence.", obj({"run": STRING})),
        "progress_snapshot": ("Read completed progress records without taking the execution writer lock; not current acceptance.", obj({"run": STRING})),
        "next": ("Get ready modules and their evidence obligations.", obj({"run": STRING})),
        "run": ("Execute an agent-authored supervised module packet; never auto-accept.", obj({"run": STRING, "packet": PACKET_SCHEMA})),
        "review": ("Submit independently inspected checks, retaining all failures.", obj({"run": STRING, "report": REPORT_SCHEMA})),
        "feedback": ("Record a causal revision and invalidate affected descendants.", obj({"run": STRING, "revision": OBJECT})),
        "recover": ("Record an interrupted owned attempt after its processes have exited.", obj({"run": STRING, "module": STRING, "reason": STRING})),
        "world_bind": ("Export a current native scene/task/policy handoff; missing gates block it.",
                       obj({"run": STRING, "output": STRING, "task_id": STRING, "backend_id": STRING})),
        "world_verify": ("Recheck native gates, frozen source and contract identities.", obj({"binding": STRING})),
        "backend_register": ("Pin a scene-aware executable adapter; registration is not native qualification.", obj({"manifest": OBJECT})),
        "backend_catalog": ("Discover registered method adapters for any task/backend, including stale code.",
                            obj({"task_id": STRING, "backend_id": STRING}, [])),
        "release_promote": ("Activate a shared backend release after current paired development regressions on at least two tasks.",
                            obj({"name": STRING, "candidates": {"type": "array", "items": STRING, "minItems": 2},
                                 "regressions": {"type": "array", "items": OBJECT, "minItems": 2}, "reason": STRING})),
        "release_rollback": ("Restore the previous qualified backend release without rewriting old experiment evidence.",
                             obj({"name": STRING, "reason": STRING})),
        "release_status": ("Inspect active task-specific backend paths and verify their code and regression evidence.", obj({"name": STRING})),
        "policy_execute": ("Run a registered bound collector/trainer/evaluator under the outer module budgets.",
                           obj({"binding": STRING, "backend": STRING, "operation": {"enum": list(backends.OPERATIONS)}, "parameters": OBJECT})),
        "policy_report": ("Translate a bound worker receipt and independently authored checks into a review report.",
                          obj({"binding": STRING, "receipt": STRING, "artifact_paths": OBJECT, "checks": OBJECT,
                               "failures": {"type": "array"}}, ["binding", "receipt", "artifact_paths", "checks"])),
        "policy_review": ("Submit a saved policy review without exposing sealed evaluation traces to the planner.",
                          obj({"run": STRING, "report_path": STRING})),
        "learning_catalog": ("Inspect integrated Astra method tools and their actual connection status.", obj({})),
        "learning_inspect": ("Call a read-only Astra task/tool/experience inspection tool.",
                             obj({"name": {"enum": sorted(READ_ONLY_LEARNING)}, "arguments": OBJECT})),
        "learning_prepare": ("Ask an existing learning tool for a non-executing plan; legacy scene paths are not admitted to the new world.",
                             obj({"name": {"enum": ["sim_collect", "policy_train", "sim_evaluate"]}, "arguments": OBJECT})),
        "experience_recall": ("Search procedural reconstruction experience and optional policy library separately, preserving their scopes.",
                              obj({"context": OBJECT, "policy_query": OBJECT}, ["context"])),
        "experience_learn": ("Store scoped procedural evidence with native transfer-validation rules.", obj({"entry": OBJECT})),
    }
    return [{"type": "function", "name": name, "description": description, "parameters": schema}
            for name, (description, schema) in definitions.items()]


class Harness:
    def __init__(self, root, *, execute=False, learning_root=None, python=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.execute = execute
        self.learning_root = (Path(learning_root).resolve() if learning_root
                              else Path(__file__).resolve().parents[3] / "unified")
        self.python = python or sys.executable

    def learning_call(self, request):
        service = self.learning_root / "tool_service.py"
        if not service.is_file() and request.get("name") in {"library_inspect", "library_search"}:
            # Load the same work-library implementation used by the station.
            from importlib.util import module_from_spec, spec_from_file_location
            from ..layout import REPO
            spec = spec_from_file_location("astra_work_library", REPO / "policy_development/library/work_library.py")
            library = module_from_spec(spec)
            spec.loader.exec_module(library)
            root = self.root / "policy_library"
            if request["name"] == "library_inspect":
                result = library.status(root=root)
            else:
                args = request["arguments"]
                result = library.search(json.loads(args["query_json"]), limit=args.get("limit", 8), root=root)
            return {"id": request["id"], "result": result}
        if not service.is_file():
            raise ValueError("This optional inspection requires a tool_service.py provider configured with --learning-root.")
        env = {**os.environ,
               "ASTRA_WORK_LIBRARY_ROOT": str(self.root / "policy_library"),
               "ASTRA_TOOL_LIBRARY_ROOT": str(self.root / "policy_tool_library"),
               "ASTRA_TASK_REGISTRY_ROOT": str(self.root / "task_registry")}
        # Execution is deliberately absent. A legacy backend cannot acquire a new
        # world binding merely because its tool name matches a requested method.
        process = subprocess.run([self.python, str(service), "--run", str(self.root / "learning_transport")],
                                 input=json.dumps(request, allow_nan=False) + "\n", text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
                                 cwd=service.parent, timeout=60)
        if process.returncode:
            raise ValueError("Learning inspection failed: " + process.stderr[-1500:])
        lines = process.stdout.splitlines()
        if len(lines) != 1:
            raise ValueError("Expected exactly one learning tool response")
        result = json.loads(lines[0])
        if "error" in result:
            raise ValueError("Learning inspection: " + str(result["error"]))
        return result

    def dispatch(self, name, arguments):
        from ..oversight.contracts import validate
        schema = next((t["parameters"] for t in tool_schemas() if t["name"] == name), None)
        if schema is None:
            raise ValueError("Unknown unified tool: " + str(name))
        validate(arguments, schema)
        if name.startswith("chain_"):
            from ..chain import Chain
            from ..reconstruction import Workflow, catalog
            from ..workflow_memory import memory
            if name == "chain_catalog":
                return catalog()
            if name == "chain_init":
                return Chain.create(arguments["run"], arguments["request"]).status()
            chain = Chain(arguments["run"])
            if name == "chain_verify":
                return chain.verify_twin()
            if name == "chain_next":
                return chain.next()
            if name == "chain_plan":
                return Workflow(chain).choose(arguments["plan"])
            if name in {"chain_recall", "chain_learn"}:
                return memory(chain, name.removeprefix("chain_"), arguments["request"], self.root)
            if not self.execute and (name in {"chain_step", "chain_campaign_step"} or arguments.get("execute")):
                return {"status": "execution_disabled"}
            if name == "chain_campaign_step":
                return chain.campaign_step()
            if name == "chain_step":
                return Workflow(chain).perform(arguments["id"], arguments["arguments"], arguments.get("evidence", []))
            return chain.advance(delivery=arguments.get("delivery"), execute=arguments.get("execute", False))
        if name == "intake":
            return intake(arguments["spec"])
        if name == "plan":
            return redact_holdout(build_plan(arguments["spec"]))
        if name in {"init", "status", "progress_snapshot", "next", "run", "review", "feedback", "recover"}:
            from ..oversight.controller import Controller, read
            ctrl = Controller(arguments["run"])
            if name == "init":
                return Controller.create(arguments["run"], arguments["spec"]).status()
            if name == "progress_snapshot":
                return ctrl.progress_snapshot()
            if name in {"status", "next"}:
                result = ctrl.status()
                if name == "next":
                    plan = read(ctrl.root / "plan.json")
                    result["work"] = redact_holdout({m: plan["modules"][m] for m in result["ready"]})
                return result
            if name == "run":
                if not self.execute:
                    return {"status": "execution_disabled", "packet": arguments["packet"]}
                return ctrl.run(arguments["packet"])
            if name == "review":
                result = ctrl.accept(arguments["report"])
                return {"accepted": result["accepted"], "module": "holdout", "details": "operator_only"} if arguments["report"]["module"] == "holdout" else result
            if name == "feedback":
                return ctrl.feedback(arguments["revision"])
            return ctrl.recover(arguments["module"], arguments["reason"])
        if name == "world_bind":
            binding = export_binding(arguments["run"], arguments["output"], task_id=arguments["task_id"], backend_id=arguments["backend_id"])
            return {"path": arguments["output"], **public_binding(binding)}
        if name == "world_verify":
            return public_binding(verify_binding(arguments["binding"]))
        if name == "backend_register":
            return backends.register_backend(self.root / "backends", arguments["manifest"])
        if name == "backend_catalog":
            return {"backends": backends.catalog(self.root / "backends", **arguments)}
        if name in {"release_promote", "release_rollback", "release_status"}:
            from . import releases
            function = {"release_promote": releases.promote, "release_rollback": releases.rollback,
                        "release_status": releases.status}[name]
            return function(self.root / "shared_releases", **arguments)
        if name == "policy_execute":
            operation = backends.execute if self.execute else backends.prepare_execution
            result = operation(arguments["binding"], arguments["backend"], arguments["operation"], arguments["parameters"])
            return {**result, "execution_enabled": self.execute}
        if name == "policy_report":
            report = translate_receipt(arguments["binding"], arguments["receipt"],
                                       artifact_paths=arguments["artifact_paths"], checks=arguments["checks"],
                                       failures=arguments.get("failures"))
            if report["module"] == "holdout":
                path = self.root / "operator_reviews" / (digest(report) + ".json")
                if not path.exists():
                    backends.write_new(path, report)
                return {"report_path": str(path), "module": "holdout", "details": "operator_only", "accepted": False}
            return report
        if name == "policy_review":
            from ..oversight.controller import Controller
            report = backends.read(arguments["report_path"])
            result = Controller(arguments["run"]).accept(report)
            return {"accepted": result["accepted"], "module": "holdout", "details": "operator_only"} if report["module"] == "holdout" else result
        if name == "learning_catalog":
            connected = (self.learning_root / "tool_service.py").is_file()
            result = self.learning_call({"operation": "list_tools"}) if connected else {"tools": [
                {"name": "library_inspect", "description": "Inspect local policy memory."},
                {"name": "library_search", "description": "Search local policy memory by query_json and optional limit."},
            ]}
            descriptor = self.learning_root / "bound_capabilities.json"
            capabilities = backends.read(descriptor)["capabilities"] if descriptor.is_file() else []
            return {**result, "inspection_provider_connected": connected, "bound_capabilities": capabilities,
                    "execution_route": "policy_execute with a registered world-bound adapter",
                    "legacy_execution": "not_admitted_to_new_scene", "simulation_only": True}
        if name in {"learning_inspect", "learning_prepare"}:
            request = {"id": "inspect-" + str(time.time_ns()), "name": arguments["name"], "arguments": arguments["arguments"]}
            return {"legacy_result": self.learning_call(request), "executes_learning_job": False,
                    "scene_binding": "requires_world_bind_and_bound_adapter"}
        from ..oversight.knowledge import lookup, record
        if name == "experience_learn":
            for path in arguments["entry"].get("evidence", []):
                if _sealed_evidence(path):
                    raise ValueError("Sealed holdout evidence cannot enter the development work library")
            return record(self.root / "procedural_knowledge", arguments["entry"])
        result = {"reconstruction": lookup(self.root / "procedural_knowledge", arguments["context"]),
                  "reuse_rule": "Procedures may transfer; source fits and policy artifacts require current-world validation."}
        if "policy_query" in arguments:
            result["policy"] = self.learning_call({"id": "recall-" + str(time.time_ns()),
                                                    "name": "library_search", "arguments": arguments["policy_query"]})
        return result


def serve(stream, output, harness):
    """Durable IDs prevent repeated commands after client retries/interruption."""
    db = sqlite3.connect(harness.root / "requests.sqlite3", timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS requests(id TEXT PRIMARY KEY, signature TEXT, status TEXT, response TEXT)")
    try:
        for line in stream:
            identity, reserved = None, False
            try:
                if len(line.encode()) > 262144:
                    raise ValueError("Request exceeds 256 KiB")
                request = json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError("Non-finite JSON")))
                if request == {"operation": "list_tools"}:
                    response = {"tools": tool_schemas(), "simulation_only": True}
                else:
                    if not isinstance(request, dict) or set(request) != {"id", "name", "arguments"}:
                        raise ValueError("Expected id, name and arguments")
                    identity = request["id"]
                    if not isinstance(identity, str) or not 1 <= len(identity) <= 96:
                        raise ValueError("Invalid request id")
                    signature = digest({"request": request, "execute": harness.execute})
                    db.execute("BEGIN IMMEDIATE")
                    old = db.execute("SELECT signature,status,response FROM requests WHERE id=?", (identity,)).fetchone()
                    if old:
                        db.commit()
                        if old[0] != signature:
                            raise ValueError("Request id reused with different content or execution authority")
                        response = json.loads(old[2]) if old[2] else {"id": identity, "status": old[1],
                                    "message": "Prior reservation retained; inspect/recover the owned module instead of replaying it."}
                        if request["name"] in {"chain_verify", "chain_next", "chain_catalog", "chain_recall", "status", "next", "world_verify", "release_status", "backend_catalog",
                                               "experience_recall", "learning_inspect", "learning_catalog"}:
                            # A prior successful read is historical evidence, not a
                            # current qualification. Keep the original ledger intact.
                            with redirect_stdout(sys.stderr):
                                current = harness.dispatch(request["name"], request["arguments"])
                            response = {"id": identity, "result": current, "cached": True,
                                        "cache_note": "Revalidated read-only; original response retained."}
                    else:
                        db.execute("INSERT INTO requests VALUES(?,?,'reserved',NULL)", (identity, signature))
                        db.commit()
                        reserved = True
                        with redirect_stdout(sys.stderr):
                            result = harness.dispatch(request["name"], request["arguments"])
                        response = {"id": identity, "result": result}
                        with db:
                            db.execute("UPDATE requests SET status='completed',response=? WHERE id=?",
                                       (json.dumps(response, allow_nan=False), identity))
            except Exception as exc:
                db.rollback()
                response = {"id": identity, "error": {"type": type(exc).__name__, "message": str(exc)[:2000]}}
                if reserved:
                    with db:
                        db.execute("UPDATE requests SET status='failed',response=? WHERE id=?", (json.dumps(response), identity))
            output.write(json.dumps(response, allow_nan=False) + "\n")
            output.flush()
    finally:
        db.close()
