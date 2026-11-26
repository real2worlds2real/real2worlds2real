"""Bounded, audited Astra tool selection over the world-bound unified harness.

The runner starts after source reconstruction is independently accepted. It
chooses no manipulation method itself and makes no automatic semantic reviews.
Final outcomes are operator artifacts and never become another model prompt.
"""
import argparse
from contextlib import contextmanager
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from ..oversight.contracts import validate
from ..oversight.evidence import digest, file_hash
from ..oversight.policy_bridge import verify_binding
from .backends import load_backend, read, write_new
from .service import Harness, serve


SCHEMA = "astra_unified_agent_v1"
MODEL = "gpt-6-astra"
_PRIVATE_KEYS = {"oversight", "holdout", "final", "confirmation", "heldout", "raw_observations",
                 "raw_frames", "holdout_seeds", "final_seeds", "password", "api_key", "access_token"}
_PATH = re.compile(r"(?:[A-Za-z]:[\\/]|/(?:home|Users|var|tmp|mnt|etc|run)/|~/)")
_SECRET = re.compile(r"(?i)-----BEGIN [\w ]*PRIVATE KEY-----|\bBearer\s+[\w.=-]+|\b(?:sk-|gh[pousr]_)[\w-]{12,}")
# Match the real transport's credential-field policy even when a value is only
# administrative prose. The complete grant remains in the operator report.
_SENSITIVE_KEY = re.compile(r"(?i)^(?:api[_-]?key|authorization|password|cookie|credential|secret|access_token|refresh_token|private_key)$")
EMPTY = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
TEXT = {"type": "string", "minLength": 1, "maxLength": 1000}
REVISION = {"type": "object", "properties": {k: TEXT for k in (
    "diagnosis", "transfer_hypothesis", "expected_real_benefit", "anti_hack_review", "ablation_plan")},
    "required": ["diagnosis", "transfer_hypothesis", "expected_real_benefit", "anti_hack_review", "ablation_plan"],
    "additionalProperties": False}
MEMORY = {"type": "object", "properties": {k: TEXT for k in (
    "procedure", "applicability", "limitations", "negative_results")},
    "required": ["procedure", "applicability", "limitations", "negative_results"], "additionalProperties": False}
RECALL = {"type": "object", "properties": {"context": {"type": "object", "properties": {
    key: TEXT for key in ("task_id", "backend_id", "validation_scope", "policy_io_sha256")},
    "additionalProperties": False}}, "additionalProperties": False}


class DecisionContractError(ValueError):
    """A real model response needs correction before any worker tool executes."""


def public_summary(value, depth=0):
    """Small development summaries, never raw paths, credentials or split payloads."""
    if depth > 6:
        return "nested_summary_omitted"
    if isinstance(value, dict):
        result = {}
        for key, item in list(value.items())[:40]:
            normalized = key.lower().replace("-", "_")
            if (_SENSITIVE_KEY.fullmatch(key) or normalized in _PRIVATE_KEYS
                    or any(normalized.startswith(k + "_") for k in ("holdout", "final", "heldout", "confirmation"))):
                continue
            if _SECRET.search(key):
                raise ValueError("Credential-like keys must not enter a model prompt")
            safe_key = "artifact_field_" + str(len(result)) if _PATH.search(key) else key
            result[safe_key] = public_summary(item, depth + 1)
        return result
    if isinstance(value, list):
        return [public_summary(x, depth + 1) for x in value[:12]] + (["additional_items_omitted"] if len(value) > 12 else [])
    if isinstance(value, str):
        if _SECRET.search(value):
            raise ValueError("Credential-like text must not enter a model prompt")
        return "artifact_reference_omitted" if _PATH.search(value) else value[:1000]
    if value is None or type(value) in (bool, int, float):
        return value
    raise ValueError("Unsupported model summary value")


def bounded_prompt(facts, limit=23000):
    """Fit the actual transport's 24 KiB bound without truncating tool contracts.

    Full receipts/history remain on the worker. Only redundant diagnostics and
    older prose are compacted; task criteria, latest verdict and tool schemas are
    preserved. This does not change execution budgets or evidence acceptance.
    """
    facts = json.loads(json.dumps(facts, allow_nan=False))
    notes = []
    def size():
        return len(json.dumps(facts, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode())
    diagnosis = facts.get("latest_diagnosis", {})
    if isinstance(diagnosis, dict) and "development" in diagnosis:
        diagnosis.pop("development")  # Same receipt/verdict already appears in latest_development.
    def compact(value, text=300, count=8):
        if isinstance(value, str):
            return value[:text] + (" [worker retains full text]" if len(value) > text else "")
        if isinstance(value, list):
            return [compact(v, text, count) for v in value[:count]] + ([{"additional_items_omitted": len(value)-count}] if len(value)>count else [])
        if isinstance(value, dict):
            return {k: compact(v, text, count) for k, v in value.items()}
        return value
    if size() > limit:
        facts["earlier_development"] = compact(facts.get("earlier_development", []))
        notes.append("Older development prose compacted; complete revisions remain in the worker journal.")
    while size() > limit and facts.get("earlier_development"):
        facts["earlier_development"].pop(0)
        if "Older development records omitted to retain current tool and task contracts." not in notes:
            notes.append("Older development records omitted to retain current tool and task contracts.")
    if size() > limit:
        experience = facts.get("experience", {})
        if isinstance(experience, dict):
            experience["policy_library"] = compact(experience.get("policy_library", {}), text=240, count=4)
            experience["procedures"] = compact(experience.get("procedures", []), text=240, count=6)
        notes.append("Historical experience prose compacted; full scoped entries remain retrievable.")
    while size() > limit and facts.get("experience", {}).get("procedures"):
        facts["experience"]["procedures"].pop(0)
    if size() > limit:
        latest = facts.get("latest_development", {}).get("review")
        if isinstance(latest, dict) and "diagnostics" in latest:
            latest["diagnostics"] = compact(latest["diagnostics"], text=300, count=8)
        facts["latest_diagnosis"] = compact(facts.get("latest_diagnosis", {}), text=300, count=4)
        notes.append("Detailed development diagnostics compacted; latest acceptance verdict is unchanged.")
    if notes:
        facts["summary_omissions"] = notes
    if size() > 24576:
        raise ValueError("Task/tool contracts and current diagnostic summary exceed the audited model transport limit")
    return facts


def experience_summary(result):
    """Bound library prose as well as artifact size before sending model facts."""
    entries = []
    for row in result.get("reconstruction", [])[-6:]:
        entry = row["entry"]
        entries.append({"id": entry["id"], "status": row["status"], "context": entry["context"],
            **{key: str(entry[key])[:360] for key in
               ("problem", "procedure", "applicability", "limitations", "negative_results")}})
    return public_summary({"procedures": entries, "policy_library": result.get("policy", {}),
                           "reuse_rule": result.get("reuse_rule", "Validate applicability on the current task.")})


class IsolatedTransport:
    def __init__(self, learning_root, python=sys.executable):
        self.root, self.python = Path(learning_root).resolve(), python

    def call(self, request, timeout=660):
        script = self.root / "astra_unified_decide.py"
        proc = subprocess.run([self.python, "-B", str(script)], input=json.dumps(request, allow_nan=False),
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              cwd=self.root, timeout=timeout)
        if proc.returncode:
            raise RuntimeError("Astra/reviewer transport failed: " + proc.stderr[-1600:])
        rows = proc.stdout.splitlines()
        if len(rows) != 1:
            raise ValueError("Expected exactly one structured Astra/reviewer response")
        return json.loads(rows[0])


class DecisionSpool:
    """Authenticated client relays public decisions; jobs stay on this worker."""
    def __init__(self, root, timeout=900):
        self.root, self.timeout = Path(root), timeout
        self.root.mkdir(parents=True, exist_ok=True)

    def call(self, request):
        phase = request["phase"]
        if not re.fullmatch(r"step-\d{3}", phase):
            raise ValueError("Invalid decision spool phase")
        path = self.root / (phase + ".request.json")
        if path.exists():
            if read(path) != request:
                raise ValueError("Decision request changed during relay recovery")
        else:
            write_new(path, request)
        response = self.root / (phase + ".response.json")
        deadline = time.monotonic() + self.timeout
        while not response.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("Authenticated decision relay did not respond before its deadline")
            time.sleep(0.25)
        envelope = read(response)
        if envelope.get("schema") != "astra_decision_relay_v1" or envelope.get("request_sha256") != digest(request):
            raise ValueError("Decision relay response does not bind the current request")
        if "error" in envelope:
            raise RuntimeError("Authenticated decision relay failed: " + envelope["error"])
        return envelope["response"]


class UnifiedAgentRunner:
    def __init__(self, config, *, harness=None, decision_transport=None, reviewer=None):
        self.config = read(config) if isinstance(config, (str, Path)) else config
        required = {"schema", "campaign", "binding", "state", "backends", "max_model_calls", "reviewer", "execution_mode"}
        if not required <= set(self.config) or self.config["schema"] != SCHEMA:
            raise ValueError("Expected a complete unified-agent configuration")
        if self.config["execution_mode"] not in {"production", "offline_fixture"}:
            raise ValueError("Explicit production or offline_fixture execution mode required")
        if self.config["execution_mode"] == "production" and (decision_transport is not None or reviewer is not None):
            raise ValueError("Production cannot substitute fixture model/reviewer callbacks")
        if self.config["execution_mode"] == "offline_fixture" and decision_transport is None:
            raise ValueError("Offline fixtures require an explicitly injected test decision transport")
        if self.config["execution_mode"] == "production":
            from .preflight import verify as verify_runtime_preflight
            self.runtime_preflight = verify_runtime_preflight(self.config)
        if type(self.config["max_model_calls"]) is not int or not 1 <= self.config["max_model_calls"] <= 100:
            raise ValueError("A finite model-call budget from 1 to 100 is required")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,59}", self.config["campaign"]):
            raise ValueError("Invalid campaign identifier")
        self.root = Path(self.config["state"]).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.harness = harness or Harness(self.config.get("harness_state", self.root / "tools"), execute=True,
                                          learning_root=self.config.get("learning_root"))
        if not self.harness.execute:
            raise ValueError("The agent execution runner requires an execution-enabled harness")
        self.transport = decision_transport
        self.reviewer = reviewer
        self.isolated = IsolatedTransport(self.harness.learning_root, self.config.get("python", sys.executable))
        mode = self.config.get("decision_transport", "local_codex")
        if mode not in {"local_codex", "authenticated_relay"}:
            raise ValueError("Unsupported production decision transport")
        self.spool = DecisionSpool(self.root / "decision_spool") if mode == "authenticated_relay" else None
        self.binding = verify_binding(self.config["binding"])
        from . import continuation
        self.continuation = continuation.inspect(self.config, completed_report_exists=(self.root/'report.json').is_file())
        self.implementation = {str(path): file_hash(path) for path in (
            Path(__file__).resolve(), self.isolated.root / "astra_unified_decide.py",
            self.isolated.root / "campaign_ledger.py", self.isolated.root / "astra_decide.py",
            Path(continuation.__file__).resolve())}
        if self.config["execution_mode"] == "production":
            self.implementation[self.runtime_preflight["report"]] = self.runtime_preflight["sha256"]
            self.implementation[str(Path(verify_runtime_preflight.__code__.co_filename).resolve())] = file_hash(verify_runtime_preflight.__code__.co_filename)
        if self.continuation:
            self.implementation.update(self.continuation["evidence_inputs"])
        for path, sha in self.config.get("control_plane_inputs", {}).items():
            if not Path(path).is_absolute() or file_hash(path) != sha:
                raise ValueError("Explicit control-plane implementation changed")
            self.implementation[path] = sha
        self.config_sha = digest({"configuration": self.config, "implementation": self.implementation})
        config_path = self.root / "config.json"
        if config_path.exists():
            if read(config_path) != self.config:
                raise ValueError("Runner configuration, authority and model mode are immutable")
        else:
            write_new(config_path, self.config)
        implementation_path = self.root / "implementation.json"
        if implementation_path.exists():
            if read(implementation_path) != self.implementation:
                raise ValueError("Runner implementation changed; start a new audited campaign")
        else:
            write_new(implementation_path, self.implementation)
        self.backends = {}
        for entry in self.config["backends"]:
            package = load_backend(entry["path"])
            if any(package["manifest"][key] != self.binding[key] for key in ("task_id", "backend_id")):
                raise ValueError("Configured backend belongs to a different task/world contract")
            handle = "backend_" + package["sha256"][:12]
            self.backends[handle] = {"path": entry["path"], "package": package,
                                     "parameters": entry.get("parameter_schemas", {})}
        if not self.backends:
            raise ValueError("No executable backend has been registered")

    @contextmanager
    def _lock(self):
        path = self.root / "runner.lock"
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise ValueError("Runner lock exists; inspect its owner before recovering an interrupted process") from exc
        try:
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            yield
        finally:
            path.unlink()

    def _load(self):
        prior = None
        state = {"step": 0, "retrieved": False, "pending": None, "last": None, "development_history": [],
                 "review_result": None, "diagnosed": False, "failure_pending": False, "memory_for": None, "done": None,
                 "consecutive_retrievals": 0, "retrieval_history": []}
        history = sorted((self.root / "history").glob("*.json"))
        for path in history:
            row = read(path)
            body = {k: v for k, v in row.items() if k != "sha256"}
            if row["previous"] != prior or digest(body) != row["sha256"] or row["config_sha256"] != self.config_sha:
                raise ValueError("Agent history integrity failure")
            state, prior = row["state"], row["sha256"]
        if not history and self.continuation:
            from .continuation import restore
            state = restore(self.continuation, state, self._status())
            state["diagnosis"] = self._diagnose(state)
        return state

    def _save(self, state, event):
        directory = self.root / "history"
        directory.mkdir(exist_ok=True)
        files = sorted(directory.glob("*.json"))
        row = {"previous": read(files[-1])["sha256"] if files else None, "config_sha256": self.config_sha,
               "event": event, "state": state}
        write_new(directory / f"{len(files):06d}.json", {**row, "sha256": digest(row)})

    def _call(self, identity, name, arguments):
        output = io.StringIO()
        request = {"id": identity, "name": name, "arguments": arguments}
        serve(io.StringIO(json.dumps(request, allow_nan=False) + "\n"), output, self.harness)
        response = json.loads(output.getvalue())
        if "error" in response:
            raise ValueError(response["error"]["message"])
        return response["result"]

    def _status(self):
        if any(file_hash(path) != sha for path, sha in self.implementation.items()):
            raise ValueError("Runner implementation changed during execution")
        verify_binding(self.config["binding"])
        return self.harness.dispatch("status", {"run": self.binding["oversight_run"]})

    def _options(self, state, status):
        choices = {"stop": {"description": "Stop with retained evidence when executable progress is unavailable", "schema": EMPTY}}
        if state.get("consecutive_retrievals", 0) < 3:
            choices["retrieve"] = {"description": "Retrieve prior experience using optional exact context filters; empty context searches across tasks. At most three consecutive retrievals before inspecting diagnostics or making executable progress.", "schema": RECALL}
        if not state["retrieved"]:
            return choices
        modules = status["modules"]
        from ..oversight.controller import Controller
        plan, _ = Controller(self.binding["oversight_run"])._load()
        budget = plan["contract"]["budget"]
        failed = state.get("failure_pending", False) or any(modules[m]["state"] == "failed" for m in ("data", "train", "validation"))
        if not failed and state["last"] and "error" not in state["last"] and any(modules[m]["state"] == "awaiting_review" for m in ("data", "train", "validation")):
            choices["review"] = {"description": "Run the pinned independent evidence verifier and submit its checks", "schema": EMPTY}
        if state["last"] or failed or any(modules[m]["attempts"] for m in ("data", "train", "validation")) or state.get("consecutive_retrievals", 0) >= 3:
            choices["diagnose"] = {"description": "Inspect current development failure and verifier diagnostics", "schema": EMPTY}
        if state["diagnosed"] and status["cycle"] < budget["max_cycles"]:
            for target in ("data", "train"):
                required = ("data", "train", "validation") if target == "data" else ("train", "validation")
                if any(modules[module]["attempts"] >= budget["max_attempts_per_module"] for module in required):
                    continue
                choices["revise_" + target] = {"description": "Record a causal revision and invalidate " + target + " descendants", "schema": REVISION}
        if failed:
            # Instructions alone are insufficient: even a ready failed module
            # cannot start another job before diagnosis and an explicit revision.
            return choices
        for handle, item in self.backends.items():
            load_backend(item["path"])
            for operation, command in item["package"]["manifest"]["operations"].items():
                module = {"collect": "data", "import_data": "data", "train": "train", "develop": "validation", "confirm": "holdout"}[operation]
                if module not in status["ready"]:
                    continue
                if modules[module]["attempts"] >= budget["max_attempts_per_module"]:
                    continue
                if operation == "confirm" and state["memory_for"] != self._development_identity():
                    continue
                key = ("finalize" if operation == "confirm" else operation) + "__" + handle
                choices[key] = {"description": ("Consume the sealed final evaluation once and end model interaction" if operation == "confirm" else "Execute a registered " + operation + " method"),
                                "method": command["method"], "backend_handle": handle, "operation": operation,
                                "schema": item["parameters"].get(operation, EMPTY)}
        if modules["validation"]["current"] and state["memory_for"] != self._development_identity():
            choices["remember"] = {"description": "Store development-only experience with explicit applicability and limitations", "schema": MEMORY}
        return choices

    def _development_identity(self):
        from ..oversight.controller import Controller
        _, state = Controller(self.binding["oversight_run"])._load()
        item = state["modules"]["validation"]["accepted"]
        return item["id"] if item else None

    def _scope(self):
        from ..oversight.controller import Controller
        plan, _ = Controller(self.binding["oversight_run"])._load()
        strata = len(self.binding["success_contract"]["evaluation"]["strata"])
        case_counts = {public: len(self.binding["splits"][split]["groups"])
                               * len(self.binding["splits"][split]["seeds"]) * strata
                       for public, split in (("training", "train"), ("development", "validation"),
                                             ("sealed_confirmation", "holdout"))}
        return {"validation_scope": plan["contract"].get("validation_scope", "deployment"),
                "task_scope": plan["contract"]["task"], "completion_claim": plan["completion_claim"],
                "data_requirements": {key: plan["contract"]["data"][key] for key in ("minimum_episodes", "coverage", "failure_policy")},
                "public_case_counts": case_counts,
                "collection_repetition": {
                    "meaning": "If a collector exposes episodes_per_case, it repeats each frozen training case identity without adding split groups or seed identities. Whether repetitions draw different physical initial states depends on the adapter; repetitions must not be assumed identical.",
                    "requested_episode_total": "training case count multiplied by episodes_per_case",
                    "one_repetition_requests": case_counts["training"],
                    "minimum_accepted_successful_episodes": plan["contract"]["data"]["minimum_episodes"],
                    "acceptance_note": "Minimum episodes is a total successful-demonstration requirement, not a per-case repetition target; failed episodes do not count."},
                "development_checks": {module: plan["modules"][module]["checks"] for module in ("data", "train", "validation")},
                "execution_budget": {key: plan["contract"]["budget"][key] for key in
                                     ("max_cycles", "max_attempts_per_module", "max_jobs", "wall_seconds", "command_timeout_seconds")},
                "real_validation": "pending"}

    def _summary(self, state):
        result = state["last"]
        if not result:
            return {}
        if "error" in result:
            return {"error": public_summary(result["error"])}
        receipt = read(result["receipt"])
        if result.get("runner_receipt_sha256") != file_hash(result["receipt"]):
            raise ValueError("Development receipt changed after execution")
        if receipt.get("operation") == "confirm":
            raise ValueError("A final receipt cannot become a model summary")
        return {"operation": receipt.get("operation"), "facts": public_summary(receipt.get("facts", {})),
                "review": public_summary(state["review_result"])}

    def _diagnose(self, state):
        """Read bounded exception summaries from owned development attempts only."""
        from ..oversight.controller import Controller
        _, current = Controller(self.binding["oversight_run"])._load()
        if current.get("holdout_consumed"):
            raise ValueError("Sealed execution diagnostics cannot enter development")
        attempts = []
        for module in ("data", "train", "validation"):
            row = current["modules"][module]
            if not row["attempts"] or row["status"] not in {"failed", "awaiting_review", "invalidated"}:
                continue
            attempt = row["attempts"][-1]
            messages = []
            for log in sorted(Path(attempt["directory"]).glob("command-*.log")):
                with log.open("rb") as stream:
                    stream.seek(max(0, log.stat().st_size - 16384))
                    tail = stream.read().decode("utf-8", "replace")
                for line in tail.splitlines():
                    if re.search(r"\b\w*(?:Error|Exception)\b|\bFAILED\b", line) and not _SECRET.search(line):
                        messages.append(public_summary(line.strip())[:400])
            attempts.append({"module": module, "state": row["status"], "attempt": attempt["number"],
                             "worker_error": public_summary(attempt.get("error")),
                             "exception_summaries": messages[-4:]})
        latest_feedback = None
        for path in sorted((Path(self.binding["oversight_run"])/"history").glob("*.json"), reverse=True):
            row = read(path)
            if row["event"] == "feedback_revision":
                latest_feedback = public_summary({"revision":row["details"].get("revision"),
                                                  "invalidated":row["details"].get("invalidated")})
                break
        return {"development": self._summary(state), "owned_attempts": attempts, "latest_causal_feedback":latest_feedback,
                "retrieval_history":public_summary(state.get("retrieval_history", [])[-6:]),
                "invalidation_semantics":"An explicit causal revision cleared dependent acceptance; ready invalidated modules can execute under unchanged checks. Earlier attempt exceptions remain historical until new execution.",
                "scope": "Development-only exception summaries; raw logs remain on the worker."}

    def _decide(self, state, status, choices):
        facts = {"task": {"id": self.binding["task_id"], "backend": self.binding["backend_id"],
                           "success": public_summary(self.binding["success_contract"]["definition"]),
                           "policy_io": public_summary(self.binding["policy_io"]), "scope": public_summary(self._scope())},
                 "development_modules": {k: v for k, v in status["modules"].items() if k not in {"holdout", "delivery"}},
                 "available_tools": [{"id": k, **v} for k, v in choices.items()],
                 "experience": experience_summary(state.get("experience", {})), "latest_development": self._summary(state),
                 "earlier_development": public_summary(state["development_history"][-6:]),
                 "latest_diagnosis": public_summary(state.get("diagnosis", {})),
                 "latest_tool_error": public_summary(state.get("tool_error")),
                 "continuation": public_summary(state.get("continuation_context")),
                 "retrieval_history": public_summary(state.get("retrieval_history", [])[-6:]),
                 "retrieval_budget": {"consecutive_used":state.get("consecutive_retrievals", 0),"limit":3,
                    "rule":"Repeated reads do not create evidence. Inspect diagnosis/causal feedback, choose another available tool, or stop; no method is selected by this rule."},
                 "decision_error": public_summary(state.get("decision_error")),
                 "remaining_model_calls": self.config["max_model_calls"] - state["step"]}
        facts = bounded_prompt(facts)
        request = {"operation": "decide", "ledger_root": str(self.root / "astra_ledger"),
                   "campaign": self.config["campaign"], "task_id": "task-" + digest(self.binding["task_id"])[:24],
                   "max_model_calls": self.config["max_model_calls"], "config_sha256": self.config_sha,
                   "phase": f"step-{state['step']:03d}", "facts": facts, "tool_ids": list(choices)}
        response = self.transport(request) if self.transport else (self.spool.call(request) if self.spool else self.isolated.call(request))
        mode = self.config["execution_mode"]
        if response.get("execution_mode") != mode:
            raise ValueError("Model transport mode differs from the frozen runner mode")
        if mode == "production":
            audit = response.get("audit", {})
            if audit.get("requested_model") != MODEL or audit.get("transport") != "codex_exec" or audit.get("exit_code") != 0 or not audit.get("usage"):
                raise ValueError("Production requires a real Astra transport and token-usage audit")
        decision = response["decision"]
        if set(decision) != {"reason", "tool_id", "tool_manifest_json"} or decision["tool_id"] not in choices:
            raise DecisionContractError("Model selected an unavailable tool")
        try:
            arguments = json.loads(decision["tool_manifest_json"], parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite arguments")))
            validate(arguments, choices[decision["tool_id"]]["schema"])
        except (ValueError, TypeError) as exc:
            raise DecisionContractError("Tool arguments do not meet the advertised schema: " + str(exc)) from exc
        return {"decision": decision, "arguments": arguments, "choice": choices[decision["tool_id"]],
                "audit": response.get("audit", {}), "facts_sha256": digest(facts)}

    def _review(self, identity, result):
        if not self.config["reviewer"] and not self.reviewer:
            raise ValueError("No independent verifier is configured; execution is not acceptance")
        directory = self.root / "reviews" / identity
        directory.mkdir(parents=True, exist_ok=True)
        audit_path = directory / "verifier.json"
        if audit_path.exists():
            reviewed = read(audit_path)
        else:
            if self.reviewer:
                reviewed = self.reviewer(self.config["binding"], result, output_dir=directory)
            else:
                reviewed = self.isolated.call({"operation": "review", "reviewer": self.config["reviewer"],
                    "binding": self.config["binding"], "result": result, "output_dir": str(directory)}, timeout=300)
            if not isinstance(reviewed, dict) or not {"checks", "failures"} <= set(reviewed):
                raise ValueError("Independent verifier did not return checks and failures")
            write_new(audit_path, reviewed)
        report = self._call(identity + "-report", "policy_report", {"binding": self.config["binding"],
            "receipt": result["receipt"], "artifact_paths": result["artifact_paths"],
            "checks": reviewed["checks"], "failures": reviewed["failures"]})
        if "report_path" in report:
            accepted = self._call(identity + "-review", "policy_review", {"run": self.binding["oversight_run"], "report_path": report["report_path"]})
        else:
            accepted = self._call(identity + "-review", "review", {"run": self.binding["oversight_run"], "report": report})
        return {**accepted, "diagnostics": reviewed.get("diagnostics", {}), "verifier": str(audit_path)}

    def _execute(self, state, pending):
        name, args, choice = pending["decision"]["tool_id"], pending["arguments"], pending["choice"]
        identity = self.config["campaign"] + f"-{state['step']:03d}"
        if name == "stop":
            state["done"] = "agent_stopped_unmet"
        elif name == "retrieve":
            context = args.get("context", self.config.get("recall_context", {}))
            state["experience"] = self._call(identity, "experience_recall", {"context": context,
                **({"policy_query": self.config["policy_query"]} if self.config.get("policy_query") else {})})
            state["retrieved"] = True
            state["consecutive_retrievals"] = state.get("consecutive_retrievals", 0) + 1
            signature = digest(state["experience"])
            prior = state.setdefault("retrieval_history", [])
            prior.append({"context":context,"matched_procedures":len(state["experience"].get("reconstruction", [])),
                          "result_sha256":signature,"no_new_evidence":any(row["result_sha256"]==signature for row in prior),
                          "step":state["step"]})
        elif name == "diagnose":
            state["diagnosis"] = self._diagnose(state)
            state["diagnosed"] = True
        elif name.startswith("revise_"):
            summary = self._summary(state)
            if not state["last"] or "receipt" not in state["last"]:
                evidence = self.root / "diagnostics" / (identity + ".json")
                content = {"error": state["last"], "observed": state.get("diagnosis"), "diagnosis": args}
                if evidence.exists():
                    if read(evidence) != content:
                        raise ValueError("Diagnostic evidence changed")
                else:
                    write_new(evidence, content)
            else:
                evidence = Path(state["last"]["receipt"])
            self._call(identity, "feedback", {"run": self.binding["oversight_run"], "revision": {
                "targets": [name.removeprefix("revise_")], **args, "evidence": [str(evidence)]}})
            state["development_history"].append({"observed": summary, "revision": args, "target": name.removeprefix("revise_")})
            state.update(last=None, review_result=None, diagnosed=False, failure_pending=False, memory_for=None, diagnosis=None)
        elif name == "review":
            state["review_result"] = self._review(identity, state["last"])
            state["diagnosed"] = False
            state["failure_pending"] = not state["review_result"]["accepted"]
        elif name == "remember":
            identity_dev = self._development_identity()
            entry = {"id": identity.replace(".", "_"), "problem": self.binding["success_contract"]["definition"],
                     "context": {"task_id": self.binding["task_id"], "backend_id": self.binding["backend_id"],
                                 "validation_scope": self._scope()["validation_scope"],
                                 "policy_io_sha256": digest(self.binding["policy_io"])},
                     "procedure": args["procedure"], "applicability": args["applicability"],
                     "limitations": args["limitations"], "negative_results": [args["negative_results"]],
                     "transfer_rationale": "Reuse only after checking the current source, world and policy I/O.",
                     "evidence": [state["last"]["receipt"], state["review_result"]["verifier"]], "validation_runs": []}
            state["memory_write"] = self._call(identity, "experience_learn", {"entry": entry})
            state["memory_readback"] = self._call(identity + "-readback", "experience_recall", {"context": entry["context"]})
            if not any(row["entry"]["id"] == entry["id"] and row["path"] == state["memory_write"]["path"]
                       for row in state["memory_readback"].get("reconstruction", [])):
                raise ValueError("Written experience could not be retrieved with its declared scope")
            state["memory_for"] = identity_dev
        else:
            handle, operation = choice["backend_handle"], choice["operation"]
            result = self._call(identity, "policy_execute", {"binding": self.config["binding"],
                "backend": self.backends[handle]["path"], "operation": operation, "parameters": args})
            result = {**result, "runner_receipt_sha256": file_hash(result["receipt"])}
            state.update(last=result, review_result=None, diagnosed=False)
            if operation == "confirm":
                state["review_result"] = self._review(identity + "-sealed", result)
                state["done"] = "simulation_policy_accepted" if state["review_result"]["accepted"] else "sealed_evaluation_failed"
        if name not in {"retrieve", "diagnose", "stop"}:
            state["consecutive_retrievals"] = 0

    def run(self):
        with self._lock():
            state = self._load()
            completed = self.root/'report.json'
            if state['done'] and completed.exists():
                report = read(completed)
                seal = self.root/'report.integrity.json'
                if (report.get('config_sha256') != self.config_sha or report.get('state') != state
                        or report.get('binding_sha256') != self.binding['binding_sha256']
                        or (seal.exists() and read(seal).get('report_sha256') != file_hash(completed))):
                    raise ValueError('Completed agent report differs from its immutable history/configuration')
                return report
            if self.continuation and not list((self.root/"history").glob("*.json")):
                self._save(state, "explicit_continuation_restored")
            while not state["done"]:
                if state["pending"] is None:
                    if state["step"] >= self.config["max_model_calls"]:
                        state["done"] = "model_budget_exhausted"
                        self._save(state, "budget_exhausted")
                        break
                    status = self._status()
                    if status["closed"] or status["modules"]["holdout"]["state"] not in {"pending", "invalidated"}:
                        state["done"] = "sealed_stage_already_started"
                        self._save(state, "sealed_stop")
                        break
                    if status.get("budget_stop"):
                        state["done"] = "execution_budget_exhausted"
                        self._save(state, "execution_budget_stop")
                        break
                    choices = self._options(state, status)
                    if (state.get("failure_pending") and state["diagnosed"]
                            and not any(name.startswith("revise_") for name in choices)):
                        state["done"] = "development_budget_exhausted"
                        self._save(state, "development_budget_exhausted")
                        break
                    try:
                        state["pending"] = self._decide(state, status, choices)
                    except DecisionContractError as exc:
                        state["decision_error"] = str(exc)
                        state["step"] += 1
                        self._save(state, "invalid_decision_retained")
                        continue
                    except Exception as exc:
                        state["done"] = "model_decision_failed"
                        state["tool_error"] = {"tool":"decision_transport",
                            "error":type(exc).__name__ + ": " + str(exc)}
                        self._save(state, "model_decision_failed")
                        break
                    state["decision_error"] = None
                    self._save(state, "decision_reserved")
                try:
                    self._execute(state, state["pending"])
                    state["tool_error"] = None
                except Exception as exc:
                    finalizing = state["pending"]["choice"].get("operation") == "confirm"
                    if "operation" in state["pending"]["choice"] or state["pending"]["decision"]["tool_id"] == "review":
                        state["failure_pending"] = True
                        state["last"] = {"error": type(exc).__name__ + ": " + str(exc)}
                        state["review_result"] = None
                    # A failed memory/retrieval call must not discard a current
                    # accepted development receipt or force another training job.
                    state["tool_error"] = {"tool": state["pending"]["decision"]["tool_id"],
                                           "error": type(exc).__name__ + ": " + str(exc)}
                    if finalizing:
                        state["done"] = "sealed_evaluation_incomplete"
                state["pending"] = None
                state["step"] += 1
                self._save(state, "tool_completed")
            report = {"schema": "astra_unified_agent_report_v1", "status": state["done"],
                      "execution_mode": self.config["execution_mode"], "model": MODEL,
                      "binding_sha256": self.binding["binding_sha256"], "config_sha256": self.config_sha,
                      "decision_steps": state["step"], "memory_written": bool(state["memory_for"]),
                      "real_validation": "pending", "scope": self._scope(), "state": state}
            accounting = state['grant_accounting'] if self.continuation else {
                "schema":"astra_model_grant_accounting_v1", "grant":{"id":self.config["campaign"],
                    "additional_model_calls":self.config["max_model_calls"],"reason":"Initial explicit campaign grant"},
                "prior_campaigns":[],"prior_model_call_reservations":0,"outer_budgets_reset":False}
            reserved = state["step"] + int(state["done"] == "model_decision_failed")
            report["budget_accounting"] = {**accounting, "this_grant_model_call_reservations":reserved,
                "cumulative_model_call_reservations":accounting["prior_model_call_reservations"]+reserved,
                "remaining_in_this_grant":max(0,self.config["max_model_calls"]-reserved)}
            from ..oversight.controller import Controller
            _, outer_state = Controller(self.binding["oversight_run"])._load()
            report["development_snapshot"] = {name:{"state":outer_state["modules"][name]["status"],
                "attempts":len(outer_state["modules"][name]["attempts"])} for name in ("data","train","validation")}
            target = self.root / "report.json"
            if target.exists():
                if read(target) != report:
                    raise ValueError("Completed agent report changed")
            else:
                write_new(target, report)
            seal = self.root/'report.integrity.json'
            if not seal.exists():
                write_new(seal, {'report_sha256':file_hash(target),'config_sha256':self.config_sha})
            return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    report = UnifiedAgentRunner(args.config).run()
    print(json.dumps({k: report[k] for k in ("status", "execution_mode", "decision_steps", "memory_written", "real_validation")}))
    return 0 if report["status"] == "simulation_policy_accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
