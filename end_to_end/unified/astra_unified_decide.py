"""Isolated real-Astra transport and pinned review hook for the unified runner.

This subprocess retains the original flat-module learning imports. It never
executes a model-selected shell command or substitutes a fixture for Astra.
"""
import importlib
from contextlib import redirect_stdout
import json
from pathlib import Path
import re
import sys

from campaign_ledger import CampaignLedger, schema, strict_json
from astra_e2e.oversight.evidence import file_hash


def decide_request(request):
    """Actual Astra only; also used by an authenticated local decision relay."""
    if request["operation"] != "decide":
        raise ValueError("Expected a decision request")
    def no_worker(_):
        raise RuntimeError("Only the external unified harness executes tools")
    ledger = CampaignLedger(request["ledger_root"], request["campaign"], no_worker,
                            task_id=request["task_id"], version="unified-agent-v1",
                            worker_identity="world-bound-unified-harness",
                            budget={"max_model_calls": request["max_model_calls"]},
                            configuration={"runner_config_sha256": request["config_sha256"]})
    response_schema = schema({"tool_id": {"type": "string", "enum": request["tool_ids"]},
                              "tool_manifest_json": {"type": "string"}})
    def validate(value):
        arguments = strict_json(value["tool_manifest_json"])
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be a JSON object")
    decision = ledger.decide(request["phase"], request["facts"], response_schema,
        "Choose exactly one available tool and its arguments as tool_manifest_json. "
        "The tools are task-independent contracts; choose methods from their capabilities and evidence. "
        "Review means invoking a pinned independent verifier, not granting arbitrary approval. "
        "Diagnose failures before revising data or training. Retain failures and never change the success definition. "
        "Store scoped development experience before finalization. Finalization consumes the sealed evaluation once "
        "and ends model interaction. Stop with empty arguments if useful executable progress is unavailable.", validate)
    calls = ledger.status()["calls"]
    row = next(c for c in reversed(calls) if c["id"] == "model:" + request["phase"])
    return {"decision": decision, "audit": row["audit"], "execution_mode": "production"}


def main():
    request = strict_json(sys.stdin.read())
    if request["operation"] == "decide":
        result = decide_request(request)
    elif request["operation"] == "review":
        descriptor = request["reviewer"]
        module, function = descriptor["factory"].split(":")
        if not re.fullmatch(r"[A-Za-z_]\w*", module) or not re.fullmatch(r"[A-Za-z_]\w*", function):
            raise ValueError("Reviewer must name a checked-in module:function")
        source = Path(__file__).resolve().parent / (module + ".py")
        inputs = descriptor["inputs"]
        if inputs.get(str(source)) != file_hash(source):
            raise ValueError("Reviewer source is not pinned by the runner contract")
        if not all(Path(p).is_file() and file_hash(p) == sha for p, sha in inputs.items()):
            raise ValueError("Reviewer implementation or dependencies changed")
        hook = getattr(importlib.import_module(module), function)
        with redirect_stdout(sys.stderr):
            result = hook(request["binding"], request["result"], output_dir=Path(request["output_dir"]))
    else:
        raise ValueError("Unsupported transport operation")
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
