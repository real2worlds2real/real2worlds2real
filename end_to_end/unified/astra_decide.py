"""Bounded simulation decisions via a logged-in, explicitly selected Codex Astra model."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import tempfile
import time


MODEL = "gpt-6-astra"
PHASES = ("collect", "fit", "select", "fit", "select")
METHODS = ("ridge_bc", "mlp_bc")
DISABLED_FEATURES = ("shell_tool", "unified_exec", "multi_agent", "plugins", "apps",
                     "computer_use", "browser_use", "image_generation", "view_image", "goals")
SECRET = re.compile(r"(?i)-----BEGIN [\w ]*PRIVATE KEY-----|\bBearer\s+[\w.=-]+|\b(?:sk-|gh[pousr]_)[\w-]{12,}")
SENSITIVE_KEY = re.compile(r"(?i)^(?:api[_-]?key|authorization|password|cookie|credential|secret|access_token|refresh_token|private_key)$")
PRIVATE_PATH = re.compile(r"(?:[A-Za-z]:[\\/]|(?:^|\s)/(?:home|Users|var|tmp|mnt|etc|run)/|(?:^|\s)~/)")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def reviewed_facts(facts):
    if not isinstance(facts, dict) or not facts:
        raise ValueError("Facts must be a nonempty structured summary")
    encoded = canonical(facts)
    if len(encoded.encode()) > 24576 or SECRET.search(encoded) or PRIVATE_PATH.search(encoded):
        raise ValueError("Facts must be a small summary without credentials, local paths, or raw artifacts")

    def check(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if SENSITIVE_KEY.fullmatch(key):
                    raise ValueError("Sensitive fact field rejected")
                check(child)
        elif isinstance(value, list):
            if len(value) > 128:
                raise ValueError("Send aggregate statistics instead of raw arrays")
            for child in value:
                check(child)
    check(facts)
    return encoded


def stage_contract(phase, prior):
    fields = {"reason": {"type": "string"}}
    if phase == "collect":
        fields.update(controller={"type": "string", "enum": ["mpc", "servo"]},
                      candidate_source={"type": "string"})
        instruction = ("Choose one registered collector controller to generate simulation demonstrations. "
                       "Also author a compatibility classifier as Python source: a single function "
                       "def classify(record, request) returning accept, expert_only or reject. "
                       "Use the public classifier contract in the facts. No imports, loops or external state. "
                       "Return raw source in candidate_source without Markdown. Never claim a result before execution.")
    elif phase == "fit":
        used = {decision["method"] for decision in prior if "method" in decision}
        fields["method"] = {"type": "string", "enum": [method for method in METHODS if method not in used]}
        instruction = "Choose the registered behavior-cloning learner using the observed collection summary."
    else:
        actions = ["freeze"] if len(prior) == 4 else ["freeze", "try_other"]
        fields["action"] = {"type": "string", "enum": actions}
        instruction = ("Decide whether to freeze this trial or run the alternate learner. The worker selects "
                       "the frozen candidate by the predeclared development selection rule. Do not ask for "
                       "hidden test seeds or outcomes. A freeze is valid even when performance is low.")
    return ({"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}, instruction)


def validate_decision(value, schema):
    fields = schema["properties"]
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError("Decision keys do not match the phase contract")
    for name, spec in fields.items():
        answer = value[name]
        limit = 16000 if name == "plans_json" else 10000 if name in {"candidate_source", "candidate_json", "tool_manifest_json"} else 1200
        if not isinstance(answer, str) or not answer.strip() or len(answer) > limit or SECRET.search(answer):
            raise ValueError("Invalid decision field: " + name)
        if "enum" in spec and answer not in spec["enum"]:
            raise ValueError("Unregistered decision: " + name)
    return value


def invoke(codex, schema, prompt):
    with tempfile.TemporaryDirectory(prefix="astra-decision-") as directory:
        clean = Path(directory)
        schema_file = clean / "decision.schema.json"
        schema_file.write_text(canonical(schema), encoding="utf-8")
        args = [codex, "--no-daemon", "-a", "never", "exec", "--ignore-user-config", "--ignore-rules",
                "--skip-git-repo-check", "--ephemeral", "-m", MODEL, "--sandbox", "read-only",
                "-c", 'web_search="disabled"', "-c", "project_doc_max_bytes=0", "--output-schema",
                str(schema_file), "--json", "--color", "never"]
        for name in DISABLED_FEATURES:
            args.extend(["--disable", name])
        args.append("-")
        allowed = {"path", "pathext", "userprofile", "appdata", "localappdata", "systemroot", "systemdrive",
                   "comspec", "programfiles", "programfiles(x86)", "programdata", "temp", "tmp", "home",
                   "username", "userdomain", "homedrive", "homepath", "lang", "lc_all", "codex_home"}
        env = {k: v for k, v in os.environ.items() if k.lower() in allowed}
        proc = subprocess.run(args, input=prompt.encode("utf-8"), cwd=clean, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600, check=False)
    events = []
    for line in proc.stdout.decode("utf-8", "replace").splitlines():
        if line.strip():
            events.append(json.loads(line))
    metadata = {"requested_model": MODEL, "transport": "codex_exec", "exit_code": proc.returncode,
                "stdout_sha256": digest(proc.stdout), "stderr_sha256": digest(proc.stderr),
                "thread_id": None, "usage": None, "item_types": []}
    output = None
    unexpected = False
    for event in events:
        kind = event.get("type", "")
        if kind == "thread.started":
            metadata["thread_id"] = event.get("thread_id")
        if kind == "turn.completed":
            metadata["usage"] = event.get("usage")
        if kind.startswith("item."):
            item = event.get("item", {})
            item_type = item.get("type", "unknown")
            metadata["item_types"].append(item_type)
            if item_type not in {"reasoning", "agent_message"}:
                unexpected = True
            if kind == "item.completed" and item_type == "agent_message":
                output = item.get("text")
    if proc.returncode or unexpected or not output or metadata["usage"] is None:
        error = RuntimeError("Codex transport failed or used an unexpected tool; exit=" + str(proc.returncode)
                             + ", events=" + str(metadata["item_types"]) + ", stderr_sha256=" + metadata["stderr_sha256"])
        error.audit = metadata
        raise error
    return validate_decision(json.loads(output), schema), metadata


def decide(run: Path, phase: str, facts: dict) -> dict:
    if phase not in set(PHASES):
        raise ValueError("Unknown decision phase")
    facts_json = reviewed_facts(facts)
    facts_hash = digest(facts_json.encode())
    audit_dir = Path(run).resolve() / "astra_decisions"
    audit_dir.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(audit_dir / "calls.sqlite3", timeout=30)
    db.row_factory = sqlite3.Row
    try:
        db.execute("CREATE TABLE IF NOT EXISTS calls (slot INTEGER PRIMARY KEY, phase TEXT NOT NULL, "
                   "facts_hash TEXT NOT NULL, facts TEXT NOT NULL, model TEXT NOT NULL, status TEXT NOT NULL, "
                   "reserved REAL NOT NULL, completed REAL, prompt_sha TEXT NOT NULL, decision TEXT, audit TEXT)")
        db.execute("BEGIN IMMEDIATE")
        rows = db.execute("SELECT * FROM calls ORDER BY slot").fetchall()
        matching = next((r for r in reversed(rows) if r["facts_hash"] == facts_hash and r["phase"] == phase), None)
        repeated_phase_due = (len(rows) in (3, 4) and PHASES[len(rows)] == phase
                              and rows[-1]["status"] == "completed"
                              and json.loads(rows[-1]["decision"]).get("action") != "freeze")
        if matching and not repeated_phase_due:
            if matching["status"] != "completed":
                raise RuntimeError(f"Slot {matching['slot']} is {matching['status']}; reservation retained, no automatic retry")
            return json.loads(matching["decision"])
        if any(r["status"] != "completed" for r in rows):
            raise RuntimeError("An unresolved reservation exists; no automatic retry")
        prior = [json.loads(r["decision"]) for r in rows]
        slot = len(prior)
        if slot >= len(PHASES) or PHASES[slot] != phase or (prior and prior[-1].get("action") == "freeze"):
            raise ValueError("Out-of-order decision or exhausted campaign budget")
        schema, instruction = stage_contract(phase, prior)
        prior_choices = [{k: v for k, v in d.items() if k in {"controller", "method", "action"}} for d in prior]
        prompt = ("You are Astra's registered decision component for a simulation-only robotics campaign. "
                  "Use only the structured facts below. Do not access the filesystem, network, terminal or any tools. "
                  "Return exactly the requested JSON. Explain your choice briefly. " + instruction + "\n"
                  + canonical({"phase": phase, "previous_choices": prior_choices, "facts": facts}))
        codex = shutil.which("codex")
        if not codex:
            raise FileNotFoundError("The authenticated Codex CLI is unavailable")
        db.execute("INSERT INTO calls(slot,phase,facts_hash,facts,model,status,reserved,prompt_sha) "
                   "VALUES(?,?,?,?,?,'reserved',?,?)",
                   (slot, phase, facts_hash, facts_json, MODEL, time.time(), digest(prompt.encode())))
        db.commit()
        try:
            decision, audit = invoke(codex, schema, prompt)
        except Exception as exc:
            audit = getattr(exc, "audit", {})
            audit["error_type"] = type(exc).__name__
            db.execute("UPDATE calls SET status='failed',completed=?,audit=? WHERE slot=?", (time.time(), canonical(audit), slot))
            db.commit()
            raise
        db.execute("UPDATE calls SET status='completed',completed=?,decision=?,audit=? WHERE slot=?",
                   (time.time(), canonical(decision), canonical(audit), slot))
        db.commit()
        return decision
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--phase", choices=sorted(set(PHASES)), required=True)
    parser.add_argument("--facts", type=Path, required=True)
    options = parser.parse_args()
    summary = json.loads(options.facts.read_text(encoding="utf-8-sig"))
    print(canonical(decide(options.run, options.phase, summary)))
