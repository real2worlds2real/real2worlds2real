"""Agent-neutral decision interface.

The controller sends one self-contained prompt per decision and expects one JSON object back.  Any reasoning agent can be
plugged in: a Claude Code session, a Codex/GPT session, an arbitrary command (request JSON on stdin, response JSON on
stdout) or scripted fixtures for tests.  Every transport returns `(decision, audit)`; the audit records the model that
actually answered and its usage so reports can state who decided.  Each decision is a fresh, tool-less session: the agent
decides, the harness executes and measures.

The same interface also launches *worker sessions* (an agent that iterates inside a confined workspace, e.g. the program
synthesis method); see `worker_session`.
"""
from __future__ import annotations

import json
import re
import shlex
import subprocess
import tempfile
from pathlib import Path

DECISION_KEYS = ("hypothesis", "tool", "arguments", "rationale")


def parse_decision(text: str) -> dict:
    """The last top-level JSON object in the text that has a `tool` field."""
    best = None
    for m in re.finditer(r"\{", text):
        depth, i = 0, m.start()
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[i:j + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict) and "tool" in obj:
                        best = obj
                    break
    if best is None:
        raise ValueError("the agent's answer contains no JSON object with a 'tool' field")
    return best


class ClaudeCLI:
    """Claude Code headless (`claude -p`), every tool disabled: the session can only answer."""
    name = "claude_cli"
    DENY = ["Bash", "Read", "Write", "Edit", "Glob", "Grep", "WebSearch", "WebFetch", "Agent", "Task", "NotebookEdit"]

    def __init__(self, model="claude-opus-5-5", binary="claude", timeout=1800):
        self.model, self.binary, self.timeout = model, binary, timeout

    def decide(self, prompt: str):
        with tempfile.TemporaryDirectory(prefix="astra-decision-") as d:
            p = subprocess.run([self.binary, "-p", prompt, "--model", self.model, "--output-format", "json", "--strict-mcp-config",
                                "--disallowedTools", *self.DENY], cwd=d, capture_output=True, text=True, timeout=self.timeout)
        try:
            meta = json.loads(p.stdout)
        except json.JSONDecodeError:
            raise RuntimeError(f"claude returned no JSON (exit {p.returncode}): {p.stderr[-800:]}")
        text = meta.get("result", "") or ""
        audit = {"transport": self.name, "requested_model": self.model, "answered_models": list((meta.get("modelUsage") or {}).keys()),
                 "usage": meta.get("usage"), "cost_usd": meta.get("total_cost_usd"), "session_id": meta.get("session_id"),
                 "is_error": meta.get("is_error"), "exit_code": p.returncode, "prompt_chars": len(prompt), "answer": text[-6000:]}
        if meta.get("is_error") or p.returncode != 0:
            raise RuntimeError(f"claude session error: {text[-800:]}")
        return parse_decision(text), audit


class CodexCLI:
    """Codex CLI (`codex exec`) in a read-only sandbox; the final message carries the decision."""
    name = "codex_cli"

    def __init__(self, model="gpt-5", binary="codex", timeout=1800):
        self.model, self.binary, self.timeout = model, binary, timeout

    def decide(self, prompt: str):
        with tempfile.TemporaryDirectory(prefix="astra-decision-") as d:
            last = Path(d) / "last.txt"
            p = subprocess.run([self.binary, "exec", "--model", self.model, "--sandbox", "read-only", "--skip-git-repo-check",
                                "--output-last-message", str(last), "-"], input=prompt, cwd=d, capture_output=True, text=True,
                               timeout=self.timeout)
            text = last.read_text() if last.exists() else ""
        audit = {"transport": self.name, "requested_model": self.model, "exit_code": p.returncode, "prompt_chars": len(prompt),
                 "answer": text[-6000:], "stderr_tail": p.stderr[-800:]}
        if p.returncode != 0:
            raise RuntimeError(f"codex exited {p.returncode}: {p.stderr[-800:]}")
        return parse_decision(text), audit


class CommandAgent:
    """Any agent behind a command: stdin {"prompt": ...}; stdout {"answer": text, "audit": {...}} or a decision object."""
    name = "command"

    def __init__(self, argv, timeout=1800):
        self.argv, self.timeout = list(argv), timeout

    def decide(self, prompt: str):
        p = subprocess.run(self.argv, input=json.dumps({"prompt": prompt}), capture_output=True, text=True, timeout=self.timeout)
        if p.returncode != 0:
            raise RuntimeError(f"agent command exited {p.returncode}: {p.stderr[-800:]}")
        out = json.loads(p.stdout)
        if "tool" in out:
            return out, {"transport": self.name, "argv": self.argv}
        return parse_decision(out.get("answer", "")), {"transport": self.name, "argv": self.argv, **(out.get("audit") or {})}


class FixtureAgent:
    """Scripted decisions (tests and operator dry runs); never valid for a reported campaign."""
    name = "fixture"

    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.prompts = []

    def decide(self, prompt: str):
        self.prompts.append(prompt)
        if not self.decisions:
            raise RuntimeError("fixture exhausted")
        d = self.decisions.pop(0)
        return (d(prompt) if callable(d) else d), {"transport": self.name}


def make(cfg: dict):
    kind = cfg.get("transport", "claude_cli")
    if kind == "claude_cli":
        return ClaudeCLI(model=cfg.get("model", "claude-opus-5-5"), binary=cfg.get("binary", "claude"))
    if kind == "codex_cli":
        return CodexCLI(model=cfg.get("model", "gpt-5"), binary=cfg.get("binary", "codex"))
    if kind == "command":
        return CommandAgent(cfg["argv"])
    raise ValueError(f"unknown decision transport {kind!r}")


# ---------------------------------------------------------------------- worker sessions
def worker_session(cfg: dict, workspace: Path, prompt: str, tool_command: str, out_json: Path, timeout_s: int) -> dict:
    """Launch one confined agent session inside `workspace`.  The only shell command it may run is `tool_command` (a
    harness-provided wrapper); files only inside the workspace; no web, no sub-agents.  Returns an audit dict."""
    kind = cfg.get("transport", "claude_cli")
    if kind == "claude_cli":
        argv = [cfg.get("binary", "claude"), "-p", prompt, "--model", cfg.get("model", "claude-opus-5-5"),
                "--permission-mode", "acceptEdits", "--allowedTools", f"Bash({tool_command}:*)",
                "--disallowedTools", "WebSearch", "WebFetch", "Agent", "Task", "--strict-mcp-config", "--output-format", "json"]
        stdin = None
    elif kind == "codex_cli":
        argv = [cfg.get("binary", "codex"), "exec", "--model", cfg.get("model", "gpt-5"), "--sandbox", "workspace-write",
                "--skip-git-repo-check", "-"]
        stdin = prompt
    elif kind == "command":                    # any agent: argv runs in the workspace, prompt on stdin, JSON summary on stdout
        argv = list(cfg["worker_argv"]) + ["--tool", tool_command]
        stdin = prompt
    else:
        raise ValueError(f"worker sessions are not available for transport {kind!r}")
    with open(out_json, "w") as fh:
        try:
            p = subprocess.run(argv, input=stdin, cwd=workspace, stdout=fh, stderr=subprocess.PIPE, text=True, timeout=timeout_s)
            rc, err = p.returncode, p.stderr[-2000:]
        except subprocess.TimeoutExpired:
            rc, err = 124, f"session killed after {timeout_s} s"
    audit = {"transport": kind, "model": cfg.get("model"), "exit_code": rc, "stderr_tail": err, "argv0": shlex.quote(argv[0])}
    try:
        meta = json.loads(Path(out_json).read_text())
        audit.update({k: meta.get(k) for k in ("is_error", "num_turns", "session_id", "total_cost_usd")},
                     answered_models=list((meta.get("modelUsage") or {}).keys()), result=(meta.get("result") or "")[-4000:])
    except (json.JSONDecodeError, OSError):
        pass
    return audit
