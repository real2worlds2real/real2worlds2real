"""Registered method tools (`astra_station_tool_v1`).

A tool is one executable method behind a fixed contract: JSON arguments (validated against the manifest's schema) and an
adapter command that receives `--binding --request --result` and writes an `astra_station_receipt_v1`.  Adapter code,
the shared code the site names in `tool_code` (sandbox runtime, station client) and manifests are pinned by hash when a
campaign starts; editing them mid-campaign stops the campaign.  New methods join by
adding a manifest + adapter; nothing in the controller names a task or a method.  A manifest may list the policy
interfaces it supports (`interfaces`, matched against the task spec's `policy_io.interface`); a tool without the list
applies to every task.  A manifest may also list files it needs at the site (`requires_files`, with `{repo}` and
`{station_root}`) and the site's compute entries it uses (`requires_compute`, e.g. a training host); when one is missing
the tool is loaded as unavailable, so no decision can choose it.
"""
from __future__ import annotations

from pathlib import Path

from .common import digest, file_hash, read, validate

TOOL_SCHEMA = {
    "type": "object",
    "required": ["schema", "id", "family", "description", "arguments", "adapter", "status"],
    "properties": {
        "schema": {"const": "astra_station_tool_v1"},
        "id": {"type": "string", "pattern": "[a-z][a-z0-9_]*"},
        "family": {"enum": ["evaluation", "data", "imitation", "program", "rl", "composition"]},
        "description": {"type": "string", "minLength": 20},
        "arguments": {"type": "object"},
        "costs": {"type": "object"},
        "cap_key": {"type": "string"},
        "adapter": {"type": "object", "required": ["argv", "timeout_seconds"],
                    "properties": {"argv": {"type": "array", "items": {"type": "string"}, "minItems": 2},
                                   "timeout_seconds": {"type": "integer", "minimum": 60}}},
        "requires_compute": {"type": "array", "items": {"type": "string"}},
        "interfaces": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "requires_files": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "status": {"enum": ["ready", "unavailable"]},
        "status_note": {"type": "string"},
    },
}

RECEIPT_SCHEMA = {
    "type": "object",
    "required": ["schema", "status", "tool", "request_sha256", "facts", "created", "usage"],
    "properties": {
        "schema": {"const": "astra_station_receipt_v1"},
        "status": {"enum": ["completed", "failed"]},
        "tool": {"type": "string"}, "request_sha256": {"type": "string"},
        "facts": {"type": "object"},
        "created": {"type": "object", "properties": {
            "candidates": {"type": "array", "items": {"type": "object", "required": ["name", "kind", "path", "note"]}},
            "datasets": {"type": "array", "items": {"type": "object", "required": ["name", "path", "episodes", "note"]}},
            "evaluations": {"type": "array", "items": {"type": "object", "required": ["candidate", "starts", "n", "success"]}}}},
        "usage": {"type": "object", "additionalProperties": {"type": "number"}},
        "error": {"type": "string"},
    },
}


def load(tools_dir, repo: Path, site_cfg: dict) -> dict:
    """id -> {manifest, sha256, inputs (pinned adapter files)}."""
    out = {}
    manifests = sorted(Path(tools_dir).glob("*.json"))
    if not manifests:  # a campaign would otherwise offer its agent no method at all and end with nothing to deliver
        raise ValueError(f"no tool manifests in {tools_dir} (the site's tools_dir, relative to the repository root)")
    for p in manifests:
        m = read(p)
        validate(m, TOOL_SCHEMA)
        if m["id"] in out:
            raise ValueError(f"duplicate tool id {m['id']}")
        # files the adapter needs at this site ({repo}, {station_root}); a tool whose prerequisites are absent is not offered
        missing = [f for f in m.get("requires_files", []) if not Path(expand(f, repo, site_cfg)).is_file()]
        missing += [f"compute entry {c!r}" for c in m.get("requires_compute", []) if c not in (site_cfg.get("compute") or {})]
        if missing and m["status"] == "ready":
            m = {**m, "status": "unavailable", "status_note": "missing at this site: " + ", ".join(missing)}
        argv = [expand(a, repo, site_cfg) for a in m["adapter"]["argv"]]
        scripts = [Path(a) for a in argv if a.endswith(".py") and Path(a).is_file()]
        helpers = [h for sc in scripts for h in sorted(sc.parent.glob("*.py"))] + [h for sc in scripts for h in sorted((sc.parent / "prompts").glob("*.md"))]
        shared = [f for d in site_cfg.get("tool_code", []) for f in sorted((repo / d).rglob("*"))
                  if f.is_file() and "__pycache__" not in f.parts]           # code the adapters copy into sandboxes and candidates
        inputs = {str(f): file_hash(f) for f in [*scripts, *helpers, *shared]}      # the adapter and every module it may use
        out[m["id"]] = {"manifest": m, "path": str(p), "sha256": file_hash(p), "argv": argv, "inputs": inputs}
    return out


def expand(arg: str, repo: Path, site_cfg: dict) -> str:
    return (arg.replace("{repo}", str(repo)).replace("{worker_python}", site_cfg["worker_python"])
               .replace("{station_root}", str(Path(site_cfg["station_root"]).expanduser())))


def pins(tools: dict) -> dict:
    return {k: {"manifest": v["sha256"], "inputs": v["inputs"]} for k, v in tools.items()}


def check_pins(tools: dict, pinned: dict) -> None:
    if pins(tools) != pinned:
        raise ValueError("a tool manifest or adapter changed during the campaign; finish it or start a new campaign")


def menu(tools: dict) -> list:
    """Model-facing view of the executable tools."""
    return [{"tool": k, "family": v["manifest"]["family"], "description": v["manifest"]["description"],
             "arguments": v["manifest"]["arguments"], "costs": v["manifest"].get("costs", {})}
            for k, v in tools.items() if v["manifest"]["status"] == "ready"]


def request_sha(request: dict) -> str:
    return digest(request)


def applies(manifest: dict, interface: str | None) -> bool:
    """A tool applies to a task when it lists no interfaces or lists the task's."""
    allowed = manifest.get("interfaces")
    return not allowed or interface in allowed
