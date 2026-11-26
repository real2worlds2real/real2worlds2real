"""Immutable task-to-station binding (`astra_station_binding_v1`).

The binding pins what a campaign is allowed to assume: the station code (its judge, safety layer and evaluation runner),
the sealed partition files (by hash only), the task documents, the real demonstrations, the robot-native policy I/O and
the public split facts.  `verify` recomputes every hash; any change means a new binding and a new campaign.  Sealed
start ids never appear in a binding.
"""
from __future__ import annotations

from pathlib import Path

from .common import digest, file_hash, read, tree_hash, validate, write_new
from .site import Site

TASK_SCHEMA = {
    "type": "object",
    "required": ["schema", "task_id", "station_task", "doc", "station_doc", "data", "policy_io", "tags"],
    "properties": {
        "schema": {"const": "astra_station_task_v1"},
        "task_id": {"type": "string", "pattern": "[a-z0-9_]+"},
        "station_task": {"type": "string"},
        "doc": {"type": "string"}, "station_doc": {"type": "string"}, "data": {"type": "string"},
        "text": {"type": "string"},
        "policy_io": {"type": "object", "required": ["observation", "action_dim", "control_hz", "max_steps", "deliverables"]},
        "tags": {"type": "object", "required": ["scene", "objects", "phases", "robot"]},
        "labels": {"type": "object"},
        "real_constraints": {"type": "array", "items": {"type": "string"}},
        "privileged": {"type": "object"},
        "station": {"type": "object"},
        "real": {"type": "object"},
        "twin": {"type": "object"},
    },
}


def _identity(site: Site, task: dict) -> dict:
    root = site.root
    files = {rel: file_hash(root / rel) for rel in site.cfg["pinned_files"]}
    sealed = {rel: file_hash(root / rel) for rel in site.sealed_files(task)}
    data = site.task_path(task, task["data"])
    n_demo = len(list(data.glob("*.npz"))) if data.is_dir() else 0
    out = {
        "station": {"root": str(root), "files": files, "sealed_files_sha256": sealed},
        "task_documents": {k: {"path": task[k], "sha256": file_hash(site.task_path(task, task[k]))} for k in ("doc", "station_doc")},
        "real_demonstrations": {"path": task["data"], "episodes": n_demo,
                                "tree_sha256": tree_hash(data) if data.is_dir() else None},
    }
    km = (task.get("station") or {}).get("kinematics_model")
    if km:                                             # the arm model programs use behind the policy interface
        out["station"]["kinematics_model_sha256"] = tree_hash(site.task_path(task, km))
    tw = task.get("twin")
    if tw and tw.get("delivery"):                      # a real2sim twin by its delivery record: the record and every file it pins
        from .twin_task import delivery_pins, verify_delivery
        record = verify_delivery(tw["delivery"])
        out["twin"] = {"session": tw["session"], "delivery": tw["delivery"], "delivery_sha256": file_hash(tw["delivery"]),
                       "route": record["route"], "pinned": {pin["path"]: pin["sha256"] for pin in delivery_pins(record)}}
        if record["route"] == "scene_pipeline":
            out["twin"]["freeze_sha256"] = record["acceptance"]["sha256"]
    elif tw:                                           # a frozen scene-pipeline session: its freeze manifest
        out["twin"] = {"session": tw["session"], "freeze_sha256": file_hash(Path(tw["session"]) / "qa" / "independent_freeze.json")}
    return out


def _spec(task: dict) -> dict:
    """The task spec as written (without the loader's `_dir`)."""
    return {k: v for k, v in task.items() if k != "_dir"}


def create(site_path, task_id: str, output) -> dict:
    site = Site(site_path)
    task = site.task_spec(task_id)
    validate(_spec(task), TASK_SCHEMA)
    sp = site.cfg["splits"]
    payload = {
        "schema": "astra_station_binding_v1",
        "site": str(site.path), "site_sha256": site.sha256,
        "task_id": task["task_id"], "station_task": task["station_task"], "task_spec_sha256": digest(_spec(task)),
        "task": {"text": task.get("text", ""), "tags": task["tags"], "labels": task.get("labels", {}),
                 "real_constraints": task.get("real_constraints", []), "privileged": task.get("privileged", {})},
        "policy_io": task["policy_io"],
        "splits": {"development": {"ids": list(sp["development"]), "use": "open; every candidate may be evaluated on them"},
                   "unseen_check": {"count": len(sp["unseen_check"]), "use": "run once by the harness on the candidate being frozen"},
                   "sealed": {"count": sp["sealed_count"], "use": "coordinator only, once, after the campaign ended"}},
        "success": "the station reports success, termination reason and steps per episode; the judge is hidden",
        "validation_scope": "simulation_twin", "real_validation": "pending",
        **_identity(site, task),
    }
    out = {**payload, "binding_sha256": digest(payload)}
    write_new(Path(output), out)
    return out


def verify(binding) -> dict:
    b = read(binding) if isinstance(binding, (str, Path)) else binding
    if b.get("schema") != "astra_station_binding_v1":
        raise ValueError("expected an astra_station_binding_v1")
    body = {k: v for k, v in b.items() if k != "binding_sha256"}
    if digest(body) != b.get("binding_sha256"):
        raise ValueError("binding integrity failure")
    site = Site(b["site"])
    if site.sha256 != b["site_sha256"]:
        raise ValueError("site configuration changed since binding; create a new binding and campaign")
    task = site.task_spec(b["task_id"])
    if digest(_spec(task)) != b["task_spec_sha256"]:
        raise ValueError("task spec changed since binding; create a new binding and campaign")
    cur = _identity(site, task)
    for key in ("station", "task_documents", "real_demonstrations", "twin"):
        if cur.get(key) != b.get(key):
            raise ValueError(f"{key} changed since binding (station code, sealed partitions, documents or demonstrations); "
                             "create a new binding and campaign")
    return b


def public(b: dict) -> dict:
    """What a deciding agent may see."""
    return {"task_id": b["task_id"], "task": b["task"], "policy_io": b["policy_io"], "splits": b["splits"],
            "success": b["success"], "real_demonstrations": {"episodes": b["real_demonstrations"]["episodes"]},
            "validation_scope": b["validation_scope"], "binding_sha256": b["binding_sha256"]}
