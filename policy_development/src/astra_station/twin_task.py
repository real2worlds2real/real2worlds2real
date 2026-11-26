"""A station task spec for a real2sim twin: the bridge from the reconstruction pipeline to policy development.

A frozen twin session (sessions/<name>, accepted gates, qa/independent_freeze.json) plus its task contract
(configs/<name>/task.json: the task in words and the objects' roles; scene_graph.json `task`: kind, target, goal,
support) become a task spec of the task registry (astra_station_task_v1) served by the twin station backend
(policy_development/station/server/twin.py).  The binding then pins the twin by its freeze manifest, so a campaign can
only run on the exact frozen reconstruction.

    python -m astra_station new-twin-task --session sessions/<name> --config configs/<name> --task-id <id> \
        --factory <file.py>:<function> --policy-io JSON --out <task registry>/<id>
    python -m astra_station new-twin-task --delivery <name>.delivery.json --task-id <id> \
        --factory <file.py>:<function> --policy-io JSON --out <task registry>/<id> [--setup <setup.json>]

The second form starts from a native v1 delivery or a v2 delivery with checked model revisions from either route,
a frozen scene-pipeline session or an accepted scene of a task workspace. The binding then pins
the record and the files it names.  This module reads those JSON files only; it imports nothing from real2sim.

What a twin task still needs before a campaign: its environment factory (worked examples for both routes:
real2sim/examples/station/), sealed partitions (station/server/partitions.py) and, for real-robot work, a real chain in
the task directory.
"""
from __future__ import annotations

import json
from pathlib import Path

from .binding import TASK_SCHEMA
from .common import read, validate
from .delivery_files import LEGACY_SCHEMA as DELIVERY_SCHEMA, changed_files, delivery_pins

PHASES = {      # scene_graph task kind -> the station phases (experience and reward retrieval tags)
    "insert": ["reach", "grasp", "align", "insert"],
    "place": ["reach", "grasp", "lift", "carry", "place", "release"],
    "stack": ["reach", "grasp", "lift", "carry", "place", "release"],
    "lift": ["reach", "grasp", "lift", "hold"],
    "push": ["reach", "push"],
    "open": ["reach", "grasp", "pull"],
}


def spec_from_twin(session: Path, config: Path, task_id: str, factory: str, policy_io: dict, robot="vega_right_sharpa",
                   num_envs: int = 4, max_steps: int = 600) -> dict:
    session, config = Path(session).resolve(), Path(config).resolve()
    if not (session / "qa" / "independent_freeze.json").is_file():
        raise ValueError(f"{session} is not a frozen twin (run python -m calibrated_scene_reconstruction.scene.twin.freeze {session} first)")
    contract = read(config / "task.json")
    graph = read(config / "scene_graph.json").get("task") or {}
    kind = graph.get("kind", "")
    objects = sorted(contract.get("objects", {}))
    spec = {
        "schema": "astra_station_task_v1", "task_id": task_id, "station_task": task_id,
        "text": contract["task"],
        "doc": "task.md", "station_doc": "station_api.md", "data": f"station:task_data/{task_id}",
        "policy_io": {"control_hz": 20, "max_steps": max_steps, "deliverables": ["policy", "program"], **policy_io},
        "tags": {"scene": config.name, "objects": objects, "phases": PHASES.get(kind, [kind] if kind else []), "robot": robot},
        "labels": {},
        "station": {"partition_kind": task_id, "backend": "<astra-robotlearning>/policy_development/station/server/twin.py:TwinStation",
                    "runtime": "python", "num_envs": num_envs, "need_mb": 12000,
                    "args": ["--twin-session", str(session), "--twin-factory", factory, "--twin-task-spec", "{spec}",
                             "--twin-max-steps", str(max_steps)]},
        "twin": {"session": str(session), "config": str(config), "kind": kind, "target": graph.get("target"),
                 "goal": graph.get("goal"), "support": graph.get("support")},
        "real_constraints": [],
    }
    validate(spec, TASK_SCHEMA)
    return spec


def verify_delivery(path: Path) -> dict:
    """The record, after re-hashing the files it pins (ValueError when one is missing or changed)."""
    record = read(path)
    changed = changed_files(record)
    if changed:
        raise ValueError(f"the delivered twin changed after acceptance: {changed}")
    return record


def spec_from_delivery(delivery: Path, task_id: str, factory: str, policy_io: dict, robot="vega_right_sharpa",
                       num_envs: int = 4, max_steps: int = 600, setup: dict | None = None) -> dict:
    """A station task spec for the twin a delivery record names.

    A scene-pipeline record gives the same spec as `spec_from_twin` (its session stays the station's twin).  A
    task-workspace record must deliver a scene; the station then starts from the record itself.  Either way the twin
    must come with a robot interaction setup: the environment factory places and drives the robot in it.

    `setup` is the station author's own declaration for the factory (the robot, its base, which participant is the
    target, the task kind and its thresholds).  A task-workspace record states none of these, so the declaration is
    kept in the spec as `twin.setup` with its source named; it is no statement of the twin.
    """
    delivery = Path(delivery).resolve()
    record = verify_delivery(delivery)
    if setup is not None and not isinstance(setup, dict):
        raise ValueError("the setup is a JSON object")
    if record.get("level") != "scene" or not record.get("scene"):
        raise ValueError("a station needs a delivered scene; this record delivers a single object: compose it into a "
                         "scene with a robot interaction setup first")
    if record["route"] == "scene_pipeline":
        spec = spec_from_twin(Path(record["source"]["session"]), Path(record["source"]["config"]), task_id, factory,
                              policy_io, robot, num_envs, max_steps)
    else:
        task = record.get("task") or {}
        if not (task.get("text") or (setup or {}).get("text")):
            raise ValueError("the delivery record states no task; record the task in words before a campaign")
        declared = setup or {}
        kind = task.get("kind") or declared.get("kind") or ""
        spec = {
            "schema": "astra_station_task_v1", "task_id": task_id, "station_task": task_id,
            "text": declared.get("text") or task["text"],        # the run's objective describes building the scene
            "doc": "task.md", "station_doc": "station_api.md", "data": f"station:task_data/{task_id}",
            "policy_io": {"control_hz": 20, "max_steps": max_steps, "deliverables": ["policy", "program"], **policy_io},
            "tags": {"scene": record["name"], "objects": sorted(task.get("objects") or {}),
                     "phases": PHASES.get(kind, [kind] if kind else []), "robot": robot},
            "labels": {},
            "station": {"partition_kind": task_id,
                        "backend": "<astra-robotlearning>/policy_development/station/server/twin.py:TwinStation",
                        "runtime": "python", "num_envs": num_envs, "need_mb": 12000,
                        "args": ["--twin-factory", factory, "--twin-task-spec", "{spec}", "--twin-max-steps", str(max_steps)]},
            "twin": {"session": record["source"]["run"], "kind": kind,
                     **{key: task.get(key) or declared.get(key) for key in ("target", "goal", "support")}},
            "real_constraints": [],
        }
    spec["twin"].update(delivery=str(delivery), route=record["route"], scene=record["scene"]["path"])
    if setup is not None:
        spec["twin"].update(setup=setup, setup_source="station author; not a statement of the delivered twin")
    spec["station"]["args"] += ["--twin-delivery", str(delivery)]
    validate(spec, TASK_SCHEMA)
    return spec


def write_task(spec: dict, out: Path, backend_path: Path) -> Path:
    out = Path(out)
    if (out / "task.json").exists():
        raise ValueError(f"{out / 'task.json'} exists")
    out.mkdir(parents=True, exist_ok=True)
    spec = json.loads(json.dumps(spec).replace("<astra-robotlearning>/policy_development/station/server/twin.py", str(backend_path)))
    (out / "task.json").write_text(json.dumps(spec, indent=1, ensure_ascii=False) + "\n")
    t = spec["twin"]
    roles = [f"{k} {t[k]}" for k in ("target", "goal", "support") if t.get(k)]
    stated = f"task kind {t['kind']}: " if t.get("kind") else "no task kind stated; "   # a twin may leave its kind open
    (out / "task.md").write_text(f"# Task: {spec['text']}\n\nScene: the accepted real2sim twin `{Path(t['session']).name}` "
                                 f"(objects: {', '.join(spec['tags']['objects'])}; {stated}{', '.join(roles)}).\n")
    io = spec["policy_io"]
    (out / "station_api.md").write_text(
        "# Station (real2sim twin)\n\nObservation: " + "; ".join(io.get("observation", [])) + f"\nAction: {io.get('action', '')} "
        f"({io['action_dim']}-D) at {io['control_hz']} Hz, at most {io['max_steps']} steps.\nDeliverables: a station policy module "
        "(`class Policy: __init__(info) / reset(env) / act(obs)`) or a program (station program contract).\n")
    return out / "task.json"
