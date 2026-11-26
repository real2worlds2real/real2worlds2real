"""One delivery record for an accepted twin, whichever route produced it.

real2sim has two routes to a twin an agent has accepted:

* a **task workspace** of the agent harness (``real2sim-harness task init`` ...): an editable, operable object
  (``final.json``) or a composed interactive scene (``final_scene.json``);
* a **scene pipeline** session (``pipeline/twin.sh``, ``python -m calibrated_scene_reconstruction.agent.tool``): a metric, robot-ready scene that
  passed the gates S0-S6 and was frozen (``qa/independent_freeze.json``).

Downstream users (policy development, the end-to-end chain) should not need to know which route ran.  This module
writes the same small JSON record for both and re-checks it later.  It reads files only: no simulator, no model and
none of the heavy reconstruction dependencies.  The record states what the route's own acceptance record states; it is
not a new review and adds no claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from .delivery_files import LEGACY_SCHEMA as SCHEMA, changed_files
ROUTES = ("scene_pipeline", "task_workspace")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pin(path: Path) -> dict:
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": sha256_file(path)}


def _session_references(node):
    if isinstance(node, str):
        if node.startswith("sessions/") and not any(c.isspace() for c in node):   # a path, not a note
            yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _session_references(value)
    elif isinstance(node, list):
        for value in node:
            yield from _session_references(value)


def _check_session_references(config: Path) -> None:
    """The scene graph names session files relative to the repository (``sessions/<name>/...``): the simulator, the
    station factories and the episode recorder resolve them from the repository root, so the session must be found
    there (a link is enough) or a delivered twin cannot be built anywhere else."""
    root = config.parents[1]
    missing = sorted({ref for ref in _session_references(_read(config / "scene_graph.json")) if not (root / ref).exists()})
    if missing:
        raise ValueError(f"the scene graph names {len(missing)} session file(s) under {root}, which do not exist there "
                         f"(first: {missing[0]}); put the session at {root}/sessions/<name> (a symbolic link is enough)")


def _frozen_observations(manifest, session, config):
    """Collect original capture hashes only from documents pinned by this freeze."""
    pins = {(row["base"], row["path"].replace("\\", "/")): row["sha256"]
            for row in manifest.get("files", []) if isinstance(row, dict)}

    def frozen_json(base, relative):
        if (base, relative) not in pins:
            return {}
        root = config if base == "config" else session
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha256_file(path) != pins[base, relative]:
            raise ValueError(f"Frozen observation record changed: {path}")
        return _read(path)

    sources = set(frozen_json("session", "capture/observations.json").get("sha256", []))
    if not sources and ("session", "capture/keyframes.json") in pins:
        # Earlier video captures recorded the original digest in the selected exposure round.
        pointer = frozen_json("session", "capture/exposure/current.json")
        if pointer:
            relative = (Path(pointer["directory"]) / "manifest.json").as_posix()
            if pins.get(("session", relative)) != pointer["manifest_sha256"]:
                raise ValueError("Exposure evidence is missing from the frozen capture")
            sources.add(frozen_json("session", relative)["source_sha256"])
    plan = frozen_json("config", "inputs.json")
    sources.update(item["sha256"] for item in plan.get("assets", []))
    return sorted(sources)


def from_scene_pipeline(session: Path, config: Path | None = None) -> dict:
    """A frozen scene-pipeline session.  ``config`` defaults to ``configs/<session name>`` next to ``sessions/``."""
    session = Path(session).resolve(strict=True)
    freeze = session / "qa" / "independent_freeze.json"
    if not freeze.is_file():
        raise ValueError(f"{session} is not frozen: run the delivery stage (S6) and the freeze first")
    config = Path(config).resolve(strict=True) if config else session.parents[1] / "configs" / session.name
    if not (config / "task.json").is_file():
        raise ValueError(f"{config} has no task.json; pass the session's configuration directory")
    _check_session_references(config)
    contract, graph = _read(config / "task.json"), _read(config / "scene_graph.json").get("task") or {}
    manifest = _read(freeze)
    exports = sorted((session / "export" / "usd").glob("*.usdz")) if (session / "export" / "usd").is_dir() else []
    if len(exports) > 1:
        raise ValueError(f"{session / 'export' / 'usd'} holds several USDZ files; deliver one background per session")
    from calibrated_scene_reconstruction.scene.twin.process_result import state as process_state
    process = process_state(session, contract)
    physics_assets = {}
    if process["applicable"]:
        if not process["passed"]:
            raise ValueError("Physical calibration evidence is stale or missing: " + process["reason"])
        proof = session / "qa/physics_process.json"
        frozen = {row["path"]: row["sha256"] for row in manifest["files"] if row["base"] == "session"}
        if frozen.get("qa/physics_process.json") != sha256_file(proof):
            raise ValueError("Physical calibration is outside this scene freeze; deliver a reviewed derivative")
        physics_assets["physics_receipt"] = _pin(proof)
        for index, (path, expected) in enumerate(sorted(_read(proof)["files"].items())):
            physics_assets[f"physics_{index}"] = {"path": path, "sha256": expected}
    return {
        "schema": SCHEMA, "name": session.name, "route": "scene_pipeline", "level": "scene",
        "observation_sha256": _frozen_observations(manifest, session, config),
        "task": {"text": contract.get("task"), "kind": graph.get("kind"), "target": graph.get("target"),
                 "goal": graph.get("goal"), "support": graph.get("support"),
                 "objects": {name: row.get("role") for name, row in sorted((contract.get("objects") or {}).items())}},
        "scene": {**_pin(exports[0]), "format": "usdz"} if exports else None,
        "source": {"session": str(session), "config": str(config)},
        "acceptance": {"kind": "freeze_manifest", **_pin(freeze), "frozen_at": manifest.get("frozen_at"),
                       "files": len(manifest.get("files") or []), "basis": manifest.get("basis")},
        **({"assets": physics_assets} if physics_assets else {}),
        "statements": {"metric_scale": True, "robot_placed": bool(_read(config / "scene_graph.json").get("photo_robot")),
                       "real_physics_calibrated": bool(physics_assets)},
        "recheck": ["python -m calibrated_scene_reconstruction.scene.twin.freeze <session> --verify", "python -m calibrated_scene_reconstruction.scene.twin.gates <session> status"],
    }


def from_task_workspace(run: Path) -> dict:
    """An accepted task workspace of the agent harness: a scene if one was accepted, otherwise the object."""
    run = Path(run).resolve(strict=True)
    record = _read(run / "run.json")
    if record.get("status") != "accepted":
        raise ValueError(f"{run} is not accepted (status {record.get('status')!r})")
    base = {"schema": SCHEMA, "name": run.name, "route": "task_workspace", "source": {"run": str(run)},
            "recheck": ["real2sim-harness task call <run> audit_task_evidence --request <request.json>"]}
    if (run / "final_scene.json").is_file():
        final = _read(run / "final_scene.json")
        if final.get("status") != "accepted":
            raise ValueError("final_scene.json does not record an accepted scene")
        for path, expected in final.get("evidence_files", {}).items():
            if not Path(path).is_file() or sha256_file(path) != expected:
                raise ValueError(f"Scene acceptance evidence changed: {path}")
        scene = run / final["scene"]
        bundle = scene.parent / "scene_bundle.json"
        if sha256_file(bundle) != final["scene_bundle_sha256"]:
            raise ValueError("The scene bundle changed after acceptance; reopen and accept the scene again")
        composed = _read(bundle)
        frame = composed.get("frame") or {}
        metric_scale = frame.get("scale_verified")
        if metric_scale is not None and not isinstance(metric_scale, bool):
            raise ValueError("scene bundle frame.scale_verified must be a boolean when supplied")
        # every file of the composed package, so a station can tell that nothing under the scene changed (the scene
        # file alone only references them)
        package = {}
        for name, digest in sorted((composed.get("files") or {}).items()):
            package[name] = _pin(scene.parent / name)
            if package[name]["sha256"] != digest:
                raise ValueError(f"{name} changed after composition; reassemble and accept the scene again")
        # the composition report lists every participant and background it checked; roles live in the scene manifest
        objects = {row["id"]: None for row in composed.get("checks", []) if isinstance(row, dict) and row.get("id")}
        inputs = run / "input_manifest.json"
        sources = _read(inputs)["inputs"] if inputs.is_file() else record.get("references", [])
        return {**base, "level": "scene", "observation_sha256": sorted({row["sha256"] for row in sources}),
                "task": {"text": record.get("objective"), "kind": None, "target": None, "goal": None, "support": None,
                         "objects": objects},
                "scene": {**_pin(scene), "format": "usda"}, "assets": package,
                "acceptance": {"kind": "accepted_scene", **_pin(run / "final_scene.json"), "reason": final.get("reason"),
                               "scene_id": final.get("scene_id"), "runtime": final.get("runtime"),
                               "backend": composed.get("backend")},
                "statements": {"metric_scale": metric_scale, "robot_placed": None,
                               "real_physics_calibrated": bool(final.get("real_physics_calibrated"))}}
    final = _read(run / "final.json")
    if final.get("status") != "accepted":
        raise ValueError("final.json does not record an accepted object")
    artifacts = {name: _pin(run / rel) for name, rel in sorted(final.get("artifacts", {}).items())
                 if name.endswith((".glb", ".usd", ".usda", ".usdz", ".blend")) and (run / rel).is_file()}
    return {**base, "level": "object",
            "task": {"text": record.get("objective"), "kind": None, "target": None, "goal": None, "support": None,
                     "objects": {}},
            "scene": None, "assets": artifacts,
            "acceptance": {"kind": "accepted_object", **_pin(run / "final.json"), "reason": final.get("reason"),
                           "attempt": final.get("attempt_id")},
            "statements": {"metric_scale": None, "robot_placed": False, "real_physics_calibrated": False}}


def build(*, session=None, config=None, run=None) -> dict:
    if (session is None) == (run is None):
        raise ValueError("Give exactly one of a scene-pipeline session or a task-workspace run")
    record = from_scene_pipeline(session, config) if session is not None else from_task_workspace(run)
    record["created"] = datetime.now(UTC).isoformat()
    return record


def verify(record: dict) -> dict:
    """Re-hash every file the record pins.  ``ok`` is false when one is missing or changed."""
    changed = changed_files(record)
    return {"ok": not changed, "changed": changed, "name": record["name"], "route": record["route"]}


def write(record: dict, output: Path) -> Path:
    output = Path(output)
    if output.exists():
        raise ValueError(f"{output} exists; a delivery record is never overwritten")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return output


def command(args: argparse.Namespace) -> int:
    if args.verify:
        report = verify(_read(args.verify))
        print(json.dumps(report, indent=1))
        return 0 if report["ok"] else 1
    record = build(session=args.session, config=args.config, run=args.run)
    if args.output:
        write(record, args.output)
    print(json.dumps(record, indent=1, ensure_ascii=False))
    return 0


def add_parser(commands) -> None:
    deliver = commands.add_parser(
        "deliver", help="write or re-check the delivery record of an accepted twin (either route); reads files only")
    deliver.add_argument("--session", type=Path, help="a frozen scene-pipeline session (sessions/<name>)")
    deliver.add_argument("--config", type=Path, help="its configuration directory (default configs/<name>)")
    deliver.add_argument("--run", type=Path, help="an accepted task workspace of the agent harness")
    deliver.add_argument("--output", type=Path, help="new file to write the record to")
    deliver.add_argument("--verify", type=Path, help="re-hash the files an existing record pins")
    deliver.set_defaults(func=command)
