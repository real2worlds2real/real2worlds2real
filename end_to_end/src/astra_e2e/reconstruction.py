"""Agent-selected native operations inside one reconstruction stage."""
from __future__ import annotations

import fcntl
import uuid
from contextlib import contextmanager, redirect_stdout
import sys
from pathlib import Path

from astra_station.common import digest, file_hash, read, validate, write, write_new

from . import model_modules

ID = {"type": "string", "pattern": "[a-z][a-z0-9_]{0,63}"}
TEXT = {"type": "string", "minLength": 1}
STEP = {"type": "object", "additionalProperties": False, "required": ["id", "operation", "source", "why"],
        "properties": {"id": ID, "operation": TEXT, "source": TEXT, "why": TEXT}}
PLAN = {"type": "object", "additionalProperties": False, "required": ["steps", "why"],
        "properties": {"why": TEXT, "restart_from": ID, "steps": {"type": "array", "minItems": 1, "items": STEP}}}

BRIDGES = {
    "workspace.init": "Create a native task from the chain observations under source (the runs directory).",
    "exchange.asset": "Recheck an accepted task object and package its native runtime representation for a scene.",
    "exchange.geometry": "Transfer the exact fitted surface, source files and model-to-world transform as a prior.",
    "exchange.views": "Copy registered images, masks and calibrated poses into the destination task workspace.",
    "exchange.placements": "Write a scene manifest from calibrated object fits and registered runtime bundles.",
    "workspace.cousins": "Start a native cousin family from an accepted task.",
    "family.complete": "Verify every variant in the native cousin family and retain the family record.",
    "family.variant": "Open or resume one independently reviewed native cousin task.",
    "delivery.session": "Verify the native freeze and build its delivery record.",
    "delivery.workspace": "Recheck the accepted object or scene and build its delivery record.",
}


def _args(properties, required=()):
    return {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}


BOOL = {"type": "boolean"}
OBJECT = {"type": "object"}
MATRIX = {"type": "array", "minItems": 4, "maxItems": 4, "items": {
    "type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "number"}}}
FRAME = _args({"name": TEXT, "mask": TEXT, "ignore": TEXT, "depth": TEXT}, ["name"])
PART = _args({"id": TEXT, "fit": TEXT, "bundle": TEXT, "bundle_to_model": MATRIX},
             ["id", "fit", "bundle", "bundle_to_model"])
BRIDGE_INPUTS = {
    "workspace.init": _args({"blender": TEXT, "blender_timeout": {"type": "integer", "minimum": 1},
                             "video_frames": {"type": "integer"}, "asset_kind": TEXT, "front_axis": TEXT,
                             "required_part": {"type": "array", "items": TEXT}, "scene": BOOL,
                             "scene_runner": TEXT, "gaussian_runner": TEXT}, ["blender"]),
    "exchange.geometry": _args({"task": TEXT, "round_path": TEXT}, ["task", "round_path"]),
    "exchange.asset": _args({"asset_id": ID}, ["asset_id"]),
    "exchange.views": _args({"task": TEXT, "frames": {"type": "array", "items": FRAME, "minItems": 1, "maxItems": 100},
                             "verified": BOOL, "bundle_to_world": MATRIX}, ["task", "frames"]),
    "exchange.placements": _args({"task": TEXT, "scene": OBJECT, "participants": {
        "type": "array", "items": PART, "minItems": 1}}, ["task", "scene", "participants"]),
    "workspace.cousins": _args({"prompt": TEXT, "count": {"type": "integer", "minimum": 1}}, ["prompt"]),
    "family.variant": _args({"execute_plan": BOOL, "variant_index": {"type": "integer", "minimum": 1}}),
    "family.complete": _args({}),
    "delivery.session": _args({}), "delivery.workspace": _args({}),
}


def catalog(selected=None):
    """Resolve only the implementations an agent selected when validating its plan."""
    def want(prefix):
        return selected is None or any(name.startswith(prefix) and name not in BRIDGES for name in selected)
    rows = {}
    if want("session."):
        from calibrated_scene_reconstruction.agent.tool import schemas
        for name, contract in schemas().items():
            schema = contract["input"]
            schema["properties"].pop("session", None)
            schema["required"] = [key for key in schema["required"] if key != "session"]
            rows["session." + name] = schema
    if want("workspace."):
        from real2sim_harness.task_tools import available_tools
        for mode in ("twin", "scene", "cousin", "scene_cousin", "cousin_planning", "scene_cousin_planning"):
            rows.update({"workspace." + v["name"]: v["parameters"] for v in available_tools(mode)})
    for name, description in BRIDGES.items():
        rows[name] = {**BRIDGE_INPUTS[name], "description": description}
    rows.update(model_modules.schemas())
    return rows


def _workspace(path):
    task = Path(path).resolve(strict=True)
    work = Path(read(task / "external/context.json")["workspace"]).resolve(strict=True)
    if not work.is_relative_to(task) or work == task:
        raise ValueError("Invalid native task workspace")
    return work


def _observed(chain, source, workspace):
    """Bind the final native task to its intake (supplementary views may be derived)."""
    if workspace:
        hashes = {r["sha256"] for r in read(source / "input_manifest.json")["inputs"]}
    else:
        paths = []
        keyframes = source / "capture/keyframes.json"
        if keyframes.is_file():
            paths.append(read(keyframes)["video"])
        frames = source / "splat/frames.json"
        if frames.is_file():
            paths.extend(read(frames).get("videos", []))
        hashes = {file_hash(source.parents[1] / p) for p in set(paths)}
    if not hashes.intersection(chain.done("observations")["files"].values()):
        raise ValueError("The native source does not match any chain observation")
    return hashes


def _native_delivery(chain, source, workspace):
    from real2sim_harness import delivery

    _observed(chain, source, workspace)
    if workspace:
        from real2sim_harness.artifacts import RunStore
        from real2sim_harness.scene_review import _current_runtime, scene_folder
        record = delivery.build(run=source)
        store = RunStore(source)
        if record["level"] == "scene":
            from real2sim_harness.scene_cousins import validate_variant
            scene_id = record["acceptance"]["scene_id"]
            review = _current_runtime(scene_folder(store, scene_id))
            config = read(source / "run.json").get("scene_runner")
            if config and Path(config).resolve() != Path(review["runner_config"]).resolve():
                raise ValueError("Scene runtime configuration changed")
            validate_variant(store, scene_id)
        else:
            checked = store.accept(record["acceptance"]["attempt"], record["acceptance"]["reason"], commit=False)
            if checked["delivery_status"] != "simulation_verified":
                raise ValueError("The object needs source-bound native simulation review before delivery")
            original = read(source / "final.json")
            if any(checked[key] != original.get(key) for key in ("report", "artifacts", "delivery_status")):
                raise ValueError("Object evidence changed since native acceptance")
    else:
        from calibrated_scene_reconstruction.scene.twin.freeze import verify
        from calibrated_scene_reconstruction.scene.session import Session
        session = Session(source)
        # The delivery adapter also supports relocatable sessions and configs.
        session.config_dir = source.parents[1] / "configs" / source.name
        checked = verify(session)
        freeze = read(source / "qa/independent_freeze.json")
        if not checked["unchanged"] or not checked["files"] or not freeze.get("status", {}).get("passed"):
            raise ValueError("Missing or stale native freeze evidence")
        record = delivery.build(session=source)
        if record["scene"]:
            export = Path(record["scene"]["path"]).relative_to(source).as_posix()
            frozen = {row["path"]: row["sha256"] for row in freeze["files"] if row["base"] == "session"}
            if frozen.get(export) != record["scene"]["sha256"]:
                raise ValueError("Scene export is outside the verified native freeze")
    chain.check_delivery(record)
    return record


def _complete_family(source, *, verify_only=False):
    from real2sim_harness.external_agent import next_variant

    family = read(source / "family.json")
    rows = family["variants"]
    if family.get("family_kind") != "scene":
        parent = family["parent"]
        root = Path(parent["run"])
        final = read(root / "final.json")
        model = root / "attempts" / final["attempt_id"] / "scene.blend"
        if final["attempt_id"] != parent["attempt_id"] or file_hash(model) != parent["source_scene_sha256"]:
            raise ValueError("Cousin family was built from a different parent revision")
    if not rows or len(rows) != family.get("count") or not all(row.get("run") for row in rows):
        raise ValueError("Finish each requested cousin before family review")
    scene = family.get("family_kind") == "scene"
    name = "final_scene.json" if scene else "final.json"
    pins = {}
    for row in rows:
        path = source / row["run"] / name
        if not path.is_file() or read(path).get("status") != "accepted":
            raise ValueError("A cousin still needs native acceptance")
        pins[str(path)] = file_hash(path)
    checked = next_variant(source, verify_only=verify_only)
    if checked["status"] != "complete":
        raise ValueError("Native cousin family review is incomplete")
    pins[str(source / "family.json")] = file_hash(source / "family.json")
    return checked, pins


def _execute(chain, step, arguments, folder, *, heads=None):
    from real2sim_harness.artifacts import RunStore

    op, source = step["operation"], Path(step["source"])
    if op.startswith("delivery."):
        from real2sim_harness.delivery import write as write_delivery
        if arguments:
            raise ValueError("Delivery reads its native source without overrides")
        record = _native_delivery(chain, source, op == "delivery.workspace")
        record = model_modules.bind_delivery(heads or {}, record, source.resolve())
        path = write_delivery(record, folder / "delivery.json")
        return {"ok": True, "delivery": str(path)}, {str(path): file_hash(path)}
    if op == "workspace.init":
        from real2sim_harness.cli import parser
        from real2sim_harness.external_agent import create_task
        argv = ["task", "init", "--runs", str(source), "--objective", chain.request["task"]]
        for obs in chain.done("observations")["files"]:
            argv.extend(["--reference", obs])
        for key, value in arguments.items():
            if key == "scene":
                if value:
                    argv.append("--scene")
                continue
            for item in value if isinstance(value, list) else [value]:
                argv.append(f"--{key.replace('_', '-')}={item}")
        try:
            args = parser().parse_args(argv)
        except SystemExit as exc:
            raise ValueError("Invalid workspace.init options; include the Blender executable") from exc
        result = create_task(args)
        manifest = Path(result["input_manifest"])
        return result, {str(manifest): file_hash(manifest)}
    if op == "exchange.asset":
        from real2sim_harness.blender import BlenderExecutor
        from real2sim_harness.external_agent import task_lock
        from real2sim_harness.scene_agent import package_object_result
        from real2sim_harness.sim_bundle import verify_bundle
        store = RunStore(source)
        runtime = read(source / "external/runtime.json")
        with task_lock(store):
            if read(source / "run.json").get("status") != "accepted":
                raise ValueError("Scene participants require current native acceptance")
            executor = BlenderExecutor(Path(runtime["blender"]), timeout_seconds=runtime["blender_timeout"])
            result = package_object_result(executor, store, arguments["asset_id"])
            if not result["ok"]:
                return result, {}
            bundle = Path(result["bundle"])
            check = verify_bundle(bundle)
            if not check["ok"]:
                raise ValueError(f"Invalid accepted runtime bundle: {check['failures']}")
            pins = {str(bundle / p): h for p, h in read(bundle / "asset_bundle.json")["files"].items()}
            for p in (bundle / "asset_bundle.json", source / "final.json"):
                pins[str(p)] = file_hash(p)
            return result, pins
    if op == "workspace.cousins":
        from real2sim_harness.external_agent import plan_cousins
        return plan_cousins(RunStore(source), **arguments), {}
    if op == "family.complete":
        return _complete_family(source)
    if op == "family.variant":
        from real2sim_harness.external_agent import next_variant
        return next_variant(source, **arguments), {}
    if op.startswith("session."):
        from calibrated_scene_reconstruction.agent.tool import execute
        if "session" in arguments:
            raise ValueError("The plan already binds this session")
        result = execute(op.split(".", 1)[1], {**arguments, "session": str(source)}, folder)
        return result, {str(folder / "result.json"): file_hash(folder / "result.json")}
    if op.startswith("workspace."):
        from real2sim_harness.external_agent import call
        result = call(RunStore(source), op.split(".", 1)[1], arguments)
        calls = Path(result["operation_record"])
        return result, {str(calls / p): file_hash(calls / p) for p in ("request.json", "result.json")}
    _observed(chain, source, False)
    from real2sim_harness.exchange import export_geometry, export_views, placed_manifest
    work = _workspace(arguments["task"])
    target = work / ("exchange-" + folder.name)
    if op in {"exchange.views", "exchange.geometry"}:
        if op == "exchange.views":
            result = export_views(source, arguments["frames"], target, bundle_to_world=arguments.get("bundle_to_world"),
                                  verified=arguments.get("verified", False))
        else:
            result = export_geometry(source, arguments["round_path"], target)
        files = {str(p): file_hash(p) for p in target.iterdir() if p.is_file()}
        return {"ok": True, "verified_scale": arguments.get("verified", False),
                **{k: v for k, v in result.items() if k != "inputs"}}, {**files, **result["inputs"]}
    spec, pins = placed_manifest(source, arguments["scene"], arguments["participants"])
    path = target.with_suffix(".json")
    write_new(path, spec)
    return {"ok": True, "manifest": str(path), "manifest_path": path.name,
            "verified_scale": spec["frame"]["scale_verified"]}, {**pins, str(path): file_hash(path)}


class Workflow:
    def __init__(self, chain):
        self.chain = chain
        self.root = chain.run / "reconstruction"

    @contextmanager
    def lock(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".lock").open("a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            yield

    def _state(self):
        path = self.root / "current.json"
        return read(path) if path.is_file() else {"steps": [], "receipts": [], "revision": 0}

    def choose(self, spec):
        validate(spec, PLAN)
        steps = spec["steps"]
        operations = catalog([s["operation"] for s in steps])
        ids = [s["id"] for s in steps]
        if spec.get("restart_from") is not None and spec["restart_from"] not in ids:
            raise ValueError("restart_from must identify a step in the revised plan")
        if len(ids) != len(set(ids)) or any(s["operation"] not in operations for s in steps):
            raise ValueError("Use unique step IDs and operations from catalog")
        if any(not Path(s["source"]).is_absolute() for s in steps):
            raise ValueError("Every native source must be an absolute path")
        if [s["operation"].startswith("delivery.") for s in steps] != [False] * (len(steps)-1) + [True]:
            raise ValueError("End the plan with exactly one native delivery")
        with self.lock():
            self._open()
            old = self._state()
            retained = []
            for a, b, receipt in zip(steps, old["steps"], old["receipts"]):
                if a != b or b["id"] == spec.get("restart_from") or not self._fresh(receipt):
                    break
                retained.append(receipt)
            revision = old["revision"] + 1
            plan = {**spec, "revision": revision, "receipts": retained}
            write_new(self.root / "plans" / f"{revision}-{uuid.uuid4().hex}.json", plan)
            write(self.root / "current.json", plan)
            return self.status()

    def _open(self):
        if self.chain.current() != "real2sim" or "route" in self.chain.request:
            raise ValueError("Module selection requires an open, unpinned reconstruction stage")

    @staticmethod
    def _fresh(pin):
        path = Path(pin["path"])
        if not path.is_file() or file_hash(path) != pin["sha256"]:
            return False
        receipt = read(path)
        if not all(Path(p).is_file() and file_hash(p) == h for p, h in receipt["files"].items()):
            return False
        if receipt["step"]["operation"] in {"exchange.views", "exchange.placements"}:
            if "verified_scale" not in receipt["result"]:
                return False
            if not receipt["result"]["verified_scale"]:
                return True
            from real2sim_harness.exchange import scale_evidence
            try:
                _, current = scale_evidence(Path(receipt["step"]["source"]), True)
            except (ValueError, OSError, KeyError):
                return False
            return all(receipt["files"].get(p) == h for p, h in current.items())
        return True

    def status(self):
        state = self._state()
        i = next((i for i, pin in enumerate(state["receipts"]) if not self._fresh(pin)), len(state["receipts"]))
        heads = model_modules.models(state["receipts"][:i])
        return {**({"models": heads} if heads else {}), "revision": state["revision"], "steps": state["steps"], "completed": i, "receipts": state["receipts"][:i],
                "next": next(iter(state["steps"][i:]), None)}

    def perform(self, step_id, arguments, evidence=()):
        with self.lock():
            self._open()
            state, progress = self._state(), self.status()
            step = progress["next"]
            if not step or step["id"] != step_id:
                raise ValueError("Run the next current step or revise the plan")
            if step["operation"] in BRIDGE_INPUTS:
                validate(arguments, BRIDGE_INPUTS[step["operation"]])
            model_schema = model_modules.schemas().get(step["operation"])
            if model_schema:
                if step["operation"] in {"object.build", "environment.build"}:
                    arguments = {"method": "blender", **arguments}
                validate(arguments, model_schema)
            inputs = self.chain.done("observations")["files"]
            if any(not Path(p).is_file() or file_hash(p) != h for p, h in inputs.items()):
                raise ValueError("Chain observations changed")
            directory = self.root / "attempts" / uuid.uuid4().hex
            directory.mkdir(parents=True)
            write_new(directory / "call.json", {"step": step, "arguments": arguments, "revision": state["revision"]})
            try:
                with redirect_stdout(sys.stderr):
                    event = None
                    if model_schema:
                        if step["operation"] in model_modules.METHODS:
                            native = model_modules.METHODS[step["operation"]][arguments["method"]]
                            _observed(self.chain, Path(step["source"]), native.startswith("workspace."))
                        result, files, event = model_modules.run(
                            self.chain, step, arguments, directory, progress["receipts"], _execute, _workspace)
                    else:
                        result, files = _execute(self.chain, step, arguments, directory, heads=progress.get("models"))
                for p in evidence:
                    path = Path(p).resolve(strict=True)
                    files[str(path)] = file_hash(path)
                if result.get("ok") is False:
                    raise ValueError(f"Native operation failed: {result}")
            except Exception as exc:
                write_new(directory / "failure.json", {"error": str(exc)})
                raise
            receipt = directory / "receipt.json"
            write_new(receipt, {"step": step, "result": result, "files": files, "observations": inputs,
                                **({"model_event": event} if event else {})})
            state["receipts"] = progress["receipts"] + [{"path": str(receipt), "sha256": file_hash(receipt)}]
            write(self.root / "current.json", state)
            return {"result": result, "receipt": str(receipt)}

    def delivery(self):
        progress = self.status()
        if not progress["steps"] or progress["next"]:
            raise ValueError("Complete the current module plan before advancing")
        receipt = read(progress["receipts"][-1]["path"])
        path = Path(receipt["result"]["delivery"])
        source = Path(receipt["step"]["source"])
        workspace = receipt["step"]["operation"] == "delivery.workspace"
        current = _native_delivery(self.chain, source, workspace)
        current = model_modules.bind_delivery(progress.get("models", {}), current, source.resolve())
        seen = _observed(self.chain, source, workspace)
        placements = {}
        families = {}
        for pin in progress["receipts"][:-1]:
            prior = read(pin["path"])
            op = prior["step"]["operation"]
            seen.update(prior["files"].values())
            if op == "family.complete":
                family = Path(prior["step"]["source"])
                batch = read(family / "family.json")
                parent = batch["parent"]
                if Path(parent.get("parent_run") or parent["run"]).resolve() == source:
                    checked, _ = _complete_family(family, verify_only=True)
                    families[str(family)] = checked["count"]
            if op.startswith("exchange.") and op != "exchange.asset":
                seen.update(_observed(self.chain, Path(prior["step"]["source"]), False))
            if op == "exchange.placements":
                manifest = prior["result"]["manifest"]
                placements[str(Path(manifest).parent)] = manifest
        if "cousins" in self.chain.request and self.chain.request["cousins"] != sum(families.values()):
            raise ValueError("The delivery needs the requested count of independently accepted cousins of this twin")
        missing = [p for p, sha in self.chain.done("observations")["files"].items() if sha not in seen]
        if missing:
            raise ValueError(f"Observations lack native input or step evidence: {missing}")
        if workspace and current["level"] == "scene" and placements:
            manifest = placements.get(str(_workspace(source)))
            if manifest:
                bundle = Path(current["scene"]["path"]).parent / "scene_bundle.json"
                if read(bundle)["source_dependencies"].get(manifest) != file_hash(manifest):
                    raise ValueError("The accepted scene did not consume its current calibrated placements")
        saved = read(path)
        if {k: v for k, v in current.items() if k != "created"} != {k: v for k, v in saved.items() if k != "created"}:
            raise ValueError("Native acceptance changed since the delivery step")
        inputs = self.chain.done("observations")["files"]
        if any(not Path(p).is_file() or file_hash(p) != h for p, h in inputs.items()):
            raise ValueError("Chain observations changed")
        return path, digest(progress)
