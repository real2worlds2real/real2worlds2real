"""Tasks and domain commands for an existing Codex/Claude coding session.

The caller owns reasoning, native tools, conversation history and delegation.
This module never starts a model or reads model credentials.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import uuid
from contextlib import contextmanager, redirect_stdout
from pathlib import Path

from .artifacts import RunStore, atomic_json
from .blender import BlenderExecutor
from .skills import SkillRegistry
from .task_tools import NATIVE_HELPERS, available_tools, execute


def _read(path):
    return json.loads(Path(path).read_text())


def _store(path):
    store = RunStore(Path(path).resolve())
    if not (store.root / "run.json").is_file():
        raise ValueError("Use an existing Real2Sim task directory containing run.json")
    return store


def _runtime(store):
    return _read(store.root / "external/runtime.json")


def _guidance(mode):
    registry = SkillRegistry()
    startup = registry.startup(mode)
    catalog = registry.catalog()
    startup["capabilities_sha256"] = hashlib.sha256(json.dumps(
        {"tools": available_tools(mode), "skills": catalog}, sort_keys=True,
    ).encode()).hexdigest()
    return startup, catalog


def prepare_task(store, *, blender=None, inherit=None):
    """Initialize portable context for a new or existing task, retaining its artifacts."""
    record = _read(store.root / "run.json")
    # Adoption must retain the full simulator gate, even for early prototype runs
    # whose tool list never advertised target simulation.
    record["tools"] = available_tools(record.get("generation_mode", "twin"))
    record["agent_entrypoint"] = "external"
    atomic_json(store.root / "run.json", record)
    external = store.root / "external"
    external.mkdir(exist_ok=True)
    runtime_path = external / "runtime.json"
    if runtime_path.is_file():
        runtime = _read(runtime_path)
    elif inherit is not None:
        runtime = dict(_runtime(inherit))
    else:
        runtime = {"blender": None, "blender_timeout": 600}
    if blender is not None:
        if hasattr(blender, "blender"):
            runtime.update(blender=str(getattr(blender, "_original_blender", blender.blender)),
                           blender_timeout=blender.timeout_seconds)
        else:
            path = Path(shutil.which(str(blender)) or blender).resolve()
            if not path.is_file():
                raise ValueError("Blender executable not found")
            runtime["blender"] = str(path)
    if not runtime.get("blender"):
        raise ValueError("Configure Blender with task init/adopt --blender")
    atomic_json(runtime_path, runtime)
    workspace = store.root / "workspace"
    if (external / "context.json").is_file():
        workspace = Path(_read(external / "context.json")["workspace"])
    elif (store.root / "codex/workspace/task.json").is_file():
        workspace = store.root / "codex/workspace"
    workspace = workspace.resolve()
    if not workspace.is_relative_to(store.root.resolve()) or workspace == store.root.resolve():
        raise ValueError("Use a separate workspace inside this task")
    workspace.mkdir(exist_ok=True, parents=True)
    motion_help = Path(__file__).resolve().parents[2] / "docs/physics_motion_review.md"
    shutil.copy2(motion_help, workspace / "physics_motion_review.md")
    mode = record.get("generation_mode", "twin")
    startup, catalog = _guidance(mode)
    atomic_json(external / "startup_skills.json", {k: v for k, v in startup.items() if k != "instructions"})
    (external / "startup.md").write_text(startup["instructions"] + "\n")
    context = {
        "schema_version": 1,
        "entrypoint": "external-agent",
        "task": str(store.root),
        "workspace": str(workspace),
        "source": str(Path(__file__).resolve().parents[2]),
        "objective": record["objective"],
        "generation_mode": mode,
        "asset_contract": record.get("asset_contract"),
        "contract_inference_required": record.get("contract_inference_required", False),
        "startup_instructions": str(external / "startup.md"),
        "available_skills": catalog,
        "references": [str(store.root / row["path"]) for row in record.get("references", [])],
        "input_manifest": str(store.root / "input_manifest.json"),
        "parent_context": str(store.root / "context") if (store.root / "context").is_dir() else None,
        "measurements": str(store.root / "measurements/inputs.json") if (store.root / "measurements/inputs.json").is_file() else None,
        "required_finish": "submit_cousin_plan" if mode.endswith("planning") else "accept_scene" if mode in {"scene", "scene_cousin"} else "accept_result",
        "commands": {"python": sys.executable, "module": "real2sim_harness.cli", "operation_help": ["task", "schemas", "--run", str(store.root)]},
        "access": "Use source and this task's records for investigation; write generated work in workspace. Native session permissions belong to your coding client. Repository-generated Blender code uses its own task boundary.",
        "delegation": "The caller owns subagents and concurrency. Each participant/variant has its own task. Do not infer permission to allocate extra compute or start more model sessions.",
    }
    atomic_json(external / "context.json", context)
    # Existing managed workspaces retain their task format and original inputs.
    if not (workspace / "task.json").is_file():
        atomic_json(workspace / "task.json", context)
    return {"ok": True, **context}


def create_task(args):
    from .capabilities import TASK_TOOLS
    from .measurement import retain_inputs
    from .video import prepare_reference_inputs
    from .video_review import retain_input_videos

    if not args.objective.strip() or args.blender_timeout <= 0:
        raise ValueError("Supply a nonempty objective and a positive tool timeout")
    if args.video_frames != 0 and not 2 <= args.video_frames <= 120:
        raise ValueError("Optional video overview count must be 2..120, or 0 for original-only input")
    with tempfile.TemporaryDirectory(prefix="real2sim-inputs-") as temporary:
        refs, inputs = prepare_reference_inputs(args.reference, Path(temporary), video_frames=args.video_frames)
        store = RunStore.create(args.runs, references=refs, objective=args.objective,
                                model="external-agent", max_turns=12, tool_schemas=TASK_TOOLS,
                                asset_kind=args.asset_kind, front_axis=args.front_axis,
                                required_parts=args.required_part)
    retain_input_videos(store.root, inputs)
    atomic_json(store.root / "input_manifest.json", {"inputs": inputs})
    record = _read(store.root / "run.json")
    record.update(agent_entrypoint="external", generation_mode="scene" if args.scene else "twin")
    if args.scene:
        record["contract_inference_required"] = False
    for key in ("scene_runner", "gaussian_runner"):
        value = getattr(args, key, None)
        if value is not None:
            value = value.resolve()
            if not value.is_file():
                raise ValueError("Runtime configuration does not exist: " + key)
            record[key] = str(value)
    atomic_json(store.root / "run.json", record)
    if args.measurements:
        retain_inputs(store, args.measurements)
    result = prepare_task(store, blender=args.blender)
    runtime = _runtime(store)
    runtime["blender_timeout"] = args.blender_timeout
    atomic_json(store.root / "external/runtime.json", runtime)
    return result


@contextmanager
def task_lock(store):
    path = store.root / "external/operation.lock"
    path.parent.mkdir(exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another operation owns this task; use distinct child tasks for parallel work") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _pending_updates(store):
    from .steering import pending_for_acceptance_reply
    rows = pending_for_acceptance_reply(store, store.root / "external", "external-agent", "acceptance")
    if rows:
        raise ValueError("Review new task updates before accepting, then retry: " + json.dumps(rows, ensure_ascii=False))


def _refresh_guidance(store):
    record = _read(store.root / "run.json")
    startup, _ = _guidance(record.get("generation_mode", "twin"))
    metadata = {k: v for k, v in startup.items() if k != "instructions"}
    path = store.root / "external/startup_skills.json"
    if path.is_file() and _read(path) == metadata:
        return None
    context = prepare_task(store)
    return {"startup_instructions": context["startup_instructions"],
            "layers": startup["layers"], "capabilities_sha256": startup["capabilities_sha256"],
            "instructions": startup["instructions"], "available_skills": context["available_skills"],
            "reason": "Framework guidance or capability descriptions changed. Apply these current "
                      "instructions and reconsider affected findings before continuing or accepting."}


def call(store, name, arguments, *, executor=None):
    if not isinstance(arguments, dict):
        raise ValueError("Operation request must be a JSON object")
    context = _read(store.root / "external/context.json")
    workspace = Path(context["workspace"]).resolve()
    if not workspace.is_relative_to(store.root.resolve()) or workspace == store.root.resolve():
        raise ValueError("Workspace must remain inside this task")
    with task_lock(store):
        folder = store.root / "external/calls" / uuid.uuid4().hex
        folder.mkdir(parents=True)
        atomic_json(folder / "request.json", {"operation": name, "arguments": arguments})
        guidance = None
        try:
            guidance = _refresh_guidance(store)
            if guidance and name in {"accept_result", "accept_scene"}:
                raise ValueError("Task guidance was updated. Review the returned guidance_update, "
                                 "recheck affected evidence, then retry acceptance.")
            if name in {"accept_result", "accept_scene"}:
                _pending_updates(store)
            if executor is None:
                runtime = _runtime(store)
                executor = BlenderExecutor(Path(runtime["blender"]), timeout_seconds=runtime["blender_timeout"]).for_task(store)
            result = execute(store, executor, workspace, name, arguments).document()
        except (OSError, ValueError, TypeError, RuntimeError, TimeoutError, subprocess.SubprocessError) as error:
            result = {"ok": False, "error": str(error), "error_type": type(error).__name__}
        if guidance is not None:
            result["guidance_update"] = guidance
        result["operation_record"] = str(folder)
        atomic_json(folder / "result.json", result)
        store.append_event({"type": "external_tool_call", "operation": name, "ok": result.get("ok", True), "record": str(folder.relative_to(store.root))})
        return result


def plan_cousins(store, *, prompt, count=None):
    from .cousins import create_planning_run
    runtime = _runtime(store)
    executor = BlenderExecutor(Path(runtime["blender"]), timeout_seconds=runtime["blender_timeout"]).for_task(store)
    with task_lock(store):
        family, planner = create_planning_run(store.root, executor, prompt=prompt, count=count)
        context = prepare_task(planner, inherit=store)
    return {"ok": True, "family": str(family), "status": "awaiting_plan", "planner": context}


def next_variant(family, *, execute_plan=False, variant_index=None, verify_only=False):
    """Return one retained variant task. The caller chooses how to run/delegate it."""
    family = Path(family).resolve()
    batch = _read(family / "family.json")
    if verify_only and (batch["status"] != "complete" or execute_plan or variant_index is not None):
        raise ValueError("Read-only review requires a complete native family")
    planner = _store(family / batch["planner"])
    if batch["status"] in {"awaiting_plan", "needs_input"}:
        return {"ok": True, "status": batch["status"], "planner": prepare_task(planner)}
    if not batch.get("count"):
        raise ValueError("A resolved cousin plan is required")
    if variant_index is not None and not 1 <= variant_index <= batch["count"]:
        raise ValueError("Variant index must be within the recorded user-requested count")
    with task_lock(planner):
        if batch.get("execute") is False and not execute_plan:
            return {"ok": True, "status": "planned", "execute": False, "family": str(family)}
        if execute_plan:
            batch.update(execute=True, execution_request="The caller explicitly starts this retained cousin plan; preserve its design requirements.")
            atomic_json(family / "family.json", batch)
        scene = batch.get("family_kind") == "scene"
        if scene:
            from .scene_cousins import create_child, validate_variant, verify_parent
            from .scene_review import _current_runtime, scene_folder
            verify_parent(family, batch)
        else:
            from .cousins import create_child
            parent = _store(batch["parent"]["run"])
            parent_final = _read(parent.root / "final.json")
            parent.accept(parent_final["attempt_id"], parent_final["reason"], commit=False)
        for index, row in enumerate(batch["variants"]):
            child = _store(family / row["run"]) if row.get("run") else None
            final_path = child.root / ("final_scene.json" if scene else "final.json") if child else None
            if final_path and final_path.is_file():
                final = _read(final_path)
                if scene:
                    _current_runtime(scene_folder(child, final["scene_id"]))
                    validate_variant(child, final["scene_id"])
                else:
                    child.accept(final["attempt_id"], final["reason"], commit=False)
                if row["status"] != "accepted":
                    if verify_only:
                        raise ValueError("Incomplete cousin acceptance")
                    row["status"] = "accepted"
                    atomic_json(family / "family.json", batch)
                continue
            if row["status"] == "accepted" or verify_only:
                raise ValueError("An accepted cousin has lost its final receipt")
            if variant_index is not None and index + 1 != variant_index:
                continue
            if child is None:
                child = create_child(family, index, max_turns=12)
            context = prepare_task(child, inherit=planner)
            return {"ok": True, "status": "needs_variant_work", "family": str(family), "variant_index": index + 1,
                    "variant": context, "next_action": "Complete this variant using the ordinary task commands, then request the next variant. No model session was started."}
        if variant_index is not None:
            return {"ok": True, "status": "variant_accepted", "variant_index": variant_index, "family": str(family)}
        if not scene:
            from .cousin_contract import validate_cousin
            # Native subagents may complete siblings concurrently. Check diversity
            # against all final siblings, including those absent at child creation.
            structures = [_read(family / row["run"] / "final.json")["report"]["functional_structure"] for row in batch["variants"]]
            for index, current in enumerate(structures):
                child_record = _read(family / batch["variants"][index]["run"] / "run.json")
                result = validate_cousin(batch["parent"]["functional_structure"], current,
                                          allow_surface_only=child_record["cousin"]["allow_surface_only"],
                                          sibling_geometry=[s["geometry_fingerprint"] for i, s in enumerate(structures) if i != index])
                if not result["ok"]:
                    if not verify_only:
                        batch["status"] = "incomplete"
                        atomic_json(family / "family.json", batch)
                    raise ValueError("Final cousin family fails structure/diversity checks: " + json.dumps(result))
        if batch["status"] != "complete":
            batch["status"] = "complete"
            atomic_json(family / "family.json", batch)
        return {"ok": True, "status": "complete", "family": str(family), "count": batch["count"]}


def add_parser(commands):
    root = commands.add_parser("task", help="domain tasks for your existing coding-agent session; starts no model")
    tasks = root.add_subparsers(dest="task_command", required=True)
    create = tasks.add_parser("init", help="retain inputs and prepare startup instructions")
    create.add_argument("--reference", type=Path, action="append", default=[])
    create.add_argument("--objective", required=True)
    create.add_argument("--runs", type=Path, default=Path("runs"))
    create.add_argument("--blender", required=True)
    create.add_argument("--blender-timeout", type=int, default=600)
    create.add_argument("--video-frames", type=int, default=0, help="optional overview frame count; 0 retains original video without decoding")
    create.add_argument("--asset-kind", choices=("auto", "rigid", "articulated", "deformable", "solid", "strand", "fluid", "hybrid", "fracture"), default="auto")
    create.add_argument("--front-axis", choices=("+X", "-X", "+Y", "-Y"), default="+Y")
    create.add_argument("--required-part", action="append", default=[])
    create.add_argument("--measurements", type=Path)
    create.add_argument("--scene", action="store_true")
    create.add_argument("--scene-runner", type=Path)
    create.add_argument("--gaussian-runner", type=Path)
    create.set_defaults(task_function=lambda a: create_task(a))
    adopt = tasks.add_parser("adopt", help="use an existing run without rebuilding or starting a model")
    adopt.add_argument("run", type=Path)
    adopt.add_argument("--blender", required=True)
    adopt.set_defaults(task_function=lambda a: prepare_task(_store(a.run), blender=a.blender))
    schemas = tasks.add_parser("schemas", help="list operations or inspect one exact JSON request contract")
    schemas.add_argument("operation", nargs="?")
    schemas.add_argument("--run", type=Path)
    schemas.set_defaults(task_function=_schemas)
    invoke = tasks.add_parser("call", help="execute one domain operation with a workspace JSON request")
    invoke.add_argument("run", type=Path)
    invoke.add_argument("operation")
    invoke.add_argument("--request", type=Path, required=True)
    invoke.set_defaults(task_function=lambda a: call(_store(a.run), a.operation, _read(a.request)))
    cousins = tasks.add_parser("cousins", help="prepare post-twin cousin planning for this agent")
    cousins.add_argument("parent", type=Path)
    cousins.add_argument("--prompt", default="")
    cousins.add_argument("--count", type=int)
    cousins.set_defaults(task_function=lambda a: plan_cousins(_store(a.parent), prompt=a.prompt, count=a.count))
    next_task = tasks.add_parser("next", help="return a retained cousin variant; never launches another agent")
    next_task.add_argument("family", type=Path)
    next_task.add_argument("--execute", action="store_true", help="explicitly start a previously planning-only family")
    next_task.add_argument("--index", type=int, help="prepare a particular variant for caller-managed delegation; starts no model")
    next_task.set_defaults(task_function=lambda a: next_variant(a.family, execute_plan=a.execute, variant_index=a.index))
    root.set_defaults(func=run_command)


def _schemas(args):
    from .capabilities import TASK_TOOLS
    rows = available_tools(_read(_store(args.run).root / "run.json").get("generation_mode", "twin")) if args.run else [row for row in TASK_TOOLS if row["name"] not in NATIVE_HELPERS]
    if args.operation:
        rows = [row for row in rows if row["name"] == args.operation]
        if not rows:
            raise ValueError("Unknown or unavailable operation")
        return rows[0]
    return {"operations": [{"name": row["name"], "description": row["description"]} for row in rows],
            "next_action": "Select an operation by name to read its JSON request schema."}


def run_command(args):
    try:
        # Domain commands may print progress; reserve stdout for one JSON document.
        with redirect_stdout(sys.stderr):
            result = args.task_function(args)
    except (OSError, ValueError, TypeError, RuntimeError, TimeoutError, subprocess.SubprocessError) as error:
        result = {"ok": False, "error": str(error), "error_type": type(error).__name__}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok", True) else 2
