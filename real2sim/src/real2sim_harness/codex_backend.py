from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from .agent import image_block
from .artifacts import RunStore, atomic_json
from .blender import BlenderExecutor
from .capabilities import TASK_TOOLS as CODEX_TOOLS
from .capabilities import tool_enabled as _tool_enabled
from .reference_tools import (
    _align_reference_prior as _align_reference_prior,
)
from .reference_tools import (
    _attempt_directory as _attempt_directory,
)
from .reference_tools import (
    _attempt_items as _attempt_items,
)
from .reference_tools import (
    _camera_alignment_items as _camera_alignment_items,
)
from .reference_tools import (
    _compact_attempt_summary as _compact_attempt_summary,
)
from .reference_tools import (
    _final_render_items as _final_render_items,
)
from .reference_tools import (
    _inspection_items as _inspection_items,
)
from .reference_tools import (
    _reference_paths as _reference_paths,
)
from .reference_tools import (
    _reference_prior_items as _reference_prior_items,
)
from .reference_tools import (
    _reference_report as _reference_report,
)
from .reference_tools import (
    _run_reference_geometry as _run_reference_geometry,
)
from .reference_tools import (
    _run_reference_prior as _run_reference_prior,
)
from .reference_tools import (
    _solve_reference_camera as _solve_reference_camera,
)
from .reference_tools import (
    _workspace_script as _workspace_script,
)
from .scene_agent import SCENE_INSTRUCTIONS, SCENE_TOOLS
from .skills import SkillRegistry
from .task_tools import SHARED_OPERATIONS
from .task_tools import execute as execute_task_tool

CODEX_INSTRUCTIONS = """You are Astra, the real2sim object-generation coding agent.
The harness has loaded common working methods and this task's workflow as startup skills.
Read task.json and the provided references. If context/scene_parent.json exists, preserve
its relevant scene constraints and participant identity while generating the assigned object.
Use the available native file/image/code capabilities to investigate and author editable
Blender source in the workspace. Call run_blender with a workspace-relative script path.
When contract_inference_required is true, declare the inferred contract before the first build.
After choosing the representation, read the applicable specialized skill and its detailed
resources when needed. Do not preload or require every unrelated tool or physical model.
Use actual tool reports, task records and source for investigation; the common startup skill
explains access. Task measurement inputs and original videos remain available beyond summaries.
Keep explicit metric observations, learned geometry priors and generated assumptions distinct.
Optional outside search is subject to configured availability and user instructions.
Follow the required part names, metre frame and semantic orientation in the asset contract.
Implement real structure and the selected physical representation, not visual-only substitutes.
Read failures at their responsible construction/control/conversion/runtime/presentation layer.
Final delivery requires the current source's selected target-simulator checks plus adequate
visual and functional review, including additional gates required by that representation.
A failed or unavailable adapter is not simulation readiness. Reuse valid retained evidence,
repair demonstrated defects and regenerate affected evidence rather than hiding the failure.
Finish with accept_result only when the supported evidence meets the user's request.
If the original request also asks for cousins, retain that request without inventing variants;
the optional cousin stage follows successful twin acceptance.
"""

COUSIN_PLANNING_INSTRUCTIONS = """You are planning an optional post-twin cousin stage, not building another twin.
Read task.json and context/parent.json, the accepted parent visuals and editable source as useful. Read the
generate-functional-cousins skill when useful. Interpret the user's natural-language request yourself: extract
the requested count, shared requirements and any per-variant directions without asking the user to split fields.
Use submit_cousin_plan with exactly one concrete prompt per requested cousin. There is no fixed count.
Infer execution intent too: execute=false for a planning-only request, true when generation is requested.
An explicit count in task.json overrides an ambiguous textual quantity. If no count can be inferred or the
request is materially contradictory, submit a concise question rather than scheduling invented work.
Preserve the object's function. Articulated cousins preserve joint identifiers/roles, types and connectivity,
mechanical loops and coupling relationships while dimensions, shapes, proportions and joint placement may vary.
Rigid cousins may change nonessential parts, such as a mug becoming a handleless drinking glass.
Default to meaningful geometry redesign and diverse variants, rather than only color/texture edits. Surface-only
variants are appropriate when the user explicitly requests them. Record that distinction in the internal plan.
The original prompt remains authoritative context; each per-variant prompt should resolve its intended scope.
Do not run Blender construction, declare an asset contract, perform simulation or call accept_result in this phase.
Viewing strategies and optional tools are your choice. Explicitly finish with submit_cousin_plan.
"""

COUSIN_INSTRUCTIONS = """This run generates one functional cousin of an accepted twin. Read context/cousin_task.json
and the generate-functional-cousins skill when useful. The supplied images and source describe the PARENT;
they are not pixel-matching targets for this variant. If context records a later explicit execution_request,
it supersedes earlier planning-only wording: now generate while keeping the original design requirements.
Preserve the original prompt as provenance. Follow the raw user request and this variant's planned
prompt, interpreting their scope yourself. Reuse the editable source or parent_scene.blend where useful.
Preserve function and the parent simulation asset kind. Preserve articulated joint identifiers, types and parent
connectivity, mechanism loops and cloth attachment ownership. Joint locations, mounting orientations, limits,
size, shape, proportions, mesh topology and materials may adapt to the new geometry. Revalidate all motion.
Rigid nonessential details need not survive; infer this cousin's essential semantic parts before construction.
Default to meaningful geometry change and differentiate accepted siblings. Exact geometry clones are rejected
when the plan expects geometry change. Fingerprint differences do not establish meaningful semantic diversity;
judge the redesign, function and user-prompt compliance from rendered and simulated evidence yourself.
Do not create further generations or call request_cousins. Each cousin uses the normal modeling, optional tools,
mandatory final target-simulation feedback and accept_result loop. Parent success does not certify its children.
"""


def _load_codex_runtime():
    from .codex_transport import NATIVE_ITEMS, PROVIDER, AppServerTransport, server_launch
    return NATIVE_ITEMS, PROVIDER, AppServerTransport, server_launch


def _configure_reference_search(argv, config, mode):
    if mode not in {"live", "cached", "disabled"}:
        raise ValueError("Invalid web-search mode")
    argv = list(argv)
    for index in range(1, len(argv)):
        if argv[index - 1] == "-c" and argv[index].startswith("web_search="):
            argv[index] = "web_search=" + json.dumps(mode)
            break
    else:
        argv.extend(["-c", "web_search=" + json.dumps(mode)])
    argv.extend(["-c", "tools.web_search=" + json.dumps(mode != "disabled")])
    return argv, {**config, "web_search": mode, "tools.web_search": mode != "disabled"}


def _expired_authentication(report):
    error = report.get("terminal_error", {}).get("error", {})
    info = error.get("codexErrorInfo")
    status = info.get("httpConnectionFailed", {}).get("httpStatusCode") if isinstance(info, dict) else None
    message = error.get("message", "").lower()
    return status == 401 and ("token_expired" in message or ("token" in message and "expired" in message))


def run_codex_agent(executor: BlenderExecutor, store: RunStore, **options) -> dict[str, Any]:
    """Retain one bounded expired-token recovery within the caller's time budget."""
    if hasattr(executor, "for_task"):
        executor = executor.for_task(store, auth_dir=options.get("auth_dir"))
    if options.get("timeout_seconds", 3600) <= 0:
        raise ValueError("Invalid Codex time budget")
    deadline = time.monotonic() + options.get("timeout_seconds", 3600)
    resume_existing = options.pop("resume_existing", False)
    options.pop("timeout_seconds", None)
    for renewal in range(2):
        try:
            return _run_codex_session(
                executor, store, **options, resume_existing=resume_existing,
                timeout_seconds=max(1, int(deadline - time.monotonic())),
            )
        except RuntimeError:
            path = store.root / "codex" / "report.json"
            report = json.loads(path.read_text()) if path.is_file() else {}
            if renewal or time.monotonic() >= deadline or not _expired_authentication(report):
                raise
            store.append_event({"type": "expired_token_recovery", "action": "restart with current explicit authorization and resume retained thread"})
            resume_existing = True
    raise RuntimeError("Expired-token recovery exhausted")


def _run_codex_session(
    executor: BlenderExecutor,
    store: RunStore,
    *,
    model: str,
    objective: str,
    auth_mode: str,
    auth_dir: Path | None,
    codex: str,
    seed_script: Path | None = None,
    web_search: str = "disabled",
    max_turns: int = 8,
    max_calls: int = 24,
    reasoning_effort: str = "high",
    timeout_seconds: int = 3600,
    resume_existing: bool = False,
) -> dict[str, Any]:
    if auth_mode not in {"chatgpt_plan", "api_key"}:
        raise ValueError("Choose chatgpt_plan or api_key")
    if not 1 <= max_turns <= 30 or not 1 <= max_calls <= 100 or timeout_seconds <= 0:
        raise ValueError("Invalid Codex budgets")
    if web_search not in {"live", "cached", "disabled"}:
        raise ValueError("Invalid web-search mode")
    native_items, provider, transport_type, server_launch = _load_codex_runtime()
    service_call_limit = max(24, max_calls * 4)
    runtime = store.root / "codex"
    skill_registry = SkillRegistry()
    workspace = runtime / "workspace"
    home = runtime / "home"
    if resume_existing and not all(path.is_file() for path in (runtime / "thread.json", runtime / "report.json", workspace / "task.json")):
        raise ValueError("Resume requires a retained thread, report and agent workspace")
    workspace.mkdir(parents=True, exist_ok=resume_existing)
    home.mkdir(mode=0o700, exist_ok=resume_existing)
    (workspace / ".real2sim-agent-root").touch()
    run_input_path = store.root / 'measurements/inputs.json'
    if run_input_path.is_file():
        from .measurement import inspect_inputs
        measurement_context = inspect_inputs(store)
        (workspace / 'measurement_context.json').write_text(json.dumps(measurement_context, indent=2))
    refs_dir = workspace / "references"
    refs_dir.mkdir(exist_ok=resume_existing)
    shutil.copy2(Path(__file__).resolve().parents[2] / "docs/physics_motion_review.md",
                 workspace / "physics_motion_review.md")
    input_manifest = store.root / "input_manifest.json"
    if input_manifest.is_file():
        shutil.copy2(input_manifest, workspace / "input_manifest.json")
    from .video_review import video_sources
    original_videos = [record for record in video_sources(store).values() if record["role"] == "reference"]
    for record in original_videos:
        target = workspace / "videos" / Path(record["path"]).name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(store.root / record["path"], target)
    run = json.loads((store.root / "run.json").read_text())
    reference_paths = []
    mode = run.get("generation_mode", "twin")
    from .scene_cousins import SCENE_COUSIN_INSTRUCTIONS, SCENE_PLANNING_INSTRUCTIONS
    scene_mode = mode in {"scene", "scene_cousin"}
    planning_mode = mode in {"cousin_planning", "scene_cousin_planning"}
    if scene_mode:
        instructions = SCENE_INSTRUCTIONS + (SCENE_COUSIN_INSTRUCTIONS if mode == "scene_cousin" else "")
    elif planning_mode:
        instructions = SCENE_PLANNING_INSTRUCTIONS if mode == "scene_cousin_planning" else COUSIN_PLANNING_INSTRUCTIONS
    else:
        instructions = CODEX_INSTRUCTIONS + (COUSIN_INSTRUCTIONS if mode == "cousin" else "")
    startup = skill_registry.startup(mode)
    instructions = startup["instructions"] + "\n\n" + instructions
    atomic_json(runtime / "startup_skills.json", {key:value for key,value in startup.items() if key != "instructions"})
    if scene_mode:
        shutil.copy2(Path(__file__).resolve().parents[2] / "docs/SCENES.md", workspace / "scene_contract.md")
    if (store.root / "context").is_dir():
        shutil.copytree(store.root / "context", workspace / "context", dirs_exist_ok=True)
    for index, reference in enumerate(run["references"]):
        source = store.root / reference["path"]
        target = refs_dir / f"{index:02d}_{Path(reference['source_name']).name}"
        shutil.copy2(source, target)
        reference_paths.append(str(target.relative_to(workspace)))
    if not resume_existing:
        (workspace / "README.md").write_text(
            "# Agent workspace\n\nWrite Blender source here. Reference images are in `references/`.\n",
            encoding="utf-8",
        )
    seed_name = json.loads((workspace / "task.json").read_text()).get("seed_script") if resume_existing else None
    if seed_script is not None:
        seed_script = seed_script.resolve()
        if seed_script.suffix != ".py" or not seed_script.is_file():
            raise ValueError("seed_script must name an existing .py file")
        seed_name = "seed_scene.py"
        shutil.copy2(seed_script, workspace / seed_name)
    atomic_json(
        workspace / "task.json",
        {
            "objective": objective,
            "generation_mode": mode,
            "cousin": run.get("cousin"),
            "scene_cousin": run.get("scene_cousin"),
            "cousin_explicit_count": run.get("cousin_explicit_count"),
            "references": reference_paths,
            "inspection_context": "project_access.json",
            "input_manifest": "input_manifest.json" if input_manifest.is_file() else None,
            "original_videos": [{**record, "workspace_path": "videos/" + Path(record["path"]).name} for record in original_videos],
            "seed_script": seed_name,
            "budgets": {
                "turns": max_turns,
                "blender_attempts": max_calls,
                "final_renders": 6,
                "service_calls": service_call_limit,
            },
            "asset_contract": run.get("asset_contract", {}),
            "contract_inference_required": run.get("contract_inference_required", False),
            "required_finish": "remo.submit_cousin_plan" if planning_mode else "remo.accept_scene" if scene_mode else "remo.accept_result",
            "available_skills": skill_registry.catalog(),
            "startup_skills": startup["layers"],
            "measurement_context": 'measurement_context.json' if (workspace / 'measurement_context.json').is_file() else None,
        },
    )
    resolved_codex = shutil.which(codex) or codex
    argv, env, config, credential = server_launch(
        workspace,
        home,
        auth_mode=auth_mode,
        auth_dir=str(auth_dir) if auth_dir is not None else "",
        codex=resolved_codex,
    )
    # Search remains an explicit per-run choice in the isolated worker.
    argv, config = _configure_reference_search(argv, config, web_search)
    from .project_inspection import configure as configure_inspection
    argv, config = configure_inspection(argv, config, provider=provider,
        project=Path(__file__).resolve().parents[2], run=store.root,
        workspace=workspace, auth_dir=auth_dir)
    report: dict[str, Any] = {
        "status": "starting",
        "backend": "codex-app-server",
        "model": model,
        "turns": 0,
        "service_calls": 0,
        "blender_calls": 0,
        "final_render_calls": 0,
        "web_search_calls": 0,
        "native_calls": 0,
        "usage": {},
        "text": "",
        "workspace": str(workspace),
    }
    report_path = runtime / "report.json"
    if resume_existing:
        previous = json.loads(report_path.read_text())
        if previous.get("status") == "accepted":
            raise ValueError("The retained run is already accepted")
        resume_index = previous.get("resumes", 0) + 1
        atomic_json(runtime / f"report-before-resume-{resume_index:03d}.json", previous)
        report.update(previous)
        report.update(status="starting", resumes=resume_index)
        report.pop("error_type", None)
        report.pop("terminal_error", None)
        # Recover results completed by the host but possibly lost to the model on disconnect.
        recovered = []
        for attempt in sorted(store.attempts_dir.glob("attempt-*"))[-1:]:
            destination = workspace / "recovery" / attempt.name
            destination.mkdir(parents=True, exist_ok=True)
            paths = [attempt / "report.json", *sorted(attempt.glob("hero_*.png")), *sorted(attempt.glob("preview_*.png"))]
            paths += sorted(attempt.glob("surface-review-*/*.png"))
            paths += sorted(attempt.glob("deformation-*/*.png"))
            copied = []
            for source in paths:
                if source.is_file():
                    relative = source.relative_to(attempt)
                    target = destination / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
                    copied.append(str(target.relative_to(workspace)))
            recovered.append({"attempt_id": attempt.name, "evidence_files": copied})
        atomic_json(workspace / "recovery" / "index.json", {"attempts": recovered})
        if run.get("latest_rejection"):
            review_path = store.root / run["latest_rejection"]
            shutil.copytree(review_path.parent, workspace / "recovery" / "human_review", dirs_exist_ok=True)
    atomic_json(runtime / "launch.json", {"argv": argv, "config": config})
    transport = None
    seen: set[str] = set()
    steering_barrier = []
    deadline = time.monotonic() + timeout_seconds
    try:
        transport = transport_type(argv, workspace, env, secret=credential)
        report["server_identity"] = transport.request(
            "initialize",
            {
                "clientInfo": {"name": "real2sim_harness", "title": "Real2Sim Harness", "version": "0.1.0"},
                "capabilities": {"experimentalApi": True},
            },
        )
        transport.send({"method": "initialized", "params": {}})
        tools = [
            {
                "type": "namespace",
                "name": "remo",
                "description": "Build, inspect and accept persistent Blender object-generation attempts",
                "tools": [
                    {
                        "type": "function",
                        "name": tool["name"],
                        "description": tool["description"],
                        "inputSchema": tool["parameters"],
                        "deferLoading": False,
                    }
                    for tool in CODEX_TOOLS
                    if _tool_enabled(mode, tool["name"])
                ],
            }
        ]
        if resume_existing:
            retained_id = json.loads((runtime / "thread.json").read_text())["thread_id"]
            response = transport.request("thread/resume", {
                "threadId": retained_id, "excludeTurns": True,
                "model": model, "modelProvider": provider, "cwd": str(workspace),
                "approvalPolicy": "never", "permissions": provider,
                "developerInstructions": instructions,
                "config": {"model_reasoning_effort": reasoning_effort},
            })
            if response.get("thread", {}).get("id") != retained_id:
                raise RuntimeError("Codex resumed a different thread")
            recent = transport.request("thread/turns/list", {"threadId": retained_id, "limit": 2, "itemsView": "notLoaded"})
            safe_recent = json.loads(json.dumps(recent).replace(credential, "[REDACTED]"))
            atomic_json(runtime / f"turn-status-before-resume-{report['resumes']:03d}.json", safe_recent)
            store.append_event({"type": "codex_thread_resumed", "thread_id": retained_id, "resumes": report["resumes"]})
        else:
            response = transport.request("thread/start", {
                "model": model,
                "modelProvider": provider,
                "cwd": str(workspace),
                "approvalPolicy": "never",
                "permissions": provider,
                "ephemeral": False,
                "config": {"model_reasoning_effort": reasoning_effort},
                "dynamicTools": tools,
                "baseInstructions": "Plan functional cousins from the user request." if planning_mode else "Build and review an operable scene." if scene_mode else "Build and visually iterate one editable 3D object from the supplied references.",
                "developerInstructions": instructions,
            })
        if response.get("model") != model or response.get("modelProvider") != provider:
            raise RuntimeError("Codex returned a different model or provider")
        active_profile = response.get("activePermissionProfile")
        if active_profile is not None and active_profile.get("id") != provider:
            raise RuntimeError("Codex returned a different task permission profile")
        atomic_json(runtime / "permissions_receipt.json", {
            "requested_profile": provider, "active_profile": active_profile,
            "operation": "resume" if resume_existing else "start",
            "cwd": response.get("cwd", str(workspace)),
            "scope": "Codex native command/file access; host tool executors have their own execution boundary.",
        })
        thread_id = response["thread"]["id"]
        report.update(status="running", thread_id=thread_id)
        if resume_existing:
            run["status"] = "running"
            atomic_json(store.root / "run.json", run)
        atomic_json(runtime / "thread.json", {"thread_id": thread_id})
        atomic_json(report_path, report)
        starting_turn = report.get("turns", 0) if resume_existing else 0
        # A deliberate resume grants this invocation's bounded turn budget.
        # Retain cumulative numbering without making a previously exhausted
        # task impossible to continue at the same configured limit.
        for turn_index in range(starting_turn, starting_turn + max_turns):
            prompt = (
                ("The previous acceptance has been withdrawn after review feedback. "
                 "Read recovery/human_review/review.json and the supplied numerical, source and visual evidence. "
                 "Separate a demonstrated asset defect from a test-controller, conversion or presentation issue. "
                 "Choose checks appropriate to the asset's selected simulation backend and representation. "
                 "Repair only the demonstrated cause; preserve unchanged valid source and evidence. "
                 "If changing test controls or physics assumptions, regenerate matching simulator evidence and inspect the actual movie. "
                 "Do not hide geometry, disable required contacts, alter acceptance thresholds merely to pass, or assume previous endpoint success rules out transient defects. "
                 if run.get("latest_rejection") else
                 "Resume the same unfinished object after the transport failure. ")
                + "Existing source and artifacts are retained. "
                "The latest host-completed reports and images are available inside this workspace at recovery/index.json. "
                "Inspect those recovered images and reports; do not rebuild or repeat a completed unchanged attempt just because its tool response was interrupted. "
                "Continue reviewing, revising if needed, and explicitly accept an evidence-supported result."
                if resume_existing and turn_index == starting_turn
                else
                json.dumps(
                    {
                        "objective": objective,
                        "reference_files": reference_paths,
                        "seed_script": seed_name,
                        "instruction": "Inspect the images and any supplied seed, implement the object, execute it and iterate from rendered evidence.",
                    },
                    ensure_ascii=False,
                )
                if turn_index == 0
                else "Continue the same unfinished object from the latest code and renders. Do not repeat an unchanged attempt."
            )
            if planning_mode:
                prompt = json.dumps({"request": objective, "explicit_count": run.get("cousin_explicit_count"),
                                     "parent_context": "context/parent_scene.json" if mode == "scene_cousin_planning" else "context/parent.json", "reference_files": reference_paths,
                                     "instruction": "Interpret the user's scope and quantity yourself. Inspect useful parent context, then submit_cousin_plan; do not build an asset in this planning stage."}, ensure_ascii=False)
            if scene_mode:
                prompt = json.dumps({"objective": objective, "reference_files": reference_paths,
                                     "task": "task.json", "contract": "scene_contract.md",
                                     "instruction": "Continue the retained scene plan and artifacts. Inspect inputs, choose tools, generate or reuse complete participants, assemble, run the configured simulator, inspect feedback and repair. Finish with accept_scene only after verified scene-level success."}, ensure_ascii=False)
            turn_id = transport.request(
                "turn/start",
                {
                    "threadId": thread_id,
                    "cwd": str(workspace),
                    "model": model,
                    "effort": reasoning_effort,
                    "approvalPolicy": "never",
                    "permissions": provider,
                    "input": [{"type": "text", "text": prompt}],
                },
            )["turn"]["id"]
            report["turns"] = turn_index + 1
            while True:
                from .steering import deliver_pending
                steering_barrier.extend(deliver_pending(store, runtime, transport, thread_id, turn_id))
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("real2sim Codex task exceeded its time budget")
                try:
                    event = transport.next_event(timeout=min(30, remaining))
                except TimeoutError:
                    continue
                method, params = event.get("method"), event.get("params", {})
                if params.get("threadId") not in (None, thread_id):
                    raise RuntimeError("Codex event belongs to another thread")
                if method == "item/tool/call":
                    if params.get("turnId") != turn_id or params.get("namespace") not in (
                        None,
                        "remo",
                    ):
                        raise RuntimeError("Stale turn or unknown tool namespace")
                    tool_name = params["tool"]
                    if report["service_calls"] >= service_call_limit and tool_name not in {"accept_result", "accept_scene"}:
                        report["status"] = "budget_exhausted"
                        store.finish_without_acceptance("budget_exhausted")
                        return report
                    call_id = params["callId"]
                    if call_id in seen:
                        raise RuntimeError("Duplicate tool call; refusing re-execution")
                    seen.add(call_id)
                    arguments = params["arguments"]
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    try:
                        if not _tool_enabled(mode, tool_name):
                            raise ValueError("Tool is unavailable in this generation mode")
                        if tool_name in {tool["name"] for tool in SCENE_TOOLS}:
                            from .scene_agent import build_background, build_object
                            from .scene_physics import plan_scene_physics
                            from .scene_review import (
                                accept_scene,
                                assemble_scene,
                                run_scene_simulation,
                            )
                            if tool_name == "plan_scene_physics":
                                result = plan_scene_physics(**arguments)
                            elif tool_name == "build_scene_background":
                                if report["blender_calls"] >= max_calls:
                                    raise ValueError("Scene authoring budget exhausted")
                                report["blender_calls"] += 1
                                result = build_background(executor, store, workspace, **arguments)
                            elif tool_name in {"build_scene_object", "build_scene_cousin_object"}:
                                def worker(child_executor, child_store, **child_options):
                                    return run_codex_agent(child_executor, child_store, model=model,
                                        auth_mode=auth_mode, auth_dir=auth_dir, codex=codex,
                                        web_search=web_search, max_turns=min(max_turns, 12), reasoning_effort=reasoning_effort,
                                        timeout_seconds=max(1, int(deadline - time.monotonic())), **child_options)
                                arguments["max_calls"] = min(arguments["max_calls"], max_calls)
                                if tool_name == "build_scene_cousin_object":
                                    from .scene_cousins import build_object as build_cousin_object
                                    result = build_cousin_object(executor, store, workspace, worker, **arguments)
                                else:
                                    result = build_object(executor, store, workspace, worker, **arguments)
                            elif tool_name == "assemble_scene":
                                result = assemble_scene(store, workspace, **arguments)
                            elif tool_name == "run_scene_simulation":
                                arguments["timeout_seconds"] = max(1, min(arguments["timeout_seconds"], int(deadline - time.monotonic())))
                                result = run_scene_simulation(store, **arguments)
                            elif tool_name == "run_gaussian_background":
                                from .gaussian_background import run_background
                                arguments["timeout_seconds"] = max(1, min(arguments["timeout_seconds"], int(deadline - time.monotonic())))
                                result = run_background(store, workspace, **arguments)
                            else:
                                steering_barrier.extend(deliver_pending(store, runtime, transport, thread_id, turn_id))
                                from .steering import pending_for_acceptance_reply
                                steering_barrier.extend(pending_for_acceptance_reply(store, runtime, thread_id, turn_id))
                                if steering_barrier:
                                    updates = [row["message"] for row in steering_barrier]
                                    steering_barrier.clear()
                                    raise ValueError("Review new scene steering before acceptance: " + json.dumps(updates, ensure_ascii=False))
                                final = accept_scene(store, **arguments)
                                result = final
                            items = [{"type": "inputText", "text": json.dumps(result, ensure_ascii=False)}]
                            if tool_name == "build_scene_background" and result.get("ok"):
                                from .scene_review import _media_path
                                for name in result.get("preview_images", [])[:4]:
                                    path = _media_path(Path(result["directory"]), name)
                                    items.append({"type": "inputImage", "imageUrl": image_block(path)["image_url"]})
                            if tool_name == "run_scene_simulation" and result.get("report"):
                                from .scene_review import _media_path, scene_folder
                                evidence_root = scene_folder(store, arguments["scene_id"]) / result["output"]
                                for name in result["report"].get("images", [])[:8]:
                                    items.append({"type": "inputImage", "imageUrl": image_block(_media_path(evidence_root, name))["image_url"]})
                            if tool_name == "run_gaussian_background" and result.get("ok"):
                                from .scene_video import evidence_path
                                for name in result.get("preview_images", [])[:4]:
                                    items.append({"type": "inputImage", "imageUrl": image_block(evidence_path(result["directory"], name))["image_url"]})
                            success = result.get("ok", True)
                        elif tool_name in SHARED_OPERATIONS:
                            if tool_name == "run_blender":
                                if report["blender_calls"] >= max_calls:
                                    raise ValueError("Blender attempt budget exhausted")
                                report["blender_calls"] += 1
                            elif tool_name == "render_final":
                                if report["final_render_calls"] >= 6:
                                    raise ValueError("Final-render budget exhausted")
                                report["final_render_calls"] += 1
                            result = execute_task_tool(store, executor, workspace, tool_name, arguments)
                            items = _final_render_items(result)
                            success = result.ok
                        elif tool_name == "request_cousins":
                            from .cousins import request_cousins
                            result = request_cousins(store, **arguments)
                            items = [{"type": "inputText", "text": json.dumps(result, ensure_ascii=False)}]
                            success = True
                        elif tool_name == "submit_cousin_plan":
                            from .cousins import submit_plan
                            planned = submit_plan(store, **arguments)
                            items = [{"type": "inputText", "text": json.dumps(planned, ensure_ascii=False)}]
                            success = True
                        elif tool_name == "list_video_sources":
                            from .video_review import list_video_sources
                            items = [{"type": "inputText", "text": json.dumps(list_video_sources(store), ensure_ascii=False)}]
                            success = True
                        elif tool_name == "inspect_video":
                            from .video_review import inspect_video
                            result = inspect_video(store, **arguments)
                            items = _final_render_items(result)
                            success = result.ok
                        elif tool_name == "promote_video_frames":
                            from .video_review import promote_video_frames
                            promoted = promote_video_frames(store, **arguments)
                            for frame in promoted["references"]:
                                target = workspace / frame["path"]
                                target.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(store.root / frame["path"], target)
                                frame["workspace_path"] = frame["path"]
                                if frame["reference_index"] == len(reference_paths):
                                    reference_paths.append(frame["path"])
                            task_path = workspace / "task.json"
                            task = json.loads(task_path.read_text())
                            task["references"] = reference_paths
                            atomic_json(task_path, task)
                            items = [{"type": "inputText", "text": json.dumps(promoted, ensure_ascii=False)}]
                            success = True
                        elif tool_name == "render_motion_video":
                            from .motion_review import prepare_candidate_motion
                            from .video_review import list_video_sources
                            store.accept(arguments["attempt_id"], arguments["assumptions"], commit=False)
                            result = prepare_candidate_motion(executor, store, arguments["attempt_id"])
                            summary = {**result.summary, "video_sources": list_video_sources(store),
                                       "next_action": "Use inspect_video to review the rendered motion, intervals and details before acceptance. The original editable source and actual simulation cache are retained."}
                            items = [{"type": "inputText", "text": json.dumps(summary, ensure_ascii=False)}]
                            success = True
                        elif tool_name == "accept_result":
                            steering_barrier.extend(deliver_pending(store, runtime, transport, thread_id, turn_id))
                            from .steering import pending_for_acceptance_reply
                            steering_barrier.extend(pending_for_acceptance_reply(store, runtime, thread_id, turn_id))
                            if steering_barrier:
                                updates = [row["message"] for row in steering_barrier]
                                steering_barrier.clear()
                                raise ValueError("New task/runtime updates arrived. Review them and any affected evidence before accepting: " + json.dumps(updates, ensure_ascii=False))
                            acceptance = dict(arguments)
                            requested_cousins = acceptance.pop("cousin_request", None)
                            if requested_cousins:
                                from .cousins import request_cousins
                                request_cousins(store, request=requested_cousins)
                            final = store.accept(**acceptance)
                            items = [{"type": "inputText", "text": json.dumps(final, ensure_ascii=False)}]
                            success = True
                        elif tool_name == "search_skills":
                            items = [
                                {
                                    "type": "inputText",
                                    "text": json.dumps(
                                        skill_registry.search(**arguments), ensure_ascii=False
                                    ),
                                }
                            ]
                            success = True
                        elif tool_name == "read_skill":
                            items = [
                                {
                                    "type": "inputText",
                                    "text": json.dumps(
                                        skill_registry.read(**arguments), ensure_ascii=False
                                    ),
                                }
                            ]
                            success = True
                        elif tool_name == "inspect_silhouette":
                            items = _inspection_items(store, kind="silhouette", **arguments)
                            success = True
                        elif tool_name == "inspect_articulation":
                            items = _inspection_items(store, kind="articulation", **arguments)
                            success = True
                        elif tool_name == "inspect_references":
                            items = [
                                {
                                    "type": "inputText",
                                    "text": json.dumps(
                                        _reference_report(store), ensure_ascii=False
                                    ),
                                }
                            ]
                            success = True
                        elif tool_name == "solve_reference_camera":
                            items = [
                                {
                                    "type": "inputText",
                                    "text": json.dumps(
                                        _solve_reference_camera(store, **arguments),
                                        ensure_ascii=False,
                                    ),
                                }
                            ]
                            success = True
                        elif tool_name == "inspect_camera_alignment":
                            items = _camera_alignment_items(store, **arguments)
                            success = True
                        elif tool_name == "run_reference_geometry":
                            geometry = _run_reference_geometry(store, **arguments)
                            items = [
                                {
                                    "type": "inputText",
                                    "text": json.dumps(geometry, ensure_ascii=False),
                                }
                            ]
                            success = bool(geometry["ok"])
                        elif tool_name == "run_reference_prior":
                            resolution = arguments["resolution"]
                            prior = _run_reference_prior(store, **arguments)
                            items = _reference_prior_items(store, prior, resolution)
                            success = bool(prior["ok"])
                        elif tool_name == "align_reference_prior":
                            alignment = _align_reference_prior(store, **arguments)
                            items = [
                                {
                                    "type": "inputText",
                                    "text": json.dumps(alignment, ensure_ascii=False),
                                }
                            ]
                            success = bool(alignment["ok"])
                        else:
                            raise ValueError(f"Unknown tool: {tool_name}")
                    except (
                        ValueError,
                        TypeError,
                        OSError,
                        RuntimeError,
                        json.JSONDecodeError,
                    ) as exc:
                        items = [
                            {
                                "type": "inputText",
                                "text": json.dumps(
                                    {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                                ),
                            }
                        ]
                        success = False
                    transport.send(
                        {"id": event["id"], "result": {"success": success, "contentItems": items}}
                    )
                    report["service_calls"] += 1
                    store.append_event(
                        {
                            "type": "codex_tool_call",
                            "tool": tool_name,
                            "success": success,
                        }
                    )
                    if tool_name in {"accept_result", "accept_scene"} and success:
                        report["status"] = "accepted"
                        report["final"] = final
                        return report
                    if tool_name == "submit_cousin_plan" and success:
                        report.update(status=planned["status"], cousin_family=planned["family"], cousin_plan=planned["plan"])
                        store.finish_without_acceptance(planned["status"])
                        return report
                elif method == "thread/tokenUsage/updated":
                    report["usage"] = params.get("tokenUsage", {})
                elif method == "item/completed":
                    item = params.get("item", {})
                    kind = item.get("type")
                    if kind == "agentMessage":
                        report["text"] = item.get("text", "")
                    elif kind in native_items or kind == "webSearch":
                        if kind == "webSearch":
                            report["web_search_calls"] += 1
                            store.append_event({"type": "web_search_completed", "item": item})
                        report["native_calls"] += 1
                        if report["native_calls"] >= max(40, max_calls * 8):
                            report["status"] = "budget_exhausted"
                            store.finish_without_acceptance("budget_exhausted")
                            return report
                elif method == "turn/completed":
                    turn = params.get("turn", {})
                    if turn.get("id") != turn_id or turn.get("status") != "completed":
                        if turn.get("error"):
                            report["terminal_error"] = {"error": json.loads(json.dumps(turn["error"]).replace(credential, "[REDACTED]"))}
                        raise RuntimeError("Codex turn failed or completed a different turn")
                    break
                elif method == "error" and not params.get("willRetry", False):
                    safe_error = json.loads(json.dumps(params).replace(credential, "[REDACTED]"))
                    report["terminal_error"] = safe_error
                    store.append_event({"type": "codex_terminal_error", "detail": safe_error})
                    raise RuntimeError("Codex reported a terminal error")
                elif "id" in event and method:
                    raise RuntimeError(f"Unexpected Codex capability request: {method}")
                atomic_json(report_path, report)
        report["status"] = "budget_exhausted"
        store.finish_without_acceptance("budget_exhausted")
        return report
    except BaseException as exc:
        report.update(status="error", error_type=type(exc).__name__)
        store.finish_without_acceptance("error")
        raise
    finally:
        if transport is not None:
            transport.close()
        atomic_json(report_path, report)
