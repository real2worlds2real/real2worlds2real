from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any, Protocol

from .artifacts import RunStore, atomic_json
from .blender import AttemptResult
from .skills import SkillRegistry

INSTRUCTIONS = """You are the object-generation coding agent for real2sim.
Study the supplied reference images and build one editable, physically plausible 3D object.
You control Blender by submitting complete Python scripts through build_object.
Never merely describe code: use the tool, inspect its real renders and geometry report, then revise when needed.
Prefer clean named parts, modifiers and materials over one opaque blob. Use metres and put the object on z=0.
Geometry must be genuinely three-dimensional and must make sense from unseen views.
Follow run.json's asset_contract: declare real2sim_asset_kind, real2sim_front_axis, real2sim_up_axis and real2sim_units on the scene,
and include every required_parts term in a clear object name. Articulated parts need real joint parents and limits.
Do not use external downloads, network access, pre-existing asset libraries, or image-to-3D services in phase 1.
The host adds neutral cameras, lights and a ground plane after build_scene() returns.
Failed executions and rejected attempts remain evidence. Only call accept_result for a successful attempt whose
three rendered views are adequate for the stated objective. If the budget is insufficient, report that honestly.
"""

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "build_object",
        "description": "Execute a complete Blender Python object-generation script and return real geometry statistics plus three rendered views.",
        "parameters": {
            "type": "object",
            "properties": {
                "plan": {
                    "type": "string",
                    "description": "What geometry/material strategy this attempt uses.",
                },
                "change_summary": {
                    "type": "string",
                    "description": "For revisions, what changed from the prior attempt; use 'initial' first.",
                },
                "script": {
                    "type": "string",
                    "description": "Complete Python source importing bpy and defining def build_scene():. It must construct the object from scratch.",
                },
            },
            "required": ["plan", "change_summary", "script"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "accept_result",
        "description": "Accept one successful Blender attempt as the final phase-1 result.",
        "parameters": {
            "type": "object",
            "properties": {
                "attempt_id": {"type": "string", "pattern": "^attempt-[0-9]{3}$"},
                "reason": {
                    "type": "string",
                    "description": "Evidence-based reason the geometry and all three views are adequate.",
                },
            },
            "required": ["attempt_id", "reason"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


class Executor(Protocol):
    def execute(
        self,
        store: RunStore,
        *,
        script: str,
        plan: str,
        change_summary: str,
    ) -> AttemptResult: ...


def image_block(path: Path) -> dict[str, str]:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "input_image", "image_url": f"data:{mime};base64,{encoded}"}


def attempt_output(result: AttemptResult) -> list[dict[str, str]]:
    return [
        {"type": "input_text", "text": json.dumps(result.summary, ensure_ascii=False)},
        *[image_block(path) for path in result.images],
    ]


def initial_content(store: RunStore, objective: str) -> list[dict[str, str]]:
    run = json.loads((store.root / "run.json").read_text())
    content: list[dict[str, str]] = [
        {
            "type": "input_text",
            "text": json.dumps(
                {
                    "objective": objective,
                    "phase": "single object, Blender code generation and visual iteration",
                    "reference_count": len(run["references"]),
                    "access": {
                        "task_records_read_root":str(store.root),
                        "project_source_root":str(Path(__file__).resolve().parents[2]),
                        "workspace_write_root":str(store.root/"workspace"),
                        "native_shell_available":False,
                        "execution":"Only exposed function tools; generated Blender code uses the task boundary.",
                    },
                },
                ensure_ascii=False,
            ),
        }
    ]
    for reference in run["references"]:
        content.append(image_block(store.root / reference["path"]))
    return content


def run_agent(
    client: Any,
    executor: Executor,
    store: RunStore,
    *,
    model: str,
    objective: str,
    max_turns: int = 8,
    max_output_tokens: int = 16_000,
    reasoning_effort: str = "high",
    auth_dir: Path | None = None,
) -> dict[str, Any]:
    if hasattr(executor, "for_task"):
        executor = executor.for_task(store, auth_dir=auth_dir)
    (store.root/"workspace").mkdir(exist_ok=True)
    if not 1 <= max_turns <= 30:
        raise ValueError("max_turns must be between 1 and 30")
    history: list[Any] = [{"role": "user", "content": initial_content(store, objective)}]
    trace = []
    auth_mode = getattr(client, "auth_mode", "api_key")
    mode = json.loads((store.root / "run.json").read_text()).get("generation_mode", "twin")
    startup = SkillRegistry().startup(mode)
    atomic_json(store.root / "startup_skills.json", {key:value for key,value in startup.items() if key != "instructions"})
    for turn in range(max_turns):
        request: dict[str, Any] = {
            "model": model,
            "instructions": startup["instructions"] + "\n\n" + INSTRUCTIONS,
            "input": history,
            "tools": TOOLS,
            "parallel_tool_calls": False,
            "store": False,
            "reasoning": {"effort": reasoning_effort},
        }
        if auth_mode != "chatgpt_plan":
            request["max_output_tokens"] = max_output_tokens
        response = client.responses.create(**request)
        history.extend(response.output)
        calls = [item for item in response.output if item.type == "function_call"]
        row = {
            "turn": turn,
            "response_id": getattr(response, "id", None),
            "tools": [call.name for call in calls],
        }
        usage = getattr(response, "usage", None)
        if usage is not None and hasattr(usage, "model_dump"):
            row["usage"] = usage.model_dump()
        trace.append(row)
        store.append_event({"type": "model_turn", **row})
        if not calls:
            store.finish_without_acceptance("model_finished_without_acceptance")
            return {
                "status": "model_finished_without_acceptance",
                "text": getattr(response, "output_text", ""),
                "trace": trace,
                "run": str(store.root),
            }
        for call in calls:
            try:
                arguments = json.loads(call.arguments)
                if call.name == "build_object":
                    result = executor.execute(store, **arguments)
                    output = attempt_output(result)
                elif call.name == "accept_result":
                    final = store.accept(**arguments)
                    output = [{"type": "input_text", "text": json.dumps(final, ensure_ascii=False)}]
                    history.append(
                        {"type": "function_call_output", "call_id": call.call_id, "output": output}
                    )
                    return {
                        "status": "accepted",
                        "final": final,
                        "trace": trace,
                        "run": str(store.root),
                    }
                else:
                    raise ValueError(f"Unknown tool: {call.name}")
            except (TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
                output = [
                    {
                        "type": "input_text",
                        "text": json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}),
                    }
                ]
            history.append(
                {"type": "function_call_output", "call_id": call.call_id, "output": output}
            )
    store.finish_without_acceptance("budget_exhausted")
    return {"status": "budget_exhausted", "trace": trace, "run": str(store.root)}
