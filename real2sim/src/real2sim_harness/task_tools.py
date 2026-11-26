"""Domain operations for a caller-owned coding agent; no model sessions are launched."""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .artifacts import RunStore, atomic_json
from .capabilities import TASK_TOOLS, tool_enabled
from .observation_tools import OPERATIONS
from .reference_tools import (
    _align_reference_prior,
    _run_reference_geometry,
    _run_reference_prior,
    _solve_reference_camera,
    _workspace_script,
)


@dataclass
class ToolOutcome:
    data: dict[str, Any]
    images: list[Path] = field(default_factory=list)

    @property
    def summary(self):
        return self.data

    @property
    def ok(self):
        return bool(self.data.get("ok", True))

    def document(self):
        return {**self.data, "images": [str(p.resolve()) for p in self.images]}


def outcome(result):
    if hasattr(result, "summary"):
        return ToolOutcome(
            {**result.summary, "ok": result.ok, "attempt_id": result.attempt_id},
            list(result.images),
        )
    return ToolOutcome({"ok": True, **result})


NATIVE_HELPERS = {
    "search_skills", "read_skill", "inspect_references", "inspect_silhouette",
    "inspect_articulation", "list_video_sources", "inspect_video", "promote_video_frames",
}

# Both model transports and caller-owned sessions invoke these same services.
SHARED_OPERATIONS = {
    "declare_asset_contract", "run_blender", "render_final", "inspect_surface",
    "run_deformation", "run_strands", "run_solids", "run_fracture", "run_fluid",
    "run_simulation", "run_hybrid", "run_rigid_contact_behavior",
    "tetrahedralize_control_mesh", "run_deformable_behavior",
    "declare_measurements", "calibrate_measurement_camera", "measure_reference_board",
    "review_measurements", *OPERATIONS,
}


def available_tools(mode):
    return [row for row in TASK_TOOLS if row["name"] not in NATIVE_HELPERS and tool_enabled(mode, row["name"])]


def execute(store: RunStore, executor, workspace: Path, name: str, arguments: dict):
    """Reuse asset services and acceptance gates, returning original reports and file paths."""
    arguments = copy.deepcopy(arguments)
    record = json.loads((store.root / "run.json").read_text())
    mode = record.get("generation_mode", "twin")
    if name not in {row["name"] for row in available_tools(mode)}:
        raise ValueError("Unknown operation or operation unavailable in this task mode: " + name)
    if name in OPERATIONS:
        from .observation_tools import run_tool
        result = run_tool(store, workspace, name, executor=executor, **arguments)
        return ToolOutcome(result, [Path(result["directory"]) / p for p in result.get("preview_images", [])])
    if name in {"tetrahedralize_control_mesh", "run_deformable_behavior"}:
        from .deformable_tools import run_tool
        return outcome(run_tool(store, executor, workspace, name, **arguments))
    if name == "run_rigid_contact_behavior":
        from .rigid_contact_tools import run_tool
        return outcome(run_tool(store, executor, workspace, **arguments))
    if name == "declare_asset_contract":
        result = store.declare_asset_contract(**arguments)
        task = json.loads((workspace / "task.json").read_text())
        task.update(asset_contract=result["asset_contract"], contract_inference_required=False)
        atomic_json(workspace / "task.json", task)
        return outcome(result)
    if name == "run_blender":
        script = _workspace_script(workspace, arguments.pop("script_path"))
        return outcome(executor.execute(store, script=script.read_text(), **arguments))
    executor_operations = {
        "render_final", "run_deformation", "run_strands", "run_solids", "run_fracture",
        "run_fluid", "run_simulation", "run_hybrid", "inspect_surface",
    }
    if name in executor_operations:
        return outcome(getattr(executor, name)(store, **arguments))
    if name == "accept_result":
        requested = arguments.pop("cousin_request", None)
        if requested:
            from .cousins import request_cousins
            request_cousins(store, request=requested)
        return outcome(store.accept(**arguments))
    if name in {"request_cousins", "submit_cousin_plan"}:
        from .cousins import request_cousins, submit_plan
        result = (request_cousins if name == "request_cousins" else submit_plan)(store, **arguments)
        if name == "submit_cousin_plan":
            store.finish_without_acceptance(result["status"])
        return outcome(result)
    if name in {"declare_measurements", "calibrate_measurement_camera", "measure_reference_board", "review_measurements"}:
        from .calibration import calibrate_store, measure_board
        from .measurement import declare_dimensions, review_asset
        if name == "declare_measurements":
            result = declare_dimensions(store, **arguments)
            atomic_json(workspace / "measurement_context.json", result)
            task = json.loads((workspace / "task.json").read_text())
            task["measurement_context"] = "measurement_context.json"
            atomic_json(workspace / "task.json", task)
        elif name == "calibrate_measurement_camera":
            result = calibrate_store(store, **arguments)
        elif name == "measure_reference_board":
            result = measure_board(store, **arguments)
        else:
            result = review_asset(executor, store, **arguments)
        return ToolOutcome(result, [store.root / result["annotation"]] if result.get("annotation") else [])
    if name == "solve_reference_camera":
        return outcome(_solve_reference_camera(store, **arguments))
    if name == "run_reference_geometry":
        return outcome(_run_reference_geometry(store, **arguments))
    if name == "run_reference_prior":
        result = _run_reference_prior(store, **arguments)
        root = store.root / "reference_geometry" / f"vggt_omega_{arguments['resolution']}"
        panels = result.get("artifacts", {}).get("evidence_panels", []) if result.get("ok") else []
        return ToolOutcome(result, [root / p for p in panels])
    if name == "align_reference_prior":
        return outcome(_align_reference_prior(store, **arguments))
    if name == "inspect_camera_alignment":
        from .reference_tools import camera_alignment_evidence
        camera, paths = camera_alignment_evidence(store, **arguments)
        return ToolOutcome(camera, paths)
    if name == "render_motion_video":
        from .motion_review import prepare_candidate_motion
        from .video_review import list_video_sources
        store.accept(arguments["attempt_id"], arguments["assumptions"], commit=False)
        result = outcome(prepare_candidate_motion(executor, store, arguments["attempt_id"]))
        result.data["video_sources"] = list_video_sources(store)
        return result
    from .scene_agent import SCENE_TOOLS
    if name in {row["name"] for row in SCENE_TOOLS}:
        return _scene_operation(store, executor, workspace, name, arguments)
    raise ValueError("Operation has no external-agent handler: " + name)


def _scene_operation(store, executor, workspace, name, arguments):
    from .scene_agent import build_background, build_object
    from .scene_physics import plan_scene_physics
    from .scene_review import accept_scene, assemble_scene, run_scene_simulation

    if name == "plan_scene_physics":
        return outcome(plan_scene_physics(**arguments))
    if name == "build_scene_background":
        result = build_background(executor, store, workspace, **arguments)
        return ToolOutcome(result, [Path(result["directory"]) / p for p in result.get("preview_images", [])] if result.get("ok") else [])
    if name in {"build_scene_object", "build_scene_cousin_object"}:
        # A handoff is task data, not a second hidden model process.
        from .external_agent import prepare_task
        def worker(child_executor, child_store, **options):
            prepare_task(child_store, blender=child_executor, inherit=store)
            return {"status": "awaiting_external_agent"}
        if name == "build_scene_cousin_object":
            from .scene_cousins import build_object as build_cousin_object
            result = build_cousin_object(executor, store, workspace, worker, **arguments)
        else:
            result = build_object(executor, store, workspace, worker, **arguments)
        if result.get("worker_status") == "awaiting_external_agent":
            child = RunStore(Path(result["run"]))
            result.update(ok=True, status="needs_object_work", task=str(child.root / "external/context.json"),
                          next_action="Complete this child task in the current agent session, then repeat this operation with its asset_id to validate and package it.")
        return outcome(result)
    if name == "assemble_scene":
        return outcome(assemble_scene(store, workspace, **arguments))
    if name == "run_scene_simulation":
        result = run_scene_simulation(store, **arguments)
        from .scene_review import _media_path, scene_folder
        root = scene_folder(store, arguments["scene_id"]) / result["output"] if result.get("report") else None
        return ToolOutcome(result, [_media_path(root, p) for p in result.get("report", {}).get("images", [])])
    if name == "run_gaussian_background":
        from .gaussian_background import run_background
        result = run_background(store, workspace, **arguments)
        return ToolOutcome(result, [Path(result["directory"]) / p for p in result.get("preview_images", [])] if result.get("ok") else [])
    if name == "accept_scene":
        return outcome(accept_scene(store, **arguments))
    raise ValueError("Unknown scene operation: " + name)
