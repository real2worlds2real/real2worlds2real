"""Optional target-backend validation for an Astra Blender attempt."""

from __future__ import annotations

import json
import re

from .artifacts import atomic_json, sha256_file
from .blender import FinalRenderResult
from .sim_bundle import prepare_bundle, verify_bundle
from .sim_parameters import load_parameters


def failure_evidence(output):
    """Expose bounded diagnostics to the agent without hiding backend failures."""
    evidence = {}
    for name in ("conversion_validation.json", "rigid_validation.json", "physx_validation.json", "newton_validation.json"):
        path = output / name
        if path.is_file():
            try:
                evidence[name] = json.loads(path.read_text())
            except (ValueError, OSError):
                pass
    evidence["log_tails"] = {
        path.name: path.read_text(errors="replace")[-3000:]
        for path in sorted((output / "conversion_logs").glob("*.log"))[-3:]
    }
    return evidence


def run_simulation(executor, store, *, attempt_id, steps, render_views, parameters_json, assumptions):
    if not re.fullmatch(r"attempt-[0-9]{3}", attempt_id) or steps not in {60, 120, 240, 480}:
        raise ValueError("Choose a saved attempt and 60/120/240/480 simulation steps")
    if not assumptions.strip():
        raise ValueError("Record the purpose and parameter assumptions of this check")
    run = json.loads((store.root / "run.json").read_text())
    if run["asset_contract"]["kind"] not in {"rigid", "articulated", "deformable"}:
        raise ValueError("This validator supports rigid, revolute articulated, and cloth assets; use run_fluid for particle materials")
    attempt = store.attempts_dir / attempt_id
    report_path = attempt / "report.json"
    report = json.loads(report_path.read_text())
    source = attempt / "scene.blend"
    if not report.get("ok") or not source.is_file():
        raise ValueError("Use a successful Blender attempt")
    report["selected_simulation_review"] = "simulation"
    report["simulation"] = {"ok": False, "status": "running", "source_scene_sha256": sha256_file(source)}
    atomic_json(report_path, report)
    index = len(list(attempt.glob("simulator-[0-9][0-9][0-9]"))) + 1
    output = attempt / f"simulator-{index:03d}"
    parameters = json.loads(parameters_json) if parameters_json else None
    store.append_event({"type": "simulation_review_started", "attempt_id": attempt_id})
    bundle = prepare_bundle(source, output, executor.blender, steps=steps, render=render_views,
                            timeout=executor.timeout_seconds, physics_parameters=parameters)
    detail = {"ok": bool(bundle.get("ok")), "directory": str(output.relative_to(attempt)),
              "assumptions": assumptions, "source_scene_sha256": sha256_file(source),
              "bundle_manifest": str((output / "asset_bundle.json").relative_to(attempt)),
              "bundle_sha256": sha256_file(output / "asset_bundle.json")}
    images = []
    if detail["ok"]:
        detail["integrity"] = verify_bundle(output)
        detail["ok"] = detail["integrity"]["ok"]
        detail["parameters"] = load_parameters(output)
        detail["conversion_validation"] = json.loads((output / "conversion_validation.json").read_text())
        detail["validation"] = json.loads((output / bundle["validation"]).read_text())
        if render_views:
            images = [output / "isaac_views" / name for name in ("front.png", "side.png", "top.png")]
        detail["images"] = [str(path.relative_to(attempt)) for path in images]
        detail["visual_render_verified"] = bool(bundle.get("visual_render_verified"))
    else:
        detail["error"] = bundle.get("error")
        detail["failed_stage"] = bundle.get("failed_stage", "unknown")
        detail["diagnostics"] = failure_evidence(output)
        images = [output / name for name in bundle.get("failure_images", [])
                  if (output / name).resolve().is_relative_to(output.resolve()) and (output / name).is_file()]
        detail["images"] = [str(path.relative_to(attempt)) for path in images]
        if bundle.get("failure_render_error"):
            detail["failure_render_error"] = bundle["failure_render_error"]
        stage = detail["failed_stage"]
        if stage == "simulation_validation":
            detail["failure_scope"] = "operability_check_failed_origin_unresolved"
            detail["next_action"] = "Inspect diagnostics and conversion evidence before attributing failure to the source. Correct a demonstrated source or parameter defect and resubmit; report integration defects explicitly. Do not deliver while validation fails."
        else:
            detail["failure_scope"] = "conversion_or_runtime"
            detail["next_action"] = "Inspect the failing integration stage and logs. Do not blindly change source geometry or material parameters to hide a conversion or runtime defect. Resubmit after the actual cause is addressed; delivery remains blocked."
    report["simulation"] = detail
    report.setdefault("artifacts", {})[detail["bundle_manifest"]] = detail["bundle_manifest"]
    atomic_json(report_path, report)
    store.append_event({"type": "simulation_review_finished", "attempt_id": attempt_id, "ok": detail["ok"]})
    return FinalRenderResult(attempt_id, detail["ok"], detail, images)
