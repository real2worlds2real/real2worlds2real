"""Blender-to-Newton mixed-asset tool with actual motion and final-render feedback."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from .artifacts import atomic_json, sha256_file
from .blender import FinalRenderResult
from .sim_parameters import load_parameters


def run_hybrid(executor, store, *, attempt_id, frames, parameters_json, assumptions):
    if not re.fullmatch(r"attempt-[0-9]{3}", attempt_id) or frames not in {96, 120, 180, 240}:
        raise ValueError("Choose a saved attempt and 96/120/180/240 frames")
    if not assumptions.strip():
        raise ValueError("Record the functional motion and physical assumptions")
    run = json.loads((store.root / "run.json").read_text())
    if run["asset_contract"]["kind"] != "hybrid":
        raise ValueError("Use run_hybrid for articulated rigid bodies with attached cloth")
    attempt = store.attempts_dir / attempt_id
    report_path = attempt / "report.json"
    report = json.loads(report_path.read_text())
    source = attempt / "scene.blend"
    if not report.get("ok") or not source.is_file():
        raise ValueError("Use a successful saved Blender attempt")
    index = len(list(attempt.glob("hybrid-[0-9][0-9][0-9]"))) + 1
    output = attempt / f"hybrid-{index:03d}"
    output.mkdir()
    root = Path(__file__).resolve().parents[2]
    source_digest = sha256_file(source)
    detail = {"ok": False, "directory": str(output.relative_to(attempt)), "source_scene_sha256": source_digest,
              "assumptions": assumptions, "backend": "newton_vbd_native_attachments"}
    images, stage = [], "initialization"

    def runtime(path):
        return str(Path("/workspace/real2sim_harness") / Path(path).resolve().relative_to(root))

    def execute(name, command, *, allow_motion_failure=False):
        nonlocal stage
        stage = name
        with (output / f"{name}.log").open("w") as stream:
            process = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT,
                                     timeout=executor.timeout_seconds, check=False)
        if process.returncode and not (allow_motion_failure and process.returncode == 2):
            tail = (output / f"{name}.log").read_text(errors="replace")[-5000:]
            raise RuntimeError(f"{name} failed ({process.returncode}): {tail}")

    def isaac(name, script, *arguments):
        execute(name, [str(root / "scripts/run_isaac_lab3.sh"), runtime(root / "scripts" / script), *map(str, arguments)])

    store.append_event({"type": "hybrid_started", "attempt_id": attempt_id, "frames": frames})
    try:
        execute("export", [str(executor.blender), "--background", str(source), "--python",
                           str(root / "scripts/export_blend_for_isaac.py"), "--", str(output)])
        parameters = load_parameters(output, create=True)
        if parameters_json:
            overrides = json.loads(parameters_json)
            if not isinstance(overrides, dict):
                raise ValueError("Parameter overrides must be a JSON object")
            for key, value in overrides.items():
                if isinstance(value, dict) and isinstance(parameters.get(key), dict):
                    parameters[key].update(value)
                else:
                    parameters[key] = value
            atomic_json(output / "physics_parameters.json", parameters)
            load_parameters(output)
        isaac("binding", "isaac_surface_binding.py", runtime(output))
        execute("simulation", [str(root / "scripts/run_isaac_lab3.sh"), runtime(root / "scripts/hybrid_python_entry.py"),
            runtime(root / "scripts/simulate_newton_hybrid.py"), runtime(output), "--frames", str(frames)], allow_motion_failure=True)
        detail.update(json.loads((output / "hybrid_validation.json").read_text()))
        detail["parameters"] = load_parameters(output)
        execute("intersections", [str(executor.blender), "--background", str(source), "--python",
                str(root / "scripts/inspect_hybrid_intersections.py"), "--", str(output),
                "--output", str(output / "surface_intersections.json")])
        intersection = json.loads((output / "surface_intersections.json").read_text())
        detail["surface_intersections"] = {key: value for key, value in intersection.items() if key != "failures"}
        failing = intersection["failures"]
        extra_frames = []
        if failing:
            extra_frames = [failing[index]["cache_frame"] for index in
                            sorted({0, len(failing)//3, 2*len(failing)//3, len(failing)-1})]
            worst = max(failing, key=lambda row: sum(item["crossing_triangle_pairs"] for item in row["crossings"]))
            extra_frames.append(worst["cache_frame"])
        execute("preview", [str(executor.blender), "--background", str(source), "--python",
                            str(Path(__file__).with_name("blender_hybrid_harness.py")), "--", str(output),
                            "--sample-frames", json.dumps(extra_frames)])
        replay = json.loads((output / "preview_report.json").read_text())
        detail["cache_verification"] = replay
        images.extend(output / name for name in replay["images"])
        isaac("snapshot", "write_hybrid_snapshot.py", runtime(output))
        isaac("target_render", "render_usd_asset.py", runtime(output / "simulation_final.usda"),
              "--output", runtime(output / "isaac_views"), "--resolution", "640",
              "--viz", "none", "--device", "cuda:0")
        images.extend(output / "isaac_views" / f"{name}.png" for name in ("front", "side", "top"))
        detail["target_render"] = json.loads((output / "isaac_views/render_report.json").read_text())
        if sha256_file(source) != source_digest:
            raise ValueError("The source changed during hybrid review")
        detail["ok"] = bool(detail["ok"] and intersection["ok"] and replay["ok"] and detail["target_render"]["ok"])
        if not detail["ok"]:
            detail["failed_stage"] = "simulation_validation"
            detail["next_action"] = "Inspect motion samples, control tracking, loop/attachment gaps, stretch and surface_intersections.json. It lists every failing cached frame, control/visual layer, rigid object and geometric witness. Repair real crossings; do not hide parts or disable required collision. Successful rendering does not override failed physics."
        detail["cache_manifest"] = {str(path.relative_to(attempt)): sha256_file(path)
            for path in sorted(output.rglob("*")) if path.is_file() and path.suffix != ".log"}
    except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired) as error:
        detail["ok"] = False
        detail["failed_stage"] = stage
        detail["error"] = str(error)
        detail["next_action"] = "Inspect the failing stage before blaming the model. Correct demonstrated source/parameter defects; report integration defects. Delivery remains blocked until a source-matched hybrid review passes."
    detail["images"] = [str(path.relative_to(attempt)) for path in images if path.is_file()]
    report["hybrid"] = detail
    relative = str((output / "review_report.json").relative_to(attempt))
    report.setdefault("artifacts", {})[relative] = relative
    atomic_json(output / "review_report.json", detail)
    atomic_json(report_path, report)
    store.append_event({"type": "hybrid_finished", "attempt_id": attempt_id, "ok": detail["ok"]})
    compact = {key: value for key, value in detail.items() if key not in {"cache_manifest", "rest_matrices"}}
    return FinalRenderResult(attempt_id, detail["ok"], compact, images)
