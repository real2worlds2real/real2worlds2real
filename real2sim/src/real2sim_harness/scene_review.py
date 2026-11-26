"""Scene tools, simulator adapter protocol and source-bound final review."""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
from pathlib import Path

from .artifacts import atomic_json, sha256_file
from .scene import compose_scene, verify_scene


def scene_folder(store, scene_id):
    if not isinstance(scene_id, str) or not re.fullmatch(r"scene-[0-9]{3}", scene_id):
        raise ValueError("Choose a retained scene-NNN ID")
    return store.root / "scenes" / scene_id


def assemble_scene(store, workspace, *, manifest_path):
    source = (Path(workspace) / manifest_path).resolve()
    if not source.is_relative_to(Path(workspace).resolve()) or not source.is_file():
        raise ValueError("Use a scene manifest file inside your workspace")
    root = store.root / "scenes"
    index = len(list(root.glob("scene-[0-9][0-9][0-9]"))) + 1
    scene_id = f"scene-{index:03d}"
    result = compose_scene(source, scene_folder(store, scene_id))
    store.append_event({"type": "scene_composed", "scene_id": scene_id, "ok": result["ok"]})
    return {"scene_id": scene_id, "directory": str(scene_folder(store, scene_id)), **result}


def _media_path(folder, value):
    if not isinstance(value, str) or not value:
        raise ValueError("Runtime media must use output-relative paths")
    path = (folder / value).resolve()
    if Path(value).is_absolute() or not path.is_relative_to(folder.resolve()) or not path.is_file():
        raise ValueError("Runtime evidence is missing or outside its output directory")
    return path


def run_scene_runtime(folder, config_path, *, timeout_seconds=900, check_sources=False, record_video=False,
                      physics_hz=None):
    """The caller supplies its simulator runner; this layer selects no engine version."""
    folder, config_path = Path(folder).resolve(), Path(config_path).resolve()
    current = verify_scene(folder, check_sources=check_sources)
    if not current["ok"]:
        raise ValueError("Reassemble changed or failed scene inputs: " + ", ".join(current["failures"]))
    config = json.loads(config_path.read_text())
    bundle = json.loads((folder / "scene_bundle.json").read_text())
    if not isinstance(record_video, bool):
        raise ValueError("record_video must be a boolean")
    if physics_hz is not None:
        from .scene_timing import SceneTiming
        SceneTiming(physics_hz, steps=1, capture_fps=30 if record_video else 0)
    from .scene_runners import select_runner
    selection=select_runner(config,bundle['required_capabilities'],record_video=record_video,
                            configurable_timestep=physics_hz is not None)
    if not selection['ok']:
        return {"ok": False, "status": "runtime_capability_missing", "missing_capabilities": selection['missing_capabilities'],
                'routes_considered':selection['routes_considered'],
                "next_action": "Choose a compatible runner or explicitly adapt the representation; do not weaken acceptance."}
    active=selection['route']
    index = len(list(folder.glob("runtime-[0-9][0-9][0-9]"))) + 1
    output = folder / f"runtime-{index:03d}"
    output.mkdir()
    request = {"schema_version": 1, "record_video": record_video, "fps": 30, "time_basis": "simulation_seconds"}
    if physics_hz is not None:
        request["physics_hz"] = physics_hz
    atomic_json(output / "review_request.json", request)
    request_digest = sha256_file(output / "review_request.json")
    before = sha256_file(folder / "scene.usda")
    bundle_digest, config_digest = sha256_file(folder / "scene_bundle.json"), sha256_file(config_path)
    replacements = {"{scene}": str(folder / "scene.usda"), "{output}": str(output),
                    "{manifest}": str(folder / "request.json"), "{review_request}": str(output / "review_request.json")}
    argv = []
    for argument in active["argv"]:
        for token, value in replacements.items():
            argument = argument.replace(token, value)
        argv.append(argument)
    result = {"schema_version": 1, "ok": False, "status": "runtime_failed", "output": output.name,
              "scene_sha256": before, "bundle_sha256": bundle_digest,
              "runner_config_sha256": config_digest, "runner_config": str(config_path),
              'runner_route':active['id'],
              "record_video_requested": record_video, "physics_hz_requested": physics_hz,
              "review_request_sha256": request_digest}
    try:
        with (output / "runner.log").open("w") as stream:
            process = subprocess.run(argv, cwd=config_path.parent, stdout=stream, stderr=subprocess.STDOUT,
                                     timeout=timeout_seconds, check=False)
        result["returncode"] = process.returncode
        report = json.loads((output / "runtime_report.json").read_text())
        if not isinstance(report, dict) or not isinstance(report.get("videos", []), list):
            raise ValueError("Runtime report must contain an object and a video path list")
        if report.get("frame_sequence") and not report.get("videos"):
            from .scene_video import encode_sequence
            encoded = encode_sequence(output, report["frame_sequence"])
            report.setdefault("videos", []).append(encoded["path"])
            report.setdefault("video_metadata", {})[encoded["path"]] = encoded
            atomic_json(output / "runtime_report.json", report)
        result["report"] = report
        failures = []
        if sha256_file(output / "review_request.json") != request_digest:
            failures.append("review_request_changed_during_runtime")
        if physics_hz is not None:
            timing = report.get("timing", {})
            actual_dt = timing.get("runtime_physics_dt_s") if isinstance(timing, dict) else None
            if (not isinstance(actual_dt, (int, float)) or isinstance(actual_dt, bool)
                    or not math.isfinite(actual_dt)
                    or not math.isclose(actual_dt, 1 / physics_hz, rel_tol=1e-6, abs_tol=1e-12)):
                failures.append("requested_physics_timestep_not_verified")
        external=report.get('runtime_dependencies',[])
        if not isinstance(external,list) or len(external)>50:
            raise ValueError('Runtime dependencies must be a bounded list')
        for dependency in external:
            if not isinstance(dependency,dict) or set(dependency)!={'path','sha256'}:
                raise ValueError('Invalid runtime dependency record')
            path=Path(dependency['path'])
            if not path.is_absolute() or not path.is_file() or sha256_file(path)!=dependency['sha256']:
                failures.append('runtime_dependency_changed')
        if process.returncode or report.get("ok") is not True:
            failures.append("runner_failed")
        if report.get("scene_sha256") != before:
            failures.append("runtime_report_scene_mismatch")
        if report.get("schema_version") != 1 or not isinstance(report.get("runtime"), dict) or not report["runtime"].get("name") or not report["runtime"].get("version"):
            failures.append("missing_runtime_identity")
        checks = report.get("checks", [])
        required = {"scene_loaded", "finite_state", "environment_contacts"}
        if bundle.get('backend') == 'newton' and 'fluid' not in bundle['required_capabilities']:
            required.add('contact_capacity')
        if 'fluid' in bundle['required_capabilities']:
            required.update({'fluid_state','surface_integrity','static_environment_integrity'})
        if "articulated" in bundle["required_capabilities"]:
            required.add("joint_motion")
        if "hybrid" in bundle["required_capabilities"]:
            required.update({"joint_motion", "attachment_integrity"})
        if "deformable" in bundle["required_capabilities"] or "hybrid" in bundle["required_capabilities"]:
            required.add("deformation_integrity")
        for capability in ("rigid_deformable_contact", "rigid_fluid_contact", "deformable_fluid_contact"):
            if capability in bundle["required_capabilities"]:
                required.add(capability)
        if (not isinstance(checks, list) or not checks or not all(isinstance(v, dict) and v.get("ok") is True for v in checks)
                or not required <= {v.get("name") for v in checks if isinstance(v, dict)}):
            failures.append("scene_checks_missing_or_failed")
        expected_ids = {v["id"] for v in bundle["checks"] if "asset_kind" in v}
        if not isinstance(report.get("checked_asset_ids"), list) or set(report["checked_asset_ids"]) != expected_ids:
            failures.append("not_all_assets_checked")
        if bundle.get("support_requirements"):
            from .scene_support import audit_support_requirements
            coverage = audit_support_requirements(report, bundle["support_requirements"])
            result["support_coverage"] = coverage
            if not coverage["ok"]:
                failures.append("required_support_relations_not_verified")
        images, videos = report.get("images", []), report.get("videos", [])
        if not isinstance(images, list) or not isinstance(videos, list) or len(images) + len(videos) > 200:
            raise ValueError("Return bounded lists of rendered evidence")
        media = [_media_path(output, value) for value in [*images, *videos]]
        state_files = report.get("state_files", [])
        if not isinstance(state_files, list) or len(state_files) > 200:
            raise ValueError("Return a bounded list of numerical state evidence")
        state_evidence = []
        for row in state_files:
            if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
                raise ValueError("Numerical state evidence needs a path and source hash")
            path = _media_path(output, row["path"])
            if sha256_file(path) != row["sha256"]:
                raise ValueError("Numerical state evidence does not match its reported hash")
            state_evidence.append(path)
        if not media:
            failures.append("missing_visual_evidence")
        for value in images:
            from PIL import Image
            with Image.open(_media_path(output, value)) as image:
                image.verify()
        if record_video and not videos:
            failures.append("requested_video_missing")
        from .scene_video import inspect_movie
        result["decoded_videos"] = {value: inspect_movie(_media_path(output, value)) for value in videos}
        current = verify_scene(folder, check_sources=check_sources)
        if not current["ok"] or sha256_file(folder / "scene.usda") != before or sha256_file(folder / "scene_bundle.json") != bundle_digest:
            failures.append("scene_changed_during_runtime")
        if sha256_file(config_path) != config_digest:
            failures.append("runner_changed_during_runtime")
        result.update(ok=not failures, status="simulation_verified" if not failures else "runtime_failed", failures=failures,
                      evidence_files={str(p.relative_to(folder)): sha256_file(p) for p in [output / "runtime_report.json", output / "review_request.json", *media, *state_evidence]})
    except (ImportError, OSError, ValueError, RuntimeError, TypeError, subprocess.TimeoutExpired) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["log_tail"] = (output / "runner.log").read_text(errors="replace")[-2500:] if (output / "runner.log").is_file() else ""
    atomic_json(output / "review.json", result)
    atomic_json(folder / "latest_runtime.json", {"path": f"{output.name}/review.json", "sha256": sha256_file(output / "review.json")})
    return result


def _current_runtime(folder):
    current = verify_scene(folder, check_sources=True)
    if not current["ok"]:
        raise ValueError("Scene package or inputs changed; reassemble and rerun")
    latest = json.loads((folder / "latest_runtime.json").read_text())
    path = _media_path(folder, latest["path"])
    if sha256_file(path) != latest["sha256"]:
        raise ValueError("Runtime review changed; rerun")
    review = json.loads(path.read_text())
    if not review.get("ok") or review["scene_sha256"] != sha256_file(folder / "scene.usda") or review["bundle_sha256"] != sha256_file(folder / "scene_bundle.json"):
        raise ValueError("A successful current scene runtime is required")
    for name, digest in review.get("evidence_files", {}).items():
        if sha256_file(_media_path(folder, name)) != digest:
            raise ValueError("Scene evidence changed; rerun")
    for dependency in review['report'].get('runtime_dependencies',[]):
        path=Path(dependency['path'])
        if not path.is_file() or sha256_file(path)!=dependency['sha256']:
            raise ValueError('Runtime dependency changed; rerun')
    config = Path(review["runner_config"])
    if not config.is_file() or sha256_file(config) != review["runner_config_sha256"]:
        raise ValueError("Runner configuration changed; rerun")
    return review


def run_scene_simulation(store, *, scene_id, timeout_seconds=900, record_video=False, physics_hz=None):
    # Runner selection belongs to the operator, never to model-generated shell strings.
    run = json.loads((store.root / "run.json").read_text())
    config = run.get("scene_runner") or os.environ.get("REAL2SIM_SCENE_RUNNER")
    if not config:
        raise ValueError("Configure --scene-runner or REAL2SIM_SCENE_RUNNER for your simulator installation")
    folder = scene_folder(store, scene_id)
    result = run_scene_runtime(folder, config, timeout_seconds=timeout_seconds, check_sources=True,
                               record_video=record_video, physics_hz=physics_hz)
    result["scene_id"] = scene_id
    result["video_sources"] = []
    if result.get("report"):
        from .video_review import register_video
        for index, name in enumerate(result.get("decoded_videos", {})):
            record = register_video(store, _media_path(folder / result["output"], name),
                                    video_id=f"{scene_id}-{result['output']}-{index}", role="scene_simulation")
            result["video_sources"].append(record)
    return result


def accept_scene(store, *, scene_id, reason):
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Record your visual and functional scene verdict")
    folder = scene_folder(store, scene_id)
    runtime = _current_runtime(folder)
    record=json.loads((store.root/'run.json').read_text())
    configured=record.get('scene_runner') or os.environ.get('REAL2SIM_SCENE_RUNNER')
    if configured and Path(configured).resolve()!=Path(runtime['runner_config']).resolve():
        raise ValueError('Configured scene runner changed; rerun this scene with the current operator adapter')
    from .scene_cousins import validate_variant
    cousin = validate_variant(store, scene_id)
    # Retain the verdict for this exact trial even if a later run changes latest_runtime.json.
    trial = folder / runtime["output"] / "review.json"
    evidence = {str(folder / name): digest for name, digest in runtime["evidence_files"].items()}
    evidence[str(trial)] = sha256_file(trial)
    evidence[runtime["runner_config"]] = runtime["runner_config_sha256"]
    evidence.update({pin["path"]: pin["sha256"] for pin in runtime["report"].get("runtime_dependencies", [])})
    final = {"status": "accepted", "scene_id": scene_id, "reason": reason,
             "scene": str((folder / "scene.usda").relative_to(store.root)),
             "scene_bundle_sha256": sha256_file(folder / "scene_bundle.json"),
             "runtime": runtime["report"]["runtime"], "runtime_review": trial.relative_to(folder).as_posix(),
             "evidence_files": evidence, "real_physics_calibrated": False}
    if cousin is not None:
        final['cousin'] = cousin
    atomic_json(store.root / "final_scene.json", final)
    run = json.loads((store.root / "run.json").read_text())
    run.update(status="accepted", final_scene="final_scene.json")
    atomic_json(store.root / "run.json", run)
    store.append_event({"type": "scene_accepted", "scene_id": scene_id})
    return final


def reopen_scene_review(store, *, reason):
    """Operator-requested follow-up review, retaining the previous accepted state."""
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Describe the additional evidence or reported scene defect")
    run_path, final_path = store.root / "run.json", store.root / "final_scene.json"
    run = json.loads(run_path.read_text())
    if run.get("status") != "accepted" or not final_path.is_file():
        raise ValueError("Only an accepted scene can be reopened")
    final = json.loads(final_path.read_text())
    root = store.root / "scene-reviews"
    output = root / f"review-{len(list(root.glob('review-*'))) + 1:03d}"
    output.mkdir(parents=True)
    atomic_json(output / "previous_final_scene.json", final)
    review = {"status": "needs_scene_review", "scene_id": final["scene_id"], "reason": reason}
    atomic_json(output / "review.json", review)
    report_path = store.root / "codex/report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text())
        atomic_json(output / "previous_codex_report.json", report)
        report.pop("final", None)
        report.update(status="needs_scene_review")
        atomic_json(report_path, report)
    run.pop("final_scene", None)
    run.update(status="needs_scene_review", latest_scene_review=str((output / "review.json").relative_to(store.root)))
    atomic_json(run_path, run)
    final_path.unlink()
    from .steering import queue_instruction
    queue_instruction(store, "The operator reopened this scene for review: " + reason
                      + ". Preserve retained assets and completed evidence. Inspect the new evidence, repair demonstrated defects, and explicitly accept_scene only when satisfied.", source="runtime")
    store.append_event({"type": "scene_review_reopened", **review})
    return review
