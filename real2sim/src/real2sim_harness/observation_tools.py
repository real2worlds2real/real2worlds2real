"""Optional shared object/scene tools, invoked by the agent with retained provenance."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .artifacts import atomic_json, sha256_file

OPERATIONS = {
    "project_motion_to_plane": "motion_geometry",
    "prepare_motion_observations": "motion_observations",
    "compare_physics_observations": "physics_observations",
    "fit_multiview_geometry": "silhouette_fit",
    "align_multiview_camera_rig": "camera_rig_alignment",
    "bake_reference_texture": "reference_texture",
    "recover_support_footprint": "support_footprint",
    "register_metric_reconstruction": "metric_registration",
    "recover_video_cameras": "video_cameras",
    "filter_gaussian_visibility": "gaussian_visibility",
    "segment_reference_views": "segmentation",
    "audit_task_evidence": "evidence_audit",
}


def schemas():
    descriptions = {
        "audit_task_evidence": "Audit retained evidence: convex sphere-start clearance, named rigid-body window statistics, command/readback tracking, relative screw motion, or declared support coverage. Requires explicit units/identity assumptions and retained data. Read review-numerical-evidence. Numeric scope only; does not generate geometry, prove contact from fitted motion, or accept an asset.",
        "project_motion_to_plane": "Optionally map image-point tracks to metric plane coordinates using registered fixed/per-frame cameras and an explicit point-on-plane assumption. Reuses camera conventions and metric evidence; does not infer scale or recover general 3D motion. Read calibrate-observed-physics and motion geometry/fitting guidance.",
        "prepare_motion_observations": "Prepare timed image-motion evidence from an existing tracker NPZ, selected per-frame masks or measured image points. Uses original video PTS, explicit frame mapping and missing/occlusion flags; returns full-size overlays and comparison-compatible tracks. Does not run or select a tracker. Read calibrate-observed-physics.",
        "compare_physics_observations": "Compare observed and simulated registered motion tracks/events on an explicitly synchronized clock. Returns coverage, residual curves and worst times; missing rollout evidence cannot score as success. Read calibrate-observed-physics. Does not propose parameters, run a model or accept an asset.",
        "segment_reference_views": "Obtain SAM3 instance masks, scores and conservative association through operator-owned REAL2SIM_SEGMENTATION_RUNNER. Per-view prompt_boxes can supply caller-reviewed inclusion/exclusion boxes when text grounding fails. Use boundaries as evidence, retain candidates for identity review, then inspect current geometry. Read segment-reference-views; no login or weight download.",
        "filter_gaussian_visibility": "Optionally clean an existing Gaussian PLY using source-matched actual renderer particle visibility and/or registered depth free-space evidence. Preserves input and returns per-particle reasons plus a candidate PLY; read prepare-gaussian-background. Frustum counts are not render visibility.",
        "register_metric_reconstruction": "Optionally align a whole reconstructed point/camera frame to metres using observed multi-view ChArUco corners or independently measured 3D landmarks. Returns a robust similarity and transformed camera records without changing inputs; read recover-multiview-geometry.",
        "recover_video_cameras": "Optionally recover calibrated cameras from one or multiple static-scene videos or ordered frame streams. Supports per-stream calibration/self-calibration, sequential/cross-stream/loop matching and retained numeric geometry; read recover-multiview-geometry.",
        "fit_multiview_geometry": "Inspect current NPZ/Blender geometry against registered masks with evaluate_only:true, or refine bounded agent-authored parameters. Returns current-source hashes, per-view overlays and local residuals. Fixed cameras and single-view limits remain explicit; read fit-multiview-geometry. Neither inspection nor fitting is simulator acceptance.",
        "align_multiview_camera_rig": "Align an existing shared relative-camera rig to explicit model-to-image correspondences with one positive similarity. Preserves relative rig geometry; returns candidate model-world cameras and per-view reprojection. Assumed dimensions are not independent metric evidence. Read recover-multiview-geometry; no existing camera or asset is accepted/overwritten.",
        "bake_reference_texture": "Optionally bake posed RGB photographs onto an existing mesh UV atlas with self-visibility, masks/depth, obliqueness and robust color weighting. Read bake-reference-texture for the request schema. Unobserved texture remains explicit; this is not intrinsic albedo recovery.",
        "recover_support_footprint": "Optionally recover observed support-plane footprint from multi-view support/occluder/background labels and registered cameras. Optional depth consistency checks the supplied plane against registered camera-Z depth. Occluded pixels do not vote against the plane, unseen cells stay unknown. Read recover-support-footprint. Provides numeric modeling evidence, not final geometry or collisions.",
    }
    return [
        {
            "type": "function",
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "request_path": {
                        "type": "string",
                        "description": "Workspace-relative request JSON; input files may be in this workspace or retained run.",
                    }
                },
                "required": ["request_path"],
                "additionalProperties": False,
            },
            "strict": True,
        }
        for name, description in descriptions.items()
    ]


def execute(operation, request_path, output):
    from .posed_observations import load_request

    request, base = load_request(request_path)
    if operation == "evidence_audit":
        from .evidence_audit import run_audit
        return run_audit(request, base, output)
    if operation == "motion_geometry":
        from .motion_geometry import prepare
        return prepare(request, base, output)
    if operation == "motion_observations":
        from .motion_observations import prepare
        return prepare(request, base, output)
    if operation == "physics_observations":
        from .physics_inference import compare_files
        return compare_files(request, base, output)
    if operation == "silhouette_fit":
        from .silhouette_fit import fit

        return fit(request, base, output)
    if operation == 'camera_rig_alignment':
        from .camera_rig import align
        return align(request, base, output)
    if operation == "reference_texture":
        from .reference_texture import bake

        return bake(request, base, output)
    if operation == "support_footprint":
        from .support_footprint import recover

        return recover(request, base, output)
    if operation == "metric_registration":
        from .metric_registration import register
        return register(request, base, output)
    if operation == "video_cameras":
        from .video_geometry import recover
        return recover(request, base, output)
    if operation == "gaussian_visibility":
        from .gaussian_visibility import filter_visibility
        return filter_visibility(request, base, output)
    if operation == "segmentation":
        from .segmentation import run_backend
        return run_backend(request_path, output)
    raise ValueError("Unknown observation operation")


def run_tool(store, workspace, name, request_path, *, timeout_seconds=300, executor=None):
    if name not in OPERATIONS:
        raise ValueError("Unknown observation tool")
    workspace = Path(workspace).resolve()
    if Path(request_path).is_absolute():
        raise ValueError("Use a workspace-relative request_path")
    path = (workspace / request_path).resolve()
    if not path.is_relative_to(workspace) or not path.is_file():
        raise ValueError("Request must be an existing file inside the agent workspace")
    request = json.loads(path.read_text())
    if name == 'fit_multiview_geometry':
        # Leave time for retained reports/overlays before the process hard limit.
        reserve = min(30., timeout_seconds * .2)
        budget = request.get('time_budget_seconds', timeout_seconds - reserve)
        if (isinstance(budget, bool) or not isinstance(budget, (int, float))
                or not math.isfinite(budget) or budget <= 0):
            raise ValueError('time_budget_seconds must be finite and positive')
        request['time_budget_seconds'] = min(budget, timeout_seconds - reserve)
    # Agent JSON cannot grant additional host filesystem/runtime access.
    request.pop("blender_executable", None)
    blender_runtime = None
    if request.get("model_format") == "blender":
        if executor is None:
            raise ValueError("Task Blender fitting needs its operator-configured executor")
        blender_runtime = str(getattr(executor, "_original_blender", executor.blender))
        request["blender_executable"] = blender_runtime
    root = store.root.resolve()
    parent = root / "reference_tools"
    parent.mkdir(exist_ok=True)
    number = 1
    while (parent / f"{name}-{number:03d}").exists():
        number += 1
    output = parent / f"{name}-{number:03d}"
    inputs = output / "inputs"
    inputs.mkdir(parents=True)
    retained = {}

    def retain(value):
        if not isinstance(value, str):
            raise ValueError("Input paths must be strings")
        source = (path.parent / value).resolve()
        if (
            not (source.is_relative_to(workspace) or source.is_relative_to(root))
            or not source.is_file()
        ):
            raise ValueError("Inputs must come from this workspace or retained run")
        target = inputs / f"{len(retained):03d}{source.suffix.lower()}"
        shutil.copy2(source, target)
        retained[str(target.relative_to(output))] = {
            "source": str(
                source.relative_to(workspace if source.is_relative_to(workspace) else root)
            ),
            "sha256": sha256_file(target),
        }
        return str(target.relative_to(inputs))

    for view in request.get("views", []):
        for key in ("image", "mask", "ignore", "depth", "particle_visibility"):
            if key in view:
                view[key] = retain(view[key])
    file_keys = ["mesh", "model_script", "surface_points", "gaussians"]
    if name == "audit_task_evidence":
        file_keys.extend(["trajectory", "support_report"])
    if name == "compare_physics_observations":
        file_keys.extend(["observations", "rollout"])
    if name == "project_motion_to_plane":
        file_keys.append("observations")
    if name == "prepare_motion_observations":
        file_keys.append("video")
    for key in file_keys:
        if key in request:
            request[key] = retain(request[key])
    for stream in request.get("streams", []):
        if "video" in stream:
            stream["video"] = retain(stream["video"])
        for frame in stream.get("frames", []):
            frame["image"] = retain(frame["image"])
    if name == "prepare_motion_observations":
        for track in request.get("tracks", []):
            if "archive" in track:
                track["archive"] = retain(track["archive"])
            for sample in track.get("samples", []):
                if "mask" in sample:
                    sample["mask"] = retain(sample["mask"])
    atomic_json(inputs / "request.json", request)
    snapshot_digest = sha256_file(inputs / "request.json")
    def verify_retained_inputs():
        if sha256_file(inputs / "request.json") != snapshot_digest:
            raise RuntimeError("Tool changed its retained request snapshot")
        for relative, item in retained.items():
            target = output / relative
            if not target.is_file() or sha256_file(target) != item["sha256"]:
                raise RuntimeError("Tool changed retained input evidence")
    atomic_json(
        output / "provenance.json",
        {"tool": name, "request_sha256": sha256_file(path), "retained_inputs": retained},
    )
    environment = dict(
        os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]), PYTHONDONTWRITEBYTECODE="1"
    )
    argv = [
        sys.executable,
        "-m",
        "real2sim_harness.observation_tools",
        OPERATIONS[name],
        "--request",
        str(inputs / "request.json"),
        "--output",
        str(output),
    ]
    try:
        if name == "segment_reference_views":
            from .segmentation import run_backend
            result = run_backend(inputs / "request.json", output, run=store.root,
                                 timeout_seconds=timeout_seconds)
            verify_retained_inputs()
            result["directory"] = str(output)
            store.append_event({"type": "observation_tool_finished", "tool": name,
                                "ok": result["ok"], "output": str(output.relative_to(root))})
            return result
        from .task_execution import policy_for, sandbox_command
        execution = policy_for(store.root, writable=[output], executable=blender_runtime)
        argv = sandbox_command(argv, execution, cwd=output)
        environment = execution["environment"]
        with (output / "execution.log").open("w") as log:
            process = subprocess.run(
                argv,
                env=environment,
                cwd=output,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
                check=False,
            )
        if process.returncode or not (output / "report.json").is_file():
            raise RuntimeError(
                "Observation tool failed; inspect retained execution.log and revise inputs/model"
            )
        result = json.loads((output / "report.json").read_text())
        verify_retained_inputs()
    except (subprocess.TimeoutExpired, RuntimeError) as error:
        result = {
            "ok": False,
            "operation": OPERATIONS[name],
            "error": str(error),
            "preview_images": [],
        }
        if name == 'fit_multiview_geometry' and (output / 'search_progress.json').is_file():
            progress = json.loads((output / 'search_progress.json').read_text())
            mesh = output / str(progress.get('mesh', ''))
            if (mesh.parent == output and mesh.is_file()
                    and sha256_file(mesh) == progress.get('mesh_sha256')):
                verify_retained_inputs()
                result['partial_candidate'] = progress
                result['partial_candidate_requires_review'] = True
        atomic_json(output / "report.json", result)
    result["directory"] = str(output)
    store.append_event(
        {
            "type": "observation_tool_finished",
            "tool": name,
            "ok": result["ok"],
            "output": str(output.relative_to(root)),
        }
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=sorted(OPERATIONS.values()))
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    execute(args.operation, args.request, args.output)


if __name__ == "__main__":
    main()
