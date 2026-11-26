"""Reference computation and evidence presentation, independent of model transport."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from .agent import image_block
from .artifacts import RunStore, atomic_json
from .camera import solve_pnp
from .multiview import run_colmap
from .prior_alignment import align_vggt_prior
from .reference import analyze_reference_images
from .vggt import run_vggt, vggt_status


def _workspace_script(workspace: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise ValueError("script_path must be a workspace-relative path")
    candidate = (workspace / relative).resolve()
    try:
        candidate.relative_to(workspace.resolve())
    except ValueError as exc:
        raise ValueError("script_path escapes the workspace") from exc
    if candidate.suffix != ".py" or not candidate.is_file():
        raise ValueError("script_path must name an existing .py file in the workspace")
    return candidate


def _compact_attempt_summary(result) -> dict[str, Any]:
    summary = result.summary
    geometry = summary.get("geometry") or {}
    articulation = summary.get("articulation") or {}
    silhouette = summary.get("silhouette") or {}
    camera_alignment = summary.get("camera_alignment_previews") or []
    compact = {
        "attempt_id": result.attempt_id,
        "ok": result.ok,
        "geometry": {
            key: geometry.get(key)
            for key in ("mesh_objects", "vertices", "polygons", "materials", "size")
        },
        "contract": summary.get("contract"),
        "export": summary.get("export"),
        "articulation": {
            "ok": articulation.get("ok"),
            "required": articulation.get("required"),
            "joint_count": articulation.get("joint_count"),
            "issues": [
                {"joint": joint["name"], "errors": joint.get("schema_errors", [])}
                for joint in articulation.get("joints", []) if not joint.get("valid")
            ],
        },
        "silhouette": {
            "ok": silhouette.get("ok"),
            "mean_iou": silhouette.get("mean_iou"),
            "missing": silhouette.get("missing", []),
        }
        if silhouette
        else None,
        "available_inspections": [
            name
            for name, available in (
                ("inspect_silhouette", bool(silhouette)),
                ("inspect_articulation", bool(articulation.get("joint_count"))),
                ("inspect_camera_alignment", bool(camera_alignment)),
            )
            if available
        ],
        "error": summary.get("error"),
    }
    if not result.ok:
        compact["stdout_tail"] = summary.get("stdout_tail", "")[-2000:]
        compact["stderr_tail"] = summary.get("stderr_tail", "")[-2000:]
    return compact


def _attempt_items(result) -> list[dict[str, Any]]:
    return [
        {"type": "inputText", "text": json.dumps(_compact_attempt_summary(result), ensure_ascii=False)},
        *[
            {"type": "inputImage", "imageUrl": image_block(path)["image_url"]}
            for path in result.images
        ],
    ]


def _final_render_items(result) -> list[dict[str, Any]]:
    items = [{"type": "inputText", "text": json.dumps(result.summary, ensure_ascii=False)}]
    captions = {item["path"]: item["caption"] for item in result.summary.get("image_evidence", [])}
    for path in result.images:
        if str(path) in captions:
            items.append({"type": "inputText", "text": captions[str(path)]})
        items.append({"type": "inputImage", "imageUrl": image_block(path)["image_url"]})
    return items


def _attempt_directory(store: RunStore, attempt_id: str) -> Path:
    if not re.fullmatch(r"attempt-[0-9]{3}", attempt_id):
        raise ValueError("Invalid attempt id")
    attempt = store.attempts_dir / attempt_id
    if not attempt.is_dir():
        raise ValueError("Unknown attempt")
    return attempt


def _inspection_items(store: RunStore, attempt_id: str, kind: str) -> list[dict[str, Any]]:
    attempt = _attempt_directory(store, attempt_id)
    report = json.loads((attempt / "report.json").read_text())
    if kind == "silhouette":
        detail = report.get("silhouette")
        paths = sorted(attempt.glob("mask_*.png"))
    elif kind == "articulation":
        detail = report.get("articulation")
        paths = sorted(attempt.glob("joint_*.png"))
    else:
        raise ValueError("Unknown inspection kind")
    if detail is None:
        raise ValueError(f"Attempt has no {kind} evidence")
    return [
        {"type": "inputText", "text": json.dumps(detail, ensure_ascii=False)},
        *[
            {"type": "inputImage", "imageUrl": image_block(path)["image_url"]}
            for path in paths
        ],
    ]


def _reference_paths(store: RunStore) -> list[Path]:
    run = json.loads((store.root / "run.json").read_text())
    return [store.root / reference["path"] for reference in run["references"]]


def _reference_report(store: RunStore) -> dict[str, Any]:
    return analyze_reference_images(_reference_paths(store))


def _solve_reference_camera(store: RunStore, **arguments) -> dict[str, Any]:
    reference_index = arguments.pop("reference_index")
    report = _reference_report(store)
    if not 0 <= reference_index < report["count"]:
        raise ValueError("reference_index is out of range")
    view = report["views"][reference_index]
    width, height = view["display_size"]
    solution = solve_pnp(width=width, height=height, **arguments)
    solution.update(
        {
            "reference_index": reference_index,
            "reference_name": view["name"],
            "reference_sha256": view["sha256"],
            "object_points": arguments["object_points"],
            "image_points": arguments["image_points"],
        }
    )
    from .calibration import comparison_reference
    comparison_reference(store, reference_index, solution)
    output = store.root / "camera_solutions" / f"reference_{reference_index:03d}.json"
    atomic_json(output, solution)
    store.append_event(
        {
            "type": "camera_solved",
            "reference_index": reference_index,
            "mean_reprojection_px": solution["reprojection"]["mean_px"],
        }
    )
    return solution


def camera_alignment_evidence(store: RunStore, attempt_id: str, reference_index: int):
    attempt = _attempt_directory(store, attempt_id)
    references = _reference_paths(store)
    if not 0 <= reference_index < len(references):
        raise ValueError("reference_index is out of range")
    solution_path = store.root / "camera_solutions" / f"reference_{reference_index:03d}.json"
    aligned = attempt / f"aligned_reference_{reference_index:03d}.png"
    mask = attempt / f"aligned_mask_reference_{reference_index:03d}.png"
    if not solution_path.is_file():
        raise ValueError("Reference camera has not been solved")
    if not aligned.is_file() or not mask.is_file():
        raise ValueError("Attempt predates this camera solution; run Blender again first")
    detail = json.loads(solution_path.read_text())
    from .artifacts import sha256_file
    provenance_path = attempt / 'aligned_camera_provenance.json'
    provenance = json.loads(provenance_path.read_text()).get(str(reference_index), {}) if provenance_path.is_file() else {}
    if provenance.get('camera_sha256') != sha256_file(solution_path):
        raise ValueError('Aligned images do not match the current camera solution; rerun Blender')
    if any(provenance.get('images', {}).get(path.name) != sha256_file(path) for path in (aligned, mask)):
        raise ValueError('Aligned camera images changed; rerun Blender')
    comparison = references[reference_index]
    if detail.get('comparison_reference'):
        candidate = (store.root / detail['comparison_reference']['path']).resolve()
        if not candidate.is_relative_to(store.root.resolve()) or sha256_file(candidate) != detail['comparison_reference']['sha256']:
            raise ValueError('Undistorted comparison reference is missing or changed')
        comparison = candidate
    return detail, [comparison, aligned, mask]


def _camera_alignment_items(
    store: RunStore, attempt_id: str, reference_index: int
) -> list[dict[str, Any]]:
    detail, paths = camera_alignment_evidence(store, attempt_id, reference_index)
    return [
        {"type": "inputText", "text": json.dumps(detail, ensure_ascii=False)},
        *[
            {"type": "inputImage", "imageUrl": image_block(path)["image_url"]}
            for path in paths
        ],
    ]


def _run_reference_geometry(store: RunStore, features: str) -> dict[str, Any]:
    if features not in {"sift", "aliked"}:
        raise ValueError("Unsupported COLMAP feature backend")
    output = store.root / "reference_geometry" / f"colmap_{features}"
    report_path = output / "report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text())
        return {**report, "cached": True}
    try:
        report = run_colmap(_reference_paths(store), output, features=features)
        store.append_event(
            {
                "type": "reference_geometry_finished",
                "backend": "pycolmap",
                "features": features,
                "ok": True,
                "registered_images": report["registered_images"],
            }
        )
        return {**report, "cached": False}
    except (ValueError, OSError, RuntimeError) as exc:
        report = {
            "schema_version": 1,
            "ok": False,
            "backend": "pycolmap",
            "features": features,
            "error": f"{type(exc).__name__}: {exc}",
        }
        atomic_json(report_path, report)
        store.append_event(
            {
                "type": "reference_geometry_finished",
                "backend": "pycolmap",
                "ok": False,
                "error": report["error"],
            }
        )
        return {**report, "cached": False}


def _run_reference_prior(store: RunStore, resolution: int) -> dict[str, Any]:
    output = store.root / "reference_geometry" / f"vggt_omega_{resolution}"
    status_report = vggt_status()
    if not status_report["checkpoint_ready"]:
        return {
            "schema_version": 1,
            "ok": False,
            "status": "waiting_for_checkpoint",
            "backend": "vggt-omega",
            "backend_status": status_report,
        }
    try:
        report = run_vggt(_reference_paths(store), output, resolution=resolution)
        store.append_event(
            {
                "type": "reference_prior_finished",
                "backend": "vggt-omega",
                "ok": True,
                "frames": len(report["images"]),
            }
        )
        return report
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        report = {
            "schema_version": 1,
            "ok": False,
            "status": "failed",
            "backend": "vggt-omega",
            "error": f"{type(exc).__name__}: {exc}",
        }
        store.append_event(
            {
                "type": "reference_prior_finished",
                "backend": "vggt-omega",
                "ok": False,
                "status": "failed",
            }
        )
        return report


def _reference_prior_items(
    store: RunStore, report: dict[str, Any], resolution: int
) -> list[dict[str, Any]]:
    items = [{"type": "inputText", "text": json.dumps(report, ensure_ascii=False)}]
    output = store.root / "reference_geometry" / f"vggt_omega_{resolution}"
    panels = report.get("artifacts", {}).get("evidence_panels", []) if report.get("ok") else []
    for index, relative in enumerate(panels):
        path = output / relative
        if not path.is_file():
            continue
        items.extend(
            (
                {
                    "type": "inputText",
                    "text": json.dumps(
                        {
                            "reference_index": index,
                            "panel_order": [
                                "model input",
                                "relative depth (near dark, far bright)",
                                "confidence (higher bright)",
                            ],
                        }
                    ),
                },
                {"type": "inputImage", "imageUrl": image_block(path)["image_url"]},
            )
        )
    return items


def _align_reference_prior(store: RunStore, **arguments) -> dict[str, Any]:
    resolution = arguments.pop("resolution")
    output = store.root / "reference_geometry" / f"vggt_omega_{resolution}"
    result = align_vggt_prior(
        output,
        store.root / "camera_solutions",
        **arguments,
    )
    store.append_event(
        {
            "type": "reference_prior_aligned",
            "backend": "vggt-omega",
            "ok": result["ok"],
            "anchor_reference_index": arguments["reference_index"],
            "written_camera_indices": result.get("written_camera_indices", []),
            "status": result.get("status"),
        }
    )
    return result
