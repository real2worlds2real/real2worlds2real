"""Source-bound tetrahedral solid conversion, native review and material replay."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from .artifacts import atomic_json, sha256_file
from .blender import FinalRenderResult


def run_solids(executor, store, *, attempt_id, frames, assumptions):
    if (
        not re.fullmatch(r"attempt-[0-9]{3}", attempt_id)
        or frames not in {120, 144, 192, 240}
        or not assumptions.strip()
    ):
        raise ValueError(
            "Choose a saved solid attempt, 120/144/192/240 frames and explicit assumptions"
        )
    if json.loads((store.root / "run.json").read_text())["asset_contract"]["kind"] != "solid":
        raise ValueError("Declare asset kind solid before volumetric acceptance")
    attempt = store.attempts_dir / attempt_id
    report = json.loads((attempt / "report.json").read_text())
    source = attempt / "scene.blend"
    if not report.get("ok") or not source.is_file():
        raise ValueError("Use a successful saved Blender attempt")
    root = Path(__file__).resolve().parents[2]
    folder = attempt / f"solid-{len(list(attempt.glob('solid-[0-9][0-9][0-9]'))) + 1:03d}"
    folder.mkdir()
    digest = sha256_file(source)
    detail = {
        "ok": False,
        "directory": str(folder.relative_to(attempt)),
        "source_scene_sha256": digest,
        "assumptions": assumptions,
        "attempt_id": attempt_id,
        "evidence_directory": str(folder),
    }
    commands = [
        (
            "export",
            [
                str(executor.blender),
                "--background",
                str(source),
                "--python",
                str(root / "scripts/export_blend_solid.py"),
                "--",
                str(folder),
            ],
        ),
        (
            "meshing_binding",
            [
                sys.executable,
                str(root / "scripts/prepare_solid_native.py"),
                str(folder),
                "--frames",
                str(frames),
            ],
        ),
        (
            "simulation",
            [
                sys.executable,
                str(root / "scripts/simulate_deformable_behavior.py"),
                str(folder / "behavior_request.json"),
                "--output",
                str(folder),
            ],
        ),
        (
            "preview",
            [
                str(executor.blender),
                "--background",
                str(source),
                "--python",
                str(root / "scripts/render_deformable_behavior.py"),
                "--",
                str(folder),
                "--source-replay",
            ],
        ),
    ]
    images = []
    stage = "initialization"
    store.append_event({"type": "solid_started", "attempt_id": attempt_id})
    try:
        for stage, command in commands:
            with (folder / f"{stage}.log").open("w") as log:
                process = subprocess.run(
                    command,
                    cwd=root,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=executor.timeout_seconds,
                    check=False,
                )
            if process.returncode and not (
                stage == "simulation"
                and process.returncode == 2
                and (folder / "motion.npz").is_file()
            ):
                raise RuntimeError(
                    f"Solid {stage} failed: "
                    + (folder / f"{stage}.log").read_text(errors="replace")[-5000:]
                )
        native = json.loads((folder / "validation.json").read_text())
        replay = json.loads((folder / "preview_report.json").read_text())
        if (
            native.get("source_scene_sha256") != digest
            or replay.get("source_scene_sha256") != digest
            or sha256_file(source) != digest
        ):
            raise ValueError("Solid evidence does not match the saved source")
        detail.update(native)
        detail.update(
            ok=bool(native["ok"] and replay["ok"] and replay.get("source_material_replay")),
            cache_verification=replay,
            delivery_scope="Source-bound native volumetric solid and source-material replay; no robot task or physical calibration.",
        )
        detail["cache_manifest"] = {
            str(path.relative_to(attempt)): sha256_file(path)
            for path in folder.iterdir()
            if path.is_file() and path.suffix in {".json", ".npz"}
        }
        images = [folder / name for name in replay["images"]]
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        detail.update(
            ok=False,
            failed_stage=stage,
            error=str(error),
            next_action="Repair source geometry, selections, material assumptions or test controls and rerun native solid review.",
        )
        if (folder / "validation.json").is_file():
            detail["failed_native_report"] = json.loads((folder / "validation.json").read_text())
    report["solid"] = detail
    atomic_json(attempt / "report.json", report)
    atomic_json(folder / "review_report.json", detail)
    store.append_event({"type": "solid_finished", "attempt_id": attempt_id, "ok": detail["ok"]})
    return FinalRenderResult(attempt_id, detail["ok"], detail, images)


def prepare_solid_motion(executor, store, attempt_id, report):
    attempt = store.attempts_dir / attempt_id
    detail = report["solid"]
    source = attempt / "scene.blend"
    folder = (attempt / detail["directory"]).resolve()
    if (
        not detail.get("ok")
        or not folder.is_relative_to(attempt.resolve())
        or detail.get("source_scene_sha256") != sha256_file(source)
        or not detail.get("cache_manifest")
    ):
        raise ValueError("Use a source-matched successful native solid cache")
    for name, digest in detail["cache_manifest"].items():
        path = (attempt / name).resolve()
        if (
            not path.is_relative_to(attempt.resolve())
            or not path.is_file()
            or sha256_file(path) != digest
        ):
            raise ValueError("Native solid construction or motion changed")
    output = (
        attempt
        / f"motion-review-{len(list(attempt.glob('motion-review-[0-9][0-9][0-9]'))) + 1:03d}"
    )
    output.mkdir()
    root = Path(__file__).resolve().parents[2]
    commands = [
        (
            "render",
            [
                str(executor.blender),
                "--background",
                str(source),
                "--python",
                str(root / "scripts/render_deformable_behavior.py"),
                "--",
                str(folder),
                "--source-replay",
                "--mode",
                "render",
                "--output",
                str(output),
                "--resolution",
                "768",
                "--samples",
                "32",
            ],
        ),
        (
            "encode",
            [
                sys.executable,
                str(root / "scripts/encode_object_demo.py"),
                str(output),
                "--name",
                "candidate_motion",
            ],
        ),
    ]
    for stage, command in commands:
        with (output / f"{stage}.log").open("w") as log:
            subprocess.run(
                command,
                cwd=root,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=executor.timeout_seconds,
                check=True,
            )
    if sha256_file(source) != detail["source_scene_sha256"]:
        raise ValueError("Solid source changed during movie rendering")
    review = {
        "status": "awaiting_agent_review",
        "directory": str(output.relative_to(attempt)),
        "source_scene_sha256": sha256_file(source),
        "motion_sha256": sha256_file(folder / "motion.npz"),
        "video": str((output / "candidate_motion.mp4").relative_to(attempt)),
        "frames": detail["frames"],
        "complete_video_encoded": True,
        "backend": detail["backend"],
    }
    report["motion_review"] = review
    atomic_json(attempt / "report.json", report)
    return FinalRenderResult(attempt_id, True, review, [])
