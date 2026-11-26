"""Generic impact-fracture tool and source-bound native evidence."""

import json
import re
import subprocess
import sys
from pathlib import Path

from .artifacts import atomic_json, sha256_file
from .blender import FinalRenderResult


def run_fracture(executor, store, *, attempt_id, frames, assumptions):
    if (
        not re.fullmatch(r"attempt-[0-9]{3}", attempt_id)
        or frames not in {144, 192, 240}
        or not assumptions.strip()
    ):
        raise ValueError(
            "Choose a saved attempt, 144/192/240 frames, and explicit fracture assumptions"
        )
    run = json.loads((store.root / "run.json").read_text())
    if run["asset_contract"]["kind"] != "fracture":
        raise ValueError("Declare asset kind fracture before native impact testing")
    attempt = store.attempts_dir / attempt_id
    report = json.loads((attempt / "report.json").read_text())
    source = attempt / "scene.blend"
    if not report.get("ok") or not source.is_file():
        raise ValueError("Use a successful saved Blender attempt")
    folder = attempt / f"fracture-{len(list(attempt.glob('fracture-[0-9][0-9][0-9]'))) + 1:03d}"
    folder.mkdir()
    root = Path(__file__).resolve().parents[2]
    digest = sha256_file(source)
    detail = {
        "ok": False,
        "directory": str(folder.relative_to(attempt)),
        "source_scene_sha256": digest,
        "assumptions": assumptions,
    }
    images = []
    stage = "export"
    commands = [
        (
            "export",
            [
                str(executor.blender),
                "-b",
                str(source),
                "--python-exit-code",
                "1",
                "--python",
                str(root / "scripts/export_blend_fracture.py"),
                "--",
                str(folder),
            ],
        ),
        (
            "simulation",
            [
                str(root / "scripts/run_isaac_lab3.sh"),
                "/workspace/real2sim_harness/scripts/simulate_physx_fracture.py",
                "/workspace/real2sim_harness/" + str(folder.relative_to(root)),
                "--frames",
                str(frames),
                "--viz",
                "none",
                "--device",
                "cpu",
            ],
        ),
        (
            "preview",
            [
                str(executor.blender),
                "-b",
                str(folder / "fractured.blend"),
                "--python",
                str(root / "src/real2sim_harness/blender_fracture_harness.py"),
                "--",
                str(folder),
            ],
        ),
    ]
    store.append_event({"type": "fracture_started", "attempt_id": attempt_id})
    try:
        for stage, command in commands:
            with (folder / f"{stage}.log").open("w") as log:
                result = subprocess.run(
                    command,
                    cwd=root,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=executor.timeout_seconds,
                    check=False,
                )
            if result.returncode:
                raise RuntimeError(
                    stage
                    + " failed: "
                    + (folder / f"{stage}.log").read_text(errors="replace")[-5000:]
                )
        native = json.loads((folder / "fracture_validation.json").read_text())
        replay = json.loads((folder / "preview_report.json").read_text())
        detail.update({k: v for k, v in native.items() if k != "final_poses"})
        detail["cache_verification"] = replay
        detail["ok"] = bool(native["ok"] and replay["ok"])
        if digest != sha256_file(source) or digest != native["source_scene_sha256"]:
            raise ValueError("Source changed during fracture test")
        detail["cache_manifest"] = {
            str(p.relative_to(attempt)): sha256_file(p)
            for p in [
                folder / "fracture_configuration.json",
                folder / "fracture_validation.json",
                folder / "fracture_motion.npz",
                folder / "fractured.blend",
            ]
        }
        images = [folder / name for name in replay["images"]]
    except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired) as error:
        detail.update(
            ok=False,
            failed_stage=stage,
            error=str(error),
            next_action="Inspect closed material volumes, shard coverage, impact threshold, collision geometry and native settling. Repair and repeat; do not substitute an authored explosion animation.",
        )
        if (folder / "fracture_validation.json").is_file():
            failed = json.loads((folder / "fracture_validation.json").read_text())
            detail["failed_native_report"] = {k: v for k, v in failed.items() if k != "final_poses"}
    detail["images"] = [str(p.relative_to(attempt)) for p in images]
    report["fracture"] = detail
    atomic_json(folder / "review_report.json", detail)
    atomic_json(attempt / "report.json", report)
    store.append_event({"type": "fracture_finished", "attempt_id": attempt_id, "ok": detail["ok"]})
    return FinalRenderResult(attempt_id, detail["ok"], detail, images)


def prepare_fracture_motion(executor, store, attempt_id, report):
    attempt = store.attempts_dir / attempt_id
    native = report["fracture"]
    folder = (attempt / native["directory"]).resolve()
    if (
        not folder.is_relative_to(attempt.resolve())
        or not native.get("ok")
        or sha256_file(attempt / "scene.blend") != native["source_scene_sha256"]
    ):
        raise ValueError("Use a source-matched successful fracture cache")
    for name, digest in native["cache_manifest"].items():
        path = (attempt / name).resolve()
        if not path.is_relative_to(attempt.resolve()) or sha256_file(path) != digest:
            raise ValueError("Fracture evidence changed")
    output = (
        attempt
        / f"motion-review-{len(list(attempt.glob('motion-review-[0-9][0-9][0-9]'))) + 1:03d}"
    )
    output.mkdir()
    root = Path(__file__).resolve().parents[2]
    for stage, command in [
        (
            "render",
            [
                str(executor.blender),
                "-b",
                str(folder / "fractured.blend"),
                "--python",
                str(root / "src/real2sim_harness/blender_fracture_harness.py"),
                "--",
                str(folder),
                "--mode",
                "render",
                "--output",
                str(output),
                "--resolution",
                "1024",
                "--samples",
                "64",
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
    ]:
        with (output / f"{stage}.log").open("w") as log:
            result = subprocess.run(
                command,
                cwd=root,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=executor.timeout_seconds,
                check=False,
            )
        if result.returncode:
            raise RuntimeError(
                "Fracture movie failed: "
                + (output / f"{stage}.log").read_text(errors="replace")[-4000:]
            )
    review = {
        "status": "awaiting_agent_review",
        "directory": str(output.relative_to(attempt)),
        "backend": native["backend"],
        "source_scene_sha256": native["source_scene_sha256"],
        "motion_sha256": native["motion_sha256"],
        "frames": native["frames"],
        "video": str((output / "candidate_motion.mp4").relative_to(attempt)),
        "complete_video_encoded": True,
        "review_policy": "Agent chooses viewing coverage and judges impact, shard motion, surfaces and settling; encoding is not acceptance.",
    }
    report["motion_review"] = review
    atomic_json(attempt / "report.json", report)
    return FinalRenderResult(attempt_id, True, review, [])
