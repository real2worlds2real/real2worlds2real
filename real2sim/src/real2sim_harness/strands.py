"""Agent tool for actual Newton strand simulation and source-matched replay."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from .artifacts import atomic_json, sha256_file
from .blender import FinalRenderResult


def run_strands(executor, store, *, attempt_id, frames, assumptions):
    if not re.fullmatch(r"attempt-[0-9]{3}", attempt_id) or frames not in {96, 120, 144, 192, 240}:
        raise ValueError("Use a saved attempt and 96/120/144/192/240 frames")
    if not assumptions.strip():
        raise ValueError("Record material, endpoint controls and scale assumptions")
    run = json.loads((store.root / "run.json").read_text())
    if run["asset_contract"]["kind"] != "strand":
        raise ValueError("run_strands requires a strand asset contract")
    attempt = store.attempts_dir / attempt_id
    report = json.loads((attempt / "report.json").read_text())
    source = attempt / "scene.blend"
    if not report.get("ok") or not source.is_file():
        raise ValueError("Use a successful Blender attempt")
    folder = attempt / f"strand-{len(list(attempt.glob('strand-[0-9][0-9][0-9]'))) + 1:03d}"
    folder.mkdir()
    root = Path(__file__).resolve().parents[2]
    digest = sha256_file(source)
    detail = {
        "ok": False,
        "directory": str(folder.relative_to(attempt)),
        "source_scene_sha256": digest,
        "assumptions": assumptions,
    }
    stage = "export"
    images = []
    store.append_event({"type": "strand_started", "attempt_id": attempt_id, "frames": frames})
    try:
        commands = [
            (
                "export",
                [
                    str(executor.blender),
                    "--background",
                    str(source),
                    "--python",
                    str(root / "scripts/export_blend_strands.py"),
                    "--",
                    str(folder),
                ],
            ),
            (
                "simulation",
                [
                    sys.executable,
                    str(root / "scripts/simulate_newton_strands.py"),
                    str(folder),
                    "--frames",
                    str(frames),
                ],
            ),
            (
                "preview",
                [
                    str(executor.blender),
                    "--background",
                    str(source),
                    "--python",
                    str(Path(__file__).with_name("blender_strand_harness.py")),
                    "--",
                    str(folder),
                ],
            ),
        ]
        for stage, command in commands:
            with (folder / f"{stage}.log").open("w") as log:
                p = subprocess.run(
                    command,
                    cwd=root,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=executor.timeout_seconds,
                    check=False,
                )
            if p.returncode:
                raise RuntimeError(
                    f"{stage} failed ({p.returncode}): "
                    + (folder / f"{stage}.log").read_text(errors="replace")[-5000:]
                )
        native = json.loads((folder / "strand_validation.json").read_text())
        replay = json.loads((folder / "preview_report.json").read_text())
        detail.update(native)
        detail["cache_verification"] = replay
        if native["source_scene_sha256"] != digest or sha256_file(source) != digest:
            raise ValueError("Source changed during native strand review")
        detail["ok"] = bool(native["ok"] and replay["ok"])
        detail["cache_manifest"] = {
            str(p.relative_to(attempt)): sha256_file(p)
            for p in [
                folder / "strand_configuration.json",
                folder / "strand_validation.json",
                folder / "strand_motion.npz",
            ]
        }
        images = [folder / name for name in replay["images"]]
    except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired) as error:
        detail.update(
            ok=False,
            failed_stage=stage,
            error=str(error),
            next_action="Inspect native rod, collision proxy, material, endpoint and replay diagnostics; repair and repeat. No prescribed deformation can substitute for a successful native simulation.",
        )
        if (folder / "strand_validation.json").is_file():
            detail["failed_native_report"] = json.loads(
                (folder / "strand_validation.json").read_text()
            )
    detail["images"] = [str(p.relative_to(attempt)) for p in images]
    report["strand"] = detail
    atomic_json(folder / "review_report.json", detail)
    atomic_json(attempt / "report.json", report)
    store.append_event({"type": "strand_finished", "attempt_id": attempt_id, "ok": detail["ok"]})
    compact = {k: v for k, v in detail.items() if k not in {"cache_manifest", "configuration"}}
    compact["strands"] = [
        {k: v for k, v in row.items() if k not in {"rest_quaternions", "body_ids"}}
        for row in detail.get("strands", [])
    ]
    if "failed_native_report" in compact:
        failed = compact["failed_native_report"]
        compact["failed_native_report"] = {
            k: v for k, v in failed.items() if k not in {"configuration", "body_ids", "strands"}
        }
        compact["failed_native_report"]["strands"] = [
            {k: v for k, v in row.items() if k not in {"rest_quaternions", "body_ids"}}
            for row in failed.get("strands", [])
        ]
    return FinalRenderResult(attempt_id, detail["ok"], compact, images)
