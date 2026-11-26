from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from .artifacts import atomic_json


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def vggt_paths(root: Path | None = None) -> dict[str, Path]:
    root = (root or project_root()).resolve()
    checkpoint_dir = root / ".tools" / "checkpoints" / "vggt-omega"
    return {
        "root": root,
        "repository": Path(os.environ.get("REAL2SIM_VGGT_REPO", root / ".tools" / "vggt-omega")),
        "python": Path(os.environ.get("REAL2SIM_VGGT_PYTHON", root / ".envs" / "vggt-omega" / "bin" / "python")),
        "runner": root / "scripts" / "run_vggt_omega.py",
        "checkpoint": Path(os.environ.get("REAL2SIM_VGGT_CHECKPOINT", checkpoint_dir / "vggt_omega_1b_512.pt")).resolve(),
        "validation": checkpoint_dir / "validation-report.json",
    }


def _validation_matches_checkpoint(paths: dict[str, Path]) -> bool:
    checkpoint = paths["checkpoint"]
    validation = paths["validation"]
    if not checkpoint.is_file() or not validation.is_file():
        return False
    try:
        report = json.loads(validation.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    stat = checkpoint.stat()
    return bool(
        report.get("ok")
        and report.get("checkpoint") == str(checkpoint)
        and report.get("checkpoint_size") == stat.st_size
        and report.get("checkpoint_mtime_ns") == stat.st_mtime_ns
    )


def vggt_status(root: Path | None = None) -> dict[str, Any]:
    paths = vggt_paths(root)
    code_ready = (paths["repository"] / "vggt_omega" / "models" / "vggt_omega.py").is_file()
    environment_ready = paths["python"].is_file()
    runner_ready = paths["runner"].is_file()
    checkpoint_ready = paths["checkpoint"].is_file()
    real_smoke_passed = _validation_matches_checkpoint(paths)
    return {
        "available": bool(
            code_ready
            and environment_ready
            and runner_ready
            and checkpoint_ready
            and real_smoke_passed
        ),
        "code_ready": code_ready,
        "environment_ready": environment_ready,
        "runner_ready": runner_ready,
        "checkpoint_ready": checkpoint_ready,
        "real_smoke_passed": real_smoke_passed,
        "checkpoint": str(paths["checkpoint"]),
        "access": "ready" if checkpoint_ready else "checkpoint_missing",
        "authorization_checked": False,
        "license": "FAIR noncommercial research license",
        "role": "feed-forward camera and depth prior for images or video",
    }


def _record_validation(paths: dict[str, Path], report: dict[str, Any]) -> None:
    checkpoint = paths["checkpoint"]
    if (
        not report.get("ok")
        or report.get("structural_smoke")
        or report.get("checkpoint") != str(checkpoint)
    ):
        raise RuntimeError("VGGT-Omega report does not validate the configured checkpoint")
    checkpoint_stat = checkpoint.stat()
    atomic_json(
        paths["validation"],
        {
            "schema_version": 1,
            "ok": True,
            "checkpoint": str(checkpoint),
            "checkpoint_size": checkpoint_stat.st_size,
            "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
            "checkpoint_sha256": report.get("checkpoint_sha256"),
            "backend_commit": report.get("backend_commit"),
            "torch": report.get("torch"),
            "cuda_runtime": report.get("cuda_runtime"),
            "gpu": report.get("gpu"),
        },
    )


def run_vggt(
    images: list[Path],
    output: Path,
    *,
    resolution: int = 512,
    root: Path | None = None,
    timeout_seconds: int = 1800,
) -> dict[str, Any]:
    if resolution not in {256, 416, 512}:
        raise ValueError("Resolution must be 256, 416 or 512")
    if not images:
        raise ValueError("At least one image is required")
    missing = [str(path) for path in images if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing reference images: {missing}")

    paths = vggt_paths(root)
    status = vggt_status(root)
    prerequisites = ("code_ready", "environment_ready", "runner_ready", "checkpoint_ready")
    if not all(status[name] for name in prerequisites):
        raise RuntimeError(f"VGGT-Omega is not ready: {json.dumps(status, sort_keys=True)}")

    output = output.resolve()
    report_path = output / "report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        _record_validation(paths, report)
        return {**report, "cached": True}

    command = [
        str(paths["python"]),
        str(paths["runner"]),
        "--checkpoint",
        str(paths["checkpoint"]),
        "--output",
        str(output),
        "--resolution",
        str(resolution),
    ]
    for image in images:
        command.extend(("--image", str(image.resolve())))
    completed = subprocess.run(
        command,
        cwd=paths["root"],
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout)[-4000:]
        raise RuntimeError(f"VGGT-Omega failed with exit {completed.returncode}: {detail}")
    if not report_path.is_file():
        raise RuntimeError("VGGT-Omega exited successfully without a report")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _record_validation(paths, report)
    return {**report, "cached": False}
