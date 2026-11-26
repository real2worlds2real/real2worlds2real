"""Generic Blender material-volume to Newton MPM execution and visual review."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

from .artifacts import atomic_json, sha256_file
from .blender import FinalRenderResult


def run_fluid(executor, store, *, attempt_id: str, frames: int, assumptions: str):
    if not re.fullmatch(r"attempt-[0-9]{3}", attempt_id) or frames not in {48, 72, 96, 120, 144, 192}:
        raise ValueError("Use a saved attempt and 48/72/96/120/144/192 frames")
    if not assumptions.strip():
        raise ValueError("Record the material and scale assumptions")
    attempt = store.attempts_dir / attempt_id
    scene = attempt / "scene.blend"
    report_path = attempt / "report.json"
    report = json.loads(report_path.read_text())
    if not report.get("ok") or not scene.is_file():
        raise ValueError("Use a successful saved Blender attempt")
    number = len(list(attempt.glob("fluid-[0-9][0-9][0-9]"))) + 1
    output = attempt / f"fluid-{number:03d}"
    output.mkdir()
    root = Path(__file__).resolve().parents[2]

    def runtime(path):
        return str(Path("/workspace/real2sim_harness") / Path(path).resolve().relative_to(root))

    def execute(name, command):
        with (output / f"{name}.log").open("w") as stream:
            process = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT,
                                     timeout=executor.timeout_seconds, check=False)
        if process.returncode:
            tail = (output / f"{name}.log").read_text(errors="replace")[-5000:]
            raise RuntimeError(f"{name} failed ({process.returncode}): {tail}")

    detail = {"ok": False, "directory": str(output.relative_to(attempt)), "assumptions": assumptions}
    store.append_event({"type": "fluid_started", "attempt_id": attempt_id, "frames": frames})
    try:
        execute("export", [str(executor.blender), "--background", str(scene), "--python",
                           str(root / "scripts/export_blend_fluid.py"), "--", str(output / "fluid_configuration.json")])
        native_runner = os.environ.get("REAL2SIM_MPM_RUNNER")
        if native_runner:
            from .mpm_runtime import physics_command

            runner_path = Path(native_runner).resolve()
            command = physics_command(runner_path, output / "fluid_configuration.json", output, frames, 24)
            detail["physics_runner_sha256"] = sha256_file(runner_path)
        else:
            command = [str(root / "scripts/run_isaac_lab3.sh"), runtime(root / "scripts/simulate_particle_materials.py"),
                       runtime(output / "fluid_configuration.json"), "--output", runtime(output),
                       "--frames", str(frames), "--fps", "24", "--viz", "none", "--device", "cuda:0"]
        execute("simulation", command)
        execute("preview", [str(executor.blender), "--background", str(scene), "--python",
                            str(Path(__file__).with_name("blender_fluid_harness.py")), "--", str(output),
                            "--mode", "preview", "--resolution", "512", "--samples", "32"])
        simulation = json.loads((output / "fluid_report.json").read_text())
        verification = json.loads((output / "preview_report.json").read_text())
        detail.update(simulation)
        detail["cache_verification"] = verification
        detail["ok"] = bool(simulation.get("ok") and verification.get("ok"))
        detail["cache_manifest"] = {
            str(path.relative_to(attempt)): sha256_file(path)
            for path in [output / "fluid_configuration.json", output / "particle_motion.npz", *sorted((output / "surfaces").glob("*.npz"))]
        }
        detail["images"] = [str((output / name).relative_to(attempt)) for name in verification["images"]]
    except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
        detail["error"] = str(exc)
    report["fluid"] = detail
    for filename in ("review_report.json", "fluid_configuration.json", "simulation.blend"):
        relative = str((output / filename).relative_to(attempt))
        report.setdefault("artifacts", {})[relative] = relative
    atomic_json(report_path, report)
    atomic_json(output / "review_report.json", detail)
    store.append_event({"type": "fluid_finished", "attempt_id": attempt_id, "ok": detail["ok"]})
    compact = {key: value for key, value in detail.items() if key not in {"cache_manifest", "frame_records"}}
    compact["parts"] = [{key: value for key, value in part.items() if key != "history"}
                        for part in detail.get("parts", [])]
    images = [attempt / name for name in detail.get("images", []) if (attempt / name).is_file()]
    return FinalRenderResult(attempt_id, bool(detail["ok"]), compact, images)
