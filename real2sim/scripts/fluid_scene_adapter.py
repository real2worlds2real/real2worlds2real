"""Scene protocol adapter for liquids and finite static environment supports.

The operator selects a physics process and Blender executable. All liquid
participants share one MPM execution; render replay consumes its actual surfaces.
"""

import argparse
import json
import subprocess
from pathlib import Path

from real2sim_harness.artifacts import atomic_json, sha256_file
from real2sim_harness.mpm_runtime import physics_command
from real2sim_harness.scene_fluid import assemble_inputs
from real2sim_harness.scene_fluid_review import runtime_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("scene", "output", "manifest", "physics_runner", "blender"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--review-request", type=Path)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--height", type=int)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--stage-timeout", type=int, default=900)
    args = parser.parse_args()
    if (
        not 30 <= args.frames <= 1800
        or not 128 <= args.resolution <= 2048
        or not 1 <= args.samples <= 512
    ):
        raise ValueError("Use bounded frame, resolution and sample budgets")
    if not 1 <= args.stage_timeout <= 3600:
        raise ValueError("Use a bounded stage timeout")
    args.scene, args.output, args.manifest = map(
        lambda p: p.resolve(), (args.scene, args.output, args.manifest)
    )
    if args.manifest != args.scene.parent / "request.json":
        raise ValueError("Use the retained composed-scene manifest")
    request = (
        json.loads(args.review_request.read_text())
        if args.review_request
        else {"schema_version": 1, "record_video": False, "fps": 30}
    )
    if request.get("schema_version") != 1 or type(request.get("record_video")) is not bool:
        raise ValueError("Invalid scene review request")
    fps = request.get("fps", 30)
    if type(fps) is not int or fps <= 0 or 120 % fps:
        raise ValueError("MPM capture FPS must divide 120")
    assemble_inputs(args.scene.parent, args.output)
    runner_digest = sha256_file(args.physics_runner)
    command = physics_command(
        args.physics_runner, args.output / "fluid_configuration.json", args.output / "mpm", args.frames, fps
    )
    process = subprocess.run(
        command, cwd=args.physics_runner.resolve().parent, check=False, timeout=args.stage_timeout
    )
    if process.returncode not in (0, 2) or not (args.output / "mpm/fluid_report.json").is_file():
        raise RuntimeError("MPM scene process failed before producing inspectable state")
    report = runtime_report(args.scene.parent, args.output, expected_frames=args.frames, fps=fps)
    report["runtime_dependencies"] = [
        {"path": str(args.physics_runner.resolve()), "sha256": runner_digest}
    ]
    atomic_json(args.output / "physics_report.json", report)
    script = Path(__file__).resolve().parents[1] / "src/real2sim_harness/blender_scene_fluid_harness.py"
    command = [
        str(args.blender),
        "--background",
        "--python",
        str(script),
        "--",
        "--scene",
        str(args.scene),
        "--output",
        str(args.output),
        "--resolution",
        str(args.resolution),
        "--samples",
        str(args.samples),
    ]
    if request["record_video"]:
        command.append("--record-video")
    if args.height is not None:
        command.extend(["--height", str(args.height)])
    rendered = subprocess.run(command, check=False, timeout=args.stage_timeout)
    if rendered.returncode:
        raise RuntimeError("Whole-scene liquid replay failed; inspect retained physics output")
    raise SystemExit(0 if report["ok"] and process.returncode == 0 else 2)


if __name__ == "__main__":
    main()
