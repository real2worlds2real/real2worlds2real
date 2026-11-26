"""Run isolated native physics and target rendering through the scene protocol.

Invoke in the target rendering environment. --native-runtime selects an optional
Newton/Warp overlay only for the physics subprocess. Rendering consumes its
trajectory and never imports that overlay or advances physics.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
for name in ("scene", "output", "manifest"):
    parser.add_argument("--" + name, type=Path, required=True)
parser.add_argument("--native-runtime", type=Path)
parser.add_argument("--review-request", type=Path)
parser.add_argument("--frames", type=int, default=120)
parser.add_argument("--resolution", type=int, default=640)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--substeps", type=int)
parser.add_argument("--iterations", type=int)
parser.add_argument("--body-contact-capacity", type=int)
parser.add_argument("--disable-environment-collision", action="store_true")
parser.add_argument("--disable-interobject-collision", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parent
args.output.mkdir(parents=True, exist_ok=True)
fps = 30
if args.review_request:
    review = json.loads(args.review_request.read_text())
    if review.get("schema_version") != 1 or type(review.get("record_video")) is not bool:
        raise ValueError("Invalid scene review request")
    fps = review["fps"]
command = [
    sys.executable,
    str(root / "simulate_newton_scene.py"),
    "--scene",
    str(args.scene),
    "--output",
    str(args.output),
    "--frames",
    str(args.frames),
    "--fps",
    str(fps),
    "--device",
    args.device,
]
for name in ("native_runtime", "substeps", "iterations", "body_contact_capacity"):
    if getattr(args, name) is not None:
        command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
for name in ("disable_environment_collision", "disable_interobject_collision"):
    if getattr(args, name):
        command.append("--" + name.replace("_", "-"))
result = subprocess.run(command, check=False)
if result.returncode not in (0, 2) or not (args.output / "physics_report.json").is_file():
    raise SystemExit(result.returncode or 1)
command = [
    sys.executable,
    str(root / "render_newton_scene.py"),
    "--scene",
    str(args.scene),
    "--output",
    str(args.output),
    "--manifest",
    str(args.manifest),
    "--resolution",
    str(args.resolution),
    "--device",
    args.device,
    "--viz",
    "none",
]
if args.review_request:
    command.extend(["--review-request", str(args.review_request)])
render = subprocess.run(command, check=False)
raise SystemExit(render.returncode or result.returncode)
