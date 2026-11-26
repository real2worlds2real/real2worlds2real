"""Advance a portable mixed scene with one native Newton solver."""

import argparse
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scene", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--native-runtime", type=Path)
parser.add_argument("--frames", type=int, default=120)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--substeps", type=int)
parser.add_argument("--iterations", type=int)
parser.add_argument("--body-contact-capacity", type=int)
parser.add_argument("--capture-graph", action="store_true")
parser.add_argument("--drive-joints", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--disable-environment-collision", action="store_true")
parser.add_argument("--disable-interobject-collision", action="store_true")
args = parser.parse_args()
if args.native_runtime:
    sys.path.insert(0, str(args.native_runtime.resolve()))

from real2sim_harness.scene_newton_runtime import simulate_scene  # noqa: E402

report = simulate_scene(
    args.scene.parent,
    args.output,
    frames=args.frames,
    fps=args.fps,
    device=args.device,
    substeps=args.substeps,
    iterations=args.iterations,
    body_contact_capacity=args.body_contact_capacity,
    drive_joints=args.drive_joints,
    capture_graph=args.capture_graph,
    disable_environment_collision=args.disable_environment_collision,
    disable_interobject_collision=args.disable_interobject_collision,
)
print("NEWTON_SCENE_RESULT", report["ok"], flush=True)
# Keep failed physical trajectories available for final rendered feedback.
raise SystemExit(0 if report["ok"] else 2)
