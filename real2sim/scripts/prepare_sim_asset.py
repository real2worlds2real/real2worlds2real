"""Convert a saved Blender asset into a verified PhysX or Newton bundle."""

import argparse
import json
from pathlib import Path

from real2sim_harness.sim_bundle import prepare_bundle

parser = argparse.ArgumentParser()
parser.add_argument("blend", type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--blender", type=Path, default=Path(".tools/blender/blender"))
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--render", action="store_true")
args = parser.parse_args()
result = prepare_bundle(args.blend, args.output, args.blender, steps=args.steps, render=args.render)
print(json.dumps(result, indent=2))
raise SystemExit(0 if result["ok"] else 2)
