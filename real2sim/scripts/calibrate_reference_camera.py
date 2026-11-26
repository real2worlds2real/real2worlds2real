"""Calibrate a ChArUco camera independently of a model session."""
import argparse
import json
from pathlib import Path

from real2sim_harness.artifacts import atomic_json
from real2sim_harness.calibration import calibrate_camera

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=Path, action='append', required=True)
parser.add_argument('--board', type=Path, required=True, help='JSON containing the known board specification')
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--min-views', type=int, default=5)
args = parser.parse_args()
result = calibrate_camera(args.image, json.loads(args.board.read_text()), min_views=args.min_views)
atomic_json(args.output, result)
print(json.dumps(result, indent=2))
raise SystemExit(0 if result['ok'] else 2)
