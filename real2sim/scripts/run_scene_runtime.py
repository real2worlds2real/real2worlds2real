"""Run an operator-configured simulator adapter against a portable scene."""
import argparse
import json

from real2sim_harness.scene_review import run_scene_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_folder")
    parser.add_argument("--runner", required=True)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--check-sources", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--physics-hz", type=int, help="Override a capable runner's physics frequency, preserving duration")
    args = parser.parse_args()
    result = run_scene_runtime(args.scene_folder, args.runner, timeout_seconds=args.timeout,
                               check_sources=args.check_sources, record_video=args.record_video,
                               physics_hz=args.physics_hz)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
