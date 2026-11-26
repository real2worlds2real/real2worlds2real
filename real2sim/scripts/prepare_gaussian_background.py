"""Run optional Gaussian preparation/training without starting an Astra session."""
import argparse
import json
from pathlib import Path

from real2sim_harness.artifacts import RunStore, atomic_json
from real2sim_harness.gaussian_background import run_background


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / "run.json", {"gaussian_runner": str(args.runner.resolve()) if args.runner else None})
    result = run_background(RunStore(args.output), args.request.resolve().parent,
                            request_path=args.request.name, timeout_seconds=args.timeout)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
