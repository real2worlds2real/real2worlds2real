"""Task-bound Blender entry point; policy is created by the host, never by a model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from real2sim_harness.task_execution import run_blender_proxy  # noqa: E402

if __name__ == "__main__":
    policy = sys.argv[1]
    arguments = sys.argv[2:]
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    run_blender_proxy(policy, arguments)
