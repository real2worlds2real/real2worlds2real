"""Compatibility CLI and mutable module alias for the Newton rope backend."""
from pathlib import Path
import runpy
import importlib
import sys

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "tasks" / "rope.py"), run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("calibrated_scene_reconstruction.sim.newton.rope")
