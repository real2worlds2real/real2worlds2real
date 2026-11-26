"""Run one of openpi's own scripts (third_party/openpi/scripts: train.py, compute_norm_stats.py, serve_policy.py)
with our training configs registered in its config table (calibrated_scene_reconstruction.policy.openpi), its arguments unchanged.

    envs/openpi/bin/python tools/openpi_run.py third_party/openpi/scripts/train.py pi05_pick_place --exp-name <run>
"""
import runpy
import sys
from pathlib import Path

from calibrated_scene_reconstruction.core.paths import ROOT
if sys.argv[1:] == ["--help"]:
    print("usage: openpi_run.py <upstream-script> [args...]")
    raise SystemExit(0)
import calibrated_scene_reconstruction.policy.openpi  # noqa: E402,F401  (registers the configs)

script = sys.argv[1]
sys.argv = [script, *sys.argv[2:]]
sys.path.insert(0, str(Path(script).resolve().parent))
runpy.run_path(script, run_name="__main__")
