"""Compatibility entry point for calibrated_scene_reconstruction.assets.pinch_friction."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("calibrated_scene_reconstruction.assets.pinch_friction", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("calibrated_scene_reconstruction.assets.pinch_friction")
