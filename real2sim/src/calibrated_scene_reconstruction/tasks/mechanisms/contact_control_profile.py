"""Compatibility entry point for calibrated_scene_reconstruction.assets.contact_control_profile."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("calibrated_scene_reconstruction.assets.contact_control_profile", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("calibrated_scene_reconstruction.assets.contact_control_profile")
