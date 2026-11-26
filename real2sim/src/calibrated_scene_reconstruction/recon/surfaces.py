"""Compatibility entry point for calibrated_scene_reconstruction.scene.support.surfaces."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys
import warnings as _warnings

_warnings.warn("calibrated_scene_reconstruction.recon.surfaces moved to calibrated_scene_reconstruction.scene.support.surfaces", DeprecationWarning, stacklevel=2)
if __name__ == "__main__":
    _runpy.run_module("calibrated_scene_reconstruction.scene.support.surfaces", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("calibrated_scene_reconstruction.scene.support.surfaces")
