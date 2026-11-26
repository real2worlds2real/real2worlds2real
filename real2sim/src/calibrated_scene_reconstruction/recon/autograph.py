"""Compatibility entry point for calibrated_scene_reconstruction.scene.twin.graph."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys
import warnings as _warnings

_warnings.warn("calibrated_scene_reconstruction.recon.autograph moved to calibrated_scene_reconstruction.scene.twin.graph", DeprecationWarning, stacklevel=2)
if __name__ == "__main__":
    _runpy.run_module("calibrated_scene_reconstruction.scene.twin.graph", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("calibrated_scene_reconstruction.scene.twin.graph")
