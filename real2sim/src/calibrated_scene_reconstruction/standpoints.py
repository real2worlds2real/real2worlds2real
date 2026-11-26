"""Compatibility alias for calibrated_scene_reconstruction.robots.viewpoints."""
import importlib as _importlib
import sys as _sys
import warnings as _warnings

_warnings.warn("calibrated_scene_reconstruction.standpoints moved to calibrated_scene_reconstruction.robots.viewpoints", DeprecationWarning, stacklevel=2)
_sys.modules[__name__] = _importlib.import_module("calibrated_scene_reconstruction.robots.viewpoints")
