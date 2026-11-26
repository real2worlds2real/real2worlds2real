"""Compatibility alias for calibrated_scene_reconstruction.sim.robot_articulation."""
import importlib as _importlib
import sys as _sys
import warnings as _warnings

_warnings.warn("calibrated_scene_reconstruction.robots.articulation moved to calibrated_scene_reconstruction.sim.robot_articulation", DeprecationWarning, stacklevel=2)
_sys.modules[__name__] = _importlib.import_module("calibrated_scene_reconstruction.sim.robot_articulation")
