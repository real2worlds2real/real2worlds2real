"""Compatibility alias for the rigid task-check adapter."""
import importlib
import sys
import warnings

warnings.warn("sim.task_checks moved to tasks.rigid.task_checks", DeprecationWarning, stacklevel=2)
sys.modules[__name__] = importlib.import_module("calibrated_scene_reconstruction.tasks.rigid.task_checks")
