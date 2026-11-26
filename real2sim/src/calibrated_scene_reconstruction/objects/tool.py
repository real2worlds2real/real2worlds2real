"""Compatibility entry point for calibrated_scene_reconstruction.agent.object_tool."""
import importlib
import runpy
import sys
import warnings

warnings.warn("objects.tool moved to agent.object_tool", DeprecationWarning, stacklevel=2)
if __name__ == "__main__":
    runpy.run_module("calibrated_scene_reconstruction.agent.object_tool", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("calibrated_scene_reconstruction.agent.object_tool")
