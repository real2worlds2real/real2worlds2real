"""Compatibility entry point for calibrated_scene_reconstruction.agent.legacy_object_mcp."""
import importlib
import runpy
import sys
import warnings

warnings.warn("objects.mcp_server moved to agent.legacy_object_mcp", DeprecationWarning, stacklevel=2)
if __name__ == "__main__":
    runpy.run_module("calibrated_scene_reconstruction.agent.legacy_object_mcp", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("calibrated_scene_reconstruction.agent.legacy_object_mcp")
