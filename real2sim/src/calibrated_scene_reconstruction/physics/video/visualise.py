"""Compatibility entry point for calibrated_scene_reconstruction.physics.video.visualize."""
import importlib
import runpy
import sys
import warnings
warnings.warn("calibrated_scene_reconstruction.physics.video.visualise moved to calibrated_scene_reconstruction.physics.video.visualize", DeprecationWarning, stacklevel=2)
if __name__ == "__main__":
    runpy.run_module("calibrated_scene_reconstruction.physics.video.visualize", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("calibrated_scene_reconstruction.physics.video.visualize")
