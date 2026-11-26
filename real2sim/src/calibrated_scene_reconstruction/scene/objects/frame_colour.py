"""Compatibility entry point for calibrated_scene_reconstruction.scene.objects.frame_color."""
import importlib
import runpy
import sys
import warnings
warnings.warn("calibrated_scene_reconstruction.scene.objects.frame_colour moved to calibrated_scene_reconstruction.scene.objects.frame_color", DeprecationWarning, stacklevel=2)
if __name__ == "__main__":
    runpy.run_module("calibrated_scene_reconstruction.scene.objects.frame_color", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("calibrated_scene_reconstruction.scene.objects.frame_color")
