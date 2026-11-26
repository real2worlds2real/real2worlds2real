"""Compatibility module for calibrated_scene_reconstruction.datagen.video."""
import importlib
import runpy
import sys
import warnings

warnings.warn("calibrated_scene_reconstruction.video moved to calibrated_scene_reconstruction.datagen.video", DeprecationWarning, stacklevel=2)
if __name__ == "__main__":
    runpy.run_module("calibrated_scene_reconstruction.datagen.video", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("calibrated_scene_reconstruction.datagen.video")
