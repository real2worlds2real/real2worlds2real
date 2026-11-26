"""Compatibility for the live inventory CLI; cousin-edit experiments are archived under attic/scene."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys
import warnings as _warnings

_warnings.warn("Use calibrated_scene_reconstruction.scene.background.splat.inventory for things; splat cousin editing is archived", DeprecationWarning, stacklevel=2)
if __name__ == "__main__":
    _runpy.run_module("calibrated_scene_reconstruction.scene.background.splat.inventory", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("calibrated_scene_reconstruction.scene.background.splat.inventory")
