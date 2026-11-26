"""Deprecated path-based adapter. New apps explicitly prepare scene inputs before runtime authoring."""
import warnings

from . import splat_stage as _runtime
from calibrated_scene_reconstruction.scene.twin.runtime_inputs import prepare_background


warnings.warn("calibrated_scene_reconstruction.sim.splat is deprecated; use calibrated_scene_reconstruction.sim.splat_stage with prepared scene inputs", DeprecationWarning, stacklevel=2)

def __getattr__(name):
    return getattr(_runtime, name)


def add_background(stage, session, kind="mesh", cousin=""):
    if kind == "mesh":
        return []
    return _runtime.add_background(stage, prepare_background(session), kind, cousin)
