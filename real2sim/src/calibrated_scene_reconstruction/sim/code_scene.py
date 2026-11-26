"""Deprecated path-based adapter. New apps explicitly prepare scene inputs before runtime authoring."""
import warnings

from . import code_stage as _runtime
from calibrated_scene_reconstruction.scene.twin.runtime_inputs import prepare_background


warnings.warn("calibrated_scene_reconstruction.sim.code_scene is deprecated; use calibrated_scene_reconstruction.sim.code_stage with prepared scene inputs", DeprecationWarning, stacklevel=2)

def __getattr__(name):
    return getattr(_runtime, name)


def add_code_background(stage, session):
    return _runtime.add_code_background(stage, prepare_background(session))
