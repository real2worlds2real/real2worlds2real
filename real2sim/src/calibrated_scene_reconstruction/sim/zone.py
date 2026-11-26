"""Deprecated path-based adapter. New apps explicitly prepare scene inputs before runtime authoring."""
import warnings

from . import zone_stage as _runtime
from calibrated_scene_reconstruction.scene.twin.runtime_inputs import prepare_background


warnings.warn("calibrated_scene_reconstruction.sim.zone is deprecated; use calibrated_scene_reconstruction.sim.zone_stage with prepared scene inputs", DeprecationWarning, stacklevel=2)

def __getattr__(name):
    return getattr(_runtime, name)


def keep_task_zone(stage, s, keep_names=()):
    return _runtime.keep_task_zone(stage, prepare_background(s), keep_names)


def add_zone_background(stage, session, half=6.0, floor_visible=True):
    return _runtime.add_zone_background(stage, prepare_background(session), half, floor_visible)
