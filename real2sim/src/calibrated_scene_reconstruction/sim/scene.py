"""Deprecated path-based adapter. New apps explicitly prepare scene inputs before runtime authoring."""
import warnings

from . import scene_stage as _runtime
from calibrated_scene_reconstruction.scene.twin.runtime_inputs import prepare_scene
from calibrated_scene_reconstruction.scene.twin.variants import photo_goal, prepare_variant  # noqa: F401 -- legacy API


warnings.warn("calibrated_scene_reconstruction.sim.scene is deprecated; use calibrated_scene_reconstruction.sim.scene_stage with prepared scene inputs", DeprecationWarning, stacklevel=2)

def __getattr__(name):
    return getattr(_runtime, name)


class TwinScene(_runtime.TwinScene):
    def __init__(self, stage, session):
        super().__init__(stage, prepare_scene(session))


def build_scene(stage, session, variant):
    return _runtime.build_scene(stage, prepare_scene(session), variant)
