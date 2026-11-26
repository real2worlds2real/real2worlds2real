"""Deprecated path-based adapter. New apps explicitly prepare scene inputs before runtime authoring."""
import warnings

from . import setpiece_stage as _runtime
from calibrated_scene_reconstruction.scene.twin.setpiece_inputs import (  # noqa: F401 -- legacy API
    face_camera_yaw, furniture_kits, furniture_overrides, reference_albedo,
    kit_for_object as _kit_for_object,
)


warnings.warn("calibrated_scene_reconstruction.sim.setpieces is deprecated; use calibrated_scene_reconstruction.sim.setpiece_stage with prepared scene inputs", DeprecationWarning, stacklevel=2)

def __getattr__(name):
    return getattr(_runtime, name)


def kit_for_object(ts, name, fix, V):
    return _kit_for_object(ts.s, _runtime.support_boxes(ts), name, fix, V)
