"""Compatibility alias for calibrated_scene_reconstruction.core.paths."""
import sys as _sys
import warnings as _warnings
from .core import paths as _implementation

_warnings.warn("calibrated_scene_reconstruction.paths moved to calibrated_scene_reconstruction.core.paths", DeprecationWarning, stacklevel=2)
# Alias the module itself so private imports, mutable globals and old pickle paths remain compatible.
_sys.modules[__name__] = _implementation
