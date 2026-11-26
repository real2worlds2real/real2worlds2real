"""Grasp synthesis and arm trajectories, with an optional external tabletop solver."""

import os
from importlib.util import find_spec
from pathlib import Path


# A compatible port uses this package's historical qualified name internally.
# Keep its own directory layout so its imports and asset paths still resolve.
_setting = os.environ.get("SCENE_RECONSTRUCTION_BODEX_ROOT")
if _setting:
    _root = Path(_setting).expanduser()
    if _root.is_absolute() and _root.name == "bodex" and (_root / "__init__.py").is_file():
        __path__.append(str(_root.parent))


def require_bodex() -> None:
    """Check for the optional port before creating outputs or starting GPU work."""
    if find_spec(f"{__name__}.bodex") is None:
        raise ModuleNotFoundError(
            "Tabletop grasp synthesis needs a separately supplied compatible BODex port. "
            "Set SCENE_RECONSTRUCTION_BODEX_ROOT to its absolute bodex/ package directory "
            "before starting Python. See real2sim/docs/robots/grasping.md."
        )
