"""Repository paths and external resources (override with environment variables)."""
import os
from pathlib import Path

from calibrated_scene_reconstruction.core.environment import setting

ROOT = Path(__file__).resolve().parents[3]
CONFIGS = ROOT / "configs"
SAM3D_DIR = Path(setting("SAM3D_DIR", ROOT / "third_party" / "sam-3d-objects"))
POLYHAVEN_DIR = Path(setting("POLYHAVEN_DIR", ROOT / "assets" / "polyhaven"))
ABO_DIR = Path(setting("ABO_DIR", ROOT / "assets" / "abo"))
ROBOTS_DIR = ROOT / "assets" / "robots"
# YCB object scans (google_16k) used as retrieved cousins / distractors; default assumes a sibling checkout
YCB_ROOT = Path(setting("YCB_ROOT", ROOT / "assets/ycb/models"))


def robot_asset_path(path, *, required=False) -> Path:
    """Resolve a robot or grasping-hand input against the operator's asset directory."""
    path = Path(path).expanduser()
    local = path if path.is_absolute() else ROOT / path
    asset_root = setting("ROBOT_ASSET_ROOT")
    if asset_root and local.is_relative_to(ROOT / "assets"):
        relative = local.relative_to(ROOT / "assets")
        if relative.parts[:1] == ("robots",) or relative.parts[:2] == ("grasping", "hands"):
            external = Path(asset_root).expanduser().resolve() / relative
            if external.is_file() or not local.is_file():
                local = external
    if required and not local.is_file():
        raise FileNotFoundError(
            f"Missing robot asset: {local}. Supply your licensed files under "
            "SCENE_RECONSTRUCTION_ROBOT_ASSET_ROOT (see real2sim/assets/robots/README.md)."
        )
    return local


VEGA_USD = setting("VEGA_USD", str(robot_asset_path("assets/robots/assemblies/vega_1p_sharpa/vega_1p_sharpa.usd")))


def rel(p) -> str:
    """A path as written into data files (indexes, configurations, records): relative to the repository root when it
    lies inside the repository, so sessions and assets move with it; absolute otherwise. Symlinks are kept as they
    are (assets/ycb/models stays under assets/)."""
    absolute = Path(os.path.abspath(p))
    return absolute.relative_to(ROOT).as_posix() if absolute.is_relative_to(ROOT) else absolute.as_posix()


def resolve(p) -> Path:
    """A path read from a data file: a relative one is taken from the repository root (not the working directory)."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p
