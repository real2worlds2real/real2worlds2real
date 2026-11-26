"""Where the parts of a checkout live: one place that knows the directory partition."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

# source roots in import order: Part 1 (real2sim), Part 2 (policy development), this layer
SOURCE_DIRS = ("real2sim/src", "policy_development/src", "end_to_end/src")
# first-party import packages a frozen checkout must supply itself
PACKAGES = ("calibrated_scene_reconstruction", "real2sim_harness", "astra_station", "astra_e2e")


def source_roots(repository=REPO):
    """The existing source roots of a checkout (a frozen copy may carry only the parts a job needs)."""
    return [Path(repository) / d for d in SOURCE_DIRS if (Path(repository) / d).is_dir()]


def real2sim(repository=REPO):
    """Part 1's root: `calibrated_scene_reconstruction.core.paths.ROOT`, the home of configs/, sessions/, assets/ and envs/."""
    return Path(repository) / "real2sim"


def unified(repository=REPO):
    """The reconstruction-bound policy path (bound adapters, campaign preparation)."""
    return Path(repository) / "end_to_end" / "unified"


def pythonpath(repository=REPO, *extra):
    return os.pathsep.join(str(p) for p in (*source_roots(repository), *extra))


def first_party_sources(repository=REPO):
    """Every first-party Python file a bound job can import: the reconstruction package and this layer."""
    repository = Path(repository)
    source = repository / "real2sim/src"
    # A frozen checkout may carry the original package or the compatibility loader.
    packages = [source / "calibrated_scene_reconstruction", source / "r2s2r", repository / "end_to_end/src/astra_e2e"]
    files = [path for package in packages for path in package.rglob("*.py")]
    loader = source / "real2sim_harness/_legacy_imports.py"
    if loader.is_file():
        files.extend([loader, loader.with_name("__init__.py")])
    return [p for p in files if "__pycache__" not in p.parts]
