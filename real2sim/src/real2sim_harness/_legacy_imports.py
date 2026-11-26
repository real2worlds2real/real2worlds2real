"""Resolve pre-release imports to the canonical modules, including old pickle paths."""
import importlib
import importlib.abc
import importlib.util
import sys

_PACKAGES = {"ucb_remo": "real2sim_harness", "r2s2r": "calibrated_scene_reconstruction"}


class _LegacyLoader(importlib.abc.Loader):
    def __init__(self, target):
        self.target = target

    def exec_module(self, module):
        # Reuse the actual module so classes, singletons and mutable registries share identity.
        sys.modules[module.__name__] = importlib.import_module(self.target)

    def get_code(self, fullname):
        # runpy uses this path for existing "python -m old_package.module" invocations.
        spec = importlib.util.find_spec(self.target)
        return spec.loader.get_code(self.target)


class _LegacyPackages(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root, dot, suffix = fullname.partition(".")
        if not dot or root not in _PACKAGES:
            return None
        canonical = f"{_PACKAGES[root]}.{suffix}"
        spec = importlib.util.find_spec(canonical)
        if spec is None:
            return None
        return importlib.util.spec_from_loader(
            fullname, _LegacyLoader(canonical), is_package=spec.submodule_search_locations is not None
        )


def redirect_package(old_name):
    if not any(isinstance(finder, _LegacyPackages) for finder in sys.meta_path):
        sys.meta_path.insert(0, _LegacyPackages())
    sys.modules[old_name] = importlib.import_module(_PACKAGES[old_name])
