"""Compatibility CLI/import for tools/policy/openpi_run.py."""
from pathlib import Path
import importlib.util
import runpy
import sys

_target = Path(__file__).resolve().parent / "policy/openpi_run.py"
if __name__ == "__main__":
    runpy.run_path(str(_target), run_name="__main__")
else:
    _spec = importlib.util.spec_from_file_location(__name__, _target)
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[__name__] = _module
    _spec.loader.exec_module(_module)


def __getattr__(name):
    return getattr(_module, name)
