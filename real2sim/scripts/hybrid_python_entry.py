"""Select the isolated native-attachment runtime before importing Newton or Warp."""

import runpy
import sys
from pathlib import Path

runtime = Path(__file__).resolve().parents[1] / ".tools/newton-hybrid"
if not (runtime / "newton/__init__.py").is_file():
    raise RuntimeError("The isolated Newton 1.7 hybrid runtime is not installed")
sys.path.insert(0, str(runtime))
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
