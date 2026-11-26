"""Run before physics to reject lost or changed conversion semantics."""

import sys
from pathlib import Path

from real2sim_harness.artifacts import atomic_json
from real2sim_harness.conversion_audit import audit_conversion

if __name__ == "__main__":
    folder = Path(sys.argv[1])
    result = audit_conversion(folder)
    atomic_json(folder / "conversion_validation.json", result)
    print(result, flush=True)
    raise SystemExit(0 if result["ok"] else 1)
