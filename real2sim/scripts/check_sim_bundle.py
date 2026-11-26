"""Check that retained conversion and validation evidence matches a bundle."""

import json
import sys

from real2sim_harness.sim_bundle import verify_bundle

result = verify_bundle(sys.argv[1])
print(json.dumps(result, indent=2))
raise SystemExit(0 if result["ok"] else 2)
