"""Freeze a twin's structural contract without altering its accepted scene."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src/real2sim_harness'))
from blender_structure import snapshot_structure  # noqa: E402

output = Path(sys.argv[sys.argv.index('--')+1])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(snapshot_structure(), indent=2))
print('ASSET_STRUCTURE_SNAPSHOT', str(output), flush=True)
