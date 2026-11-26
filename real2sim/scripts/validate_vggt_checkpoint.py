"""Validate separately obtained VGGT weights in the isolated VGGT environment."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=Path, default=Path(os.environ.get(
    'REAL2SIM_VGGT_CHECKPOINT', ROOT / '.tools/checkpoints/vggt-omega/vggt_omega_1b_512.pt')))
args = parser.parse_args()
sys.path.insert(0, os.environ.get('REAL2SIM_VGGT_REPO', str(ROOT / '.tools/vggt-omega')))
import torch  # noqa: E402
from vggt_omega.models import VGGTOmega  # noqa: E402

checkpoint = args.checkpoint.resolve()
model = VGGTOmega()
model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
stat = checkpoint.stat()
report = {'ok': True, 'checkpoint': str(checkpoint), 'checkpoint_size': stat.st_size,
          'checkpoint_mtime_ns': stat.st_mtime_ns, 'scope': 'strict CPU state-dictionary load; not an inference quality check'}
output = ROOT / '.tools/checkpoints/vggt-omega/validation-report.json'
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'ok': True, 'report': str(output)}))
