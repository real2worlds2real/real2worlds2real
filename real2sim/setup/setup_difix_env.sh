#!/usr/bin/env bash
# Splat clean-up environment (envs/difix): Difix (third_party/Difix3D, NVIDIA's single-step diffusion fixer of
# novel-view renders) for calibrated_scene_reconstruction.recon.splat_fix, kept apart from the others (its diffusers 0.25 / transformers 4.38 pins).
#
#   setup/fetch_difix.sh            # pinned source; preserves existing checkouts
#   setup/setup_difix_env.sh          # ~3 min, needs network and uv
#
# The weights (nvidia/difix, nvidia/difix_ref: research and non-commercial use) download into the Hugging Face cache
# on first use. torch 2.7.0 cu128 as the other environments.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
setup/fetch_difix.sh

[ -x envs/difix/bin/python ] || uv venv envs/difix --python 3.11
PY=envs/difix/bin/python
uv pip install --python $PY torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python $PY "diffusers==0.25.1" "huggingface-hub==0.25.1" "transformers==4.38.0" "peft==0.9.0" \
  einops "imageio[ffmpeg]" "numpy<2" pillow opencv-python-headless plyfile scipy accelerate

"$ROOT/setup/install_package.sh" "$PY"

$PY - <<'PYEOF'
import sys
sys.path.insert(0, "third_party/Difix3D/src")
import torch
from pipeline_difix import DifixPipeline  # noqa: F401
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
PYEOF
