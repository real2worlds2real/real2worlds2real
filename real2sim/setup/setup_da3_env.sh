#!/usr/bin/env bash
# Multi-view geometry environment (envs/da3): Depth Anything 3 (third_party/depth-anything-3) for camera poses and
# depth of several photos of one scene in one frame (calibrated_scene_reconstruction.recon.multiview), kept apart from envs/recon (its numpy < 2,
# xformers and open3d pins).
#
#   git submodule update --init third_party/depth-anything-3
#   setup/setup_da3_env.sh          # ~3 min, needs network and uv
#
# The weights (depth-anything/DA3NESTED-GIANT-LARGE-1.1, 6.3 GB, CC BY-NC 4.0: non-commercial use) download into the
# Hugging Face cache on first use. torch 2.7.0 cu128 as the other environments; xformers 0.0.30 is its build.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

[ -x envs/da3/bin/python ] || uv venv envs/da3 --python 3.11
PY=envs/da3/bin/python
uv pip install --python $PY torch==2.7.0 torchvision==0.22.0 xformers==0.0.30 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python $PY "numpy<2" trimesh einops "huggingface_hub<1.0" imageio opencv-python-headless open3d \
  omegaconf evo e3nn "moviepy==1.0.3" plyfile pillow pillow_heif safetensors pycolmap typer requests scipy \
  addict scikit-learn fastapi uvicorn
uv pip install --python $PY --no-deps ./third_party/depth-anything-3   # a copy: the env outlives a checkout

"$ROOT/setup/install_package.sh" "$PY"

$PY - <<'PYEOF'
import torch
import xformers
from depth_anything_3.api import DepthAnything3  # noqa: F401
print("torch", torch.__version__, "xformers", xformers.__version__, "cuda", torch.cuda.is_available())
PYEOF
