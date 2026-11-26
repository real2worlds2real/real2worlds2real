#!/usr/bin/env bash
# Policy environment (envs/policy): LeRobot for the Diffusion Policy (calibrated_scene_reconstruction.policy.diffusion training, the lerobot of
# calibrated_scene_reconstruction.policy.runners.PolicyRunner), kept apart from envs/recon and the Isaac python.
#
#   setup/setup_policy_env.sh       # ~2 min, needs network and uv
#
# lerobot 0.3.3 is the last release that reads LeRobot v2.1 datasets as they are (0.4 converts them to v3.0); its pins
# allow torch 2.7.0 / torchvision 0.22.0, the cu128 builds envs/recon and the Isaac python use. Python 3.11 as the
# Isaac python: calibrated_scene_reconstruction.policy.runners appends this env's site-packages to its sys.path for lerobot (torch stays Isaac's).
# huggingface-hub < 1.0 (lerobot 0.3.3 asks for its [cli,hf-transfer] extras, gone in 1.x). Videos are decoded with
# PyAV (config "video_backend": "pyav"): torchcodec, lerobot's default, needs FFmpeg 4-7 shared libraries
# (libavutil, libavcodec, libavformat, libavdevice, ...) on the loader path.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

[ -x envs/policy/bin/python ] || uv venv envs/policy --python 3.11
PY=envs/policy/bin/python
uv pip install --python $PY torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python $PY -r setup/requirements-policy.lock.txt   # lerobot==0.3.3 and its resolved dependencies

"$ROOT/setup/install_package.sh" "$PY"

PYTHONPATH=src $PY - <<'PYEOF'
import torch
import torchvision
import lerobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: F401
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy  # noqa: F401
import calibrated_scene_reconstruction.policy.diffusion  # noqa: F401
print("torch", torch.__version__, "torchvision", torchvision.__version__, "cuda", torch.cuda.is_available(),
      "| lerobot", lerobot.__version__)
PYEOF
