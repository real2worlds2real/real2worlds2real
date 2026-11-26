#!/usr/bin/env bash
# The official Diffusion Policy environment (envs/dp_official): real-stanford/diffusion_policy (third_party/
# diffusion_policy, Chi et al., RSS 2023) trained as is, with its own train.py and workspaces, on our datasets
# (tools/export_dp_official.py, calibrated_scene_reconstruction.policy.dp_official). Only what its image-policy training path imports: none of
# its simulation benchmarks (mujoco, robosuite, gym, pybullet, ray). Its conda pins (python 3.9, torch 1.12) predate
# these GPUs: python 3.11 and torch 2.7.0 cu128 as the other environments, the library pins its code needs kept
# (diffusers 0.11.1, a huggingface_hub that still has cached_download, zarr 2).
#
#   git submodule update --init third_party/diffusion_policy
#   setup/setup_dp_official_env.sh        # ~5 min, needs network and uv
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV="$ROOT/envs/dp_official"
PY="$ENV/bin/python"

uv venv --clear --python 3.11 "$ENV"
uv pip install --python "$PY" torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$PY" "zarr>=2.16,<3" "numcodecs<0.16" numba "diffusers==0.11.1" \
  "huggingface_hub<0.26" "hydra-core==1.3.2" "omegaconf==2.3.0" "einops==0.4.1" dill tqdm wandb threadpoolctl \
  filelock scipy imagecodecs pillow pandas pyarrow
# the hybrid image policy's encoder (robomimic 0.2.0, as its conda env; its own pins are for its simulation benchmarks)
uv pip install --python "$PY" --no-deps robomimic==0.2.0
# the package has no top-level __init__.py (setup.py find_packages misses it; the README runs from the checkout):
# the checkout on the path
echo "$ROOT/third_party/diffusion_policy" > "$("$PY" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")/diffusion_policy_checkout.pth"

"$ROOT/setup/install_package.sh" "$PY"

"$PY" - <<'PYEOF'
import torch, diffusers, zarr
import diffusion_policy
from diffusion_policy.workspace.train_diffusion_unet_image_workspace import TrainDiffusionUnetImageWorkspace  # noqa: F401
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "| diffusers", diffusers.__version__,
      "| zarr", zarr.__version__, "| diffusion_policy", diffusion_policy.__file__)
PYEOF
