#!/usr/bin/env bash
# Background splat environment (envs/splat), for the optional Gaussian-splat background of a twin (run_session.sh
# --splat): 3DGRUT (third_party/3dgrut, NVIDIA: 3DGUT training, export to NuRec USDZ / ParticleField USD, which Isaac
# Sim renders with RTX beside the meshes) with its own CUDA 12.8 toolkit (no system nvcc needed; sm_120 needs 12.8),
# and pycolmap for the frames' poses. Kept apart from envs/recon and envs/da3 (its own torch build, tiny-cuda-nn,
# kaolin).
#
#   git submodule update --init third_party/3dgrut
#   setup/setup_splat_env.sh          # ~30-40 min: a 4 GB CUDA toolkit, then tiny-cuda-nn and kaolin builds
#
# 3DGRUT's own scripts build into <its checkout>/.venv, so it is cloned under envs/splat/3dgrut at the submodule's
# commit (the environment outlives a worktree): envs/splat/3dgrut/.venv/bin/python is the interpreter.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

REV=$(cd third_party/3dgrut && git rev-parse HEAD)
mkdir -p "$ROOT/envs/splat"
# envs/ may be a symlink (a worktree's to the main checkout's): the build writes absolute paths into the venv
# (shebangs, CUDA_HOME in its activation hooks), so they must be the real ones
SRC="$(cd "$ROOT/envs/splat" && pwd -P)/3dgrut"
if [ ! -d "$SRC/.git" ]; then
  git clone https://github.com/nv-tlabs/3dgrut.git "$SRC"
fi
cd "$SRC"
git checkout -q "$REV"
git submodule update --init --recursive

# the local toolkit, extracted here with --nox11: 3DGRUT's script runs the runfile itself, which outside a terminal
# (nohup, a background job) tries to open an xterm ("exec: -title: not found") and stops
URL=https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run
if [ ! -x .venv/cuda-12.8.1/bin/nvcc ]; then
  RUN="$ROOT/envs/splat/cuda_12.8.1_linux.run"
  [ -f "$RUN" ] || wget -q -O "$RUN" "$URL"
  [ -d .venv ] || uv venv .venv --python 3.11 --prompt splat
  sh "$RUN" --nox11 --toolkit --toolkitpath="$SRC/.venv/cuda-12.8.1" --silent --no-man-page --override
  rm -f "$RUN"
fi
FORCE_LOCAL_CUDA=1 CUDA_VERSION=12.8 bash scripts/create_venv_cuda.sh splat
# shellcheck disable=SC1091
source .venv/bin/activate
bash install_env_uv.sh splat
uv pip install --python .venv/bin/python pycolmap opencv-python-headless lpips 'threadpoolctl>=3.6,<4'

"$ROOT/setup/install_package.sh" "$SRC/.venv/bin/python"

python - <<'PYEOF'
import torch
import pycolmap
import threedgrut  # noqa: F401
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0),
      "pycolmap", pycolmap.__version__)
PYEOF
