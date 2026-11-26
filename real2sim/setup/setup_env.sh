#!/usr/bin/env bash
# Reproduce the reconstruction environment (envs/recon) for an sm_120 GPU with driver >= 570; another architecture:
# TORCH_CUDA_ARCH_LIST=8.6 (the source builds are compiled for that list).
# Isaac Sim 5.1 itself is NOT installed here: point ISAAC_PYTHON (tools/run_isaac.sh) at an existing Isaac python.
#
#   setup/setup_env.sh            # full setup (~30 min, needs network, conda and uv)
#   SAM3D_CHECKPOINTS=0 setup/setup_env.sh   # without the SAM 3D weights (no reconstruction: training, collection)
#
# Steps: submodules + SAM 3D configuration -> CUDA 12.8 toolkit (conda prefix, for extension builds) -> uv venv (py3.11,
# torch 2.7.0+cu128) -> locked requirements -> source builds (pytorch3d, nvdiffrast) -> MoGe (no flex-gemm) ->
# kaolin 0.18 -> this package (editable) -> checkpoints (SAM 3D is gated on Hugging Face: request access first).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "== submodules"
git submodule update --init --recursive
python3 setup/configure_sam.py third_party/sam-3d-objects

echo "== CUDA 12.8 toolkit -> envs/cuda128 (only used to compile CUDA extensions)"
[ -x envs/cuda128/bin/nvcc ] || conda create -y -q -p envs/cuda128 -c nvidia/label/cuda-12.8.1 cuda-toolkit
export CUDA_HOME="$ROOT/envs/cuda128" PATH="$ROOT/envs/cuda128/bin:$PATH"
export CPATH="$ROOT/envs/cuda128/targets/x86_64-linux/include"
export LIBRARY_PATH="$ROOT/envs/cuda128/lib:$ROOT/envs/cuda128/targets/x86_64-linux/lib"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}" FORCE_CUDA=1 MAX_JOBS="${MAX_JOBS:-16}"   # MAX_JOBS bounds nvcc RAM use

echo "== uv venv envs/recon"
[ -x envs/recon/bin/python ] || uv venv envs/recon --python 3.11
source envs/recon/bin/activate
uv pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install -r setup/requirements-recon.lock.txt
uv pip install kaolin==0.18.0 --find-links https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.7.0_cu128.html

echo "== source builds for TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
uv pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47"
uv pip install --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae"

echo "== MoGe (editable, without its flex-gemm extension, which inference does not need)"
uv pip install --no-deps -e third_party/MoGe
echo "== calibrated_scene_reconstruction (editable)"
uv pip install --no-deps -e .

echo "== checks"
python - <<'EOF'
import torch
from pytorch3d import _C  # noqa: F401
import nvdiffrast.torch, kaolin, spconv.pytorch, gsplat  # noqa: F401,E401
from moge.model.v2 import MoGeModel  # noqa: F401
from transformers import Sam3Model  # noqa: F401
from diffusers import AutoPipelineForInpainting  # noqa: F401
import calibrated_scene_reconstruction
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0), "| calibrated_scene_reconstruction", calibrated_scene_reconstruction.__version__)
EOF

echo "== checkpoints"
if [ "${SAM3D_CHECKPOINTS:-1}" = 0 ]; then
  echo "   SAM3D_CHECKPOINTS=0: SAM 3D weights skipped (reconstruction needs them)"
elif [ ! -f third_party/sam-3d-objects/checkpoints/hf/pipeline.yaml ]; then
  hf download --repo-type model --local-dir third_party/sam-3d-objects/checkpoints/hf-download facebook/sam-3d-objects
  mv third_party/sam-3d-objects/checkpoints/hf-download/checkpoints third_party/sam-3d-objects/checkpoints/hf
  rm -rf third_party/sam-3d-objects/checkpoints/hf-download
fi
echo "MoGe-2, SAM 3 (gated) and LaMa weights download on first use."
echo "Library assets: python -m calibrated_scene_reconstruction.assets.polyhaven && tools/run_isaac.sh apps/gltf_to_usd.py"
