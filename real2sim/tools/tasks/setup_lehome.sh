#!/bin/bash
# The LeHome Challenge (lehome-official/lehome-challenge: Isaac Lab garment folding, two SO-101 arms, PhysX particle
# cloth) inside our Isaac Sim 5.1 image, without a new image: the challenge, its Isaac Lab fork and the packages the
# image lacks go under envs/lehome (git-ignored). apps/tasks/lehome_run.py runs the challenge's modules with them.
#
#     tools/tasks/setup_lehome.sh
#     cd envs/lehome/lehome-challenge
#     CUDA_VISIBLE_DEVICES=<index> ../../../tools/run_isaac.sh ../../../apps/tasks/lehome_run.py -m scripts.dataset_sim replay \
#         --dataset_root Datasets/example/top_long_merged --num_replays 1 --disable_depth --device cpu --enable_cameras --headless
#
# The venv is made by the image's python with --system-site-packages and every install is constrained to the image's
# own versions (constraints_image.txt), so nothing of the image is replaced. Isaac Lab and lerobot are installed
# without their dependencies (their pins clash with the image's starlette / packaging) and what they need at run time
# is added on its own. apps/tasks/lehome_run.py loads libusb from libusb-package for Open3D.
set -euo pipefail
if [[ "${1:-}" == "--help" ]]; then
  echo "usage: tools/tasks/setup_lehome.sh [--help]"
  exit 0
fi
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
[ -f "$ROOT/.env.local" ] && . "$ROOT/.env.local"
: "${ISAAC_IMAGE:?set ISAAC_IMAGE (.env.local)}" "${ISAAC_PYTHON:?set ISAAC_PYTHON (.env.local)}"
L="$ROOT/envs/lehome"
CHALLENGE_REV=a805ad2  # 2026-04-29
ISAACLAB_REV=69f6fa54
mkdir -p "$L/logs"
cd "$L"
[ -d lehome-challenge ] || git clone -q https://github.com/lehome-official/lehome-challenge.git
git -C lehome-challenge checkout -q "$CHALLENGE_REV"
[ -d lehome-challenge/third_party/IsaacLab ] || git clone -q https://github.com/lehome-official/IsaacLab.git lehome-challenge/third_party/IsaacLab
git -C lehome-challenge/third_party/IsaacLab checkout -q "$ISAACLAB_REV"

in_image() {  # a command with the image's python (or the venv's), the repository mounted at its own path
  docker run --rm --user "$(id -u):$(id -g)" -e HOME="$ROOT/envs/isaac_docker/home" --network host -v "$ROOT:$ROOT" -w "$L" \
    --entrypoint "$1" "$ISAAC_IMAGE" "${@:2}"
}
[ -x venv/bin/python ] || in_image "$ISAAC_PYTHON" -m venv --system-site-packages venv
PY="$L/venv/bin/python"
[ -f constraints_image.txt ] || in_image "$PY" -m pip freeze | grep -v -E "^-e|@ file" > constraints_image.txt
pip_c() { in_image "$PY" -m pip install -q -c constraints_image.txt "$@"; }
I=lehome-challenge/third_party/IsaacLab/source
pip_c --no-deps -e "$I/isaaclab" -e "$I/isaaclab_assets" -e "$I/isaaclab_tasks" -e lehome-challenge/source/lehome "lerobot==0.4.3"
pip_c prettytable==3.3.0 toml hidapi==0.14.0.post2 gymnasium==1.2.1 "pyglet<2" einops flatdict==4.0.1 transforms3d num2words \
  omegaconf "pandas>=2.2,<2.3" "pyarrow==21.0.0" "deepdiff>=7.0.1,<9" psutil "h5py<3.15" plotly \
  "av>=15,<16" "datasets<4.2" "draccus==0.10.0" "huggingface-hub>=0.34.2,<0.36" jsonlines termcolor pyserial pynput \
  "torchcodec==0.5" "accelerate<1.11" "diffusers>=0.27.2,<0.36"
pip_c --only-binary=:all: "open3d>=0.19" libusb-package

# the challenge's assets (scenes, SO-101, garments, textures) and its 1000 teleoperated demonstrations (HuggingFace)
HF="$ROOT/envs/policy/bin/hf"
cd lehome-challenge
[ -d Assets/objects ] || "$HF" download lehome/asset_challenge --repo-type dataset --local-dir Assets
[ -d Datasets/example/four_types_merged ] || "$HF" download lehome/dataset_challenge_merged --repo-type dataset --local-dir Datasets/example
