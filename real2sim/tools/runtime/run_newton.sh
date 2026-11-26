#!/usr/bin/env bash
# Run a Python script with Newton (Warp physics: VBD cloth, Featherstone / MuJoCo-Warp robots) in the Isaac Lab 3
# image: the rope letters (docs/deformable.md). Always in docker, like every Isaac stack here.
#
#   CUDA_VISIBLE_DEVICES=<index> tools/run_newton.sh apps/newton_rope_letter.py [args...]
#
#   NEWTON_IMAGE  the image (default nvcr.io/nvidia/isaac-lab:3.0.0: Isaac Sim 6.2, Isaac Lab 3, Newton 1.6, Warp)
#   MEM           container memory (default 24G)
#
# The repo is mounted at its own path (outputs land in it), the container runs as the calling user, and HOME is
# envs/newton_docker/ (git-ignored): Warp's kernel cache and Newton's downloaded assets are kept there for every run.
# The image's own entry point starts Kit streaming, so the venv's python is the entry point instead. The host's network,
# as tools/run_isaac.sh: a policy server on the machine's localhost (tools/dp_official_server.py) is reachable.
set -euo pipefail
if [[ "${1:-}" == "--help" ]]; then
  echo "usage: tools/runtime/run_newton.sh <script> [args...]"
  exit 0
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${NEWTON_IMAGE:-nvcr.io/nvidia/isaac-lab:3.0.0}"
STATE="$ROOT/envs/newton_docker"
mkdir -p "$STATE"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
exec docker run --rm --init --gpus "device=$GPU" -e CUDA_VISIBLE_DEVICES=0 \
  -e PYTHONPATH="$ROOT/src" -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y --user "$(id -u):$(id -g)" --network host \
  --memory "${MEM:-24G}" --memory-swap "${MEM:-24G}" \
  -v "$ROOT:$ROOT" -w "$ROOT" -v "$STATE:/home/newton" -e HOME=/home/newton \
  --entrypoint /opt/isaaclab-venv/bin/python "$IMAGE" -u "$@"
