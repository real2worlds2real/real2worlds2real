#!/usr/bin/env bash
# One local Isaac job at a time, with a shared-GPU admission margin.
# The expected peak is an admission estimate, not a hard VRAM cap.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo 'Select one physical GPU for shared Isaac jobs.' >&2; exit 2; }
EXPECTED="${SCENE_RECONSTRUCTION_ISAAC_EXPECTED_VRAM_MIB:-${R2S2R_ISAAC_EXPECTED_VRAM_MIB:-8192}}"
RESERVE="${SCENE_RECONSTRUCTION_SHARED_GPU_RESERVE_MIB:-${R2S2R_SHARED_GPU_RESERVE_MIB:-8192}}"
[[ "$EXPECTED" =~ ^[0-9]+$ && "$RESERVE" =~ ^[0-9]+$ ]] || exit 2
# flock supervises the actual run, closing its descriptor in the child. Kit
# can start a persistent Omniverse hub; inheriting an open lock descriptor
# would let that idle service retain the GPU lease after the trial exits.
if [[ "${1:-}" != --_r2s2r_gpu_lease_held ]]; then
    exec flock --exclusive --nonblock --conflict-exit-code 75 --close \
        "/tmp/r2s2r_datagen_isaac_gpu${GPU}.lock" "$0" --_r2s2r_gpu_lease_held "$@"
fi
shift
FREE="$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits)"
FREE="${FREE//[[:space:]]/}"
[[ "$FREE" =~ ^[0-9]+$ ]] || exit 2
if (( FREE < EXPECTED + RESERVE )); then
    echo "[shared_gpu] Defer: ${FREE}MiB free; need ${EXPECTED}MiB expected peak plus ${RESERVE}MiB shared headroom." >&2
    exit 75
fi
echo "[shared_gpu] Serial Isaac job; GPU${GPU} free=${FREE}MiB, expected peak=${EXPECTED}MiB, shared headroom=${RESERVE}MiB."
export CUDA_VISIBLE_DEVICES="$GPU"
export SCENE_RECONSTRUCTION_SHARED_GPU_ACTIVE=1
export SCENE_RECONSTRUCTION_ISAAC_EXPECTED_VRAM_MIB="$EXPECTED"
export SCENE_RECONSTRUCTION_SHARED_GPU_RESERVE_MIB="$RESERVE"
# Frozen worker scripts still read these names when they join an existing GPU lease.
export R2S2R_SHARED_GPU_ACTIVE="$SCENE_RECONSTRUCTION_SHARED_GPU_ACTIVE"
export R2S2R_ISAAC_EXPECTED_VRAM_MIB="$SCENE_RECONSTRUCTION_ISAAC_EXPECTED_VRAM_MIB"
export R2S2R_SHARED_GPU_RESERVE_MIB="$SCENE_RECONSTRUCTION_SHARED_GPU_RESERVE_MIB"
exec "$ROOT/tools/run_isaac.sh" "$@"
