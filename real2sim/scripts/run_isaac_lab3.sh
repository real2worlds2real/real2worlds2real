#!/usr/bin/env bash
# Stable container-side paths are independent of the host checkout location.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="${REAL2SIM_SIM_RUNTIME:-docker}"
PROFILE="${REAL2SIM_ISAAC_PROFILE:-5.1}"
export REAL2SIM_ISAAC_PROFILE="$PROFILE"
case "$PROFILE" in
  6.2) LAB_VERSION=3.0.0; CACHE_NAME=isaac-lab-3 ;;
  5.1) LAB_VERSION=2.3.0; CACHE_NAME=isaac-sim-5.1 ;;
  *) echo 'REAL2SIM_ISAAC_PROFILE must be 6.2 or 5.1' >&2; exit 2 ;;
esac
CACHE="${REAL2SIM_SIM_CACHE:-$ROOT/.cache/$CACHE_NAME/runtime}"
if [[ $# -eq 0 ]]; then
    echo "usage: $0 SCRIPT [SCRIPT_ARGS ...]" >&2
    exit 2
fi
# Older exported runners store absolute paths inside the original container mount.
normalized=()
for argument in "$@"; do
    case "$argument" in
        /workspace/ucb_remo|/workspace/ucb_remo/*) argument="/workspace/real2sim_harness${argument#/workspace/ucb_remo}" ;;
    esac
    normalized+=("$argument")
done
set -- "${normalized[@]}"
if [[ "$PROFILE" == 5.1 ]]; then
    case "$(basename "$1")" in
      author_physx_articulation.py|audit_sim_conversion.py)
        # Pip-installed Sim 5.1 exposes pxr through Kit, not system Python.
        set -- /workspace/real2sim_harness/scripts/isaac_usd_entry.py "$@" ;;
      isaac_lab3_physx_smoke.py)
        echo 'Use scripts/isaac_physx_smoke.py for the Sim 5.1 profile' >&2; exit 2 ;;
      *newton*|*particle*|*hybrid*|*deformable*)
        echo 'The Sim 5.1 profile supports PhysX only; use the 6.2 profile for Lab 3 Newton workflows' >&2
        exit 2 ;;
    esac
fi
mkdir -p "$CACHE/home/Documents/Kit/shared" "$CACHE/kit" "$CACHE/kit_data"
case "$RUNTIME" in
  docker)
    IMAGE="${REAL2SIM_ISAAC_IMAGE:-nvcr.io/nvidia/isaac-lab:$LAB_VERSION}"
    exec docker run --rm --gpus all --entrypoint /bin/bash \
      -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=N -e PYTHONUNBUFFERED=1 \
      -e "REAL2SIM_ISAAC_PROFILE=$PROFILE" \
      -e PYTHONPATH=/workspace/real2sim_harness/src \
      -v "$ROOT:/workspace/real2sim_harness" -v "$ROOT:/workspace/ucb_remo" -v "$CACHE/home:/root" \
      -v "$CACHE/kit:/isaac-sim/kit/cache" -v "$CACHE/kit_data:/isaac-sim/kit/data" \
      -w /workspace/isaaclab "$IMAGE" /workspace/isaaclab/isaaclab.sh -p "$@"
    ;;
  apptainer)
    IMAGE="${REAL2SIM_ISAAC_IMAGE:-$ROOT/containers/isaac-lab-$LAB_VERSION.sif}"
    [[ -f "$IMAGE" ]] || { echo 'Set REAL2SIM_ISAAC_IMAGE to your local SIF file' >&2; exit 1; }
    FLAGS=()
    if [[ "${REAL2SIM_APPTAINER_FAKEROOT:-0}" == 1 ]]; then FLAGS+=(--fakeroot); fi
    exec apptainer exec "${FLAGS[@]}" --cleanenv --nv --writable-tmpfs \
      --home "$CACHE/home:/root" --env ACCEPT_EULA=Y --env PRIVACY_CONSENT=N \
      --env "REAL2SIM_ISAAC_PROFILE=$PROFILE" \
      --env PYTHONPATH=/workspace/real2sim_harness/src --env XDG_CACHE_HOME=/root/.cache \
      --env PYTHONUNBUFFERED=1 --bind "$ROOT:/workspace/real2sim_harness" --bind "$ROOT:/workspace/ucb_remo" \
      --bind "$CACHE/kit:/isaac-sim/kit/cache" --bind "$CACHE/kit_data:/isaac-sim/kit/data" \
      --pwd /workspace/isaaclab "$IMAGE" /workspace/isaaclab/isaaclab.sh -p "$@"
    ;;
  local)
    : "${REAL2SIM_ISAACLAB_ROOT:?Set to your compatible Isaac Lab checkout (2.3 for Sim 5.1, 3.0 for Sim 6.2)}"
    # Lab 2's shell entrypoint invokes `tabs`, even for redirected headless runs.
    if [[ "$PROFILE" == 5.1 && "${TERM:-dumb}" == dumb ]]; then export TERM=xterm; fi
    ARGS=()
    for argument in "$@"; do
      if [[ "$argument" == /workspace/real2sim_harness/* ]]; then
        argument="$ROOT/${argument#/workspace/real2sim_harness/}"
      fi
      ARGS+=("$argument")
    done
    export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
    cd "$REAL2SIM_ISAACLAB_ROOT"
    exec ./isaaclab.sh -p "${ARGS[@]}"
    ;;
  *) echo 'REAL2SIM_SIM_RUNTIME must be docker, apptainer, or local' >&2; exit 2 ;;
esac
