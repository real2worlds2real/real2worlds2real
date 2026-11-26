#!/usr/bin/env bash
# Run an Isaac Sim 5.1 entry point (apps/*.py) with the Isaac python, under a memory cap.
# Usage: tools/run_isaac.sh apps/<app>.py [args...]
# Settings come from the environment or from .env.local at the repo root (git-ignored). Every machine names how Isaac
# starts there, there is no default:
#   ISAAC_RUNTIME=host    ISAAC_PYTHON=<the Isaac Sim 5.1 python on this machine>
#   ISAAC_RUNTIME=docker  ISAAC_IMAGE=<an image with Isaac Sim 5.1 (pip, Python 3.11)>
#                         ISAAC_PYTHON=<that python inside the image>
#                         [ISAAC_DOCKER_ARGS="<more docker run arguments, e.g. -v for data outside the repo>"]
# The Isaac python imports this package, so it needs the pipeline's CPU dependencies too (pyproject extra scene-pipeline,
# and plyfile); an existing Isaac environment typically lacks plyfile, threadpoolctl and tifffile: install them there or
# put a directory that holds them on PYTHONPATH.
# The apps call ffmpeg (rollout videos with --video, VideoWriter); an image without it crashes at the first video
# (FileNotFoundError, then a segfault as Kit unloads): mount a static build onto its PATH through ISAAC_DOCKER_ARGS,
# e.g. -v "$ROOT/envs/bin/ffmpeg:/usr/local/bin/ffmpeg:ro".
# docker: one card per run, CUDA_VISIBLE_DEVICES=<index> (default 0); the repo is mounted at its own path (the apps reach
# envs/policy by path) and the container runs as the calling user (no root-owned files in the repo); Kit's writable
# state (HOME, kit/logs, kit/data, kit/cache) is envs/isaac_docker/, one for every run: shaders compile once, but two
# runs that start against an empty one can hang each other, so the first run on a machine goes alone.
# ISAAC_CONTAINER=<name> names the container: killing this script (or docker run) leaves the container running, a caller
# that gives up on a run stops it with `docker kill <name>` (tools/collect_place.run).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$ROOT/.env.local" ] && . "$ROOT/.env.local"
: "${ISAAC_PYTHON:?set ISAAC_PYTHON to the Isaac Sim 5.1 python (environment or .env.local)}"
export OMNI_KIT_ACCEPT_EULA=YES
# the apps run headless: with DISPLAY set (a desktop session) Kit's Vulkan start polls the X server and can hang there
# for good, before "Startup Complete"; ISAAC_KEEP_DISPLAY=1 keeps it
[ -n "${ISAAC_KEEP_DISPLAY:-}" ] || unset DISPLAY XAUTHORITY WAYLAND_DISPLAY
export MEM="${MEM:-24G}"

case "${ISAAC_RUNTIME:-}" in
host)
  exec "$ROOT/tools/run_capped.sh" "$ISAAC_PYTHON" -u "$@"
  ;;
docker)
  : "${ISAAC_IMAGE:?set ISAAC_IMAGE to the Isaac Sim 5.1 image (environment or .env.local)}"
  GPU="${CUDA_VISIBLE_DEVICES:-0}"
  [[ "$GPU" =~ ^[0-9]+$ ]] || { echo "tools/run_isaac.sh: CUDA_VISIBLE_DEVICES='$GPU': one card index per run" >&2; exit 2; }
  KIT="$(dirname "$(dirname "$ISAAC_PYTHON")")/lib/python3.11/site-packages/isaacsim/kit"
  STATE="$ROOT/envs/isaac_docker"
  mkdir -p "$STATE/home" "$STATE/kit/logs" "$STATE/kit/data" "$STATE/kit/cache"
  [[ "$PWD" == "$ROOT" || "$PWD" == "$ROOT"/* ]] && WD="$PWD" || WD="$ROOT"
  PASS=()  # the settings the apps read
  for v in $(compgen -e); do
    case "$v" in SCENE_RECONSTRUCTION_*|R2S2R_*|POLICY_ENV|VIDPHYS_*|ISAAC_KEEP_DISPLAY) PASS+=(-e "$v") ;; esac
  done
  # docker's own copy of the output (json-file, outside the repo) capped: Kit stuck repeating an error filled the disk
  # with it; the caller's redirect of the output is the record
  # shellcheck disable=SC2086  # ISAAC_DOCKER_ARGS is a list of arguments
  exec "$ROOT/tools/run_capped.sh" docker run --rm --init ${ISAAC_CONTAINER:+--name "$ISAAC_CONTAINER"} \
    --gpus "device=$GPU" -e CUDA_VISIBLE_DEVICES=0 -e NVIDIA_DRIVER_CAPABILITIES=all \
    --user "$(id -u):$(id -g)" -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
    --network host --ipc host --memory "$MEM" --memory-swap "$MEM" --log-opt max-size=10m --log-opt max-file=1 \
    -e HOME="$STATE/home" -e OMNI_KIT_ACCEPT_EULA=YES -e PYTHONUNBUFFERED=1 -e PYTHONPATH="$ROOT/src" "${PASS[@]}" \
    -v "$ROOT:$ROOT" -w "$WD" \
    -v "$STATE/kit/logs:$KIT/logs" -v "$STATE/kit/data:$KIT/data" -v "$STATE/kit/cache:$KIT/cache" \
    ${ISAAC_DOCKER_ARGS:-} --entrypoint "$ISAAC_PYTHON" "$ISAAC_IMAGE" -u "$@"
  ;;
*)
  echo "tools/run_isaac.sh: set ISAAC_RUNTIME=host or ISAAC_RUNTIME=docker (environment or .env.local)" >&2
  exit 2
  ;;
esac
