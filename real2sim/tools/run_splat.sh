#!/usr/bin/env bash
# Run a 3DGRUT entry point (train.py, render.py, ...) or a python module in the splat environment (envs/splat,
# setup/setup_splat_env.sh), its venv activated (its own CUDA 12.8 toolkit as CUDA_HOME and the build variables
# it persists), in a systemd scope with a memory cap as tools/run_capped.sh (which would put envs/recon and
# envs/cuda128 first instead).
# Usage: MEM=40G tools/run_splat.sh train.py [args...]          a script of 3DGRUT's checkout, run from it
#        MEM=40G tools/run_splat.sh -m calibrated_scene_reconstruction.<module> [args...]  a module of this repository (PYTHONPATH=src)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$(cd "$ROOT/envs/splat" && pwd -P)/3dgrut"
[ -x "$SRC/.venv/bin/python" ] || { echo "no splat environment: run setup/setup_splat_env.sh"; exit 1; }
# shellcheck disable=SC1091
source "$SRC/.venv/bin/activate"
cap() {
  local scope=(systemd-run --user --scope --quiet -p MemoryMax="${MEM:-40G}" -p MemorySwapMax=1G)
  if ! "${scope[@]}" true 2>/dev/null; then   # no systemd user session (a container, a bare remote command)
    echo "run_splat.sh: no systemd user session; running without the memory cap" >&2
    scope=()
  fi
  exec ${scope[@]+"${scope[@]}"} \
    /usr/bin/time -f "[run_splat] peak_rss_kb=%M elapsed=%e exit=%x" "$SRC/.venv/bin/python" -u "$@"
}
if [ "${1:-}" = "-m" ]; then
  export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
  cd "$ROOT"
  cap "$@"
fi
cd "$SRC"
cap "$@"
