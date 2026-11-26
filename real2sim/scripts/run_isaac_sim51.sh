#!/usr/bin/env bash
# Use the same mounted paths/entrypoints with an isolated Sim 5.1 runtime.
set -euo pipefail
export REAL2SIM_ISAAC_PROFILE=5.1
exec "$(dirname "${BASH_SOURCE[0]}")/run_isaac_lab3.sh" "$@"
