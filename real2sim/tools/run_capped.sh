#!/usr/bin/env bash
# Run a heavy job in its own systemd scope with a memory cap: an OOM then kills only this job instead of the
# whole terminal scope (which also holds the shell / agent session). Without a systemd user session the job runs uncapped.
# Usage: MEM=40G tools/run_capped.sh <cmd> [args...]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CUDA_HOME="$ROOT/envs/cuda128"
export PATH="$ROOT/envs/recon/bin:$ROOT/envs/cuda128/bin:$PATH"
export CPATH="$ROOT/envs/cuda128/targets/x86_64-linux/include"
export LIBRARY_PATH="$ROOT/envs/cuda128/lib:$ROOT/envs/cuda128/targets/x86_64-linux/lib"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"   # another architecture: e.g. 8.6
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
SCOPE=(systemd-run --user --scope --quiet -p MemoryMax="${MEM:-40G}" -p MemorySwapMax=1G)
if ! "${SCOPE[@]}" true 2>/dev/null; then   # no systemd user session (a container, a bare remote command)
  echo "run_capped.sh: no systemd user session; running without the memory cap" >&2
  SCOPE=()
fi
exec ${SCOPE[@]+"${SCOPE[@]}"} \
  /usr/bin/time -f "[run_capped] peak_rss_kb=%M elapsed=%e exit=%x" "$@"
