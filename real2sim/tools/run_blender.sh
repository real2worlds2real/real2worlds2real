#!/usr/bin/env bash
# A factory-startup Blender subprocess in its own memory-capped scope. Arguments follow --python harness.py.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BLENDER="${SCENE_RECONSTRUCTION_BLENDER:-${R2S2R_BLENDER:-$ROOT/third_party/blender/current/blender}}"
[ -x "$BLENDER" ] || { echo 'Run setup/setup_blender.sh first' >&2; exit 1; }
SCOPE=(systemd-run --user --scope --quiet -p MemoryMax="${MEM:-40G}" -p MemorySwapMax=1G)
if ! "${SCOPE[@]}" true 2>/dev/null; then   # no systemd user session (a container, a bare remote command)
  echo "run_blender.sh: no systemd user session; running without the memory cap" >&2
  SCOPE=()
fi
exec ${SCOPE[@]+"${SCOPE[@]}"} \
  "$BLENDER" --background --factory-startup --threads "${BLENDER_THREADS:-4}" --python-exit-code 1 "$@"
