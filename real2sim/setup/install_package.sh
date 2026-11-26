#!/usr/bin/env bash
# Install only this repository, editable, into existing Python environments.
# Usage: setup/install_package.sh [python ...]
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo "Usage: setup/install_package.sh [python ...]"
  echo "Install this package without changing dependencies; defaults to existing repository environments."
  exit 0
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [ "$#" -eq 0 ]; then
  PYS=()
  for candidate in envs/*/bin/python envs/splat/3dgrut/.venv/bin/python envs/lehome/venv/bin/python; do
    [ ! -x "$candidate" ] || PYS+=("$candidate")
  done
else
  PYS=("$@")
fi
[ "${#PYS[@]}" -gt 0 ] || { echo "No existing Python environment found" >&2; exit 1; }
for interpreter in "${PYS[@]}"; do
  # Do not resolve or upgrade third-party dependencies during a repository refactor.
  uv pip install --python "$interpreter" --no-deps --no-build-isolation -e "$ROOT"
  "$interpreter" -c 'import calibrated_scene_reconstruction, calibrated_scene_reconstruction.core.paths; print("calibrated_scene_reconstruction", calibrated_scene_reconstruction.__version__)'
done
