#!/usr/bin/env bash
# Compatibility launcher for tools/runtime/run_newton.sh.
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/runtime/run_newton.sh" "$@"
