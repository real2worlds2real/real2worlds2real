#!/usr/bin/env bash
# Compatibility launcher for tools/tasks/setup_lehome.sh.
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/tasks/setup_lehome.sh" "$@"
