#!/usr/bin/env bash
# Compatibility launcher.
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/physics/video_physics.sh" "$@"
