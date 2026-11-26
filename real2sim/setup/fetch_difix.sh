#!/usr/bin/env bash
# Obtain the tested optional Difix source without modifying an existing checkout.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$ROOT/third_party/Difix3D"
REV=c76edc595586e16732c91ddee82f3a6d83a8a9cc
if [ -e "$TARGET" ]; then
  ACTUAL="$(git -C "$TARGET" rev-parse HEAD)"
  [ "$ACTUAL" = "$REV" ] || { echo "Difix checkout must be at $REV; existing checkout left unchanged" >&2; exit 1; }
  [ -z "$(git -C "$TARGET" status --porcelain --untracked-files=no)" ] || { echo "Difix checkout has local changes; left unchanged" >&2; exit 1; }
else
  git clone --no-checkout --filter=blob:none https://github.com/nv-tlabs/Difix3D.git "$TARGET"
  git -C "$TARGET" checkout --detach "$REV"
fi
printf 'Difix source: %s\n' "$REV"
