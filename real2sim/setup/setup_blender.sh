#!/usr/bin/env bash
# Install the latest official Blender 4.5 LTS portable release, verified against its SHA256 manifest.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE=https://download.blender.org/release/Blender4.5
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
curl --fail --location --retry 3 "$BASE/" -o "$TMP/index"
VERSION=$(python3 - "$TMP/index" <<'PY'
import re, sys
versions = set(re.findall(r'blender-(4\.5\.\d+)-linux-x64.tar.xz', open(sys.argv[1]).read()))
if not versions:
    raise SystemExit('No Blender 4.5 Linux x64 release in official index')
print(max(versions, key=lambda v: tuple(map(int, v.split('.')))))
PY
)
ARCHIVE="blender-$VERSION-linux-x64.tar.xz"
curl --fail --location --retry 3 "$BASE/$ARCHIVE" -o "$TMP/$ARCHIVE"
curl --fail --location --retry 3 "$BASE/blender-$VERSION.sha256" -o "$TMP/checksums"
(cd "$TMP"; awk -v f="$ARCHIVE" '$2 == f || $2 == "*" f {print}' checksums > selected.sha256; test -s selected.sha256; sha256sum -c selected.sha256)
mkdir -p "$ROOT/third_party/blender"
tar -xf "$TMP/$ARCHIVE" -C "$ROOT/third_party/blender"
ln -sfn "blender-$VERSION-linux-x64" "$ROOT/third_party/blender/current"
cp "$TMP/selected.sha256" "$ROOT/third_party/blender/installed.sha256"
"$ROOT/third_party/blender/current/blender" --version
