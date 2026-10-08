#!/bin/bash
# Pack the code and the Mac bundle into one archive for the GPU (a few MB; the corpus is rebuilt there).
#     bash scale/cloud/pack.sh ~/.cache/segresearch-scale/bundle  scale_pilot.tgz
set -euo pipefail
BUNDLE=${1:?bundle dir}; OUT=${2:-scale_pilot.tgz}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
TMP=$(mktemp -d)
mkdir -p "$TMP/pilot/scale/cloud" "$TMP/pilot/results/registry"
cp "$ROOT"/scale/*.py "$ROOT"/scale/requirements.txt "$TMP/pilot/scale/"
cp "$ROOT"/scale/cloud/*.sh "$ROOT"/scale/cloud/*.py "$TMP/pilot/scale/cloud/"
cp "$ROOT"/registry.py "$TMP/pilot/"
cp -r "$BUNDLE" "$TMP/pilot/bundle"
(cd "$ROOT" && git rev-parse HEAD) > "$TMP/pilot/COMMIT"
tar czf "$OUT" -C "$TMP" pilot
rm -rf "$TMP"
echo "packed $(du -h "$OUT" | cut -f1) -> $OUT (commit $(cd "$ROOT" && git rev-parse --short HEAD))"
