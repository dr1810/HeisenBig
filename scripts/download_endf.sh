#!/usr/bin/env bash
# Download the ENDF/B-VIII.1 neutron sublibrary (ENDF-6 format, ~344 MB) from NNDC
# and extract it to data/raw/endf-b8.1/neutrons/.
#
# Usage (host or container):  bash scripts/download_endf.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
DL_DIR="$DATA_DIR/downloads"
OUT_DIR="$DATA_DIR/raw/endf-b8.1/neutrons"

URL="https://www.nndc.bnl.gov/endf-releases/releases/B-VIII.1/neutrons/neutrons-version.VIII.1.tar.gz"
MD5="1a6abeac85bd2425df47983752687a93"   # from NNDC metadata.json
TARBALL="$DL_DIR/neutrons-version.VIII.1.tar.gz"

mkdir -p "$DL_DIR" "$OUT_DIR"

if [ ! -f "$TARBALL" ]; then
  echo ">> Downloading $URL"
  curl -fL --retry 3 -o "$TARBALL.part" "$URL"
  mv "$TARBALL.part" "$TARBALL"
fi

echo ">> Verifying MD5"
if command -v md5sum >/dev/null; then
  actual="$(md5sum "$TARBALL" | awk '{print $1}')"
else
  actual="$(md5 -q "$TARBALL")"
fi
[ "$actual" = "$MD5" ] || { echo "MD5 mismatch: $actual != $MD5"; exit 1; }

echo ">> Extracting to $OUT_DIR"
# The archive contains a top-level directory; flatten so files land directly in OUT_DIR.
tar -xzf "$TARBALL" -C "$OUT_DIR" --strip-components=1
echo ">> Done: $(find "$OUT_DIR" -name '*.endf' | wc -l | tr -d ' ') ENDF files in $OUT_DIR"
