#!/usr/bin/env bash
# Fetch ONE sample ACE table (Hg-198, 80198.710nc, ~3.5 MB) used in the endf docs.
#
# LANL only ships the whole ENDF71x library (~3.5 GB .tgz), so we stream the archive
# and stop as soon as the wanted member is extracted (bsdtar -q / GNU tar --occurrence).
# Download volume depends on where the member sits in the archive.
#
# Usage:  bash scripts/download_ace_sample.sh [member-glob]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
OUT_DIR="$DATA_DIR/raw/ace"
MEMBER="${1:-80198.710nc}"
URL="https://nucleardata.lanl.gov/lib/endf71x.tgz"

mkdir -p "$OUT_DIR"
if [ -f "$OUT_DIR/$MEMBER" ]; then echo "already present: $OUT_DIR/$MEMBER"; exit 0; fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
echo ">> Streaming $URL looking for */$MEMBER"
if tar --version 2>/dev/null | grep -q bsdtar; then
  curl -fsL "$URL" | tar -xzqf - -C "$TMP" --include="*/$MEMBER" || true
else
  curl -fsL "$URL" | tar -xzf - -C "$TMP" --wildcards --occurrence=1 "*/$MEMBER" || true
fi
f="$(find "$TMP" -type f -name "$MEMBER" | head -1)"
[ -n "$f" ] || { echo "member not found"; exit 1; }
mv "$f" "$OUT_DIR/$MEMBER"
chmod 644 "$OUT_DIR/$MEMBER"
echo ">> Saved $OUT_DIR/$MEMBER"
