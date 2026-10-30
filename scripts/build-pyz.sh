#!/bin/sh
# Build dist/tracta.pyz: one file that runs with any Python 3.10+ and nothing else installed.
set -eu
HERE=$(cd "$(dirname "$0")/.." && pwd)
B=$(mktemp -d)
trap 'rm -rf "$B"' EXIT
mkdir -p "$B/src" "$HERE/dist"
cp -R "$HERE/tracta" "$B/src/tracta"
find "$B/src" -name __pycache__ -prune -exec rm -rf {} +
python3 -m zipapp "$B/src" -m "tracta.cli:main" -p "/usr/bin/env python3" -o "$HERE/dist/tracta.pyz"
echo "built $HERE/dist/tracta.pyz ($(wc -c < "$HERE/dist/tracta.pyz") bytes)"
