#!/bin/sh
# Build static tracta binaries for macOS, Linux and Windows into dist/ (no cgo, stripped).
set -eu
HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "$HERE"
for target in darwin/arm64 darwin/amd64 linux/amd64 linux/arm64 windows/amd64; do
  os=${target%/*}; arch=${target#*/}; ext=""; [ "$os" = windows ] && ext=".exe"
  CGO_ENABLED=0 GOOS=$os GOARCH=$arch go build -trimpath -ldflags "-s -w" -o "dist/tracta-$os-$arch$ext" ./cmd/tracta
done
(cd dist && shasum -a 256 tracta-* > SHA256SUMS)
ls -la dist/tracta-* && cat dist/SHA256SUMS
