#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-h200}"
CUDA_ARCH="${LATTISENSE_CUDA_ARCH:-90}"
JOBS="${JOBS:-$(nproc)}"

for command_name in cmake git go nvcc; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Missing required build command: $command_name" >&2
    exit 2
  fi
done

if [[ "$CUDA_ARCH" != "90" ]]; then
  echo "Refusing non-H200 CUDA architecture: $CUDA_ARCH (expected 90)" >&2
  exit 2
fi

if [[ ! -f "$ROOT/inference/lattisense/CMakeLists.txt" ]]; then
  git -C "$ROOT" submodule update --init --recursive
fi

cmake -S "$ROOT" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DINFERENCE_SDK_ENABLE_GPU=ON \
  -DLATTISENSE_CUDA_ARCH="$CUDA_ARCH" \
  -DLATTISENSE_BUILD_TESTS=OFF

cmake --build "$BUILD_DIR" --target inference_batch --parallel "$JOBS"

binary="$BUILD_DIR/examples/inference_batch"
runtime="$BUILD_DIR/inference/lattisense/liblattisense.so"
if [[ ! -x "$binary" || ! -f "$runtime" ]]; then
  echo "Build completed without expected runtime artifacts" >&2
  exit 1
fi

echo "Built source-only H200 artifacts:"
sha256sum "$binary" "$runtime"
cuobjdump --list-elf "$runtime"
