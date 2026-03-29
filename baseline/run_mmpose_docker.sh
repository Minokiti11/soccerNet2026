#!/usr/bin/env zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

docker run --rm -it \
  -v "${ROOT_DIR}:/workspace" \
  -w /workspace/baseline/mmpose \
  hakanardo/mmpose
