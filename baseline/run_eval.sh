#!/usr/bin/env zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CHECKPOINT_PATH="${1:-}"
CONFIG_PATH="${2:-configs/body_bev_position/spiideo_soccernet/docker_yoloxpose_m_4xb64-300e_960.py}"

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  echo "usage: ./baseline/run_eval.sh <checkpoint-path> [config-path]" >&2
  exit 1
fi

docker run --rm \
  -v "${ROOT_DIR}:/workspace" \
  -w /workspace/baseline/mmpose \
  hakanardo/mmpose \
  python tools/test.py "${CONFIG_PATH}" "/workspace/${CHECKPOINT_PATH}"
