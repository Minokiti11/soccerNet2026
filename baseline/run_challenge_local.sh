#!/usr/bin/env zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CHECKPOINT_PATH="${1:-}"
CONFIG_PATH="${2:-baseline/mmpose/configs/body_bev_position/spiideo_soccernet/local_yoloxpose_m_4xb64-300e_960.py}"

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  echo "usage: ./baseline/run_challenge_local.sh <checkpoint-path> [config-path]" >&2
  exit 1
fi

cd "${ROOT_DIR}/baseline/mmpose"
export SOCCERNET2026_ROOT="${ROOT_DIR}"
python tools/test.py "${ROOT_DIR}/${CONFIG_PATH}" "${ROOT_DIR}/${CHECKPOINT_PATH}" --challenge
