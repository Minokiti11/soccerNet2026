#!/bin/bash
#SBATCH --partition=112-partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --job-name=vitpose-synloc
#SBATCH --output=/lustre/tenant112/user112002/soccernet_2026/mmpose/logs/slurm_%j.log
#SBATCH --time=24:00:00

set -euo pipefail

ROOT_DIR="/lustre/tenant112/user112002/soccernet_2026"
MMPOSE_DIR="${ROOT_DIR}/mmpose"
PYTHON_BIN="${ROOT_DIR}/rfdetr/.venv/bin/python"
CONFIG_PATH="configs/body_2d_keypoint/topdown_heatmap/spiideo_soccernet/td-hm_ViTPose-small-simple_8xb64-210e_synloc-256x192.py"
SOURCE_DATA_ROOT="${ROOT_DIR}/data/SpiideoSynLoc/4k"
DATA_ROOT="${ROOT_DIR}/data/SpiideoSynLoc/4k_coco_kpt"
REQUESTED_GPUS="${1:-${NPROC_PER_NODE:-2}}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="vitpose_small_simple_synloc_256x192_e10_bs32x${REQUESTED_GPUS}_${RUN_STAMP}"
WORK_DIR="${MMPOSE_DIR}/work_dirs/${RUN_NAME}"

mkdir -p "${MMPOSE_DIR}/logs" "${MMPOSE_DIR}/work_dirs"

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export SOCCERNET2026_DATA_ROOT="${DATA_ROOT}"
export PYTHONPATH="${MMPOSE_DIR}:${PYTHONPATH:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-soccernet2026-mmpose}"
export WANDB_ENTITY="${WANDB_ENTITY:-macminosp}"
export WANDB_NAME="${RUN_NAME}"
if [[ -f "${HOME}/.netrc" ]]; then
  export WANDB_MODE="${WANDB_MODE:-online}"
else
  echo "W&B login was not found, switching to offline mode."
  export WANDB_MODE=offline
fi

if ! [[ "${REQUESTED_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid gpu count: ${REQUESTED_GPUS}" >&2
  echo "Usage: sbatch --gres=gpu:<N> $0 <N>" >&2
  exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi was not found. Cannot check available GPUs." >&2
  exit 2
fi

declare -a VISIBLE_GPUS=()
declare -A VISIBLE_GPU_SET=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
  for gpu_id in "${VISIBLE_GPUS[@]}"; do
    gpu_id_trimmed="$(echo "${gpu_id}" | tr -d ' ')"
    if [[ -n "${gpu_id_trimmed}" ]]; then
      VISIBLE_GPU_SET["${gpu_id_trimmed}"]=1
    fi
  done
fi

declare -a FREE_GPU_IDS=()
while IFS=',' read -r raw_idx raw_mem raw_util; do
  gpu_idx="$(echo "${raw_idx}" | tr -d ' ')"
  mem_used="$(echo "${raw_mem}" | tr -d ' ')"
  util="$(echo "${raw_util}" | tr -d ' ')"
  if [[ ${#VISIBLE_GPU_SET[@]} -gt 0 && -z "${VISIBLE_GPU_SET[${gpu_idx}]+x}" ]]; then
    continue
  fi
  if (( mem_used <= 1024 && util <= 10 )); then
    FREE_GPU_IDS+=("${gpu_idx}")
  fi
done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)

AVAILABLE_GPU_COUNT="${#FREE_GPU_IDS[@]}"
if (( AVAILABLE_GPU_COUNT < REQUESTED_GPUS )); then
  echo "Requested GPUs: ${REQUESTED_GPUS}" >&2
  echo "Available GPUs: ${AVAILABLE_GPU_COUNT}" >&2
  echo "Free GPU IDs: ${FREE_GPU_IDS[*]:-none}" >&2
  exit 1
fi

SELECTED_GPU_IDS="$(printf "%s\n" "${FREE_GPU_IDS[@]}" | head -n "${REQUESTED_GPUS}" | paste -sd, -)"
export CUDA_VISIBLE_DEVICES="${SELECTED_GPU_IDS}"
echo "Using GPUs: ${CUDA_VISIBLE_DEVICES} (requested=${REQUESTED_GPUS}, available=${AVAILABLE_GPU_COUNT})"

if [[ -f "${DATA_ROOT}/annotations/train.json" && -f "${DATA_ROOT}/annotations/val.json" ]]; then
  echo "Using existing COCO-compatible annotations at ${DATA_ROOT}"
else
  echo "Preparing COCO-compatible annotations at ${DATA_ROOT}"
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/convert_synloc_keypoints_to_coco.py" \
    --source-root "${SOURCE_DATA_ROOT}" \
    --target-root "${DATA_ROOT}" \
    --splits train val
fi

# mmpretrain imports multimodal modules on initialization, which can break
# with newer transformers API changes. Guard that optional import.
"${PYTHON_BIN}" - <<'PY'
import glob
import sys
from pathlib import Path

candidates = glob.glob(str(Path(sys.prefix) / "lib/python*/site-packages/mmpretrain/models/__init__.py"))
if candidates:
    target = Path(candidates[0])
    text = target.read_text()
    needle = "from .multimodal import *  # noqa: F401,F403"
    replacement = "try:\n    from .multimodal import *  # noqa: F401,F403\nexcept Exception:\n    pass"
    if needle in text and replacement not in text:
        target.write_text(text.replace(needle, replacement))
        print(f"Patched optional multimodal import: {target}")
    else:
        print("mmpretrain multimodal import already patched")
else:
    print("mmpretrain models __init__.py not found; skip patch")
PY

cd "${MMPOSE_DIR}"

echo "hostname: $(hostname)"
echo "job_id: ${SLURM_JOB_ID}"
echo "run_name: ${RUN_NAME}"
echo "work_dir: ${WORK_DIR}"
echo "config: ${CONFIG_PATH}"
echo "dataset_root: ${SOCCERNET2026_DATA_ROOT}"
echo "wandb_project: ${WANDB_PROJECT}"
echo "wandb_entity: ${WANDB_ENTITY}"
echo "wandb_mode: ${WANDB_MODE}"

"${PYTHON_BIN}" --version
"${PYTHON_BIN}" -c "import torch; print('torch', torch.__version__)"

"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${REQUESTED_GPUS}" \
  tools/train.py \
  "${CONFIG_PATH}" \
  --launcher pytorch \
  --work-dir "${WORK_DIR}" \
  --amp
