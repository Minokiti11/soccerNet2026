#!/bin/bash
#SBATCH --job-name=vitpose-synloc
#SBATCH --partition=<your-partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%j_vitpose_train.log

set -euo pipefail

# ==== edit these paths ====
REPO_ROOT="${REPO_ROOT:-$PWD}"
MMPOSE_ROOT="${MMPOSE_ROOT:-/path/to/mmpose_spiideo_scenes}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-configs/body_2d_keypoint/topdown_heatmap/spiideo_soccernet/td-hm_ViTPose-small-simple_8xb64-210e_synloc-256x192.py}"
SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-/path/to/SpiideoSynLoc/4k}"
DATA_ROOT="${DATA_ROOT:-/path/to/SpiideoSynLoc/4k_coco_kpt}"
REQUESTED_GPUS="${1:-${NPROC_PER_NODE:-2}}"
WORK_DIR="${WORK_DIR:-${REPO_ROOT}/artifacts/vitpose_train_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${REPO_ROOT}/logs" "${REPO_ROOT}/artifacts"

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export SOCCERNET2026_DATA_ROOT="${DATA_ROOT}"
export PYTHONPATH="${MMPOSE_ROOT}:${PYTHONPATH:-}"

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

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/convert_synloc_keypoints_to_coco.py" \
  --source-root "${SOURCE_DATA_ROOT}" \
  --target-root "${DATA_ROOT}" \
  --splits train val

cd "${MMPOSE_ROOT}"

"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${REQUESTED_GPUS}" \
  tools/train.py \
  "${CONFIG_PATH}" \
  --launcher pytorch \
  --work-dir "${WORK_DIR}" \
  --amp

echo "done: ${WORK_DIR}"
