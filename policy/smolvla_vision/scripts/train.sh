#!/usr/bin/env bash
# Train original LeRobot SmolVLA on the vision-only dataset for N epochs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

SMOLVLA_VENV="${SMOLVLA_VENV:-${REPO_ROOT}/../smolvla/.venv}"
PYTHON="${PYTHON:-${SMOLVLA_VENV}/bin/python}"
TORCHRUN="${TORCHRUN:-${SMOLVLA_VENV}/bin/torchrun}"

TASK_NAME="${TASK_NAME:-lift_bottle}"
TASK_CONFIG="${TASK_CONFIG:-demo}"
EXPERT_DATA_NUM="${EXPERT_DATA_NUM:-2}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-8}"
POLICY_PATH="${POLICY_PATH:-}"
CONFIG_PATH="${CONFIG_PATH:-scripts/train_config.yml}"

TASK_SETTINGS="${TASK_NAME}_${TASK_CONFIG}_${EXPERT_DATA_NUM}"
REPO_ID="${REPO_ID:-local/${TASK_SETTINGS}_vision}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data/${REPO_ID}}"
RUN_TAG="${RUN_TAG:-${REPO_ID##*/}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/smolvla_vision_${RUN_TAG}_e${TRAIN_EPOCHS}}"
JOB_NAME="${JOB_NAME:-smolvla_vision_${RUN_TAG}_e${TRAIN_EPOCHS}}"

NUM_GPUS="${NUM_GPUS:-1}"
NUM_NODES="${NUM_NODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

export PYTHONPATH="${REPO_ROOT}/lerobot/src:${PYTHONPATH:-}"

STEPS="$(
"${PYTHON}" - "${DATASET_ROOT}" "${REPO_ID}" "${BATCH_SIZE}" "${TRAIN_EPOCHS}" "${NUM_GPUS}" "${NUM_NODES}" <<'PY'
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
repo_id = sys.argv[2]
batch_size = int(sys.argv[3])
epochs = int(sys.argv[4])
num_gpus = int(sys.argv[5])
num_nodes = int(sys.argv[6])

sys.path.insert(0, str(Path.cwd() / "lerobot" / "src"))
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(repo_id=repo_id, root=root)
global_batch = max(1, batch_size * max(1, num_gpus) * max(1, num_nodes))
print(max(1, math.ceil(len(dataset) / global_batch) * epochs))
PY
)"

echo "[train] repo_id=${REPO_ID}"
echo "[train] dataset_root=${DATASET_ROOT}"
echo "[train] epochs=${TRAIN_EPOCHS} batch_size=${BATCH_SIZE} steps=${STEPS}"
echo "[train] output_dir=${OUTPUT_DIR}"

POLICY_ARGS=(--policy.path="${POLICY_PATH}")
if [[ -z "${POLICY_PATH}" ]]; then
    CONFIG_PATH="${SCRATCH_CONFIG_PATH:-scripts/train_config_scratch.yml}"
    POLICY_ARGS=(--policy.type=smolvla)
fi

"${TORCHRUN}" \
    --nproc_per_node="${NUM_GPUS}" \
    --nnodes="${NUM_NODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m lerobot.scripts.lerobot_train \
    --config_path "${CONFIG_PATH}" \
    "${POLICY_ARGS[@]}" \
    --policy.push_to_hub=false \
    --dataset.repo_id="${REPO_ID}" \
    --dataset.root="${DATASET_ROOT}" \
    --batch_size="${BATCH_SIZE}" \
    --steps="${STEPS}" \
    --save_freq="${STEPS}" \
    --output_dir="${OUTPUT_DIR}" \
    --job_name="${JOB_NAME}" \
    "$@"
