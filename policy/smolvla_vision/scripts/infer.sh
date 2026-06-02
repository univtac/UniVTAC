#!/usr/bin/env bash
# Run single-frame inference with a trained vision-only SmolVLA checkpoint.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

SMOLVLA_VENV="${SMOLVLA_VENV:-${REPO_ROOT}/../smolvla/.venv}"
PYTHON="${PYTHON:-${SMOLVLA_VENV}/bin/python}"

TASK_NAME="${TASK_NAME:-lift_bottle}"
TASK_CONFIG="${TASK_CONFIG:-demo}"
EXPERT_DATA_NUM="${EXPERT_DATA_NUM:-2}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-2}"
FRAME_INDEX="${FRAME_INDEX:-0}"

TASK_SETTINGS="${TASK_NAME}_${TASK_CONFIG}_${EXPERT_DATA_NUM}"
REPO_ID="${REPO_ID:-local/${TASK_SETTINGS}_vision}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data/${REPO_ID}}"
CKPT="${CKPT:-${REPO_ROOT}/outputs/smolvla_vision_${TASK_SETTINGS}_vision_e${TRAIN_EPOCHS}/checkpoints/last/pretrained_model}"

export PYTHONPATH="${REPO_ROOT}/lerobot/src:${PYTHONPATH:-}"

"${PYTHON}" scripts/infer.py \
    --ckpt "${CKPT}" \
    --dataset-root "${DATASET_ROOT}" \
    --repo-id "${REPO_ID}" \
    --frame-index "${FRAME_INDEX}" \
    "$@"
