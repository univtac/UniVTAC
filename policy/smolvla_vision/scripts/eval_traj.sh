#!/usr/bin/env bash
# Evaluate the trained vision-only SmolVLA checkpoint for one dataset epoch.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

SMOLVLA_VENV="${SMOLVLA_VENV:-${REPO_ROOT}/../smolvla/.venv}"
PYTHON="${PYTHON:-${SMOLVLA_VENV}/bin/python}"

TASK_NAME="${TASK_NAME:-lift_bottle}"
TASK_CONFIG="${TASK_CONFIG:-demo}"
EXPERT_DATA_NUM="${EXPERT_DATA_NUM:-2}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-2}"

TASK_SETTINGS="${TASK_NAME}_${TASK_CONFIG}_${EXPERT_DATA_NUM}"
REPO_ID="${REPO_ID:-local/${TASK_SETTINGS}_vision}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data/${REPO_ID}}"
CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/outputs/smolvla_vision_${TASK_SETTINGS}_vision_e${TRAIN_EPOCHS}/checkpoints/last/pretrained_model}"
CSV="${CSV:-${REPO_ROOT}/outputs/eval_${TASK_SETTINGS}_vision_e${TRAIN_EPOCHS}.csv}"

export PYTHONPATH="${REPO_ROOT}/lerobot/src:${PYTHONPATH:-}"

"${PYTHON}" scripts/eval_traj.py \
    --dataset-root "${DATASET_ROOT}" \
    --repo-id "${REPO_ID}" \
    --ckpt-dir "${CKPT_DIR}" \
    --task-name "${TASK_NAME}" \
    --task-config "${TASK_CONFIG}" \
    --expert-data-num "${EXPERT_DATA_NUM}" \
    --train-epochs "${TRAIN_EPOCHS}" \
    --csv "${CSV}" \
    "$@"
