#!/usr/bin/env bash
# Preprocess raw UniVTAC HDF5 episodes into a vision-only LeRobot v3 dataset.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

SMOLVLA_VENV="${SMOLVLA_VENV:-${REPO_ROOT}/../smolvla/.venv}"
PYTHON="${PYTHON:-${SMOLVLA_VENV}/bin/python}"

TASK_NAME="${TASK_NAME:-lift_bottle}"
TASK_CONFIG="${TASK_CONFIG:-demo}"
EXPERT_DATA_NUM="${EXPERT_DATA_NUM:-2}"
FPS="${FPS:-30}"

TASK_SETTINGS="${TASK_NAME}_${TASK_CONFIG}_${EXPERT_DATA_NUM}"
REPO_ID="${REPO_ID:-local/${TASK_SETTINGS}_vision}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/data/${REPO_ID}}"

export PYTHONPATH="${REPO_ROOT}/lerobot/src:${PYTHONPATH:-}"

"${PYTHON}" scripts/process_data.py \
    "${TASK_NAME}" \
    "${TASK_CONFIG}" \
    "${EXPERT_DATA_NUM}" \
    --out "${OUT_DIR}" \
    --repo-id "${REPO_ID}" \
    --fps "${FPS}" \
    --videos \
    "$@"
