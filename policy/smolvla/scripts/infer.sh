#!/usr/bin/env bash
# Run inference with a trained visuo-tactile SmolVLA checkpoint on a sample
# drawn from a LeRobot v3.0 dataset.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

TASK_NAME="${TASK_NAME:-lift_bottle}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data/${TASK_NAME}_lerobot}"
REPO_ID="${REPO_ID:-local/${TASK_NAME}_vitac}"
CKPT="${CKPT:-${REPO_ROOT}/outputs/smolvla_vitac_${TASK_NAME}/checkpoints/last/pretrained_model}"
FRAME_INDEX="${FRAME_INDEX:-0}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

python scripts/infer.py \
    --ckpt "${CKPT}" \
    --dataset-root "${DATASET_ROOT}" \
    --repo-id "${REPO_ID}" \
    --frame-index "${FRAME_INDEX}" \
    "$@"
