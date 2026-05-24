#!/usr/bin/env bash
# Preprocess raw HDF5 episodes into a LeRobot v3.0 dataset for visuo-tactile SmolVLA.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

TASK_NAME="${TASK_NAME:-lift_bottle}"
SRC_DIR="${SRC_DIR:-${REPO_ROOT}/data/${TASK_NAME}}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/data/${TASK_NAME}_lerobot}"
REPO_ID="${REPO_ID:-local/${TASK_NAME}_vitac}"
FPS="${FPS:-30}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

python scripts/preprocess_hdf5_to_lerobot.py \
    --src "${SRC_DIR}" \
    --out "${OUT_DIR}" \
    --repo-id "${REPO_ID}" \
    --fps "${FPS}"
