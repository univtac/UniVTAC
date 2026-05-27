#!/usr/bin/env bash
# Train SmolVLA with the visuo-tactile extension.
#
# Training defaults live in scripts/train_config.yml. Override task-specific
# paths from the environment, e.g.:
#
#     TASK_NAME=grasp_classify NUM_GPUS=4 bash scripts/train.sh
#
# Multi-node example (run on each node with appropriate rank / addr):
#     NUM_NODES=2 NODE_RANK=0 MASTER_ADDR=hostA MASTER_PORT=29500 \
#         NUM_GPUS=8 bash scripts/train.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

TASK_NAME="${TASK_NAME:-lift_bottle}"
TASK_CONFIG="${TASK_CONFIG:-demo}"
EXPERT_DATA_NUM="${EXPERT_DATA_NUM:-2}"

export TASK_SETTINGS="${TASK_NAME}_${TASK_CONFIG}_${EXPERT_DATA_NUM}"
export REPO_ID="${REPO_ID:-local/${TASK_SETTINGS}_vitac}"
export DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data/${REPO_ID}}"

RUN_TAG="${RUN_TAG:-${REPO_ID##*/}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/smolvla_vitac_${RUN_TAG}}"
JOB_NAME="${JOB_NAME:-smolvla_vitac_${RUN_TAG}}"

# Distributed launch (works for single-GPU, multi-GPU, and multi-node).
NUM_GPUS="${NUM_GPUS:-1}"
NUM_NODES="${NUM_NODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --nnodes="${NUM_NODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m lerobot.scripts.lerobot_train \
    --config_path scripts/train_config.yml \
    --dataset.repo_id="${REPO_ID}" \
    --dataset.root="${DATASET_ROOT}" \
    --output_dir="${OUTPUT_DIR}" \
    --job_name="${JOB_NAME}" \
    "$@"
