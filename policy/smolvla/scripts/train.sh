#!/usr/bin/env bash
# Train SmolVLA with the visuo-tactile extension.
#
# Defaults below set up a single-node LoRA finetune. Override any variable
# from the environment, e.g.:
#
#     NUM_GPUS=4 FUSION=cross_attn USE_LORA=1 bash scripts/train.sh
#
# Multi-node example (run on each node with appropriate rank / addr):
#     NUM_NODES=2 NODE_RANK=0 MASTER_ADDR=hostA MASTER_PORT=29500 \
#         NUM_GPUS=8 bash scripts/train.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

TASK_NAME="${TASK_NAME:-lift_bottle}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data/${TASK_NAME}_lerobot}"
REPO_ID="${REPO_ID:-local/${TASK_NAME}_vitac}"

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/smolvla_vitac_${TASK_NAME}}"
JOB_NAME="${JOB_NAME:-smolvla_vitac_${TASK_NAME}}"

# Fusion mode: "token" or "cross_attn".
FUSION="${FUSION:-token}"
USE_LORA="${USE_LORA:-1}"
LOAD_VLM_WEIGHTS="${LOAD_VLM_WEIGHTS:-1}"

BATCH_SIZE="${BATCH_SIZE:-8}"
STEPS="${STEPS:-30000}"
LR="${LR:-1e-4}"

# Distributed launch (works for single-GPU, multi-GPU, and multi-node).
NUM_GPUS="${NUM_GPUS:-1}"
NUM_NODES="${NUM_NODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

LORA_ARGS=()
if [[ "${USE_LORA}" == "1" ]]; then
    LORA_ARGS+=( --policy.use_peft=true --policy.peft_method=lora )
fi

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --nnodes="${NUM_NODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    src/lerobot/scripts/lerobot_train.py \
    --policy.type=smolvla \
    --policy.use_tactile=true \
    --policy.tactile_fusion="${FUSION}" \
    --policy.load_vlm_weights="${LOAD_VLM_WEIGHTS}" \
    --policy.optimizer_lr="${LR}" \
    --dataset.repo_id="${REPO_ID}" \
    --dataset.root="${DATASET_ROOT}" \
    --batch_size="${BATCH_SIZE}" \
    --steps="${STEPS}" \
    --output_dir="${OUTPUT_DIR}" \
    --job_name="${JOB_NAME}" \
    "${LORA_ARGS[@]}" \
    "$@"
