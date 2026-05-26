#!/bin/bash
set -e

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4:-42}
gpu_id=${5:-0}
train_config=${6:-"train_config"}

export CUDA_VISIBLE_DEVICES=${gpu_id}
export HYDRA_FULL_ERROR=1

python process_data.py "${task_name}" "${task_config}" "${expert_data_num}"

python train.py \
  --config-dir=./ \
  --config-name="${train_config}" \
  dataset_path="data/sim-${task_name}/${task_config}-${expert_data_num}" \
  task_name="${task_name}" \
  training.seed="${seed}" \
  hydra.run.dir="unit_ckpt/unit-${task_name}/${task_config}-${expert_data_num}/${train_config}"
