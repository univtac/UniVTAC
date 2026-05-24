set -euo pipefail

# 检查最后一个参数是否为 -e
last_arg="${@: -1}"
skip_to_eval=false
if [ "$last_arg" = "-e" ]; then
    skip_to_eval=true
    # 移除 -e 参数，避免影响其他参数
    set -- "${@:1:$(($#-1))}"
fi

task_name=${1}
task_config=${2}
gpu_id=${3}
train_config=${4:-"auto"}
expert_data_num=${5:-50}

lerobot_data_home=${LEROBOT_DATA_HOME:-"$HOME/.cache/lerobot"}
repo_id=local/${task_name}-${task_config}-${expert_data_num}_vitac

if [ "$train_config" = "auto" ]; then
    train_config="pi0_fast_franka_univtac_lora_${task_name}-${task_config}-${expert_data_num}"
fi

# 如果不是 -e 模式，执行完整流程
if [ "$skip_to_eval" = false ]; then
    if [ -d "${lerobot_data_home}/${repo_id}" ]; then
        echo "Processed data for $task_name already exists. Skipping data processing."
    else
        echo "Processing data for $task_name..."
        python scripts/process_data.py \
            --task-name $task_name \
            --task-config $task_config \
            --expert-data-num $expert_data_num \
            --repo-id $repo_id
    fi

    export TASK_NAME="${task_name}"
    bash scripts/train.sh
fi

# # 执行评估部分

# GREEN='\033[0;32m'
# RESET='\033[0m'

# echo -e "
# ${GREEN}=======================================
# Task  Name  : $task_name
# Run   Config: $task_config
# Train Config: $train_config
# GPU         : $gpu_id
# =======================================${RESET}
# "

# POLICY_ROOT=$(cd "$(dirname "$0")" && pwd)
# export OPENPI_DATA_HOME=$POLICY_ROOT/openpi

# cd ../../
# export TRAIN_CONFIG=$train_config
# export EP_NUM=$expert_data_num
# # bash parallel_eval.sh $task_name $task_config pi0/deploy $gpu_id
# bash eval_policy.sh $task_name $task_config pi0/deploy $gpu_id
