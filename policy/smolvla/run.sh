set -euo pipefail

export POLICY_ROOT=$(cd "$(dirname "$0")" && pwd)
export TASK_NAME=grasp_classify
export TASK_CONFIG=demo

scripts/train.sh