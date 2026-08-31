#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
SOURCE_DATASET_ID="${SOURCE_DATASET_ID:-byml2024/UniVTAC}"
# Pinned release for which all 800 trajectories and legacy result fields were reviewed.
SOURCE_REVISION="${SOURCE_REVISION:-3d4646a7}"
OXT_DATASET_ID="${OXT_DATASET_ID:-byml2024/UniVTAC-OXT}"
UNIVTAC_RAW_DIR="${UNIVTAC_RAW_DIR:-$SCRIPT_DIR}"
OXT_WORK_DIR="${OXT_WORK_DIR:-$SCRIPT_DIR/oxt/UniVTAC-OXT}"
DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-8}"
CONVERT_WORKERS="${CONVERT_WORKERS:-4}"
UPLOAD_WORKERS="${UPLOAD_WORKERS:-8}"

usage() {
    cat <<'EOF'
用法：bash data/download.sh <command> [额外参数]

command:
  setup      安装 ModelScope 和转换依赖
  download   多 worker 下载 UniVTAC HDF5/metadata
  convert    以单个 HDF5 为粒度进行多进程转换
  package    打包 task Zarr，并生成 OXT 提交 metadata/logo
  upload     在本机准备待上传的 publish 文件夹（不会远程提交）
  submit     创建新 ModelScope 数据集并提交已准备好的 publish 文件夹
  all        依次执行 download、convert、package（不会上传）

额外参数会传给相应的转换、打包或提交脚本。
EOF
}

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "缺少命令：$1。请先运行 bash data/download.sh setup" >&2
        exit 1
    fi
}

setup() {
    "$PYTHON_BIN" -m pip install -r "$REPO_ROOT/scripts/oxt/requirements.txt"
}

download() {
    require_command modelscope
    mkdir -p "$UNIVTAC_RAW_DIR"
    modelscope download \
        --dataset "$SOURCE_DATASET_ID" \
        --revision "$SOURCE_REVISION" \
        --local_dir "$UNIVTAC_RAW_DIR" \
        --include '*/clean/*.hdf5' '*/clean/metadata.json' 'README.md' \
        --max-workers "$DOWNLOAD_WORKERS"
}

convert() {
    "$PYTHON_BIN" "$REPO_ROOT/scripts/oxt/convert_univtac.py" \
        --raw-dir "$UNIVTAC_RAW_DIR" \
        --output-dir "$OXT_WORK_DIR" \
        --workers "$CONVERT_WORKERS" \
        --source-dataset "$SOURCE_DATASET_ID" \
        --source-revision "$SOURCE_REVISION" \
        "$@"
}

package() {
    "$PYTHON_BIN" "$REPO_ROOT/scripts/oxt/package_submission.py" \
        --work-dir "$OXT_WORK_DIR" \
        --dataset-id "$OXT_DATASET_ID" \
        "$@"
}

upload() {
    package "$@"
    echo "待上传数据已保存在本机：$OXT_WORK_DIR/publish"
    echo "未进行远程提交；确认内容后请显式运行：bash data/download.sh submit"
}

submit() {
    "$PYTHON_BIN" "$REPO_ROOT/scripts/oxt/upload_modelscope.py" \
        --repo-id "$OXT_DATASET_ID" \
        --folder "$OXT_WORK_DIR/publish" \
        --workers "$UPLOAD_WORKERS" \
        --license "${MODELSCOPE_LICENSE:-MIT}" \
        "$@"
}

command="${1:-help}"
if [[ $# -gt 0 ]]; then
    shift
fi

case "$command" in
    setup) setup "$@" ;;
    download) download "$@" ;;
    convert) convert "$@" ;;
    package) package "$@" ;;
    upload) upload "$@" ;;
    submit) submit "$@" ;;
    all)
        download
        convert "$@"
        package
        ;;
    help|-h|--help) usage ;;
    *)
        echo "未知 command：$command" >&2
        usage >&2
        exit 2
        ;;
esac
