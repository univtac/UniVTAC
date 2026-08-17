#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
PYTHON_BIN="${VENV_DIR}/bin/python"
CUDA_ROOT="${UNIVTAC_CUDA_HOME:-/usr/local/cuda-12.6}"
CUDA_ARCH="${UNIVTAC_CUDA_ARCH:-89}"
BUILD_JOBS="${UNIVTAC_BUILD_JOBS:-8}"
VCPKG_ROOT="${UNIVTAC_VCPKG_ROOT:-${PROJECT_ROOT}/.cache/toolchains/vcpkg}"
CUROBO_DIR="${PROJECT_ROOT}/third_party/curobo"
CUROBO_COMMIT="ebb71702f3f70e767f40fd8e050674af0288abe8"

command -v uv >/dev/null 2>&1 || {
    echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
}
command -v cmake >/dev/null 2>&1 || {
    echo "cmake is required (install cmake and build-essential first)." >&2
    exit 1
}
if [[ ! -x "${CUDA_ROOT}/bin/nvcc" ]]; then
    echo "CUDA 12.6 was not found at ${CUDA_ROOT}. Set UNIVTAC_CUDA_HOME." >&2
    exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    uv venv --python 3.11 "${VENV_DIR}"
fi

uv pip install --python "${PYTHON_BIN}" \
    "setuptools==75.8.2" "setuptools-scm==8.1.0" "wheel==0.42.0" ninja pip
uv pip install --python "${PYTHON_BIN}" flatdict==4.0.1 --no-build-isolation
uv pip install --python "${PYTHON_BIN}" \
    "isaacsim[all,extscache]==5.1.0" \
    --extra-index-url https://pypi.nvidia.com
uv pip install --python "${PYTHON_BIN}" "isaaclab==2.3.0"
uv pip install --python "${PYTHON_BIN}" "packaging==23.0" "filelock==3.13.1"

uv pip install --python "${PYTHON_BIN}" --no-build-isolation \
    -e "${PROJECT_ROOT}/third_party/TacEx/source/tacex" \
    -e "${PROJECT_ROOT}/third_party/TacEx/source/tacex_assets"
uv pip install --python "${PYTHON_BIN}" pybind11 mypy transforms3d tetgen "polyscope>=2.5,<3"

if [[ ! -x "${VCPKG_ROOT}/vcpkg" ]]; then
    mkdir -p "$(dirname "${VCPKG_ROOT}")"
    git clone https://github.com/microsoft/vcpkg.git "${VCPKG_ROOT}"
    "${VCPKG_ROOT}/bootstrap-vcpkg.sh" -disableMetrics
fi

if [[ -n "${UNIVTAC_CC:-}" && -n "${UNIVTAC_CXX:-}" ]]; then
    BUILD_CC="${UNIVTAC_CC}"
    BUILD_CXX="${UNIVTAC_CXX}"
elif command -v gcc-12 >/dev/null 2>&1 && command -v g++-12 >/dev/null 2>&1; then
    BUILD_CC="$(command -v gcc-12)"
    BUILD_CXX="$(command -v g++-12)"
elif command -v x86_64-conda-linux-gnu-gcc >/dev/null 2>&1 \
    && command -v x86_64-conda-linux-gnu-c++ >/dev/null 2>&1; then
    BUILD_CC="${PROJECT_ROOT}/scripts/toolchains/gcc12-system-ld"
    BUILD_CXX="${PROJECT_ROOT}/scripts/toolchains/gxx12-system-ld"
else
    echo "GCC 12 is required. Install gcc-12/g++-12 or set UNIVTAC_CC and UNIVTAC_CXX." >&2
    exit 1
fi

export PATH="${CUDA_ROOT}/bin:${PATH}"
export CUDA_HOME="${CUDA_ROOT}"
export CUDA_PATH="${CUDA_ROOT}"
export CUDACXX="${CUDA_ROOT}/bin/nvcc"
export CUDAHOSTCXX="${BUILD_CXX}"
export CC="${BUILD_CC}"
export CXX="${BUILD_CXX}"
export CMAKE_TOOLCHAIN_FILE="${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
export CMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"
export CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS}"
export MAX_JOBS="${BUILD_JOBS}"
export TORCH_CUDA_ARCH_LIST="8.9"

uv pip install --python "${PYTHON_BIN}" --no-build-isolation \
    -e "${PROJECT_ROOT}/third_party/TacEx/source/tacex_uipc"

if [[ ! -d "${CUROBO_DIR}/.git" ]]; then
    git clone https://github.com/NVlabs/curobo.git "${CUROBO_DIR}"
    git -C "${CUROBO_DIR}" checkout "${CUROBO_COMMIT}"
fi
uv pip install --python "${PYTHON_BIN}" --no-build-isolation -e "${CUROBO_DIR}"

uv pip check --python "${PYTHON_BIN}"
"${PYTHON_BIN}" -c \
    'from importlib.metadata import version; import torch, uipc, curobo; print("Isaac Lab", version("isaaclab"), "Torch", torch.__version__)'
echo "UniVTAC Isaac Sim 5.1 environment is ready at ${VENV_DIR}."
