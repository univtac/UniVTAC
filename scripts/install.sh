#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV_NAME="${UNIVTAC_CONDA_ENV:-UniVTAC}"
CUDA_ARCH_INPUT="${UNIVTAC_CUDA_ARCH:-89}"
BUILD_JOBS="${UNIVTAC_BUILD_JOBS:-8}"
VCPKG_ROOT="${UNIVTAC_VCPKG_ROOT:-${PROJECT_ROOT}/.cache/toolchains/vcpkg}"
VCPKG_COMMIT="dd3097e305afa53f7b4312371f62058d2e665320"
CUROBO_DIR="${PROJECT_ROOT}/third_party/curobo"
CUROBO_COMMIT="ebb71702f3f70e767f40fd8e050674af0288abe8"
UIPC_DIR="${PROJECT_ROOT}/third_party/TacEx/source/tacex_uipc"
CHECK_ONLY=0

usage() {
    cat <<EOF
Usage: $(basename "$0") [--check]

Create or reuse a Conda environment and install the UniVTAC Isaac Sim 5.1 stack.

Options:
  --check  Validate an existing environment without installing anything.
  -h, --help

Environment variables:
  UNIVTAC_CONDA_ENV      Conda environment name (default: UniVTAC)
  UNIVTAC_CUDA_HOME      External CUDA 12.6 toolkit (default: Conda environment)
  UNIVTAC_CUDA_ARCH      CUDA compute capability, 89 or 8.9 (default: 89)
  UNIVTAC_BUILD_JOBS     Parallel UIPC/cuRobo build jobs (default: 8)
  UNIVTAC_VCPKG_ROOT     Existing or project-local vcpkg path
  UNIVTAC_CC/CXX         Optional host compiler overrides
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --check)
            CHECK_ONLY=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

if [[ ! "${CONDA_ENV_NAME}" =~ ^[A-Za-z0-9._-]+$ ]] || [[ "${CONDA_ENV_NAME}" == "base" ]]; then
    echo "UNIVTAC_CONDA_ENV must be a named, non-base Conda environment." >&2
    exit 1
fi

command -v conda >/dev/null 2>&1 || {
    echo "Conda is required. Install Miniconda/Anaconda, then rerun this script." >&2
    exit 1
}
command -v git >/dev/null 2>&1 || {
    echo "Git is required." >&2
    exit 1
}

CUDA_ARCH="${CUDA_ARCH_INPUT//./}"
if [[ ! "${CUDA_ARCH}" =~ ^[0-9]{2,3}$ ]]; then
    echo "UNIVTAC_CUDA_ARCH must look like 89 or 8.9." >&2
    exit 1
fi
TORCH_CUDA_ARCH="${CUDA_ARCH:0:${#CUDA_ARCH}-1}.${CUDA_ARCH: -1}"

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

activate_conda_env() {
    # Some Conda activation scripts inspect optional variables and are not
    # compatible with Bash nounset mode.
    local status
    set +u
    conda activate "$1"
    status=$?
    set -u
    return "${status}"
}

if ! activate_conda_env "${CONDA_ENV_NAME}" >/dev/null 2>&1; then
    if [[ "${CHECK_ONLY}" -eq 1 ]]; then
        echo "Conda environment '${CONDA_ENV_NAME}' does not exist." >&2
        exit 1
    fi
    echo "[1/7] Creating Conda environment '${CONDA_ENV_NAME}' with Python 3.11."
    conda create --name "${CONDA_ENV_NAME}" --yes --override-channels \
        --channel https://repo.anaconda.com/pkgs/main python=3.11 pip
    activate_conda_env "${CONDA_ENV_NAME}"
else
    echo "[1/7] Reusing Conda environment '${CONDA_ENV_NAME}' at ${CONDA_PREFIX}."
fi

if [[ "${CHECK_ONLY}" -eq 0 ]]; then
    echo "Updating '${CONDA_ENV_NAME}' with the TacEx/libuipc Conda toolchain."
    conda env update --name "${CONDA_ENV_NAME}" \
        --file "${UIPC_DIR}/libuipc/conda/env.yaml"
    activate_conda_env "${CONDA_ENV_NAME}"
fi

PYTHON_BIN="${CONDA_PREFIX}/bin/python"
PIP=("${PYTHON_BIN}" -m pip)
CUDA_ROOT="${UNIVTAC_CUDA_HOME:-${CONDA_PREFIX}}"

if [[ "$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.11" ]]; then
    echo "Conda environment '${CONDA_ENV_NAME}' must use Python 3.11." >&2
    exit 1
fi
if [[ ! -x "${CUDA_ROOT}/bin/nvcc" ]]; then
    echo "CUDA nvcc was not found at ${CUDA_ROOT}/bin/nvcc." >&2
    echo "Run the TacEx Conda environment update or set UNIVTAC_CUDA_HOME." >&2
    exit 1
fi
if ! "${CUDA_ROOT}/bin/nvcc" --version | tail -n 1 | grep -q "release 12\.6"; then
    echo "UniVTAC must use CUDA 12.6; ${CUDA_ROOT} is a different toolkit." >&2
    exit 1
fi

export PATH="${CUDA_ROOT}/bin:${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUDA_HOME="${CUDA_ROOT}"
export CUDA_PATH="${CUDA_ROOT}"
export CUDACXX="${CUDA_ROOT}/bin/nvcc"
export CMAKE_TOOLCHAIN_FILE="${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
export CMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"
export CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS}"
export MAX_JOBS="${BUILD_JOBS}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH}"

if [[ -n "${UNIVTAC_CC:-}" || -n "${UNIVTAC_CXX:-}" ]]; then
    if [[ -z "${UNIVTAC_CC:-}" || -z "${UNIVTAC_CXX:-}" ]]; then
        echo "Set both UNIVTAC_CC and UNIVTAC_CXX, or neither." >&2
        exit 1
    fi
    export CC="${UNIVTAC_CC}"
    export CXX="${UNIVTAC_CXX}"
else
    CONDA_GCC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
    CONDA_GXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-c++"
    if [[ ! -x "${CONDA_GCC}" || ! -x "${CONDA_GXX}" ]]; then
        echo "GCC/G++ 12 are missing from '${CONDA_ENV_NAME}'." >&2
        exit 1
    fi
    # Use Conda GCC with the host linker to avoid old-linker RELR errors.
    export UNIVTAC_GCC12="${CONDA_GCC}"
    export UNIVTAC_GXX12="${CONDA_GXX}"
    export CC="${PROJECT_ROOT}/scripts/toolchains/gcc12-system-ld"
    export CXX="${PROJECT_ROOT}/scripts/toolchains/gxx12-system-ld"
fi
export CUDAHOSTCXX="${CXX}"

check_environment() {
    "${PIP[@]}" check
    "${PYTHON_BIN}" - <<'PY'
from importlib.metadata import PackageNotFoundError, version

expected = {
    "isaacsim": "5.1.0.0",
    "isaaclab": "2.3.0",
    "torch": "2.7.0",
    "tacex": "0.1.0",
    "tacex-assets": "0.1.0",
    "tacex-uipc": "0.1.0",
    "pyuipc": "0.9.0",
}
failed = False
for package, wanted in expected.items():
    try:
        installed = version(package)
    except PackageNotFoundError:
        print(f"[missing] {package}")
        failed = True
        continue
    ok = installed == wanted
    print(f"[{'ok' if ok else 'wrong'}] {package}=={installed}")
    failed |= not ok

# Editable cuRobo metadata depends on setuptools-scm and whether the source
# checkout is dirty. Both forms below are produced by the pinned revision.
try:
    curobo_version = version("nvidia-curobo")
except PackageNotFoundError:
    print("[missing] nvidia-curobo")
    failed = True
else:
    ok = curobo_version == "0.0.0" or curobo_version.startswith("0.7.7")
    print(f"[{'ok' if ok else 'wrong'}] nvidia-curobo=={curobo_version}")
    failed |= not ok

import torch
import uipc
import curobo

if torch.version.cuda != "12.6":
    print(f"[wrong] PyTorch CUDA runtime is {torch.version.cuda}, expected 12.6")
    failed = True
else:
    print(f"[ok] PyTorch CUDA {torch.version.cuda}")
if failed:
    raise SystemExit(1)
PY
    echo "Environment '${CONDA_ENV_NAME}' is ready. Activate it with: conda activate ${CONDA_ENV_NAME}"
}

if [[ "${CHECK_ONLY}" -eq 1 ]]; then
    check_environment
    exit 0
fi

echo "[2/7] Installing Isaac Sim 5.1 and Isaac Lab 2.3.0."
"${PIP[@]}" install \
    "setuptools==75.8.2" "setuptools-scm==8.1.0" "wheel==0.42.0"
"${PIP[@]}" install flatdict==4.0.1 --no-build-isolation
"${PIP[@]}" install "torch==2.7.0" "torchvision==0.22.0" \
    --index-url https://download.pytorch.org/whl/cu126
"${PIP[@]}" install \
    "isaaclab[isaacsim,all]==2.3.0" \
    --extra-index-url https://pypi.nvidia.com
# These versions match the validated Isaac 5.1/UIPC build environment.
"${PIP[@]}" install \
    "setuptools==75.8.2" "setuptools-scm==8.1.0" \
    "packaging==23.0" "filelock==3.13.1" "wheel==0.42.0"

echo "[3/7] Installing the vendored TacEx core and assets."
"${PIP[@]}" install --no-build-isolation \
    -e "${PROJECT_ROOT}/third_party/TacEx/source/tacex" \
    -e "${PROJECT_ROOT}/third_party/TacEx/source/tacex_assets"
"${PIP[@]}" install pybind11 mypy transforms3d tetgen "polyscope>=2.5,<3"

echo "[4/7] Preparing the pinned vcpkg toolchain."
if [[ ! -x "${VCPKG_ROOT}/vcpkg" ]]; then
    if [[ -e "${VCPKG_ROOT}" && ! -d "${VCPKG_ROOT}/.git" ]]; then
        echo "${VCPKG_ROOT} exists but is not a usable vcpkg checkout." >&2
        exit 1
    fi
    if [[ ! -d "${VCPKG_ROOT}/.git" ]]; then
        mkdir -p "$(dirname "${VCPKG_ROOT}")"
        git clone https://github.com/microsoft/vcpkg.git "${VCPKG_ROOT}"
    fi
    git -C "${VCPKG_ROOT}" fetch origin "${VCPKG_COMMIT}"
    git -C "${VCPKG_ROOT}" checkout --detach "${VCPKG_COMMIT}"
    "${VCPKG_ROOT}/bootstrap-vcpkg.sh" -disableMetrics
fi

echo "[5/7] Building the vendored libuipc Python binding."
UIPC_CACHE="${UIPC_DIR}/build/CMakeCache.txt"
if [[ -f "${UIPC_CACHE}" ]]; then
    CACHED_PYTHON="$(sed -n 's|^UIPC_PYTHON_EXECUTABLE_PATH:PATH=||p' "${UIPC_CACHE}" | head -n 1)"
    CACHED_TOOLCHAIN="$(sed -n 's|^CMAKE_TOOLCHAIN_FILE:FILEPATH=||p' "${UIPC_CACHE}" | head -n 1)"
    if [[ "${CACHED_PYTHON}" != "${PYTHON_BIN}" || "${CACHED_TOOLCHAIN}" != "${CMAKE_TOOLCHAIN_FILE}" ]]; then
        echo "Removing stale UIPC CMake cache from another Python/toolchain."
        cmake -E remove_directory "${UIPC_DIR}/build"
    fi
fi
"${PIP[@]}" install --no-build-isolation -e "${UIPC_DIR}"

echo "[6/7] Installing cuRobo at ${CUROBO_COMMIT}."
if [[ ! -d "${CUROBO_DIR}/.git" ]]; then
    if [[ -e "${CUROBO_DIR}" ]]; then
        echo "${CUROBO_DIR} exists but is not a Git checkout." >&2
        exit 1
    fi
    git clone https://github.com/NVlabs/curobo.git "${CUROBO_DIR}"
    git -C "${CUROBO_DIR}" checkout --detach "${CUROBO_COMMIT}"
fi
if [[ "$(git -C "${CUROBO_DIR}" rev-parse HEAD)" != "${CUROBO_COMMIT}" ]]; then
    echo "Existing cuRobo checkout is not at ${CUROBO_COMMIT}." >&2
    echo "Move it aside or check out that revision without discarding local work." >&2
    exit 1
fi
"${PIP[@]}" install --no-build-isolation -e "${CUROBO_DIR}"

echo "[7/7] Validating package versions and compiled modules."
check_environment
