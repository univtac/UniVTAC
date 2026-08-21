# Installation (Isaac Sim 5.1)

The `isaac51` branch uses the following fixed environment:

| Component | Version / source |
|---|---|
| Environment | Conda environment `UniVTAC` |
| Python | 3.11 |
| Isaac Sim | 5.1.0 |
| Isaac Lab | 2.3.0 |
| PyTorch | 2.7.0 + cu126 |
| TacEx / libuipc | vendored sources under `third_party/TacEx` |
| CUDA toolkit | Conda CUDA toolkit 12.6 |
| Host compiler | Conda GCC/G++ 12 |

Do not install public TacEx over this environment. UniVTAC depends on the
vendored camera, optical-backend, UIPC, Actor, and render-synchronization
changes.

## CUDA version

The complete environment uses CUDA 12.6. This matches Isaac Sim 5.1's PyTorch
2.7.0 cu126 runtime and the toolkit used to build the already-validated UIPC
and cuRobo extensions. The TacEx/libuipc Conda YAML is pinned to
`cuda-toolkit=12.6`; do not update it to CUDA 13 for this branch because
PyTorch rejects CUDA extensions compiled with a different major version.

## Prerequisites

- Linux with an NVIDIA RTX GPU and a sufficiently recent NVIDIA driver
- Miniconda or Anaconda, with `conda` available in the current shell
- Git and standard Linux build tools (`build-essential`, `curl`, `zip`,
  `unzip`, and `pkg-config` on Ubuntu)
- Enough memory for the libuipc C++/CUDA build; reduce the build job count on
  machines with limited RAM

For Ubuntu, the non-CUDA host prerequisites can be installed with:

```bash
sudo apt update
sudo apt install build-essential curl git pkg-config unzip zip
```

## Automated installation

```bash
git clone https://github.com/univtac/UniVTAC.git
cd UniVTAC
git checkout isaac51
bash scripts/install.sh
conda activate UniVTAC
```

The installer performs these operations in order:

1. Creates `UniVTAC` with Python 3.11, or reuses the named environment.
2. Immediately runs `conda env update` with
   `third_party/TacEx/source/tacex_uipc/libuipc/conda/env.yaml`. This installs
   CUDA 12.6, CMake 3.26, Ninja, and GCC/G++ 12.
3. Installs PyTorch 2.7.0/torchvision 0.22.0 from the cu126 index, followed by
   Isaac Sim 5.1 and Isaac Lab 2.3.0 in the same Conda environment.
4. Installs the vendored TacEx core/assets and builds vendored libuipc with
   Conda CUDA 12.6.
5. Builds the pinned cuRobo revision with the same CUDA 12.6 toolkit.
6. Checks package compatibility, exact versions, and compiled imports.

The script does not create `.venv` and does not modify shell startup files. An
old project-local `.venv` is ignored and may be removed manually after the
Conda installation has been verified.

To use another environment name or GPU architecture:

```bash
UNIVTAC_CONDA_ENV=UniVTAC-isaac51 \
UNIVTAC_CUDA_ARCH=89 \
UNIVTAC_BUILD_JOBS=4 \
bash scripts/install.sh
```

`UNIVTAC_CUDA_ARCH` accepts either `89` or `8.9`. The default targets the RTX
40-series GPU used for phase-one validation. Set the correct compute capability
for another GPU.

If vcpkg is already available, set `UNIVTAC_VCPKG_ROOT`. Otherwise the pinned
revision is cloned under the ignored project-local `.cache/toolchains/vcpkg`
directory.

## Verification

Static environment validation does not reinstall anything:

```bash
conda activate UniVTAC
bash scripts/install.sh --check
```

On the first Isaac Sim launch, review and accept the NVIDIA Omniverse EULA. On
an already-approved non-interactive machine, set `OMNI_KIT_ACCEPT_EULA=YES` for
the launch command.

Run the phase-one smoke test:

```bash
OMNI_KIT_ACCEPT_EULA=YES python scripts/smoke_isaac51.py \
  --backend taxim --headless
```

Run one headless `grasp_classify` episode with one Franka and two GelSight Mini
sensors:

```bash
OMNI_KIT_ACCEPT_EULA=YES python scripts/collect_data.py \
  grasp_classify demo --episode_num 1 --max_seed 0 --headless
```

The optical backend is selected before environment startup in task YAML:

```yaml
sensor_type: gsmini
optical_backend: taxim  # or pix2pix
```

The pretrained Pix2Pix checkpoint is already vendored under the GelSight Mini
assets; the installer does not download another copy.

## Troubleshooting

- If `conda activate` is unavailable in a non-interactive shell, run
  `conda init` once or source `<conda-base>/etc/profile.d/conda.sh`.
- The TacEx YAML uses `nodefaults` and explicit `main`/`conda-forge` channels,
  so an obsolete `pkgs/r` entry in a user's global Conda configuration cannot
  break environment solving.
- If the UIPC build directory was created by another Python environment or
  vcpkg path, the installer removes that generated CMake cache before rebuilding.
- If a build reports that detected CUDA does not match PyTorch CUDA 12.6,
  check that `CUDA_HOME`, `CUDA_PATH`, and `nvcc` all resolve inside the active
  Conda environment. Do not leave a CUDA 13 path exported in the parent shell.
- If libuipc is killed by the OOM killer, retry with
  `UNIVTAC_BUILD_JOBS=4` (or lower).
- Phase one supports `num_envs=1`; multi-environment UIPC state distribution is
  intentionally deferred.

See [Isaac Sim 5.1 migration notes](./isaacsim_5_1_migration.md) for the camera
ownership, tactile output contract, Actor pose API, and render pipeline.
