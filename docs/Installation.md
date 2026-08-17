# Installation (Isaac Sim 5.1)

The `isaac51` branch targets this fixed stack:

- Linux with an NVIDIA RTX GPU and a driver compatible with CUDA 12.6
- Python 3.11 in the project-local `.venv`
- Isaac Sim 5.1.0 and Isaac Lab 2.3.0
- PyTorch 2.7 / CUDA 12.6
- the modified TacEx and libuipc sources vendored in this repository

Do not install public TacEx over this environment. UniVTAC depends on its
local camera, UIPC, actor and optical-backend changes.

## Automated installation

Install `uv`, CMake, Git and GCC/G++ 12, then run:

```bash
git clone https://github.com/univtac/UniVTAC.git
cd UniVTAC
git checkout isaac51
./scripts/install.sh
```

The script creates `.venv`, installs Isaac Sim/Lab and the local TacEx
packages, builds the vendored libuipc Python binding, and installs cuRobo at the
revision used for this migration. It does not alter shell startup files or a
Conda environment.

The default CUDA toolkit path is `/usr/local/cuda-12.6` and the default GPU
architecture is SM 8.9. Override build detection when needed:

```bash
UNIVTAC_CUDA_HOME=/path/to/cuda-12.6 \
UNIVTAC_CUDA_ARCH=89 \
UNIVTAC_CC=/path/to/gcc-12 \
UNIVTAC_CXX=/path/to/g++-12 \
UNIVTAC_BUILD_JOBS=8 \
./scripts/install.sh
```

If vcpkg already exists, set `UNIVTAC_VCPKG_ROOT`. Otherwise it is cloned
under the ignored project-local `.cache/toolchains` directory.

## Verification

On the first Isaac Sim launch, review and accept the NVIDIA Omniverse EULA at
the prompt. For an already-approved non-interactive machine, NVIDIA also
supports setting `OMNI_KIT_ACCEPT_EULA=YES` for the launch command.

Run one headless Taxim episode with one Franka and two GelSight Mini sensors:

```bash
.venv/bin/python scripts/collect_data.py \
  grasp_classify demo --episode_num 1 --max_seed 0 --headless
```

The optical backend is selected before the environment starts in the task
YAML:

```yaml
sensor_type: gsmini
optical_backend: taxim  # or pix2pix
```

See [Isaac Sim 5.1 migration notes](./isaacsim_5_1_migration.md) for the
camera ownership, output contract, actor pose semantics and render pipeline.

## Build troubleshooting

- Keep both `CUDA_HOME` and `CUDA_PATH` on CUDA 12.6. A stale `CUDA_PATH`
  pointing at CUDA 13 can make CMake select incompatible headers.
- libuipc is memory intensive. Reduce `UNIVTAC_BUILD_JOBS` to 4 if the build
  swaps heavily or is killed by the OOM killer.
- The included `scripts/toolchains/gcc12-system-ld` wrappers address Conda GCC
  12 installations whose bundled old linker cannot read the host glibc RELR
  sections. A normal system GCC/G++ 12 installation does not need them.
- Phase one supports `num_envs=1`; multi-environment UIPC state distribution is
  deliberately left for the next phase.
