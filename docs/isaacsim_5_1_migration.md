# Isaac Sim 5.1 migration notes

This branch targets Isaac Sim 5.1, Isaac Lab 2.3.0 and Python 3.11. The exact
TacEx/libuipc revisions are recorded in
`third_party/TacEx/UNIVTAC_BASELINE.md`.

## Environment

Create/update the Conda environment, then install the complete stack:

```bash
bash scripts/install.sh
conda activate UniVTAC
```

The installer first applies TacEx's libuipc Conda YAML, then installs Isaac Sim,
Isaac Lab, the local TacEx packages, vendored libuipc, and cuRobo into that same
environment. CUDA is uniformly pinned to 12.6. See [Install.md](./Install.md)
for prerequisites, exact versions, and validation.

The TacEx Taxim dependency is pinned to the Python 3.11 / PyTorch 2.7 / CUDA
12.6 `torch_scatter` wheel used by Isaac Sim 5.1 (the upstream branch still
contained a Python 3.10 / PyTorch 2.8 URL).

## GelSight optical backend

The backend is selected before the environment starts. Task YAML files accept:

```yaml
sensor_type: gsmini
optical_backend: taxim  # or pix2pix
```

Taxim remains the default. Pix2Pix uses the vendored upstream pretrained
checkpoint. Two GelSight sensors on the same device share one cached generator
model. Both backends expose the same task-facing output:

- `tactile_rgb`: HWC `torch.uint8`, range 0–255
- `marker_rgb`: HWC `torch.uint8`, range 0–255

Pix2Pix internally returns HWC float images in the 0–1 range, while the
existing calibrated GPU Taxim path returns float images in the 0–255 range.
The sensor boundary normalizes both to the contract above.

The sensor camera must already exist in the USD at `<sensor prim>/Camera`.
`TiledCameraCfg.spawn` is always `None`. Camera config may override focal
length, focus distance, FoV/aperture and clipping range, but does not change the
camera hierarchy or pose. TiledCamera's own depth clipping is disabled because
Isaac Lab 2.3 dereferences `spawn.clipping_range` in that path; the GelSight
layer applies the configured clipping range when constructing depth and height
maps.

The calibrated GelSight Mini topology provides the 82 gel-to-case attachment
vertex ids. They are resolved before simulation starts, so startup no longer
depends on a PhysX sweep query before the timeline is playing. If the rounded
FEM surface places the nominal outer marker centres just outside its convex
hull, the complete marker grid is uniformly inset by the smallest 0.1% step
that fits (bounded to 2%); marker count and correspondence are preserved.

## Actor body and pose API

Task props are rigid by default:

```python
actor = manager.add_from_usd_file(
    name="object",
    asset_path="object.usd",
    pose=pose,
    body_type="rigid",  # or "deformable"
)
```

`actor.set_pose(pose, soft=False)` restores the initial/rest mesh at the
requested pose, clears velocity through the appropriate UIPC StateAccessor and
disables its target constraint. `soft=True` retains the dynamic state and uses
`SoftTransformConstraint` for rigid actors or `SoftPositionConstraint` for
deformable actors.

Phase one intentionally supports `num_envs=1`. Public method signatures retain
environment-index arguments, but partial/multi-environment UIPC state writes
raise `NotImplementedError` until the scene distribution work is implemented.

## Render synchronization

UniVTAC disables TacEx's automatic UIPC render-mesh physics callback. On every
requested render, `_update_render()` performs one UIPC surface-to-Fabric copy,
then one Isaac Sim render, then reads RTX/tactile outputs. This avoids duplicate
Fabric writes and the old extra render call while ensuring the RTX cameras see
the latest UIPC surface.

## Runtime verification

The reusable phase-one smoke test validates the two USD cameras, backend output
shape/dtype, Actor hard/soft pose behavior, and exactly one UIPC render copy:

```bash
OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python scripts/smoke_isaac51.py \
  --backend taxim --headless
OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python scripts/smoke_isaac51.py \
  --backend pix2pix --headless
```

The full Taxim task used for phase-one validation is:

```bash
OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python scripts/collect_data.py \
  grasp_classify demo --max_seed 0 --headless \
  --config-overrides collect_settings.episode_num=1
```
