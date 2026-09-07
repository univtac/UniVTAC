# TLabel Format Export

This page describes how to convert raw UniVTAC HDF5 episodes into the
[TLabel](https://github.com/liesliy/tlabel) tactile annotation format.

**TLabel** is a unified annotation schema for robot tactile data — the tactile
counterpart of COCO in vision. It defines a common set of per-frame tactile
features (contact, deformation, force, slip, shear, optical flow, …) so that
datasets collected with different sensors can be labelled, visualised and
benchmarked with the same toolchain.

The exporter is implemented in
[`policy/tlabel_data_preprocessor.py`](../policy/tlabel_data_preprocessor.py)
as `TLabelDataPreprocessor`, a subclass of
[`policy/_base_data_preprocessor.py`](../policy/_base_data_preprocessor.py).
It reuses the base data loading pipeline unchanged (episode discovery,
selection, down-sampling and the new/old tactile key fallback) and replaces
the ACT-style HDF5 export with TLabel JSON export. The actual tactile feature
extraction and schema construction are delegated to the upstream
`UniVTACAdapter` shipped with the `tlabel` package
(`tlabel.adapters.univtac.UniVTACAdapter`).

## Installation

`tlabel` is an **optional** dependency — the rest of UniVTAC does not require
it. To export TLabel files:

```bash
pip install tlabel        # tested with tlabel 0.22.x
```

The exporter also needs the usual preprocessing dependencies (`h5py`,
`numpy`, `opencv-python`, `tqdm`; all already required by UniVTAC). If
`tlabel` is not installed, the import and `load_data()` still work, and
calling the export raises a clear `ImportError` with install instructions.

## Minimal usage

```bash
# Convert the first 50 episodes of data/<task_name>/<task_config>/
python -m policy.tlabel_data_preprocessor <task_name> <task_config> 50 \
    --output ./data/tlabel/<task_name>-<task_config>

# Concrete example:
python -m policy.tlabel_data_preprocessor insert_hole demo 50 \
    --output ./data/tlabel/insert_hole-demo
```

Useful options:

| Option | Default | Description |
|---|---|---|
| `--output/-o DIR` | `./data/tlabel/<task>-<config>-<n>` | Output directory |
| `--tactile-cameras` | `left right` | Tactile sides to export (`left`, `right`) |
| `--visual-cameras` | `head` | Accepted for CLI parity with the base pipeline; visual streams are not part of the TLabel schema |
| `--downsample-factor N` | `1` | Frame stride, identical semantics to the base preprocessor |
| `--random-select` | off | Pick episodes randomly instead of the first N |
| `--validate` | off | Re-load every exported file with `tlabel.load()` to verify it |

You can also use the class directly:

```python
from policy.tlabel_data_preprocessor import TLabelDataPreprocessor

processor = TLabelDataPreprocessor("insert_hole", "demo")
metadata = processor.run(
    save_root_path="./data/tlabel/insert_hole-demo",
    tactile_cameras=("left", "right"),
    episode_num=50,
)
summary = TLabelDataPreprocessor.validate_output("./data/tlabel/insert_hole-demo")
```

## Output layout

For every selected episode and every exported tactile side, one
`<output_dir>/episode_{i:04d}_{side}.tlabel.json` file is written (TLabel
Schema V2 JSON). An index file `tlabel_metadata.json` summarises the export:

```
<output_dir>/
├── episode_0000_left.tlabel.json      # left sensor, episode 0
├── episode_0000_right.tlabel.json     # right sensor, episode 0
├── episode_0001_left.tlabel.json
├── ...
└── tlabel_metadata.json               # export manifest / index
```

Each `.tlabel.json` contains the standard TLabel sections: `sensor`
(sensor model / resolution / marker count / sampling rate), `episode`
(provenance: source file, original vs exported frame counts, task/config),
`capabilities` (which features are populated), and `frames` — one entry per
exported frame with the Schema V2 tactile features, `manipulation_phase`,
`confidence` and `sensor_specific` extras (sensor pose, end-effector pose,
joint states, actor object pose when present).

The files can be read back with the released tlabel toolchain, e.g.:

```python
import tlabel
data = tlabel.load("episode_0000_left.tlabel.json")
print(data.num_frames)                 # number of frames
frame = data.get_frame(10)             # TLabelFrame with schema_v2 features
```

## Supported input fields

The exporter reads the raw episode HDF5 files documented in
[`docs/Collection.md`](./Collection.md). Per tactile sensor group
(`tactile/<side>_tactile/` or `tactile/<side>_gsmini/`):

| HDF5 field | Shape | Used for |
|---|---|---|
| `depth` | `(T, H, W)` float | contact, deformation, contact area, normal field, contact centroid, edge/texture, frame-to-frame force magnitude/direction, delta force, friction-cone ratio, contact transition |
| `marker` | `(T, 2, marker_size, 2)` float | slip event/entropy, optical-flow magnitude/direction, shear-field magnitude/direction |
| `pose` | `(T, 7)` float | stored per frame in `sensor_specific.pose` (sensor mounting pose) |
| `rgb`, `rgb_marker` | JPEG byte streams | not exported (see below) |

Episode-level fields:

| HDF5 field | Used for |
|---|---|
| `step` | frame timestamps / frame count |
| `atom/tag` | `manipulation_phase` mapping (approach/grasp/lift/place/…) |
| `embodiment/ee`, `embodiment/joint` | stored per frame in `sensor_specific` (`ee_pose`, `joint_states`) |
| `actor/<object>` | first actor object pose stored in `sensor_specific` |
| `observation/head/rgb`, `observation/wrist/rgb` | loaded by the base pipeline for policy training but **not** part of TLabel (visual-only; see below) |

**Fields not exported by design.** TLabel is a *tactile* annotation schema;
visual RGB streams (`observation/head|wrist/rgb`, tactile `rgb`/`rgb_marker`)
and robot actions are not part of it. They are still loaded through the base
pipeline for consistency with the policy/data preprocessing, and the robot
state fields are preserved inside each frame's `sensor_specific` block for
provenance. Visual images are neither copied nor referenced by the exported
JSON.

## Marker handling

The marker dataset always has shape `(T, 2, marker_size, 2)`: the two
position sets are the reference and the current marker positions, each with
`(x, y)` image coordinates. **`marker_size` is inferred dynamically from the
array shape for every episode/sensor** and recorded in the output
(`sensor.layout.marker_count`); it is never hard-coded. The same code path
therefore handles:

- the current GelSight Mini collector output — 7×9 = **63** markers;
- GF225 (9×9 = 81) and XenseWS (11×20 = 220) sensor configs in
  `envs/sensors/tactile.py`;
- the older dense marker arrays — **1200** markers (see compatibility below).

## HDF5 format compatibility

UniVTAC raw episodes exist in two tactile layouts; the exporter detects the
layout automatically (the same probe as `BaseDataPreprocessor.load_data`):

| | Current collector format | Old ModelScope release format |
|---|---|---|
| Tactile groups | `tactile/left_tactile`, `tactile/right_tactile` | `tactile/left_gsmini`, `tactile/right_gsmini` |
| Markers | GelSight Mini, 7×9 = **63** (shape `(T, 2, 63, 2)`) | dense arrays, **1200** (shape `(T, 2, 1200, 2)`) |
| Probe order | tried first | automatic fallback when `*_tactile` is absent |
| Supported | ✅ marker count inferred from data | ✅ marker count inferred from data |

Frame-count semantics follow the base preprocessor: with
`downsample_factor=1` the exported frames correspond to raw frames
`0 … T-2` (the base pipeline pairs frame `t` as state with `t+1` as action),
i.e. an episode with `T = 57` raw frames exports 56 TLabel frames. Use
`--downsample-factor N` to stride frames as `arange(0, T-1, N)`.

## Verification

The `--validate` flag (or `TLabelDataPreprocessor.validate_output(dir)`)
re-loads every produced `.tlabel.json` with `tlabel.load()` and reports the
frame counts, providing a round-trip sanity check of the export.
