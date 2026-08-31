# Data Collection

UniVTAC's data synthesizer enables fully automated data collection by executing scripted manipulation policies (defined in the `envs/` directory) in combination with the cuRobo motion planner. Data collection is configured through task-shared configuration files in `task_config/`, which define parameters such as the target tactile sensor type, observation modalities, texture randomization, and the number of episodes to collect.

The pipeline iterates over random seeds, executing the scripted policy for each seed and saving observation data on success. Failed seeds are skipped automatically, and progress is tracked in `suc_map.txt` to support resuming from interruptions. The entire process is fully automated — just run a single command to get started.

Running the following command will start data collection for the specified task:

```bash
bash collect_data.sh ${task_name} ${config_name} ${gpu_id}
# Example: bash collect_data.sh lift_bottle demo 0
```

For faster collection with multiple parallel simulation workers: (Note: the parallel collection is implemented with Python's multiprocessing, so multiple Isaac Sim Apps will be launched on the same time)

```bash
bash parallel_collect.sh ${task_name} ${config_name} ${gpu_id} [num_processes]
# Example: bash parallel_collect.sh lift_bottle demo 0 3
```

All available `task_name` options correspond to Python modules in the `envs/` directory (e.g., `lift_bottle`, `insert_HDMI`, `pull_out_key`, `grasp_classify`, etc.). The `config_name` parameter specifies a YAML configuration file in `task_config/` (without the `.yml` extension). The `gpu_id` parameter specifies which GPU to use (multiple GPUs are supported).

## Task Configuration

| Field | Type | Description |
|---|---|---|
| `env_settings.frequencies.physical` | `int` | Physics frequency in Hz; defines `sim.dt`. |
| `env_settings.frequencies.collect` | `int` | Collection control frequency in Hz. |
| `env_settings.frequencies.save` | `int` | HDF5 sample frequency in Hz. |
| `env_settings.frequencies.eval` | `int` | Evaluation control frequency in Hz. |
| `env_settings.frequencies.video` | `int` | Video frequency in Hz (`0` disables video). |
| `env_settings.frequencies.render` | `int` | Render frequency in Hz (`0` disables periodic rendering). |
| `env_settings.random_texture` | `bool` | Enable random texture domain randomization. |
| `env_settings.sensor_type` | `str` | `gsmini`, `gf225`, or `xensews`. |
| `env_settings.optical_backend` | `str` | `taxim` or `pix2pix`. |
| `collect_settings.save_root_dir` | `str` | Root directory for collected datasets. |
| `collect_settings.use_seed` | `bool` | Resume deterministic seed bookkeeping. |
| `collect_settings.episode_num` | `int` | Number of successful episodes to collect. |
| `replay_settings.save_root_dir` | `str` | Root directory for evaluation artifacts. |
| `replay_settings.force_action` | `bool` | Force replayed actions when enabled. |
| `replay_settings.max_episodes` | `int` | Maximum replay episode count. |
| `observation_settings` | `dict` | Observation modalities to record (see below). |

All frequencies use `physical` as their clock reference. The launcher derives
collection decimation as `physical / collect`, HDF5 `save_frequency` as
`collect / save`, and evaluation decimation as `physical / eval`. Replay always
consumes every recorded trajectory row (`action_stride=1`); `save` controls
collection writes only and does not constrain `eval`. Non-integral clock ratios
are rejected before Isaac Sim starts. Configuration values can be overridden
with an OmegaConf dotlist:

```bash
python scripts/collect_data.py lift_bottle clean --headless \
  --config-overrides collect_settings.episode_num=30

python scripts/replay.py lift_bottle clean --headless \
  --config-overrides env_settings.frequencies.eval=30 \
                     env_settings.frequencies.video=30
```

## Data Structure

After data collection is completed, the collected data will be stored under `data/${task_name}/${config_name}/`:

- Each episode's observation and action data are saved as an individual HDF5 file in the `hdf5/` directory.
- Visualization videos of each episode (combining camera and tactile views) can be found in the `video/` directory.
- Per-episode metadata (step counts, timing, success/failure results) is stored in `metadata.json`.
- The `suc_map.txt` and `scene/` directory are auxiliary outputs generated during the data collection process.

Below is the structure of the saved observation data for each episode (stored in HDF5 format). `HDF5Handler` in `envs/utils/data.py` can be used to read and write this data format:

```json
{
    "actor": {
        "prism": "np.ndarray(7,)",
        "prism_base": "np.ndarray(7,)",
        "slot": "np.ndarray(7,)"
    },
    "atom": {
        "id": "type: <class \"numpy.int64\">",
        "tag": "type: <class \"numpy.bytes_\">"
    },
    "embodiment": {
        "ee": "np.ndarray(7,)",
        "joint": "np.ndarray(9,)"
    },
    "observation": {
        "head": {
            "rgb": "np.ndarray(270, 480, 3)"
        },
        "wrist": {
            "rgb": "np.ndarray(270, 480, 3)"
        }
    },
    "step": "type: <class \"numpy.int64\">",
    "tactile": {
        "left_tactile": {
            "depth": "np.ndarray(240, 320), legacy raw camera distance in mm",
            "press_depth": "np.ndarray(240, 320), positive indentation in mm",
            "marker": "np.ndarray(2, 63, 2)",
            "pose": "np.ndarray(7,)",
            "rgb": "np.ndarray(240, 320, 3)",
            "rgb_marker": "np.ndarray(240, 320, 3)"
        },
        "right_tactile": {
            "depth": "np.ndarray(240, 320), legacy raw camera distance in mm",
            "press_depth": "np.ndarray(240, 320), positive indentation in mm",
            "marker": "np.ndarray(2, 63, 2)",
            "pose": "np.ndarray(7,)",
            "rgb": "np.ndarray(240, 320, 3)",
            "rgb_marker": "np.ndarray(240, 320, 3)"
        }
    }
}
```

Tactile control and Atom APIs use `press_depth`: `0` means no indentation and
larger positive values mean a deeper press. The legacy `depth` field is kept in
the dataset for compatibility with existing UniVTAC data. Requesting `depth`
records both fields.
