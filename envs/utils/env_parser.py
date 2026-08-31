"""Unified OmegaConf task configuration and environment construction.

This module deliberately has no Isaac Sim imports at module import time.  The
configuration can therefore be loaded, overridden, and validated before the
``AppLauncher`` starts.  Isaac task modules are imported only by
``create_task_env`` after the caller has launched the application.
"""

from __future__ import annotations

import argparse
import importlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from omegaconf import DictConfig, MISSING, OmegaConf


CONFIG_ROOT = Path(__file__).resolve().parent.parent.parent / "task_config"
FIXED_RENDERING_MODE = "quality"
FIXED_POST_REPLAY_DELAY_STEPS = 50


@dataclass
class FrequencySettings:
    physical: int = MISSING
    collect: int = MISSING
    save: int = MISSING
    video: int = MISSING
    eval: int = MISSING
    render: int = MISSING


@dataclass
class EnvSettings:
    frequencies: FrequencySettings = field(default_factory=FrequencySettings)
    random_texture: bool = MISSING
    sensor_type: str = MISSING
    optical_backend: str = MISSING


@dataclass
class CollectSettings:
    save_root_dir: str = MISSING
    use_seed: bool = MISSING
    episode_num: int = MISSING


@dataclass
class ReplaySettings:
    save_root_dir: str = MISSING
    force_action: bool = MISSING
    max_episodes: int = MISSING


@dataclass
class TaskRunConfig:
    env_settings: EnvSettings = field(default_factory=EnvSettings)
    collect_settings: CollectSettings = field(default_factory=CollectSettings)
    replay_settings: ReplaySettings = field(default_factory=ReplaySettings)
    observation_settings: dict[str, Any] = field(default_factory=dict)
    task_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TimingPlan:
    mode: Literal["collect", "eval"]
    physical_hz: int
    control_hz: int
    collect_hz: int
    save_hz: int
    eval_hz: int
    video_hz: int
    render_hz: int
    decimation: int
    action_stride: int
    save_period_collect_steps: int
    video_period_physics_steps: int | None
    render_period_physics_steps: int | None
    video_period_control_steps: int
    render_period_control_steps: int


@dataclass
class TaskEnv:
    task: Any
    env_cfg: Any
    task_module: Any
    timing: TimingPlan
    save_dir: Path


def add_config_override_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config-overrides",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "OmegaConf dotlist overrides, for example "
            "env_settings.frequencies.eval=30 "
            "replay_settings.max_episodes=30."
        ),
    )


def resolve_config_path(config: str | Path) -> Path:
    path = Path(config)
    if path.suffix not in {".yml", ".yaml"}:
        path = CONFIG_ROOT / f"{path}.yml"
    elif not path.is_absolute() and not path.exists():
        path = CONFIG_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Task config does not exist: {path}")
    return path


def _require_int(name: str, value: Any, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {value!r}.")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        comparator = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {comparator}, got {value}.")
    return value


def _require_divides(numerator_name: str, numerator: int, name: str, value: int) -> int:
    if numerator % value != 0:
        raise ValueError(
            f"Invalid frequencies: {numerator_name}={numerator} is not evenly "
            f"divisible by {name}={value} (remainder {numerator % value})."
        )
    return numerator // value


def _event_period(
    *,
    name: str,
    event_hz: int,
    physical_hz: int,
    control_hz: int,
    decimation: int,
) -> tuple[int | None, int]:
    if event_hz == 0:
        return None, 0
    physical_period = _require_divides("physical", physical_hz, name, event_hz)
    if physical_period % decimation != 0:
        raise ValueError(
            f"Invalid frequencies for {control_hz} Hz control: {name}={event_hz} "
            f"Hz has a {physical_period}-physics-step period, which is not evenly "
            f"divisible by decimation={decimation}."
        )
    return physical_period, physical_period // decimation


def timing_plan(config: DictConfig, mode: Literal["collect", "eval"]) -> TimingPlan:
    frequencies = config.env_settings.frequencies
    physical_hz = _require_int("env_settings.frequencies.physical", frequencies.physical)
    collect_hz = _require_int("env_settings.frequencies.collect", frequencies.collect)
    save_hz = _require_int("env_settings.frequencies.save", frequencies.save)
    eval_hz = _require_int("env_settings.frequencies.eval", frequencies.eval)
    video_hz = _require_int(
        "env_settings.frequencies.video", frequencies.video, allow_zero=True
    )
    render_hz = _require_int(
        "env_settings.frequencies.render", frequencies.render, allow_zero=True
    )

    collect_decimation = _require_divides(
        "physical", physical_hz, "collect", collect_hz
    )
    eval_decimation = _require_divides("physical", physical_hz, "eval", eval_hz)
    if collect_hz % save_hz != 0:
        raise ValueError(
            f"Invalid collection frequencies: collect={collect_hz} is not evenly "
            f"divisible by save={save_hz}; an integer save frequency cannot be "
            "constructed."
        )
    save_period_collect_steps = collect_hz // save_hz
    if save_period_collect_steps < 1:
        raise ValueError(
            f"Invalid collection frequencies: save={save_hz} cannot exceed the "
            f"collect control frequency {collect_hz}."
        )
    # Every recorded trajectory row is an action and must be replayed exactly
    # once.  ``save`` only controls how often collection writes an observation;
    # it must not silently downsample replay.  Replay timing is controlled solely
    # by ``physical / eval`` through the environment decimation.
    action_stride = 1

    control_hz = collect_hz if mode == "collect" else eval_hz
    decimation = collect_decimation if mode == "collect" else eval_decimation
    video_physics, video_control = _event_period(
        name="video",
        event_hz=video_hz,
        physical_hz=physical_hz,
        control_hz=control_hz,
        decimation=decimation,
    )
    render_physics, render_control = _event_period(
        name="render",
        event_hz=render_hz,
        physical_hz=physical_hz,
        control_hz=control_hz,
        decimation=decimation,
    )
    return TimingPlan(
        mode=mode,
        physical_hz=physical_hz,
        control_hz=control_hz,
        collect_hz=collect_hz,
        save_hz=save_hz,
        eval_hz=eval_hz,
        video_hz=video_hz,
        render_hz=render_hz,
        decimation=decimation,
        action_stride=action_stride,
        save_period_collect_steps=save_period_collect_steps,
        video_period_physics_steps=video_physics,
        render_period_physics_steps=render_physics,
        video_period_control_steps=video_control,
        render_period_control_steps=render_control,
    )


def validate_task_config(config: DictConfig) -> None:
    # Resolve every mandatory structured value before Isaac Sim is launched.
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    timing_plan(config, "collect")
    timing_plan(config, "eval")

    if not isinstance(config.collect_settings.use_seed, bool):
        raise TypeError("collect_settings.use_seed must be a boolean.")
    _require_int("collect_settings.episode_num", config.collect_settings.episode_num)
    if not isinstance(config.replay_settings.force_action, bool):
        raise TypeError("replay_settings.force_action must be a boolean.")
    _require_int("replay_settings.max_episodes", config.replay_settings.max_episodes)
    if not str(config.collect_settings.save_root_dir).strip():
        raise ValueError("collect_settings.save_root_dir must not be empty.")
    if not str(config.replay_settings.save_root_dir).strip():
        raise ValueError("replay_settings.save_root_dir must not be empty.")


def load_task_config(
    config: str | Path,
    overrides: list[str] | tuple[str, ...] = (),
) -> tuple[DictConfig, Path]:
    path = resolve_config_path(config)
    schema = OmegaConf.structured(TaskRunConfig)
    loaded = OmegaConf.load(path)
    dotlist = OmegaConf.from_dotlist(list(overrides))
    merged = OmegaConf.merge(schema, loaded, dotlist)
    OmegaConf.set_struct(merged, True)
    validate_task_config(merged)
    return merged, path


def collection_data_dir(config: DictConfig, task_name: str, config_name: str) -> Path:
    return (
        Path(str(config.collect_settings.save_root_dir)) / task_name / config_name
    )


def replay_output_dir(config: DictConfig, task_name: str, config_name: str) -> Path:
    timestamp = time.strftime(r"%Y-%m-%d_%H:%M:%S")
    return (
        Path(str(config.replay_settings.save_root_dir))
        / "replay"
        / task_name
        / config_name
        / timestamp
    )


def build_task_env_cfg(
    task_name: str,
    config: DictConfig,
    config_name: str,
    mode: Literal["collect", "eval"],
    *,
    device: str | None = None,
    save_dir: Path | None = None,
) -> tuple[Any, Any, TimingPlan, Path]:
    """Import a task and build its fully resolved ``TaskCfg``."""
    task_module = importlib.import_module(f"envs.{task_name}")
    env_cfg = task_module.TaskCfg()
    timing = timing_plan(config, mode)

    env_cfg.tactile_sensor_type = config.env_settings.sensor_type
    env_cfg.tactile_optical_backend = config.env_settings.optical_backend
    env_cfg.random_texture = config.env_settings.random_texture
    env_cfg.obs_data_type = OmegaConf.to_container(
        config.observation_settings, resolve=True
    )

    env_cfg.decimation = timing.decimation
    env_cfg.save_frequency = (
        timing.save_period_collect_steps if mode == "collect" else 1
    )
    env_cfg.video_frequency = timing.video_period_control_steps
    env_cfg.render_frequency = timing.render_period_control_steps
    env_cfg.sim.dt = 1.0 / timing.physical_hz
    env_cfg.uipc_sim.dt = env_cfg.sim.dt
    env_cfg.sim.render_interval = timing.decimation
    env_cfg.sim.render.rendering_mode = FIXED_RENDERING_MODE
    for camera_cfg in env_cfg.cameras:
        camera_cfg.update_period = env_cfg.sim.dt

    task_overrides = OmegaConf.to_container(config.task_overrides, resolve=True)
    for option_name, option_value in task_overrides.items():
        if not hasattr(env_cfg, option_name):
            raise ValueError(
                f"Unknown task override {option_name!r} for task {task_name!r}."
            )
        setattr(env_cfg, option_name, option_value)

    env_cfg.scene.num_envs = 1
    if device is not None:
        env_cfg.sim.device = device

    if save_dir is None:
        if mode == "collect":
            save_dir = collection_data_dir(config, task_name, config_name)
        else:
            save_dir = replay_output_dir(config, task_name, config_name)
    save_dir = Path(save_dir)
    env_cfg.save_dir = save_dir
    return task_module, env_cfg, timing, save_dir


def create_task_env(
    task_name: str,
    config: DictConfig,
    config_name: str,
    mode: Literal["collect", "eval"],
    *,
    device: str | None = None,
    save_dir: Path | None = None,
) -> TaskEnv:
    """Build and instantiate one task environment from the unified config."""
    task_module, env_cfg, timing, save_dir = build_task_env_cfg(
        task_name,
        config,
        config_name,
        mode,
        device=device,
        save_dir=save_dir,
    )
    task = task_module.Task(env_cfg, mode=mode)
    return TaskEnv(
        task=task,
        env_cfg=env_cfg,
        task_module=task_module,
        timing=timing,
        save_dir=save_dir,
    )
