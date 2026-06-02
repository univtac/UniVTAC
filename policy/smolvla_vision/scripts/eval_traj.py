#!/usr/bin/env python
"""Evaluate a SmolVLA checkpoint on one pass of a vision-only LeRobot dataset."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import draccus
from tqdm import tqdm

SMOLVLA_VISION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SMOLVLA_VISION_ROOT / "lerobot" / "src"))

from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE  # noqa: E402
from lerobot.configs import PreTrainedConfig  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors  # noqa: E402
from lerobot.processor import (  # noqa: E402
    PolicyProcessorPipeline,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import (  # noqa: E402
    ACTION,
    OBS_PREFIX,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


def _load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _default_dataset(task_name: str, task_config: str, expert_data_num: str) -> Path:
    return SMOLVLA_VISION_ROOT / "data" / "local" / f"{task_name}_{task_config}_{expert_data_num}_vision"


def _default_ckpt(task_name: str, task_config: str, expert_data_num: str, train_epochs: str) -> Path:
    return (
        SMOLVLA_VISION_ROOT
        / "outputs"
        / f"smolvla_vision_{task_name}_{task_config}_{expert_data_num}_vision_e{train_epochs}"
        / "checkpoints"
        / "last"
        / "pretrained_model"
    )


def _infer_repo_id(dataset_root: Path) -> str:
    source_metadata = dataset_root / "source_metadata.json"
    if source_metadata.exists():
        repo_id = _load_json(source_metadata).get("repo_id")
        if repo_id:
            return str(repo_id)
    parent = dataset_root.parent.name
    return f"{parent}/{dataset_root.name}" if parent else f"local/{dataset_root.name}"


def _get_policy_config(model):
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "input_features"):
        return config
    if hasattr(model, "get_base_model"):
        config = getattr(model.get_base_model(), "config", None)
        if config is not None and hasattr(config, "input_features"):
            return config
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        config = getattr(model.base_model.model, "config", None)
        if config is not None and hasattr(config, "input_features"):
            return config
    raise AttributeError("could not locate SmolVLA config on loaded model")


def _load_model(ckpt_dir: Path, device: str):
    model_file = ckpt_dir / SAFETENSORS_SINGLE_FILE
    adapter_file = ckpt_dir / "adapter_model.safetensors"
    if model_file.exists():
        try:
            model = SmolVLAPolicy.from_pretrained(str(ckpt_dir)).to(device).eval()
        except Exception as exc:
            config_path = ckpt_dir / "config.json"
            if not config_path.exists():
                raise
            print(f"[eval_traj] standard checkpoint load failed ({type(exc).__name__}); loading SmolVLA config directly")
            config = draccus.parse(SmolVLAConfig, config_path, args=[])
            config.device = device
            model = SmolVLAPolicy(config)
            model = SmolVLAPolicy._load_as_safetensor(model, str(model_file), device, strict=False)
            model.to(device).eval()
    elif adapter_file.exists():
        from peft import PeftConfig, PeftModel

        policy_config = PreTrainedConfig.from_pretrained(str(ckpt_dir))
        policy_config.device = device
        peft_config = PeftConfig.from_pretrained(str(ckpt_dir))
        if peft_config.base_model_name_or_path:
            model = SmolVLAPolicy.from_pretrained(peft_config.base_model_name_or_path, config=policy_config)
        else:
            model = SmolVLAPolicy(policy_config)
        model = PeftModel.from_pretrained(model, str(ckpt_dir), config=peft_config, is_trainable=False).to(device).eval()
    else:
        raise FileNotFoundError(f"Neither {SAFETENSORS_SINGLE_FILE} nor adapter_model.safetensors found in {ckpt_dir}")
    return model, _get_policy_config(model)


def _load_processors(ckpt_dir: Path, config, device: str, dataset_stats: dict | None):
    preprocessor_config = ckpt_dir / f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
    postprocessor_config = ckpt_dir / f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json"
    if preprocessor_config.exists() and postprocessor_config.exists():
        pre = PolicyProcessorPipeline.from_pretrained(
            ckpt_dir,
            config_filename=preprocessor_config.name,
            overrides={"device_processor": {"device": device}},
        )
        post = PolicyProcessorPipeline.from_pretrained(
            ckpt_dir,
            config_filename=postprocessor_config.name,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )
        return pre, post
    return make_smolvla_pre_post_processors(config, dataset_stats=dataset_stats)


def _delta_timestamps(config, fps: int) -> dict[str, list[float]] | None:
    delta_timestamps = {}
    if config.action_delta_indices is not None:
        delta_timestamps[ACTION] = [i / fps for i in config.action_delta_indices]
    if config.observation_delta_indices is not None:
        for key in config.input_features:
            if key.startswith(OBS_PREFIX):
                delta_timestamps[key] = [i / fps for i in config.observation_delta_indices]
    return delta_timestamps or None


def _to_model_frame(item: dict, config) -> dict:
    frame = {
        "task": item["task"],
        "observation.state": item["observation.state"],
    }
    for key in config.input_features:
        if key == "observation.state" or key not in item:
            continue
        frame[key] = item[key]
    return frame


def _reset_model(model) -> None:
    if hasattr(model, "reset"):
        model.reset()
    elif hasattr(model, "get_base_model") and hasattr(model.get_base_model(), "reset"):
        model.get_base_model().reset()


def _episode_list(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(part) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--repo-id", type=str, default=None)
    parser.add_argument("--ckpt-dir", type=Path, default=None)
    parser.add_argument("--task-name", default="lift_bottle")
    parser.add_argument("--task-config", default="demo")
    parser.add_argument("--expert-data-num", default="2")
    parser.add_argument("--train-epochs", default="2")
    parser.add_argument("--episodes", default=None, help="Comma-separated episode ids, e.g. 0,3,5.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--csv", type=Path, default=None, help="Optional per-frame metrics output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = (args.dataset_root or _default_dataset(args.task_name, args.task_config, args.expert_data_num)).resolve()
    ckpt_dir = (
        args.ckpt_dir or _default_ckpt(args.task_name, args.task_config, args.expert_data_num, args.train_epochs)
    ).resolve()
    repo_id = args.repo_id or _infer_repo_id(dataset_root)
    episodes = _episode_list(args.episodes)

    print(f"[eval_traj] checkpoint: {ckpt_dir}")
    print(f"[eval_traj] dataset:    {dataset_root}")
    print(f"[eval_traj] repo_id:    {repo_id}")
    print(f"[eval_traj] device:     {args.device}")

    model, config = _load_model(ckpt_dir, args.device)
    fps = int(_load_json(dataset_root / "meta" / "info.json")["fps"])
    dataset = LeRobotDataset(
        repo_id,
        root=dataset_root,
        episodes=episodes,
        delta_timestamps=_delta_timestamps(config, fps),
        return_uint8=False,
    )
    pre, post = _load_processors(ckpt_dir, config, args.device, dataset.meta.stats)

    rows = []
    l2_values = []
    mse_values = []
    previous_episode = None
    total = min(len(dataset), args.max_frames) if args.max_frames is not None else len(dataset)

    for idx in tqdm(range(total), desc="Evaluating one epoch"):
        item = dataset[idx]
        episode = int(item["episode_index"].item())
        if episode != previous_episode:
            _reset_model(model)
            previous_episode = episode

        action_item = item[ACTION]
        target_action = action_item[0] if action_item.ndim == 2 else action_item
        target = target_action.detach().cpu().float().reshape(-1)

        processed = pre(_to_model_frame(item, config))
        with torch.no_grad():
            pred = model.select_action(processed)
        pred = torch.as_tensor(post(pred)).detach().cpu().float().reshape(-1)

        dim = min(pred.numel(), target.numel())
        diff = pred[:dim] - target[:dim]
        l2 = torch.linalg.vector_norm(diff).item()
        mse = torch.mean(diff.square()).item()
        l2_values.append(l2)
        mse_values.append(mse)
        rows.append(
            {
                "index": int(item["index"].item()),
                "episode_index": episode,
                "frame_index": int(item["frame_index"].item()),
                "l2": l2,
                "mse": mse,
                "pred": pred.tolist(),
                "target": target.tolist(),
            }
        )

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["index", "episode_index", "frame_index", "l2", "mse", "pred", "target"],
            )
            writer.writeheader()
            writer.writerows(rows)

    print("[eval_traj] results")
    print(f"  frames:      {len(l2_values)}")
    print(f"  mean_l2:     {float(np.mean(l2_values)) if l2_values else float('nan'):.6f}")
    print(f"  median_l2:   {float(np.median(l2_values)) if l2_values else float('nan'):.6f}")
    print(f"  mean_mse:    {float(np.mean(mse_values)) if mse_values else float('nan'):.6f}")
    print(f"  median_mse:  {float(np.median(mse_values)) if mse_values else float('nan'):.6f}")
    if args.csv is not None:
        print(f"  csv:         {args.csv}")


if __name__ == "__main__":
    main()
