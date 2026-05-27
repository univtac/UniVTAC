#!/usr/bin/env python
"""Evaluate a SmolVLA checkpoint on a LeRobot dataset trajectory.

The script loads a dataset frame-by-frame, runs policy inference, and reports
the L2 error between predicted actions and dataset actions.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


SMOLVLA_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SMOLVLA_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SMOLVLA_ROOT / "src"))

from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE  # noqa: E402
from lerobot.configs import PreTrainedConfig  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.processor import (  # noqa: E402
    PolicyProcessorPipeline,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors  # noqa: E402
from lerobot.utils.constants import (  # noqa: E402
    ACTION,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


def _load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _default_ckpt(task_name: str, task_config: str, expert_data_num: str) -> Path:
    return (
        SMOLVLA_ROOT
        / "outputs"
        / f"smolvla_vitac_{task_name}_{task_config}_{expert_data_num}_vitac"
        / "checkpoints"
        / "last"
        / "pretrained_model"
    )


def _default_dataset(task_name: str, task_config: str, expert_data_num: str) -> Path:
    return SMOLVLA_ROOT / "data" / "local" / f"{task_name}_{task_config}_{expert_data_num}_vitac"


def _infer_repo_id(dataset_root: Path) -> str:
    source_metadata = dataset_root / "source_metadata.json"
    if source_metadata.exists():
        repo_id = _load_json(source_metadata).get("repo_id")
        if repo_id:
            return str(repo_id)
    parent = dataset_root.parent.name
    if parent:
        return f"{parent}/{dataset_root.name}"
    return f"local/{dataset_root.name}"


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
        model = SmolVLAPolicy.from_pretrained(str(ckpt_dir)).to(device).eval()
    elif adapter_file.exists():
        from peft import PeftConfig, PeftModel

        policy_config = PreTrainedConfig.from_pretrained(str(ckpt_dir))
        policy_config.device = device
        peft_config = PeftConfig.from_pretrained(str(ckpt_dir))
        if peft_config.base_model_name_or_path:
            model = SmolVLAPolicy.from_pretrained(
                peft_config.base_model_name_or_path,
                config=policy_config,
            )
        else:
            model = SmolVLAPolicy(policy_config)
        model = PeftModel.from_pretrained(
            model,
            str(ckpt_dir),
            config=peft_config,
            is_trainable=False,
        ).to(device).eval()
    else:
        raise FileNotFoundError(
            f"Neither {SAFETENSORS_SINGLE_FILE} nor adapter_model.safetensors found in {ckpt_dir}"
        )
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


def _reset_model(model) -> None:
    if hasattr(model, "reset"):
        model.reset()
    elif hasattr(model, "get_base_model") and hasattr(model.get_base_model(), "reset"):
        model.get_base_model().reset()


def _to_model_frame(item: dict, config, image_scale: str) -> dict:
    frame = {
        "task": item["task"],
        "observation.state": item["observation.state"],
    }
    for key in config.input_features:
        if key == "observation.state" or key not in item:
            continue
        value = item[key]
        if image_scale == "float01" and isinstance(value, torch.Tensor):
            value = value.float()
            if value.numel() > 0 and value.max() > 1.5:
                value = value / 255.0
        frame[key] = value
    return frame


def _episode_list(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(part) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--repo-id", type=str, default=None)
    parser.add_argument("--ckpt-dir", type=Path, default=None)
    parser.add_argument("--task-name", default="grasp_classify")
    parser.add_argument("--task-config", default="demo")
    parser.add_argument("--expert-data-num", default="2")
    parser.add_argument("--episodes", default=None, help="Comma-separated episode ids, e.g. 0,3,5.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--image-scale",
        choices=("raw", "float01"),
        default="raw",
        help="'raw' mirrors LeRobot training batches; 'float01' mirrors deploy_policy image scaling.",
    )
    parser.add_argument("--csv", type=Path, default=None, help="Optional per-frame metrics output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_dir = (args.ckpt_dir or _default_ckpt(args.task_name, args.task_config, args.expert_data_num)).resolve()
    dataset_root = (
        args.dataset_root or _default_dataset(args.task_name, args.task_config, args.expert_data_num)
    ).resolve()
    repo_id = args.repo_id or _infer_repo_id(dataset_root)
    episodes = _episode_list(args.episodes)

    print(f"[eval_traj] checkpoint: {ckpt_dir}")
    print(f"[eval_traj] dataset:    {dataset_root}")
    print(f"[eval_traj] repo_id:    {repo_id}")
    print(f"[eval_traj] device:     {args.device}")
    print(f"[eval_traj] image_scale:{args.image_scale}")

    dataset = LeRobotDataset(
        repo_id,
        root=dataset_root,
        episodes=episodes,
        return_uint8=True,
    )
    model, config = _load_model(ckpt_dir, args.device)
    pre, post = _load_processors(ckpt_dir, config, args.device, dataset.meta.stats)

    rows = []
    l2_values = []
    mse_values = []
    previous_episode = None
    total = min(len(dataset), args.max_frames) if args.max_frames is not None else len(dataset)

    for idx in tqdm(range(total), desc="Evaluating trajectory"):
        item = dataset[idx]
        episode = int(item["episode_index"].item())
        if episode != previous_episode:
            _reset_model(model)
            previous_episode = episode

        target = item[ACTION].detach().cpu().float().reshape(-1)
        batch = _to_model_frame(item, config, args.image_scale)
        processed = pre(batch)

        with torch.no_grad():
            pred = model.select_action(processed)
        pred = post(pred)
        pred = torch.as_tensor(pred).detach().cpu().float().reshape(-1)

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
