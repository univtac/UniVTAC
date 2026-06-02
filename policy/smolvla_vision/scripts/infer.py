#!/usr/bin/env python
"""Run one-step inference from a vision-only SmolVLA checkpoint and dataset frame."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import draccus
import torch

SMOLVLA_VISION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SMOLVLA_VISION_ROOT / "lerobot" / "src"))

from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("smolvla_vision.infer")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    log.info("Loading policy from %s", args.ckpt)
    try:
        policy = SmolVLAPolicy.from_pretrained(args.ckpt)
        policy.to(args.device).eval()
    except Exception as exc:
        config_path = args.ckpt / "config.json"
        model_file = args.ckpt / SAFETENSORS_SINGLE_FILE
        if not config_path.exists() or not model_file.exists():
            raise
        log.info("Standard checkpoint load failed (%s); loading SmolVLA config directly", type(exc).__name__)
        config = draccus.parse(SmolVLAConfig, config_path, args=[])
        config.device = args.device
        policy = SmolVLAPolicy(config)
        policy = SmolVLAPolicy._load_as_safetensor(policy, str(model_file), args.device, strict=False)
        policy.to(args.device).eval()

    log.info("Loading dataset %s @ %s", args.repo_id, args.dataset_root)
    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.dataset_root, return_uint8=False)
    sample = dataset[args.frame_index]

    batch = {
        key: (value.to(args.device) if isinstance(value, torch.Tensor) else value)
        for key, value in sample.items()
    }
    pre, post = make_smolvla_pre_post_processors(policy.config, dataset_stats=dataset.meta.stats)
    batch = pre(batch)

    with torch.no_grad():
        action = policy.select_action(batch)
    action = post(action)

    log.info("Predicted action shape: %s", tuple(action.shape))
    log.info("Predicted action: %s", torch.as_tensor(action).detach().cpu().tolist())


if __name__ == "__main__":
    main()
