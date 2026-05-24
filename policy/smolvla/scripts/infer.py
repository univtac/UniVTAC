#!/usr/bin/env python
# Copyright 2026 The vitac_smolvla team. All rights reserved.
"""Minimal inference entry point for visuo-tactile SmolVLA.

Loads a fine-tuned SmolVLA checkpoint and the matching LeRobot v3.0 dataset,
samples one frame, and prints the predicted action chunk shape + first action.
This is intentionally bare-bones; for closed-loop evaluation use
``src/lerobot/scripts/lerobot_eval.py``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402
from lerobot.policies.smolvla.processor_smolvla import (  # noqa: E402
    make_smolvla_pre_post_processors,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("infer")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True,
                   help="Path to a fine-tuned SmolVLAPolicy directory.")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--repo-id", type=str, required=True)
    p.add_argument("--frame-index", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    log.info("Loading policy from %s", args.ckpt)
    policy = SmolVLAPolicy.from_pretrained(args.ckpt)
    policy.to(args.device).eval()

    log.info("Loading dataset %s @ %s", args.repo_id, args.dataset_root)
    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.dataset_root)

    sample = dataset[args.frame_index]
    # Move everything to the policy's device.
    batch = {
        k: (v.to(args.device) if isinstance(v, torch.Tensor) else v)
        for k, v in sample.items()
    }

    pre, post = make_smolvla_pre_post_processors(
        policy.config, dataset_stats=dataset.meta.stats
    )
    batch = pre(batch)

    with torch.no_grad():
        action_chunk = policy.predict_action_chunk(batch)
    action_chunk = post(action_chunk)

    log.info("Predicted action chunk shape: %s", tuple(action_chunk.shape))
    log.info("First predicted action: %s", action_chunk[0, 0].detach().cpu().tolist())


if __name__ == "__main__":
    main()
