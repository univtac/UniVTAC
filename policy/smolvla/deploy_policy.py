"""SmolVLA deployment policy for the UniVTAC evaluation harness.

This is the deploy-time bridge between UniVTAC's :class:`BaseTask` observation
dictionaries and the visuo-tactile SmolVLA policy living in
``UniVTAC/policy/smolvla/src/lerobot``. The class follows the same contract
as :class:`policy.ACT.deploy_policy.Policy`:

* ``__init__(args)``  -- load checkpoint + processors, read deploy.yml-ish args.
* ``encode_obs(obs)`` -- convert UniVTAC observation -> SmolVLA batch dict.
* ``eval(task, obs)`` -- run a single env step (encode -> select_action -> take_action).
* ``reset()``         -- drop the SmolVLA action queue and re-arm language.

SmolVLA differs from ACT in two ways the deploy code has to handle:

1.  It needs a language instruction; we tokenize once per episode via the
    standard SmolVLA pre-processor pipeline.
2.  It returns *chunks* of actions, but :meth:`SmolVLAPolicy.select_action`
    already takes care of the queue so we can keep the per-step ACT-style
    loop.

A LeRobot v3.0 dataset stats file (``meta/stats.json``) is optional but
*strongly* recommended at deploy time; without it the normalizer falls
back to identity which usually destroys performance.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torchvision import transforms

# Make the locally vendored lerobot importable.
_SMOLVLA_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SMOLVLA_DIR / "src"))
sys.path.append(str(_SMOLVLA_DIR.parent))  # for the relative _base_policy import

from .._base_policy import BasePolicy  # noqa: E402

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402
from lerobot.policies.smolvla.processor_smolvla import (  # noqa: E402
    make_smolvla_pre_post_processors,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_chw_float(img: torch.Tensor) -> torch.Tensor:
    """UniVTAC images are HWC uint8 0-255 torch tensors. Convert to CHW float
    in [0, 1] which is what SmolVLA's pre-processor expects."""
    if img.ndim != 3:
        raise ValueError(f"expected HWC tensor, got shape {tuple(img.shape)}")
    if img.dtype != torch.float32 and img.dtype != torch.float64:
        img = img.float()
    if img.max() > 1.5:  # heuristic: still in 0-255 range
        img = img / 255.0
    # HWC -> CHW
    return img.permute(2, 0, 1).contiguous()


def _resize_chw(img: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Resize a CHW tensor with bilinear interpolation."""
    if tuple(img.shape[-2:]) == tuple(size):
        return img
    return transforms.functional.resize(img, list(size), antialias=True)


def _load_stats(stats_path: Path | None) -> dict | None:
    """Load LeRobot-style dataset stats JSON into the dict-of-tensors format
    expected by :func:`make_smolvla_pre_post_processors`."""
    if stats_path is None or not stats_path.exists():
        return None
    with open(stats_path, "r") as f:
        raw = json.load(f)

    def _walk(node):
        if isinstance(node, dict):
            return {k: _walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return torch.as_tensor(node, dtype=torch.float32)
        return node

    return _walk(raw)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class Policy(BasePolicy):
    """Deploy-time SmolVLA wrapper for UniVTAC."""

    # Mapping from SmolVLA batch keys -> ((root, camera, key), is_tactile)
    # ``root`` and ``camera`` follow the keys exposed by UniVTAC's
    # ``BaseTask._get_observations``.
    _DEFAULT_IMAGE_MAP = {
        "observation.images.head":           ("observation", "head", "rgb",         False),
        "observation.images.wrist":          ("observation", "wrist", "rgb",        False),
        "observation.images.tactile_ll":     ("tactile",     "left_tactile",  "rgb",        True),
        "observation.images.tactile_lr":     ("tactile",     "right_tactile", "rgb",        True),
        # Marker-overlay variants -- only used if the checkpoint was trained with them.
        "observation.images.tactile_ll_marker": ("tactile", "left_tactile",  "rgb_marker", True),
        "observation.images.tactile_lr_marker": ("tactile", "right_tactile", "rgb_marker", True),
    }

    def __init__(self, args: dict):
        # ------------------------------------------------------------------ env
        self.train_config_name = os.environ.get("TRAIN_CONFIG", "train_config")
        self.ep_num = os.environ.get("EP_NUM", "50")
        self.task_name = args["task_name"]

        # Checkpoint layout mirrors ACT:
        #   policy/smolvla/smolvla_ckpt/smolvla-<task_name>/<task_config>-<ep_num>/<train_config>/
        ckpt_dir = (
            _SMOLVLA_DIR
            / "smolvla_ckpt"
            / f"smolvla-{args['task_name']}"
            / f"{args['task_config']}-{self.ep_num}"
            / self.train_config_name
        )
        ckpt_dir = Path(args.get("ckpt_dir", ckpt_dir))

        # ------------------------------------------------------------------ task settings
        with open(_SMOLVLA_DIR.parent / "task_settings.json", "r") as f:
            task_settings = json.load(f)
        assert self.task_name in task_settings, f"Task '{self.task_name}' not found in task_settings.json"
        self.camera_type = task_settings[self.task_name].get("camera_type", "head")
        print(f"[smolvla] camera_type='{self.camera_type}' task='{self.task_name}'")

        # ------------------------------------------------------------------ deploy config
        deploy_yml = _SMOLVLA_DIR / f"{self.train_config_name}.yml"
        if not deploy_yml.exists():
            deploy_yml = _SMOLVLA_DIR / "deploy.yml"
        with open(deploy_yml, "r") as f:
            cfg = yaml.safe_load(f) or {}

        # Allow caller / env to override the on-disk yaml.
        cfg.update({k: v for k, v in args.items() if k.startswith("smolvla_")})

        self.device = torch.device(
            args.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.image_size = tuple(cfg.get("smolvla_image_size", (224, 224)))
        self.tactile_size = tuple(cfg.get("smolvla_tactile_size", (224, 224)))
        self.n_action_steps = int(cfg.get("smolvla_n_action_steps", 0))  # 0 -> use config default
        self.use_marker_variants = bool(cfg.get("smolvla_use_marker_variants", False))

        # ------------------------------------------------------------------ model + processors
        print(f"[smolvla] loading policy from {ckpt_dir}")
        self.model = SmolVLAPolicy.from_pretrained(str(ckpt_dir)).to(self.device).eval()
        if self.n_action_steps > 0:
            # Cap how many actions of the predicted chunk we actually execute.
            self.model.config.n_action_steps = min(
                self.n_action_steps, self.model.config.chunk_size
            )

        stats_path = Path(args.get(
            "stats_path",
            ckpt_dir / "meta" / "stats.json",
        ))
        dataset_stats = _load_stats(stats_path)
        self.pre, self.post = make_smolvla_pre_post_processors(
            self.model.config, dataset_stats=dataset_stats
        )

        # ------------------------------------------------------------------ image mapping
        # Build the SmolVLA-key -> UniVTAC-path map dynamically so we only
        # encode the cameras the checkpoint actually expects.
        configured_keys = set(self.model.config.input_features.keys())
        self._image_map: dict[str, tuple[str, str, str, bool]] = {}
        for batch_key, (root, cam, key, is_tactile) in self._DEFAULT_IMAGE_MAP.items():
            if batch_key not in configured_keys:
                continue
            if (not self.use_marker_variants) and batch_key.endswith("_marker"):
                continue
            self._image_map[batch_key] = (root, cam, key, is_tactile)
        print(f"[smolvla] image streams: {list(self._image_map.keys())}")

        self._instruction_set = False
        self._cached_instruction: str | None = None

    # ----------------------------------------------------------------------
    # Observation encoding
    # ----------------------------------------------------------------------

    def _pick_image(self, observation: dict, root: str, cam: str, key: str) -> torch.Tensor:
        """Fetch a HWC image tensor from a UniVTAC observation, falling back
        gracefully across the variants seen in the codebase."""
        if root in observation and cam in observation[root] and key in observation[root][cam]:
            return observation[root][cam][key]
        # Legacy aliases used by the HDF5 dumps (``left_gsmini``).
        alias = cam.replace("_tactile", "_gsmini")
        if root in observation and alias in observation[root] and key in observation[root][alias]:
            return observation[root][alias][key]
        raise KeyError(f"observation missing {root}.{cam}.{key}")

    def encode_obs(self, observation: dict) -> dict:
        """Convert UniVTAC observation to SmolVLA's raw (un-tokenized) batch.

        Input (UniVTAC, see ``envs/_base_task.py``)::
            observation['observation']['head']['rgb']      HWC uint8 0-255
            observation['observation']['wrist']['rgb']     HWC uint8 0-255
            observation['tactile']['left_tactile']['rgb']  HWC uint8 0-255
            observation['tactile']['right_tactile']['rgb'] HWC uint8 0-255
            observation['embodiment']['ee']                (7,)
            observation['embodiment']['joint']             (9,)

        Output (SmolVLA, pre-processor will then tokenize / normalize)::
            observation.images.head           (3, H, W) float32 in [0, 1]
            observation.images.wrist          (3, H, W) float32 in [0, 1]
            observation.images.tactile_ll     (3, H, W) float32 in [0, 1]
            observation.images.tactile_lr     (3, H, W) float32 in [0, 1]
            observation.state                 (16,)     float32  -- [ee(7), joint(9)]
            task                              str       -- language instruction
        """
        batch: dict[str, torch.Tensor | str] = {}

        for batch_key, (root, cam, key, is_tactile) in self._image_map.items():
            # Honor the deploy.yml camera_type setting for *visual* cameras:
            # the ACT recipe lets the operator pick head-only / wrist-only.
            if (not is_tactile) and self.camera_type != "all" and cam != self.camera_type:
                continue
            img = self._pick_image(observation, root, cam, key)
            img = _to_chw_float(img)
            target = self.tactile_size if is_tactile else self.image_size
            img = _resize_chw(img, target)
            batch[batch_key] = img.to(self.device)

        # State = [ee(7), joint(9)] following the preprocessing convention
        # in scripts/preprocess_hdf5_to_lerobot.py.
        ee = observation["embodiment"]["ee"].to(self.device).float().view(-1)
        joint = observation["embodiment"]["joint"].to(self.device).float().view(-1)
        batch["observation.state"] = torch.cat([ee, joint], dim=0)

        return batch

    # ----------------------------------------------------------------------
    # Inference
    # ----------------------------------------------------------------------

    def eval(self, task, observation):
        """Run a single env step with SmolVLA and dispatch the action.

        Mirrors :meth:`policy.ACT.deploy_policy.Policy.eval` so the harness
        treats SmolVLA identically to ACT. We use
        :meth:`SmolVLAPolicy.select_action` which internally maintains an
        action chunk queue and only re-runs the heavy VLA forward pass when
        the queue is empty.
        """
        if not self._instruction_set:
            self._cached_instruction = task.instruction or self.task_name.replace("_", " ")
            self.model.reset()
            self._instruction_set = True

        batch = self.encode_obs(observation)
        batch["task"] = self._cached_instruction

        # Run the SmolVLA pre-processor (adds batch dim, tokenizes language,
        # device-places, normalizes state).
        batch = self.pre(batch)

        with torch.no_grad():
            action = self.model.select_action(batch)  # (1, action_dim) on CPU after post

        action = self.post(action)
        if isinstance(action, torch.Tensor):
            action_np = action.detach().cpu().reshape(-1).numpy()
        else:
            action_np = np.asarray(action).reshape(-1)

        action_t = torch.from_numpy(action_np).to(task.device).float()
        # SmolVLA action is the next-step joint pose; UniVTAC uses 'qpos' for joint targets.
        exec_succ, eval_succ = task.take_action(action_t, action_type="qpos")
        return exec_succ, eval_succ

    def reset(self):
        """Reset SmolVLA's internal action queue and clear cached instruction."""
        self._instruction_set = False
        self._cached_instruction = None
        if hasattr(self.model, "reset"):
            self.model.reset()
