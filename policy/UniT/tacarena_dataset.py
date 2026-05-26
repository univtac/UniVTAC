import copy
import os
from pathlib import Path
from typing import Dict

import cv2
import h5py
import numpy as np
import torch
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


class TacArenaHdf5Dataset(BaseImageDataset):
    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        horizon=16,
        pad_before=1,
        pad_after=7,
        n_obs_steps=2,
        n_latency_steps=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
    ):
        self.shape_meta = shape_meta
        self.dataset_path = Path(dataset_path)
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps
        self.n_latency_steps = n_latency_steps
        self.seed = seed

        assert self.dataset_path.is_dir(), f"{self.dataset_path} does not exist"
        self.episode_paths = sorted(
            self.dataset_path.glob("episode_*.hdf5"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        if max_train_episodes is not None:
            self.episode_paths = self.episode_paths[:max_train_episodes]
        assert len(self.episode_paths) > 0, f"No episode_*.hdf5 under {self.dataset_path}"

        self.rgb_keys = [
            k for k, v in shape_meta["obs"].items() if v.get("type", "low_dim") == "rgb"
        ]
        self.lowdim_keys = [
            k for k, v in shape_meta["obs"].items() if v.get("type", "low_dim") == "low_dim"
        ]
        self.key_to_hdf5 = {
            "cam_high": "cam_head",
            "cam_wrist": "cam_wrist",
            "tactile_left_image": "tac_left",
            "tactile_right_image": "tac_right",
            "qpos": "qpos",
        }

        rng = np.random.default_rng(seed)
        val_count = int(round(len(self.episode_paths) * val_ratio))
        val_ids = set(rng.choice(len(self.episode_paths), size=val_count, replace=False).tolist()) if val_count else set()
        self.train_episode_ids = [i for i in range(len(self.episode_paths)) if i not in val_ids]
        self.val_episode_ids = [i for i in range(len(self.episode_paths)) if i in val_ids]
        if not self.train_episode_ids:
            self.train_episode_ids = list(range(len(self.episode_paths)))

        self.episode_ids = self.train_episode_ids
        self.index = self._build_index(self.episode_ids)
        self._normalizer = None

    def _build_index(self, episode_ids):
        index = []
        for episode_id in episode_ids:
            with h5py.File(self.episode_paths[episode_id], "r") as root:
                episode_len = root["/action"].shape[0]
            for start in range(episode_len):
                index.append((episode_id, start, episode_len))
        return index

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.episode_ids = self.val_episode_ids if self.val_episode_ids else self.train_episode_ids
        val_set.index = self._build_index(val_set.episode_ids)
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        if self._normalizer is not None:
            return self._normalizer
        actions = []
        qpos = []
        for episode_path in self.episode_paths:
            with h5py.File(episode_path, "r") as root:
                actions.append(root["/action"][()].astype(np.float32))
                qpos.append(root["/observations/qpos"][()].astype(np.float32))
        normalizer = LinearNormalizer()
        normalizer["action"] = SingleFieldLinearNormalizer.create_fit(np.concatenate(actions, axis=0))
        for key in self.lowdim_keys:
            if key == "qpos":
                normalizer[key] = SingleFieldLinearNormalizer.create_fit(np.concatenate(qpos, axis=0))
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        self._normalizer = normalizer
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        actions = []
        for episode_path in self.episode_paths:
            with h5py.File(episode_path, "r") as root:
                actions.append(root["/action"][()].astype(np.float32))
        return torch.from_numpy(np.concatenate(actions, axis=0))

    def __len__(self):
        return len(self.index)

    def _sequence_indices(self, start, episode_len):
        rel = np.arange(self.horizon) - self.pad_before
        return np.clip(start + rel, 0, episode_len - 1)

    def _read_image_seq(self, root, key, indices):
        hdf5_key = self.key_to_hdf5[key]
        dataset = root[f"/observations/images/{hdf5_key}"]
        target_shape = self.shape_meta["obs"][key]["shape"]
        target_h, target_w = target_shape[1], target_shape[2]
        out = np.empty((len(indices), target_h, target_w, 3), dtype=np.float32)
        for i, frame_idx in enumerate(indices):
            image = dataset[int(frame_idx)]
            if image.shape[:2] != (target_h, target_w):
                image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            out[i] = image.astype(np.float32) / 255.0
        return np.moveaxis(out, -1, 1)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        episode_id, start, episode_len = self.index[idx]
        indices = self._sequence_indices(start, episode_len)
        obs_indices = indices[: self.n_obs_steps]

        obs = {}
        with h5py.File(self.episode_paths[episode_id], "r") as root:
            for key in self.rgb_keys:
                obs[key] = self._read_image_seq(root, key, obs_indices)
            for key in self.lowdim_keys:
                if key == "qpos":
                    qpos_ds = root["/observations/qpos"]
                    obs[key] = np.stack([qpos_ds[int(i)] for i in obs_indices], axis=0).astype(np.float32)
            action_ds = root["/action"]
            action = np.stack([action_ds[int(i)] for i in indices], axis=0).astype(np.float32)

        if self.n_latency_steps > 0:
            action = action[self.n_latency_steps :]

        return {
            "obs": {k: torch.from_numpy(v) for k, v in obs.items()},
            "action": torch.from_numpy(action),
        }
