from typing import Dict
import numba
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler,
    get_val_mask,
    downsample_mask,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer
import pdb


class RobotImageDataset(BaseImageDataset):

    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        batch_size=128,
        max_train_episodes=None,
    ):

        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            # keys=['head_camera', 'front_camera', 'left_camera', 'right_camera', 'state', 'action'],
            # keys=["head_cam", "left_tac_cam", "right_tac_cam", "agent_pos", "action"],
            keys=["head_cam", "wrist_cam", "left_tac_cam", "right_tac_cam", "agent_pos", "action"],
        )

        val_mask = get_val_mask(n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        self.batch_size = batch_size
        sequence_length = self.sampler.sequence_length
        self.buffers = {
            k: np.zeros((batch_size, sequence_length, *v.shape[1:]), dtype=v.dtype)
            for k, v in self.sampler.replay_buffer.items()
        }
        self.buffers_torch = {k: torch.from_numpy(v) for k, v in self.buffers.items()}
        for v in self.buffers_torch.values():
            v.pin_memory()

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action": self.replay_buffer["action"],
            "agent_pos": self.replay_buffer["agent_pos"],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer["head_cam"] = SingleFieldLinearNormalizer.create_identity()
        normalizer["wrist_cam"] = SingleFieldLinearNormalizer.create_identity()
        normalizer["left_tac_cam"] = SingleFieldLinearNormalizer.create_identity()
        normalizer["right_tac_cam"] = SingleFieldLinearNormalizer.create_identity()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample["agent_pos"].astype(np.float32)[..., :8]  # (agent_posx2, block_posex3)
        # head_cam = np.moveaxis(sample["head_cam"], -1, 1) / 255
        # left_tac_cam = np.moveaxis(sample["left_tac_cam"], -1, 1) / 255
        # right_tac_cam = np.moveaxis(sample["right_tac_cam"], -1, 1) / 255
        head_cam = sample["head_cam"] / 255
        if "wrist_cam" in sample:
            wrist_cam = sample["wrist_cam"] / 255
        left_tac_cam = sample["left_tac_cam"] / 255
        right_tac_cam = sample["right_tac_cam"] / 255

        data = {
            "obs": {
                "head_cam": head_cam,  # T, 3, H, W
                "left_tac_cam": left_tac_cam,
                "right_tac_cam": right_tac_cam,
                "agent_pos": agent_pos,  # T, D
            },
            "action": sample["action"].astype(np.float32)[..., :8],  # T, D
        }
        if "wrist_cam" in sample:
            data['obs']['wrist_cam'] = wrist_cam
        return data

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if isinstance(idx, slice):
            raise NotImplementedError  # Specialized
        elif isinstance(idx, int):
            sample = self.sampler.sample_sequence(idx)
            sample = dict_apply(sample, torch.from_numpy)
            return sample
        elif isinstance(idx, np.ndarray):
            assert len(idx) == self.batch_size
            for k, v in self.sampler.replay_buffer.items():
                batch_sample_sequence(
                    self.buffers[k],
                    v,
                    self.sampler.indices,
                    idx,
                    self.sampler.sequence_length,
                )
            return self.buffers_torch
        else:
            raise ValueError(idx)

    def postprocess(self, samples, device):
        agent_pos = samples["agent_pos"][..., :8].to(device, non_blocking=True)
        head_cam = samples["head_cam"].to(device, non_blocking=True) / 255.0
        if "wrist_cam" in samples:
            wrist_cam = samples["wrist_cam"].to(device, non_blocking=True) / 255.0
        left_tac_cam = samples["left_tac_cam"].to(device, non_blocking=True) / 255.0
        right_tac_cam = samples["right_tac_cam"].to(device, non_blocking=True) / 255.0

        action = samples["action"][..., :8].to(device, non_blocking=True)
        ret = {
            "obs": {
                "head_cam": head_cam,  # B, T, 3, H, W
                "left_tac_cam": left_tac_cam,
                "right_tac_cam": right_tac_cam,
                "agent_pos": agent_pos,  # T, D
            },
            "action": action,  # B, T, D
        }
        if "wrist_cam" in samples:
            ret['obs']['wrist_cam'] = wrist_cam
        return ret


def _batch_sample_sequence(
    data: np.ndarray,
    input_arr: np.ndarray,
    indices: np.ndarray,
    idx: np.ndarray,
    sequence_length: int,
):
    for i in numba.prange(len(idx)):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = indices[idx[i]]
        data[i, sample_start_idx:sample_end_idx] = input_arr[buffer_start_idx:buffer_end_idx]
        if sample_start_idx > 0:
            data[i, :sample_start_idx] = data[i, sample_start_idx]
        if sample_end_idx < sequence_length:
            data[i, sample_end_idx:] = data[i, sample_end_idx - 1]


_batch_sample_sequence_sequential = numba.jit(_batch_sample_sequence, nopython=True, parallel=False)
_batch_sample_sequence_parallel = numba.jit(_batch_sample_sequence, nopython=True, parallel=True)


def batch_sample_sequence(
    data: np.ndarray,
    input_arr: np.ndarray,
    indices: np.ndarray,
    idx: np.ndarray,
    sequence_length: int,
):
    batch_size = len(idx)
    assert data.shape == (batch_size, sequence_length, *input_arr.shape[1:])
    if batch_size >= 16 and data.nbytes // batch_size >= 2**16:
        _batch_sample_sequence_parallel(data, input_arr, indices, idx, sequence_length)
    else:
        _batch_sample_sequence_sequential(data, input_arr, indices, idx, sequence_length)
