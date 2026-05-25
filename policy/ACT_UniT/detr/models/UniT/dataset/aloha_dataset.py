from typing import Dict, List
import torch
import numpy as np
import zarr
import os
import shutil
from filelock import FileLock
from threadpoolctl import threadpool_limits
from omegaconf import OmegaConf
import cv2
import json
import hashlib
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from UniT.dataset.real_data_conversion import real_data_to_replay_buffer
from diffusion_policy.common.normalize_util import (
    get_image_range_normalizer,
)

class AlohaDataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            n_latency_steps=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
        ):
        assert os.path.isdir(dataset_path)
        
        replay_buffer = None

        replay_buffer = _get_replay_buffer(
            dataset_path=dataset_path,
        )

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)

        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon+n_latency_steps,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)
        
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.val_mask = val_mask
        self.horizon = horizon
        self.n_latency_steps = n_latency_steps
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon+self.n_latency_steps,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=self.val_mask
            )
        val_set.val_mask = ~self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        normalizer['action'] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer['action'])
        
        # low dim
        for key in self.lowdim_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                self.replay_buffer[key])
        
        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()

        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key in self.rgb_keys:
            if key in self.shape_meta['obs']:
                target_shape = self.shape_meta['obs'][key]['shape']

                target_h, target_w = target_shape[1], target_shape[2]
                # T,H,W,C,float32
                image_seq = data[key][T_slice].astype(np.float32) / 255.
                resized_image_seq = np.zeros(
                    (image_seq.shape[0], target_h, target_w, 3), dtype=np.float32)
                for image_idx in range(image_seq.shape[0]):
                    resized_image_seq[image_idx] = cv2.resize(
                        image_seq[image_idx], (target_w, target_h))

                obs_dict[key] = np.moveaxis(resized_image_seq, -1, 1)
                del data[key]
        
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            # save ram
            del data[key]

        action = data['action'].astype(np.float32)
        # handle latency by dropping first n_latency_steps action
        # observations are already taken care of by T_slice
        if self.n_latency_steps > 0:
            action = action[self.n_latency_steps:]

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(action)
        }
        return torch_data

def zarr_resize_index_last_dim(zarr_arr, idxs):
    actions = zarr_arr[:]
    actions = actions[...,idxs]
    zarr_arr.resize(zarr_arr.shape[:-1] + (len(idxs),))
    zarr_arr[:] = actions
    return zarr_arr

def _get_replay_buffer(dataset_path):
    # load data
    cv2.setNumThreads(1)
    with threadpool_limits(1):
        replay_buffer = real_data_to_replay_buffer(
            dataset_path=dataset_path,
        )
    return replay_buffer


