from typing import Dict
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
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from UniT.dataset.real_data_conversion import real_data_to_replay_buffer

class TactilePerceptionDataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            n_latency_steps=0,
            seed=42,
            val_ratio=0.2,
            max_train_episodes=None,
            resized_image_shape=(128, 160),
            data_type = '3Dpose',
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
        # print which episodes are used for validation
        print('Validation episodes:', np.where(val_mask)[0])
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
        self.val_mask = val_mask
        self.n_latency_steps = n_latency_steps
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.resized_image_shape = resized_image_shape
        # Dummy value
        self.horizon = 1
        self.n_obs_steps = 1
        self.data_type = data_type
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

        if self.data_type == '6Dpose':
            tactile_left_image = (data['tactile_left_image'][T_slice].astype(np.float32) / 255.)[0]
            H,W = self.resized_image_shape
            resized_image = cv2.resize(tactile_left_image, (W,H))
            obs_dict['image'] = resized_image
            # save ram
            del data['tactile_left_image']
        elif self.data_type == '3Dpose':

            tactile_image = (data['tactile_image'][T_slice].astype(np.float32) / 255.)[0]
            H,W = self.resized_image_shape
            resized_image = cv2.resize(tactile_image, (W,H))
            obs_dict['image'] = resized_image
            # save ram
            del data['tactile_image']

        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice][0].astype(np.float32)
            # save ram
            del data[key]
        
        return obs_dict
def _get_replay_buffer(dataset_path):
    # load data
    cv2.setNumThreads(1)
    with threadpool_limits(1):
        replay_buffer = real_data_to_replay_buffer(
            dataset_path=dataset_path,
        )

    return replay_buffer


