from typing import Dict
import torch
import numpy as np
import os
from threadpoolctl import threadpool_limits
import cv2
import copy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)

class TactileYcbClassificationDataset(BaseImageDataset):
    def __init__(self,
            dataset_path: str,
            seed=42,
            val_ratio=0.2,
            max_train_images=None,
            image_shape=(128, 160),
        ):
        assert os.path.isdir(dataset_path)
        
        # Load replay buffer
        self.replay_buffer = self._get_replay_buffer(dataset_path)
        
        # Calculate total number of images using episode_ends
        total_images = self.replay_buffer.n_steps
        
        # Create indices for all images
        all_indices = np.arange(total_images)
        np.random.seed(seed)
        np.random.shuffle(all_indices)
        
        # Split into train/val
        n_val = int(total_images * val_ratio)
        val_indices = all_indices[:n_val]
        train_indices = all_indices[n_val:]
        
        if max_train_images is not None:
            train_indices = train_indices[:max_train_images]
        
        # Store indices for training
        self.indices = train_indices
        self.val_indices = val_indices
        self.image_shape = image_shape

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.indices = self.val_indices
        val_set.val_indices = self.indices
        return val_set

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        
        # Get global index
        global_idx = self.indices[idx]
        
        # Get image and label directly from replay buffer
        data = self.replay_buffer.get_steps_slice(global_idx, global_idx + 1)
        image = data['image'][0].astype(np.float32) / 255.0
        label = data['label'][0]

        # Resize image if needed
        H, W = self.image_shape
        if image.shape[:2] != (H, W):
            image = cv2.resize(image, (W, H))

        # Convert to torch tensors
        image_tensor = torch.from_numpy(image) 
        label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            'image': image_tensor,
            'label': label_tensor
        }

    @staticmethod
    def _get_replay_buffer(dataset_path):
        cv2.setNumThreads(1)
        with threadpool_limits(1):
            replay_buffer = ReplayBuffer.create_from_path(
                zarr_path=dataset_path,
                mode='r'
            )
        return replay_buffer
