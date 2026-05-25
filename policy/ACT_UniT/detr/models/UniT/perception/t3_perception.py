from typing import Dict
import torch
import torch.nn as nn
from UniT.model.tactile.utils import quaternion_angle_loss
import hydra
import numpy as np
from t3.models import T3
from t3.models import encoder
from t3.models.decoder import CNNFCDecoder
from omegaconf import OmegaConf
from UniT.model.tactile.utils import MlpHead

class T3Perception(nn.Module):
    def __init__(self, 
            encoder_path,
            trunk_path,
            mini: nn.Module,
            shared_trunk: nn.Module,
            encoder_embed_dim,
            device = "cuda:0",
            freeze_encoder=True,
            task_type='3Dpose',
            num_classes=None,
        ):
        super().__init__()

        self.encoder = mini.to(device)
        self.encoder.load(encoder_path)
        self.trunk = shared_trunk.to(device)
        self.trunk.load(trunk_path)
        self.task_type = task_type

        if freeze_encoder:
            self.encoder.freeze()
            self.trunk.freeze()

        if task_type == '6Dpose':
            self.head = MlpHead(encoder_embed_dim, output_dim=7).to(device)
        elif task_type == 'classification':
            assert num_classes is not None, "num_classes must be specified for classification task"
            self.head = MlpHead(encoder_embed_dim, output_dim=num_classes).to(device)
        elif task_type == '3Dpose':
            self.head = MlpHead(encoder_embed_dim, output_dim=4).to(device)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        self.encoder = self.encoder.to(device)
        self.trunk = self.trunk.to(device)
        self.head = self.head.to(device)

    def forward(self, image):
        image = image.permute(0, 3, 1, 2)
        tactile_feature = self.encoder(image)
        tactile_feature = self.trunk(tactile_feature)
        output = self.head(tactile_feature[:,0])
        return output

    def compute_loss(self, batch: Dict[str, torch.Tensor]):
        image = batch['image']
        y_hat = self.forward(image)
        
        if self.task_type == '6Dpose':
            y = batch['6Dpose']
            position_loss = torch.nn.functional.mse_loss(y_hat[:, :3], y[:, :3])
            position_error = torch.nn.functional.l1_loss(y_hat[:, :3], y[:, :3])
            rotation_loss = quaternion_angle_loss(y_hat[:, 3:], y[:, 3:])
            
            total_loss = 1000 * position_loss + rotation_loss
            # training loss, position error, rotation error
            return total_loss, position_error, rotation_loss
            
        elif self.task_type == 'classification':
            y = batch['label']
            classification_loss = torch.nn.functional.cross_entropy(y_hat, y)
            accuracy = (y_hat.argmax(dim=1) == y).float().mean()
            # classification loss, accuracy, dummy loss
            return classification_loss, accuracy, torch.tensor(0.0).to(y_hat.device)
        elif self.task_type == '3Dpose':
            y = batch['3Dpose']
            loss = quaternion_angle_loss(y_hat, y)
            # angle loss/error, dummy loss, dummy loss
            return loss, torch.tensor(0.0).to(y_hat.device), torch.tensor(0.0).to(y_hat.device)
