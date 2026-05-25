from typing import Dict
import torch
import torch.nn as nn
from UniT.taming.models.vqgan import WOVQModel as SimpleAE
from UniT.model.tactile.utils import quaternion_angle_loss
from UniT.model.tactile.utils import ConvPoolingHead
from UniT.model.tactile.utils import MlpHead

class NovqPerception(nn.Module):
    def __init__(self, 
            vq_model_config,
            latent_shape=[3,16,20],
            freeze_encoder=True,
            task_type='3Dpose',
            num_classes=None,
        ):
        super().__init__()
        self.vqgan = SimpleAE(**vq_model_config)
        self.task_type = task_type

        if freeze_encoder:
            for param in self.vqgan.parameters():
                param.requires_grad = False

        self.conv_pooling_head = ConvPoolingHead(latent_shape[0])
        
        conv_output_dim = self.conv_pooling_head(torch.randn(1, latent_shape[0], latent_shape[1], latent_shape[2])).shape[1]
        
        if task_type == '6Dpose':
            self.mlp_head = MlpHead(conv_output_dim, output_dim=7)
        elif task_type == '3Dpose':
            self.mlp_head = MlpHead(conv_output_dim, output_dim=4)
        elif task_type == 'classification':
            assert num_classes is not None, "num_classes must be specified for classification task"
            self.mlp_head = MlpHead(conv_output_dim, output_dim=num_classes)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def forward(self, image):
        tactile_feature = self.vqgan.to_latent(self.vqgan.get_input({'image': image}, 'image'))
        tactile_feature = self.conv_pooling_head(tactile_feature)
        output = self.mlp_head(tactile_feature)
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