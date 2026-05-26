from typing import Dict
import torch
import torch.nn as nn
from UniT.model.tactile.utils import quaternion_angle_loss
from UniT.model.tactile.utils import ResNetPerception

class ResnetPerception(nn.Module):
    def __init__(self, 
            output_dim = 4,
            backbone = 'resnet34',
            pretrained = False,
            task_type = '3Dpose',
            num_classes = None,
            freeze = False
        ):
        super().__init__()
        
        self.task_type = task_type
        
        if task_type == '6Dpose':
            output_dim = 7  # 3 for position + 4 for quaternion
        elif task_type == '3Dpose':
            output_dim = 4
        elif task_type == 'classification':
            assert num_classes is not None, "num_classes must be specified for classification task"
            output_dim = num_classes
        else:
            raise ValueError(f"Unknown task type: {task_type}")
            
        self.cnn = ResNetPerception(num_classes=output_dim, backbone=backbone, pretrained=pretrained, freeze=freeze)

    def forward(self, image):
        # image shape B H W C
        # convert to B C H W
        image = image.permute(0, 3, 1, 2)
        output = self.cnn(image)
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