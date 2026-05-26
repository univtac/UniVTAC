from typing import Dict
import torch
import torch.nn as nn
from UniT.model.tactile.utils import quaternion_angle_loss
from UniT.model.tactile.utils import MlpHead
from torchvision import models

class ByolPerception(nn.Module):
    def __init__(self, 
            byol_config,
            freeze_encoder=True,
            task_type='3Dpose',
            num_classes=None,
        ):
        super().__init__()

        self.resnet = models.resnet50(pretrained=True)
        self.task_type = task_type

        self.resnet.load_state_dict(torch.load(byol_config['pt_path']))
        print(f"Loaded BYOL pretrained from {byol_config['pt_path']}")
        if freeze_encoder:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # Update head based on task type
        if task_type == 'pose':
            self.resnet.fc = MlpHead(self.resnet.fc.in_features, output_dim=7)
        elif task_type == 'classification':
            assert num_classes is not None, "num_classes must be specified for classification task"
            self.resnet.fc = MlpHead(self.resnet.fc.in_features, output_dim=num_classes)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        for param in self.resnet.fc.parameters():
            param.requires_grad = True
            
        self.resnet = self.resnet.to(byol_config['device'])

    def forward(self, image):
        # batch: B H W C
        image = image.permute(0, 3, 1, 2)
        output = self.resnet(image)
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