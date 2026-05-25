from typing import Dict
import torch
import torch.nn as nn
from UniT.model.mae.utils import *
from UniT.model.tactile.utils import quaternion_angle_loss
from UniT.model.tactile.utils import ConvPoolingHead
from UniT.model.tactile.utils import MlpHead

class MaePerception(nn.Module):
    def __init__(self, 
            mae_config,
            freeze_encoder=True,
            task_type='3Dpose',
            num_classes=None,
        ):
        super().__init__()

        if mae_config.get('pt_path') is not None:
            self.mae = torch.load(mae_config['pt_path'], map_location=mae_config['device'])
            print(f"Loaded MAE from {mae_config['pt_path']}")
        else:
            self.mae = MAE_ViT(**mae_config)
        
        if freeze_encoder:
            for param in self.mae.parameters():
                param.requires_grad = False
        if num_classes is not None:
            self.vit_classifier = ViT_Classifier(self.mae.encoder,num_classes=num_classes)
        else:
            if task_type == '6Dpose':
                self.vit_classifier = ViT_Classifier(self.mae.encoder,num_classes=7)
            elif task_type == '3Dpose':
                self.vit_classifier = ViT_Classifier(self.mae.encoder,num_classes=4)
        self.task_type = task_type

    # vqvae to cnn to mlp
    def forward(self, image):
        # batch: B H W C
        image = image.permute(0, 3, 1, 2)
        output = self.vit_classifier(image)
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