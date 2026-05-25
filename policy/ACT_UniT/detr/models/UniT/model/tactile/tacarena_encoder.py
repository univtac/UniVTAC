import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class TacArenaTactileEncoder(nn.Module):
    """
    TacArena 触觉编码器适配器，包装 ResNet 使其兼容 UniT 的 VQModel 接口。
    提供 get_input() 和 to_latent() 方法。
    """
    
    def __init__(
        self,
        backbone: str = 'resnet18',
        latent_dims: int = 512,
        ckpt_path: Optional[str] = None,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        
        self.backbone_name = backbone
        self.latent_dims = latent_dims
        self.freeze_encoder = freeze_encoder
        
        if backbone == 'resnet18':
            self.encoder = models.resnet18(num_classes=latent_dims)
        elif backbone == 'resnet34':
            self.encoder = models.resnet34(num_classes=latent_dims)
        elif backbone == 'resnet50':
            self.encoder = models.resnet50(num_classes=latent_dims)
        else:
            raise ValueError(f"不支持的 backbone 类型: {backbone}")
        
        if ckpt_path is not None:
            self._load_pretrained_weights(ckpt_path)
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def _load_pretrained_weights(self, ckpt_path: str):
        state_dict = torch.load(ckpt_path, map_location='cpu')
        
        backbone_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('backbone.'):
                new_key = key[len('backbone.'):]
                backbone_state_dict[new_key] = value
        
        if len(backbone_state_dict) == 0:
            backbone_state_dict = state_dict
        
        self.encoder.load_state_dict(backbone_state_dict, strict=False)
    
    def get_input(self, batch: dict, key: str = 'image') -> torch.Tensor:
        """处理输入格式：BHWC -> BCHW"""
        x = batch[key]
        
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        
        x = x.permute(0, 3, 1, 2).contiguous()
        
        x = x.float()
        if x.max() > 1.0:
            x = x / 255.0
        
        return x
    
    def to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """提取触觉特征：(B, C, H, W) -> (B, latent_dims, 1, 1)"""
        latent = self.encoder(x)
        latent = latent.view(-1, self.latent_dims, 1, 1)
        return latent
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，输入 BCHW，输出 (B, latent_dims)"""
        return self.encoder(x)
