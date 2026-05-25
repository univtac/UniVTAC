# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict
import os
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List, Literal
import sys
from pathlib import Path
from einops import rearrange

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

import IPython

e = IPython.embed


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys,
                                                             unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        # for name, parameter in backbone.named_parameters(): # only train later layers # TODO do we want this?
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        xs = self.body(tensor)
        return xs
        # out: Dict[str, NestedTensor] = {}
        # for name, x in xs.items():
        #     m = tensor_list.mask
        #     assert m is not None
        #     mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        #     out[name] = NestedTensor(x, mask)
        # return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str, train_backbone: bool, return_interm_layers: bool, dilation: bool):
        backbone = getattr(torchvision.models,
                           name)(replace_stride_with_dilation=[False, False, dilation],
                                 pretrained=is_main_process(),
                                 norm_layer=FrozenBatchNorm2d)  # pretrained # TODO do we want frozen batch_norm??
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):

    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_vision_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model

class TactileBackbone(nn.Module):
    """UniT Tactile Backbone with VQModel and ConvPoolingHead."""

    def __init__(self, name: str, ckpt: str, tac_names: List[str], train_backbone: bool, return_interm_layers: bool, position_embedding, tactile_type: Literal['feat', 'full'] = 'feat', vq_ckpt: str = None):
        """
        Args:
            name: backbone name (unused, kept for compatibility)
            ckpt: path to checkpoint (unused in this implementation)
            tac_names: list of tactile sensor names
            train_backbone: whether to train the backbone
            return_interm_layers: whether to return intermediate layers (unused in this implementation)
            position_embedding: position embedding module
            tactile_type: 'feat' or 'full' (unused in this implementation)
            vq_ckpt: path to the VQModel checkpoint
        """
        super().__init__()
        
        self.tac_names = tac_names
        self.train_backbone = train_backbone
        self.tactile_type = tactile_type
        self.num_channels = 512  # tactile_emb_dim from diffusion_unit_policy
        
        # Import VQModel and ConvPoolingHead
        try:
            from UniT.taming.models.vqgan import VQModel
            from UniT.model.tactile.utils import ConvPoolingPolicyHead as ConvPoolingHead
        except ImportError:
            print("Warning: Unable to import UniT modules. Make sure UniT path is correctly configured.")
            raise
        
        # Initialize VQModel with default config (should match UniT config)
        # The vq_ckpt should point to the checkpoint: /mnt/data/tianxing/TacArena/policy/UniT/UniT/checkpoint-epoch=340.ckpt
        vq_model_config = {
            'embed_dim': 3,
            'n_embed': 512,
            'ddconfig': {
                'double_z': False,
                'z_channels': 3,
                'resolution': 256,
                'in_channels': 3,
                'out_ch': 3,
                'ch': 128,
                'ch_mult': [1, 1, 2, 2],
                'num_res_blocks': 2,
                'attn_resolutions': [16],
                'dropout': 0.0
            },
            'lossconfig': {
                'target': 'taming.losses.vqlpips.VQLPIPSLoss'
            }
        }
        
        self.vqgan = VQModel(**vq_model_config)
        
        # Load VQModel checkpoint if provided
        if vq_ckpt and Path(vq_ckpt).exists():
            try:
                # Load from checkpoint - assume it's a PyTorch Lightning checkpoint
                checkpoint = torch.load(vq_ckpt, map_location='cpu')
                if 'state_dict' in checkpoint:
                    # PyTorch Lightning checkpoint format
                    state_dict = checkpoint['state_dict']
                    # Remove 'model.' prefix if present
                    state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
                    self.vqgan.load_state_dict(state_dict, strict=False)
                else:
                    # Direct state dict
                    self.vqgan.load_state_dict(checkpoint, strict=False)
                print(f"Loaded VQModel from {vq_ckpt}")
            except Exception as e:
                print(f"Warning: Failed to load VQModel checkpoint from {vq_ckpt}: {e}")
        
        # Freeze VQModel
        for param in self.vqgan.parameters():
            param.requires_grad = False
        
        # Infer latent channels directly from VQModel output to avoid channel mismatch
        with torch.no_grad():
            probe = torch.zeros(1, 32, 32, 3)
            probe_latent = self.vqgan.to_latent(self.vqgan.get_input({'image': probe}, 'image'))
            latent_channels = int(probe_latent.shape[1])
            temp = ConvPoolingHead(input_channels=latent_channels)
            connec_dim = temp(probe_latent).shape[1]
        
        self.cp_heads = nn.ModuleDict()
        self.tactile_emb_dim = 512
        
        for tac_name in tac_names:
            self.cp_heads[tac_name] = nn.Sequential(
                ConvPoolingHead(input_channels=latent_channels),
                nn.Linear(connec_dim, self.tactile_emb_dim)
            )
        
        # Position embedding
        self.position_embedding = position_embedding
 
    def forward(self, x, tactile_name=None):
        """
        Args:
            x: tactile image tensor [B, C, H, W]
            tactile_name: name of the tactile sensor (for selecting the right cp_head)
        
        Returns:
            feat: list of feature tensors
            pos: list of position embeddings
        """
        feat, pos = [], []
        
        # Convert to latent representation
        # Prepare input for VQModel
        x_prep = rearrange(x, 'b c h w -> b h w c')
        latent = self.vqgan.to_latent(self.vqgan.get_input({'image': x_prep}, 'image'))
        # latent shape: [B, 4, 16, 20]
        
        # Select the appropriate cp_head based on tactile_name
        if tactile_name is None and len(self.tac_names) > 0:
            tactile_name = self.tac_names[0]
        
        if tactile_name in self.cp_heads:
            tactile_feature = self.cp_heads[tactile_name](latent)  # [B, tactile_emb_dim]
        else:
            # Default: use first cp_head
            default_name = list(self.cp_heads.keys())[0]
            tactile_feature = self.cp_heads[default_name](latent)
        
        # Reshape for compatibility with transformer input [B, emb_dim, 1, 1]
        feat.append(tactile_feature.unsqueeze(-1).unsqueeze(-1))
        
        # Position embedding
        pos.append(self.position_embedding.weight.unsqueeze(-1).unsqueeze(-1))  # [1, D, 1, 1]
        
        return feat, pos

def build_tactile_backbone(args):
    train_backbone = args.lr_tactile_backbone > 0
    return_interm_layers = args.tactile_masks if hasattr(args, 'tactile_masks') else False
    tactile_type = args.tactile_type if hasattr(args, 'tactile_type') else 'feat'
    position_embedding = build_position_encoding(args)
    
    # Use the checkpoint path from args or default
    vq_ckpt = args.tactile_ckpt if hasattr(args, 'tactile_ckpt') else '/mnt/data/tianxing/TacArena/policy/UniT/UniT/checkpoint-epoch=340.ckpt'
    
    backbone = TactileBackbone(
        args.tactile_backbone if hasattr(args, 'tactile_backbone') else 'resnet18',
        args.tactile_ckpt if hasattr(args, 'tactile_ckpt') else None,
        args.tactile_names,
        train_backbone,
        return_interm_layers,
        position_embedding,
        tactile_type,
        vq_ckpt=vq_ckpt
    )
    return backbone