"""
UniT 触觉模型模块

这个模块包含了用于处理触觉数据的各种编码器和工具函数。

可用的编码器：
- TacArenaTactileEncoder: TacArena 触觉编码器的适配器，基于 ResNet
"""

from .utils import (
    quaternion_angle_loss,
    SEBlock,
    ConvPoolingPolicyHead,
    ConvPoolingHead,
    MlpHead,
    ResNetPerception,
)

from .tacarena_encoder import TacArenaTactileEncoder

__all__ = [
    # 工具函数和类
    'quaternion_angle_loss',
    'SEBlock',
    'ConvPoolingPolicyHead',
    'ConvPoolingHead',
    'MlpHead',
    'ResNetPerception',
    # TacArena 编码器适配器
    'TacArenaTactileEncoder',
]

