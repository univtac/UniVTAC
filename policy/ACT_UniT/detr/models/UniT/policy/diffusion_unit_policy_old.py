"""
UniT 的 Diffusion Policy 实现

这个文件实现了基于扩散模型的策略学习，结合视觉和触觉输入来预测机器人动作。

修改说明：
- 添加了对 TacArena 触觉编码器的支持
- 通过 use_tacarena_encoder 参数可以选择使用 VQGAN 或 TacArena 编码器
- 使用适配器模式，最小化对原有代码的修改
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from UniT.model.vision.timm_obs_encoder import TimmObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply
from UniT.taming.models.vqgan import VQModel
from UniT.model.tactile.utils import ConvPoolingPolicyHead as ConvPoolingHead
from UniT.model.tactile.tacarena_encoder import TacArenaTactileEncoder


class DiffusionUnitPolicyOld(BaseImagePolicy):
    """
    基于扩散模型的 UniT 策略
    
    这个策略结合视觉编码器和触觉编码器来处理多模态输入，
    然后使用扩散模型来预测机器人的动作序列。
    
    触觉编码器支持两种选择：
    1. VQGAN（原始 UniT 实现）：输出空间特征图，需要 ConvPoolingHead 处理
    2. TacArena 编码器（新增）：基于 ResNet，直接输出特征向量
    """
    
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: TimmObsEncoder,
            # ========== 触觉编码器配置 ==========
            # 选项 1: 使用 VQGAN
            vq_model_config: Optional[dict] = None,
            # 选项 2: 使用 TacArena 编码器
            use_tacarena_encoder: bool = False,
            tacarena_encoder_config: Optional[dict] = None,
            # ========== 其他配置 ==========
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            input_pertub=0.1,
            inpaint_fixed_action_prefix=False,
            train_diffusion_n_samples=1,
            latent_shape=[3,16,20],
            tactile_emb_dim=512,
            obs_steps=2,
            tactile=['left', 'right'],
            # parameters passed to step
            **kwargs
        ):
        super().__init__()
        
        # ========== 保存配置 ==========
        self.use_tacarena_encoder = use_tacarena_encoder
        self.tactile_emb_dim = tactile_emb_dim
        self.tactile_sensors = tactile  # ['left', 'right'] 或其子集
        
        # ========== 初始化触觉编码器 ==========
        if use_tacarena_encoder:
            default_config = {
                'backbone': 'resnet18',
                'latent_dims': 512,
                'ckpt_path': None,
                'freeze_encoder': True,
            }
            if tacarena_encoder_config is not None:
                default_config.update(tacarena_encoder_config)
            
            self.tactile_encoder = TacArenaTactileEncoder(**default_config)
            latent_shape = [512, 1, 1]
            print("[DiffusionUnitPolicy] 使用 TacArena 触觉编码器")
        else:
            # 使用原始的 VQGAN 编码器
            self.vqgan =  VQModel(**vq_model_config)
            # freeze the vqgan
            for param in self.vqgan.parameters():
                param.requires_grad = False
            print("[DiffusionUnitPolicy] 使用 VQGAN 触觉编码器")
        
        temp = ConvPoolingHead(input_channels=latent_shape[0])
        with torch.no_grad():
            connec_dim = temp(torch.randn(1, latent_shape[0], latent_shape[1], latent_shape[2])).shape[1]
        self.cp_head_right = nn.Sequential(
                    ConvPoolingHead(input_channels=latent_shape[0]),
                    nn.Linear(connec_dim, tactile_emb_dim)
                )
        self.cp_head_left = nn.Sequential(
                    ConvPoolingHead(input_channels=latent_shape[0]),
                    nn.Linear(connec_dim, tactile_emb_dim)
                )
        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        action_horizon = shape_meta['action']['horizon']
        # get feature dim
        obs_feature_dim = np.prod(obs_encoder.output_shape())

        # create diffusion model
        assert obs_as_global_cond
        input_dim = action_dim
        # count how many tactile features are there
        tactile_sensor_count = len(tactile)
        global_cond_dim = obs_feature_dim + obs_steps * tactile_emb_dim * tactile_sensor_count

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.obs_as_global_cond = obs_as_global_cond
        self.input_pertub = input_pertub
        self.inpaint_fixed_action_prefix = inpaint_fixed_action_prefix
        self.train_diffusion_n_samples = int(train_diffusion_n_samples)
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    # def _encode_tactile(self, tactile: torch.Tensor, side: str) -> torch.Tensor:
    #     """
    #     编码触觉图像
        
    #     这个方法统一处理 VQGAN 和 TacArena 两种编码器的调用。
        
    #     参数：
    #     - tactile: 触觉图像，格式为 (B*T, C, H, W)
    #     - side: 'left' 或 'right'，指定使用哪个处理头
        
    #     返回：
    #     - 触觉特征，格式为 (B, tactile_emb_dim * T)
    #     """
    #     # 获取原始 batch 大小
    #     BT = tactile.shape[0]
        
    #     # ========== 将 BCHW 转换为 BHWC（编码器期望的格式） ==========
    #     tactile_bhwc = rearrange(tactile, 'b c h w -> b h w c')
        
    #     # ========== 调用编码器 ==========
    #     # 无论是 VQGAN 还是 TacArena 编码器，都使用相同的接口
    #     tactile_input = self.tactile_encoder.get_input({'image': tactile_bhwc}, 'image')
    #     tactile_feature = self.tactile_encoder.to_latent(tactile_input)
        
    #     # ========== 通过对应的处理头 ==========
    #     if side == 'right':
    #         tactile_feature = self.cp_head_right(tactile_feature)
    #     else:  # side == 'left'
    #         tactile_feature = self.cp_head_left(tactile_feature)
        
    #     return tactile_feature

    # ========= inference  ============
    def conditional_sample(self, 
            condition_data,
            condition_mask,
            local_cond=None,
            global_cond=None,
            generator=None,
            **kwargs
        ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
         # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor], fixed_action_prefix: torch.Tensor=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        fixed_action_prefix: unnormalized action prefix
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict  # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]

        # condition through global feature
        global_cond = self.obs_encoder(nobs)

        tactile_features = {}
        for key, value in nobs.items():
            if 'tactile_right_image' in key:
                tactile = value
                B, T, _, _, _ = tactile.shape
                tactile = rearrange(tactile, 'b t c h w -> (b t) c h w')
                # tactile_feature = self._encode_tactile(tactile, side='right') # 
                tactile = rearrange(tactile, 'b c h w -> b h w c')
                tactile_feature = self.vqgan.to_latent(self.vqgan.get_input({'image': tactile},'image'))
                tactile_feature = self.cp_head_right(tactile_feature)
                # rearrange to B,-1
                tactile_feature = tactile_feature.reshape(B, -1)
                tactile_features[key] = tactile_feature
                
            if 'tactile_left_image' in key:
                tactile = value
                B, T, _, _, _ = tactile.shape
                tactile = rearrange(tactile, 'b t c h w -> (b t) c h w')
                # tactile_feature = self._encode_tactile(tactile, side='left') #
                tactile = rearrange(tactile, 'b c h w -> b h w c')
                tactile_feature = self.vqgan.to_latent(self.vqgan.get_input({'image': tactile},'image'))
                tactile_feature = self.cp_head_left(tactile_feature)
                # rearrange to B,-1
                tactile_feature = tactile_feature.reshape(B, -1)
                tactile_features[key] = tactile_feature

        for key, value in tactile_features.items():
            global_cond = torch.cat([global_cond, value], dim=1)

        # empty data for action
        cond_data = torch.zeros(size=(B, self.action_horizon, self.action_dim), device=self.device, dtype=self.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        if fixed_action_prefix is not None and self.inpaint_fixed_action_prefix:
            n_fixed_steps = fixed_action_prefix.shape[1]
            cond_data[:, :n_fixed_steps] = fixed_action_prefix
            cond_mask[:, :n_fixed_steps] = True
            cond_data = self.normalizer['action'].normalize(cond_data)

        # run sampling
        nsample = self.conditional_sample(
            condition_data=cond_data, 
            condition_mask=cond_mask,
            local_cond=None,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        assert nsample.shape == (B, self.action_horizon, self.action_dim)
        action_pred = self.normalizer['action'].unnormalize(nsample)
        
        result = {
            'action': action_pred,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        
        assert self.obs_as_global_cond
        global_cond = self.obs_encoder(nobs)
        tactile_features = {}
        for key, value in nobs.items():
            if 'tactile_right_image' in key:
                tactile = value
                B, T, _, _, _ = tactile.shape
                tactile = rearrange(tactile, 'b t c h w -> (b t) c h w')
                # tactile_feature = self._encode_tactile(tactile, side='right') #
                tactile = rearrange(tactile, 'b c h w -> b h w c')
                tactile_feature = self.vqgan.to_latent(self.vqgan.get_input({'image': tactile},'image'))
                tactile_feature = self.cp_head_right(tactile_feature)
                # rearrange to B,-1
                tactile_feature = tactile_feature.reshape(B, -1)
                tactile_features[key] = tactile_feature
                
            if 'tactile_left_image' in key:
                tactile = value
                B, T, _, _, _ = tactile.shape
                tactile = rearrange(tactile, 'b t c h w -> (b t) c h w')
                # tactile_feature = self._encode_tactile(tactile, side='left') #
                tactile = rearrange(tactile, 'b c h w -> b h w c')
                tactile_feature = self.vqgan.to_latent(self.vqgan.get_input({'image': tactile},'image'))
                tactile_feature = self.cp_head_left(tactile_feature)
                # rearrange to B,-1
                tactile_feature = tactile_feature.reshape(B, -1)
                tactile_features[key] = tactile_feature

        for key, value in tactile_features.items():
            global_cond = torch.cat([global_cond, value], dim=1)

        # train on multiple diffusion samples per obs
        if self.train_diffusion_n_samples != 1:
            global_cond = torch.repeat_interleave(global_cond, 
                repeats=self.train_diffusion_n_samples, dim=0)
            nactions = torch.repeat_interleave(nactions, 
                repeats=self.train_diffusion_n_samples, dim=0)

        trajectory = nactions
        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        # input perturbation by adding additonal noise to alleviate exposure bias
        # reference: https://github.com/forever208/DDPM-IP
        noise_new = noise + self.input_pertub * torch.randn(trajectory.shape, device=trajectory.device)

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (nactions.shape[0],), device=trajectory.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise_new, timesteps)
        
        # Predict the noise residual
        pred = self.model(
            noisy_trajectory,
            timesteps, 
            local_cond=None,
            global_cond=global_cond
        )

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()

        return loss

    def forward(self, batch):
        return self.compute_loss(batch)
