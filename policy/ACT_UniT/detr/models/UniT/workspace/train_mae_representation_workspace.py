if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
import math
from tqdm import tqdm
from UniT.model.mae.utils import *
import numpy as np
import random
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import copy
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from UniT.dataset.tactile_representation_dataset import TactileRepresentationDataset
from UniT.dataset.tactile_representation_dataset_ycb import TactileRepresentationDatasetYcb
from UniT.dataset.representation_wrapper import RepresentationWrapper

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainMaeRepresentationWorkspace(BaseWorkspace):

    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)
        # set seed
        seed = cfg.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.batch_size = cfg.batch_size
        self.max_device_batch_size = cfg.max_device_batch_size
        self.base_learning_rate = cfg.base_learning_rate
        self.weight_decay = cfg.weight_decay
        self.total_epoch = cfg.total_epoch
        self.warmup_epoch = cfg.warmup_epoch
        self.model_path = cfg.model_path
        self.cfg = cfg
        self.seed = cfg.seed
        self.device = cfg.device
        self.image_size = cfg.image_size
        self.patch_size = cfg.patch_size
        self.emb_dim = cfg.emb_dim
        self.encoder_layer = cfg.encoder_layer
        self.encoder_head = cfg.encoder_head
        self.decoder_layer = cfg.decoder_layer
        self.decoder_head = cfg.decoder_head
        self.mask_ratio = cfg.mask_ratio
    def run(self):
        cfg = copy.deepcopy(self.cfg)

        if cfg.datatype == 'ycb':
            human_demonstrations_img = TactileRepresentationDatasetYcb(**cfg.dataset)
        else:
            human_demonstrations_img = TactileRepresentationDataset(**cfg.dataset)
        data = RepresentationWrapper(human_demonstrations_img,cfg)
        dataloader = data.train_dataloader()
        device = self.device
        model = MAE_ViT(image_size=self.image_size,
                        patch_size=self.patch_size,
                        emb_dim=self.emb_dim,  
                        encoder_layer=self.encoder_layer,
                        encoder_head=self.encoder_head,
                        decoder_layer=self.decoder_layer,  
                        decoder_head=self.decoder_head,  
                        mask_ratio=self.mask_ratio).to(self.device)
        # print model parameters amount and model size
        print("Model size: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1e6))
        optim = torch.optim.AdamW(model.parameters(), lr=self.base_learning_rate * self.batch_size / 256, betas=(0.9, 0.95), weight_decay=self.weight_decay)
        lr_func = lambda epoch: min((epoch + 1) / (self.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / self.total_epoch * math.pi) + 1))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)
        step_count = 0
        optim.zero_grad()
        for e in range(self.total_epoch):
            model.train()
            losses = []
            for img in tqdm(iter(dataloader)):
                img = img['image']
                # from (B, H, W, C) to (B, C, H, W)
                img = img.permute(0, 3, 1, 2)
                step_count += 1
                img = img.to(device)
                predicted_img, mask = model(img)
                loss = torch.mean((predicted_img - img) ** 2 * mask) / self.mask_ratio
                loss.backward()
                optim.step()
                optim.zero_grad()
                losses.append(loss.item())
            lr_scheduler.step()
            avg_loss = sum(losses) / len(losses)
            print(f'In epoch {e}, average traning loss is {avg_loss}.')
            # if representation_models folder does not exist, create it
            if not os.path.exists("representation_models"):
                os.makedirs("representation_models")
            torch.save(model, self.model_path)
@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainMaeRepresentationWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
