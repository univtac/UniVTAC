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
import pathlib
import math
from tqdm import tqdm
from omegaconf import OmegaConf
from byol_pytorch import BYOL
from torchvision import models
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

class TrainByolRepresentationWorkspace(BaseWorkspace):

    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)
        # set seed
        seed = cfg.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)


        self.total_epoch = cfg.total_epoch
        self.model_path = cfg.model_path
        self.cfg = cfg
        self.seed = cfg.seed
        self.device = cfg.device
        self.image_size = cfg.image_size

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        pretrain = cfg.pretrain
        model_type = cfg.model_type
        if cfg.datatype == 'ycb':
            human_demonstrations_img = TactileRepresentationDatasetYcb(**cfg.dataset)
        else:
            human_demonstrations_img = TactileRepresentationDataset(**cfg.dataset)
        data = RepresentationWrapper(human_demonstrations_img,cfg)
        dataloader = data.train_dataloader()
        device = self.device
        resnet = None
        if model_type == 'resnet34':
            resnet = models.resnet34(pretrained=pretrain)
        elif model_type == 'resnet50':
            resnet = models.resnet50(pretrained=pretrain)
        # to device
        resnet = resnet.to(device)
        learner = BYOL(
            resnet,
            image_size = self.image_size,
            hidden_layer = 'avgpool'
        )
        opt = torch.optim.Adam(learner.parameters(), lr=3e-4)
        for e in range(self.total_epoch):
            losses = []
            for img in tqdm(iter(dataloader)):
                img = img['image']
                # from (B, H, W, C) to (B, C, H, W)
                img = img.permute(0, 3, 1, 2)
                img = img.to(device)
                loss = learner(img)
                opt.zero_grad()
                loss.backward()
                opt.step()
                learner.update_moving_average()
                losses.append(loss.item())

            avg_loss = sum(losses) / len(losses)
            print(f'In epoch {e}, average traning loss is {avg_loss}.')

            # if representation_models folder does not exist, create it
            if not os.path.exists("representation_models"):
                os.makedirs("representation_models")
            torch.save(resnet.state_dict(), self.model_path)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainByolRepresentationWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
