import copy
from torch.utils.data import DataLoader
import pytorch_lightning as pl



class RepresentationWrapper(pl.LightningDataModule):
    def __init__(self, dataset, cfg, train=None, validation=None, test=None):
        super().__init__()
        self.dataset_configs = dict()
        self.cfg = copy.deepcopy(cfg)
        self.dataset = dataset
        if train is not None:
            self.dataset_configs["train"] = train
        if validation is not None:
            self.dataset_configs["validation"] = validation
        if test is not None:
            self.dataset_configs["test"] = test

    def train_dataloader(self):
        return DataLoader(self.dataset, **self.cfg.dataloader)

    def val_dataloader(self):
        if "validation" in self.dataset_configs:
            val_dataset = self.dataset.get_validation_dataset()
            return DataLoader(val_dataset, **self.cfg.val_dataloader)
        return None
    def val_dataset(self):
        if "validation" in self.dataset_configs:
            val_dataset = self.dataset.get_validation_dataset()
            return val_dataset
        return None

    def test_dataloader(self):
        if "test" in self.dataset_configs:
            # same with validation
            val_dataset = self.dataset.get_validation_dataset()
            return DataLoader(val_dataset, **self.cfg.val_dataloader)
        return None