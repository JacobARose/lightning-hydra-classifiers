
from omegaconf import DictConfig
import pytorch_lightning as pl
from torchvision import transforms
from typing import *



class LeavesLightningDataModule(pl.LightningDataModule): #pl.LightningDataModule):
        
    def __init__(self,
                 config: DictConfig,
                 data_dir: Optional[str]=None):
        """
        Custom Pytorch lightning datamodule for accessing train, validation, and test data splits.
        """


    def train_dataloader(self):
        return self.train_loader



    def show_batch(self, batch):
        """
        Display a batch of images
        """