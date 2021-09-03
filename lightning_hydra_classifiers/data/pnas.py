"""
Utility functions for loading PNAS Leaves images into TorchVision dataloaders

Created by: Monday April 12th, 2021
Author: Jacob A Rose



"""

from torchmetrics import Accuracy
# from flash import Task
from torch import nn, optim, Generator
import torch
# import flash
# from flash.vision import ImageClassificationData, ImageClassifier
from torchvision import models
import pytorch_lightning as pl
from typing import List, Callable, Dict, Union, Type, Optional
from pytorch_lightning.callbacks import Callback

# TODO (Jacob): Hardcode the mean & std for PNAS, Extant Leaves, Imagenet, etc.. for standardization across lab

from torchvision import transforms
from pathlib import Path
from glob import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import ImageFolder
from typing import Callable, Optional, Any, Tuple
from munch import Munch
import matplotlib.pyplot as plt
import torchvision
import os
# from torchvision.datasets.vision import VisionDataset
# log the in- and output histograms of LightningModule's `forward`
# monitor = ModuleDataMonitor()

# from .common import (LeavesDataset, 
#                      LeavesLightningDataModule,
#                      TrainValSplitDataset, 
#                      SubsetImageDataset,
#                      seed_worker)



available_datasets = {"PNAS_family_100_512": "/media/data_cifs/projects/prj_fossils/data/processed_data/data_splits/PNAS_family_100_512",
                      "PNAS_family_100_1024": "/media/data_cifs/projects/prj_fossils/data/processed_data/data_splits/PNAS_family_100_1024",
                      "PNAS_family_100_1536": "/media/data_cifs/projects/prj_fossils/data/processed_data/data_splits/PNAS_family_100_1536",
                      "PNAS_family_100_2048": "/media/data_cifs/projects/prj_fossils/data/processed_data/data_splits/PNAS_family_100_2048"}
default_name = "PNAS_family_100_512"




class PNASLeavesDataset(object): #LeavesDataset):
    
    def __init__(self,
                 name: str=default_name,
                 split: str="train",
                 dataset_dir: Optional[str]=None,
                 return_paths: bool=False,
                 **kwargs: Any
                 ) -> None:
        super().__init__(name,
                         split,
                         dataset_dir=dataset_dir,
                         return_paths=return_paths, 
                         **kwargs)
   

    @property
    def available_datasets(self):
        return available_datasets



class PNASLightningDataModule(object): #LeavesLightningDataModule):
    
#     DatasetConstructor = PNASLeavesDataset
#     splits_on_disk : Tuple[str] = ("train", "val", "test") #DatasetConstructor.splits_on_disk
#     available_splits: Tuple[str] = ("train", "val", "test")
        
    image_size = 224
#     target_size = (224, 224)
    image_buffer_size = 32
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    def __init__(self,
                 name: str=default_name,
                 batch_size: int=32,
                 val_split: float=0.2,
                 num_workers=0,
                 seed: int=None,
                 debug: bool=False,
                 normalize: bool=True,
                 image_size: Union[int,str] = "auto", #None,
                 channels: int=None,
                 dataset_dir: Optional[str]=None,
                 return_paths: bool=False,
                 predict_on_split: str="val",
                 **kwargs):
        
        super().__init__(name=name,
                         batch_size=batch_size,
                         val_split=val_split,
                         num_workers=num_workers,
                         seed=seed,
                         debug=debug,
                         normalize=normalize,
                         image_size=image_size,
                         channels=channels,
                         dataset_dir=dataset_dir,
                         return_paths=return_paths,
                         predict_on_split=predict_on_split,
                         **kwargs)
    
    
    @property
    def available_datasets(self):
        return available_datasets










#     def get_dataset_split(self, split: str) -> LeavesDataset:
#         if split in ("train","val"):
#             train_dataset = self.DatasetConstructor(self.name,
#                                                     split="train",
#                                                     dataset_dir=self.dataset_dir,
#                                                     return_paths=self.return_paths)
#             if "val" in os.listdir(train_dataset.dataset_dir):
#                 val_dataset = self.DatasetConstructor(self.name,
#                                                    split="val",
#                                                    dataset_dir=self.dataset_dir,
#                                                    return_paths=self.return_paths)
#             elif self.val_split:
#                 train_dataset, val_dataset = TrainValSplitDataset.train_val_split(train_dataset,
#                                                                                   val_split=self.val_split,
#                                                                                   seed=self.seed)
#             if split == "train":
#                 return train_dataset
#             else:
#                 return val_dataset
#         elif split == "test":
#             test_dataset = self.DatasetConstructor(self.name,
#                                                    split="test",
#                                                    dataset_dir=self.dataset_dir,
#                                                    return_paths=self.return_paths)
#             return test_dataset
#         else:
#             raise Exception(f"'split' argument must be a string pertaining to one of the following: {self.available_splits}")
        
    
########################################################    
    
# if __name__ == "__main__":
    
#     seed = 873957
#     val_split = 0.2
#     batch_size = 16
#     name = "PNAS_family_100_512"
    
#     datamodule = PNASImageDataModule(name=name,
#                         batch_size=batch_size,
#                         val_split=val_split,
#                         seed=seed)
    

    
    
    
    
    
    
    
    
    
    
    
    
            
#         self.data.init_datasets(train_transform=transforms.ToTensor(),
#                            eval_transform=transforms.ToTensor())
#         num_classes=len(train_dataset)
#         train_dataloader, val_dataloader, test_dataloader = data.init_dataloaders()
#         # train_dataset.loader(train_dataset.samples[7][0]) #__dir__()
#         PNASImageDataModule.image_size = 224

#         data_module = PNASImageDataModule(train_dataset,
#                                           val_dataset,
#                                           test_dataset,
#                                           batch_size=batch_size,
#                                           num_workers=num_workers)
        
        
#         # download data, train then test
#         MNIST(self.data_dir, train=True, download=True)
#         MNIST(self.data_dir, train=False, download=True)
