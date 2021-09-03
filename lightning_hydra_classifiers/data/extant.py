"""
Utility functions for loading PNAS Leaves images into TorchVision dataloaders

Created by: Monday April 12th, 2021
Author: Jacob A Rose



"""

import torch
import pytorch_lightning as pl
from typing import List, Callable, Dict, Union, Type, Optional, Any, Tuple

# TODO (Jacob): Hardcode the mean & std for PNAS, Extant Leaves, Imagenet, etc.. for standardization across lab

# from torchvision import transforms
# from pathlib import Path
# from glob import glob
# import numpy as np
# import os
# from PIL import Image
# from torch.utils.data import Dataset, DataLoader, random_split
# from torchvision.datasets import ImageFolder
# from munch import Munch
# import matplotlib.pyplot as plt
# import torchvision
# from torchvision.datasets.vision import VisionDataset
# log the in- and output histograms of LightningModule's `forward`
# monitor = ModuleDataMonitor()

# from .common import (LeavesDataset, 
#                      LeavesLightningDataModule,
#                      TrainValSplitDataset, 
#                      SubsetImageDataset,
#                      seed_worker)


available_datasets = {"Extant_family_10_512": "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/extant_family_10/512",
                      "Extant_family_10_1024": "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/extant_family_10/1024",
                      "Extant_family_10_1536": "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/extant_family_10/1536",
                      "Extant_family_10_2048": "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/extant_family_10/2048",
                     
                     "Extant_family_20_512": "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/extant_family_20/512",
                     "Extant_family_20_1024": "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/extant_family_20/1024",
                     "Extant_family_20_1536": "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/extant_family_20/1536",
                     "Extant_family_20_2048": "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/extant_family_20/2048",
                     
                     "Extant_family_50_512": "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/extant_family_50/512",
                     "Extant_family_50_1024": "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/extant_family_50/1024",
                     "Extant_family_50_1536": "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/extant_family_50/1536",
                     "Extant_family_50_2048": "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/extant_family_50/2048",
                     
                     "Extant_family_100_512": "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/extant_family_100/512",
                     "Extant_family_100_1024": "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/extant_family_100/1024",
                     "Extant_family_100_1536": "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/extant_family_100/1536",
                     "Extant_family_100_2048": "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/extant_family_100/2048"}
default_name = "Extant_family_10_512"




# __all__ = ["ExtantLeavesDataset", "ExtantLightningDataModule"]


# class ExtantLeavesDataset(LeavesDataset):
#     splits_on_disk = ("train", "test")
#     def __init__(self,
#                  name: str=default_name,
#                  split: str="train",
#                  dataset_dir: str=None,
#                  return_paths: bool=False,
#                  **kwargs: Any
#                  ) -> None:
#         super().__init__(name,
#                          split,
#                          dataset_dir=dataset_dir, 
#                          return_paths=return_paths, 
#                          **kwargs)
   

#     @property
#     def available_datasets(self):
#         return available_datasets



class ExtantLightningDataModule(object): #LeavesLightningDataModule):
    
#     DatasetConstructor = ExtantLeavesDataset
#     splits_on_disk : Tuple[str] = DatasetConstructor.splits_on_disk
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
                 image_size: int = 'auto',
                 channels: int=None,
                 dataset_dir: str=None,
                 return_paths: bool=False,
                 predict_on_split: str="val",
                 **kwargs):
        print(f'name: {name}')
        print(f'dataset_dir:{dataset_dir}')
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
#                                                       split="val",
#                                                       dataset_dir=self.dataset_dir,
#                                                       return_paths=self.return_paths)
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
            
            
        
        
#         if split in ("train","val"):
#             train_data = self.DatasetConstructor(self.name,
#                                                  split="train")
#             train_dataset, val_dataset = TrainValSplitDataset.train_val_split(train_data,
#                                                                               val_split=self.val_split,
#                                                                               seed=self.seed)
#             if split == "train":
#                 return train_dataset
#             else:
#                 return val_dataset
#         elif split == "test":
#             test_dataset = self.DatasetConstructor(self.name,
#                                                    split="test")
#             return test_dataset
#         else:
#             raise Exception(f"'split' argument must be a string pertaining to one of the following: {self.available_splits}")