"""
lightning_hydra_classifiers/experiments/reference_transfer_experiment.py


This module is meant to replicate the formatting of transfer_experiment.py but using common reference datasets, for use in debugging pipelines.


Currently Implemented datasets:


Future dataset implementations:

-- MNIST
-- PlantVillage




Created on: Wednesday Sept 1st, 2021
Author: Jacob A Rose


"""


import argparse
from munch import Munch
import os
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from torchvision.datasets import CIFAR10
from torchvision import transforms
import os
from typing import *


from lightning_hydra_classifiers.data.utils.make_catalogs import CSV_CATALOG_DIR_V1_0, EXPERIMENTAL_DATASETS_DIR, CSVDatasetConfig, CSVDataset, DataSplitter
from lightning_hydra_classifiers.utils.common_utils import LabelEncoder

if 'TOY_DATA_DIR' not in os.environ: 
    os.environ['TOY_DATA_DIR'] = "/media/data_cifs/projects/prj_fossils/data/toy_data"        
default_root_dir = os.environ['TOY_DATA_DIR']

__all__ = ["CIFAR10DataModule"]


# class CIFAR10Dataset(torchdata.datasets.Files):

#     def __init__(self,
#                  path_schema: Path = "{family}_{genus}_{species}_{collection}_{catalog_number}",
#                  return_signature: List[str] = ["image","target"], #,"path"],
#                  eager_encode_targets: bool = False,
#                  config: Optional[BaseDatasetConfig]=None,
#                  transform=None):
#         files = files or []
#         super().__init__(files=files)
# #         self.samples_df = samples_df
#         self.path_schema = PathSchema(path_schema)
#         self._return_signature = collections.namedtuple("return_signature", return_signature)
        
#         self.x_col = "path"
#         self.y_col = "family"
#         self.id_col = "catalog_number"
#         self.config = config or {}
#         self.transform = transform
#         self.eager_encode_targets = eager_encode_targets
#         self.setup(samples_df=samples_df)        



class Subset(torch.utils.data.dataset.Subset):
    """
    Subclass that fixes the issue where train and val subset datasets are incapable of using different transforms.
    
    """
    def __init__(self,
                 subset: torch.utils.data.dataset.Subset,
                 transform=None,
                 target_transform=None) -> None:
        super().__init__(dataset=subset.dataset, indices=subset.indices)
        self.transform = transform
        self.target_transform = target_transform

        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = super().__getitem__(index)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

        
    
    
class CIFAR10DataModule(pl.LightningDataModule):

    def __init__(self, 
                 task_id=0,
                 batch_size: int=128,
                 image_size: int=224,
                 image_buffer_size: int=32,
                 num_workers: int=4,
                 pin_memory: bool=True):

        super().__init__()
        self.data_dir = default_root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        
        self.image_size = image_size
        self.image_buffer_size = image_buffer_size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        # Train augmentation policy
        self.set_task(task_id=task_id)
        self.__init_transforms()

        
    def set_task(self, task_id: int):
#         assert task_id in self.experiment.valid_tasks
        self.task_id = task_id
        
    @property
    def tasks(self) -> List[Dict[str,"Dataset"]]:
        return [{"train":self.train_dataset,
                "val":self.val_dataset,
                "test":self.test_dataset}]
        
    @property
    def current_task(self):
        return self.tasks[self.task_id]


    def prepare_data(self):
        # download data, train then test
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # we set up only relevant datasets when stage is specified
        if stage == 'fit' or stage is None:
            cifar10 = CIFAR10(self.data_dir, train=True, transform=None)
            self.train_dataset, self.val_dataset = random_split(cifar10, [45000, 5000])
            self.train_dataset = Subset(self.train_dataset, transform=self.train_transform)
            self.val_dataset = Subset(self.val_dataset, transform=self.val_transform)
            
            self.classes = cifar10.classes
            self.label_encoder = LabelEncoder(class2idx=cifar10.class_to_idx)
            self.num_classes = len(self.label_encoder)

        if stage == 'test' or stage is None:
            self.test_dataset = CIFAR10(self.data_dir, train=False, transform=self.val_transform)

    def __init_transforms(self):
        
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=self.image_size,
                                         scale=(0.25, 1.2),
                                         ratio=(0.7, 1.3),
                                         interpolation=2),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(self.mean, self.std),
            transforms.Grayscale(num_output_channels=3)
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize(self.image_size+self.image_buffer_size),
            transforms.ToTensor(),
            transforms.CenterCrop(self.image_size),
            transforms.Normalize(self.mean, self.std),
            transforms.Grayscale(num_output_channels=3)            
        ])
            
            
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.batch_size,
                                           pin_memory=self.pin_memory,
                                           shuffle=True,
                                           num_workers=self.num_workers,
                                           drop_last=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset,
                                           batch_size=self.batch_size,
                                           pin_memory=self.pin_memory,
                                           num_workers=self.num_workers)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset,
                                           batch_size=self.batch_size,
                                           pin_memory=self.pin_memory,
                                           num_workers=self.num_workers)



# class TransferExperiment:
    
#     valid_tasks = (0, 1)
    
#     def __init__(self,
#                  config=None):
#         self.parse_config(config)
        
#     def parse_config(self,
#                      config):
#         config = config or Munch()
        

# #     @staticmethod
#     def setup_task_0(self): #experiment_root_dir = Path("/media/data_cifs/projects/prj_fossils/users/jacob/experiments/July2021-Nov2021/csv_datasets/leavesdb-v1_0/")):
#         """
#         TASK 0
#         Produces train, val, and test subsets for task 0

#         train + val: 
#             Extant_Leaves_minus_PNAS
#         test:
#             Extant_Leaves_in_PNAS
            
#         Returns:
#             task_0 (Dict[str,pd.DataFrame])

#         """
#         return task_0


# #     @staticmethod
#     def setup_task_1(self): #experiment_root_dir = Path("/media/data_cifs/projects/prj_fossils/users/jacob/experiments/July2021-Nov2021/csv_datasets/leavesdb-v1_0/")):
#         """
#         TASK 1
#         Produces train, val, and test subsets for task 1

#         train + val: 
#             PNAS_minus_Extant_Leaves
#         test:
#             PNAS_in_Extant_Leaves
            
#         Returns:
#             task_1 (Dict[str,pd.DataFrame])


#         """

#         return task_1
        
#     def export_experiment_spec(self, output_root_dir=None):
        
    
    
#     def get_multitask_datasets(self,
#                                train_transform=None,
#                                train_target_transform=None,
#                                val_transform=None,
#                                val_target_transform=None):

#         return task_0, task_1


    
        
# def cmdline_args():
#     p = argparse.ArgumentParser(description="Export a series of dataset artifacts (containing csv catalog, yml config, json labels) for each dataset, provided that the corresponding images are pointed to by one of the file paths hard-coded in catalog_registry.py.")
#     p.add_argument("-o", "--output_dir", dest="output_dir", type=str,
#                    default="/media/data_cifs/projects/prj_fossils/users/jacob/experiments/July2021-Nov2021/csv_datasets/leavesdb-v1_0",
#                    help="Output root directory. Each unique dataset will be allotted its own subdirectory within this root dir.")
#     p.add_argument("-a", "--all", dest="make_all", action="store_true",
#                    help="If user provides this flag, produce all currently in-production datasets in the most recent version (currently == 'v1_0').")
#     p.add_argument("-v", "--version", dest="version", type=str, default='v1_0',
#                    help="Available dataset versions: [v0_3, v1_0].")
    
#     return p.parse_args()

    
# if __name__ == "__main__":
    
#     args = cmdline_args()
    
#     experiment = TransferExperiment()
    
#     experiment.export_experiment_spec(output_root_dir=args.output_dir)