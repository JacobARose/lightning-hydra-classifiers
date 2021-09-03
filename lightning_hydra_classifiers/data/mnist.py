"""
contrastive_learning.data.pytorch.mnist

Utility functions for loading images from the useful toy dataset MNIST into TorchVision dataloaders

Created by: Wednesday April 14th, 2021
Author: Jacob A Rose



"""

import numpy as np

# 🍦 Vanilla PyTorch
import torch
from torch.nn import functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from torchvision.datasets import MNIST
from torchvision import transforms
import os


# Replace default file cloud urls from Yann Lecun's website to offiial aws s3 bucket
new_mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'
MNIST.resources = [
                   ('/'.join([new_mirror, url.split('/')[-1]]), md5)
                   for url, md5 in MNIST.resources
                   ]


if 'TOY_DATA_DIR' not in os.environ: 
    os.environ['TOY_DATA_DIR'] = "/media/data_cifs/projects/prj_fossils/data/toy_data"
        
default_root_dir = os.environ['TOY_DATA_DIR']



class MNISTDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str=None, batch_size: int=128):
        super().__init__()
        self.data_dir = data_dir or default_root_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        # download data, train then test
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # we set up only relevant datasets when stage is specified
        if stage == 'fit' or stage is None:
            mnist = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist, [55000, 5000])
        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    # we define a separate DataLoader for each of train/val/test
    def train_dataloader(self):
        mnist_train = DataLoader(self.mnist_train, batch_size=self.batch_size)
        return mnist_train

    def val_dataloader(self):
        mnist_val = DataLoader(self.mnist_val, batch_size=10 * self.batch_size)
        return mnist_val

    def test_dataloader(self):
        mnist_test = DataLoader(self.mnist_test, batch_size=10 * self.batch_size)
        return mnist_test

if __name__ == "__main__":
    # setup data
    mnist = MNISTDataModule()
    mnist.prepare_data()
    mnist.setup()

    # grab samples to log predictions on
    samples = next(iter(mnist.val_dataloader()))


#################################



# from torchmetrics import Accuracy
# from flash import Task
# from torch import nn, optim, Generator
# import flash
# from flash.vision import ImageClassificationData, ImageClassifier
# from torchvision import models
# import pytorch_lightning as pl
# from typing import List, Callable, Dict, Union, Type, Optional
# from pytorch_lightning.callbacks import Callback
# from contrastive_learning.data.pytorch.flash_process import Preprocess, Postprocess

# # TODO (Jacob): Hardcode the mean & std for PNAS, Extant Leaves, Imagenet, etc.. for standardization across lab

# from torchvision import transforms
# from pathlib import Path
# from glob import glob
# import numpy as np
# from PIL import Image
# from torch.utils.data import Dataset, DataLoader, random_split
# from torchvision.datasets import ImageFolder
# from typing import Callable, Optional
# from munch import Munch


# from .common import SubsetImageDataset


# available_datasets = {"PNAS_family_100_512": "/media/data_cifs/projects/prj_fossils/data/processed_data/data_splits/PNAS_family_100_512",
#                       "PNAS_family_100_1024": "/media/data_cifs/projects/prj_fossils/data/processed_data/data_splits/PNAS_family_100_1024"}



# class PNASLeaves:
    
#     available_datasets = available_datasets
    
#     def __init__(self,
#                  name: str=None,
#                  val_split: float=0.2,
#                  seed: int=None):
        
#         name = name or "PNAS_family_100_512"
        
#         error_msg = "[!] val_split should be in the range [0, 1]."
#         assert ((val_split >= 0) and (val_split <= 1)), error_msg
#         assert name in self.available_datasets
        
#         self.name = name
#         self.val_split = val_split
#         self.seed = seed
        
#         self.train_dir = Path(self.available_datasets[name], 'train')
#         self.test_dir = Path(self.available_datasets[name], 'test')
        
#         self._initialized = False        
        
#     def init_datasets(self,
#                       train_transform: Optional[Callable] = None,
#                       eval_transform: Optional[Callable] = None,
#                       target_transform: Optional[Callable] = None):
        
#         self.train_dataset = ImageFolder(root=self.train_dir, transform=train_transform, target_transform=target_transform)
#         self.test_dataset = ImageFolder(root=self.test_dir, transform=eval_transform, target_transform=target_transform)
#         self.classes = self.train_dataset.classes
        
#         val_split = self.val_split
#         num_train = len(self.train_dataset)
#         split_idx = (int(np.floor((1-val_split) * num_train)), int(np.floor(val_split * num_train)))
        
#         if self.seed is None:
#             generator = None
#         else:
#             generator = Generator().manual_seed(self.seed)

#         train_data, val_data = random_split(self.train_dataset, 
#                                             [split_idx[0], split_idx[1]], 
#                                             generator=generator)
#         self.split_indices = (train_data.indices, val_data.indices)
#         self.train_dataset = SubsetImageDataset(train_data.dataset, train_data.indices)
#         self.val_dataset = SubsetImageDataset(val_data.dataset, val_data.indices)
        
#         self._initialized = True
        
#         return self.train_dataset, self.val_dataset, self.test_dataset
    
    
#     def init_dataloaders(self, 
#                          num_workers: int=0,
#                          batch_size: int=32,
#                          pin_memory: bool=False,
#                          shuffle_train: bool=True):
        
#         if not self._initialized:
#             self.init_datasets()
        
#         self.train_loader = DataLoader(self.train_dataset,
#                                   num_workers=num_workers,
#                                   pin_memory=pin_memory,
#                                   batch_size=batch_size,
#                                   shuffle=shuffle_train)
#         self.val_loader = DataLoader(self.val_dataset,
#                                 num_workers=num_workers,
#                                 pin_memory=pin_memory,
#                                 batch_size=batch_size,
#                                 shuffle=False)
#         self.test_loader = DataLoader(self.test_dataset,
#                                  num_workers=num_workers,
#                                  pin_memory=pin_memory,
#                                  batch_size=batch_size,
#                                  shuffle=False)
        
#         return self.train_loader, self.val_loader, self.test_loader
        

# #########################################
# #########################################
# #########################################
# #########################################


# class PNASImageDataModule(ImageClassificationData):
#     """Data module for image classification tasks."""
#     preprocess_cls = Preprocess
#     postprocess_cls = Postprocess
    
#     image_size = 224
#     target_size = (224, 224)
#     image_buffer_size = 32
#     mean = [0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225]
    
#     @staticmethod
#     def default_train_transforms():
#         image_size = PNASImageDataModule.image_size
#         image_buffer_size = PNASImageDataModule.image_buffer_size
#         mean, std = PNASImageDataModule.mean, PNASImageDataModule.std
#         return {
#             "pre_tensor_transform": transforms.Compose([transforms.Resize(image_size+image_buffer_size),
#                                                         transforms.RandomHorizontalFlip(p=0.5),
#                                                         transforms.RandomCrop(image_size)]),
#             "to_tensor_transform": transforms.ToTensor(),
#             "post_tensor_transform": transforms.Normalize(mean, std),
#         }
#     ##############
#     @staticmethod
#     def default_val_transforms():
#         image_size = PNASImageDataModule.image_size
#         image_buffer_size = PNASImageDataModule.image_buffer_size
#         mean, std = PNASImageDataModule.mean, PNASImageDataModule.std
#         return {
#             "pre_tensor_transform": transforms.Compose([transforms.Resize(image_size+image_buffer_size),
#                                                         transforms.CenterCrop(image_size)]),
#             "to_tensor_transform": transforms.ToTensor(),
#             "post_tensor_transform": transforms.Normalize(mean, std)
#         }

#     @classmethod
#     def instantiate_preprocess(
#         cls,
#         train_transform: Dict[str, Union[nn.Module, Callable]] = 'default',
#         val_transform: Dict[str, Union[nn.Module, Callable]] = 'default',
#         test_transform: Dict[str, Union[nn.Module, Callable]] = 'default',
#         predict_transform: Dict[str, Union[nn.Module, Callable]] = 'default',
#         preprocess_cls: Type[Preprocess] = None
#     ) -> Preprocess:
#         """
#         This function is used to instantiate ImageClassificationData preprocess object.
#         Args:
#             train_transform: Train transforms for images.
#             val_transform: Validation transforms for images.
#             test_transform: Test transforms for images.
#             predict_transform: Predict transforms for images.
#             preprocess_cls: User provided preprocess_cls.
#         Example::
#             train_transform = {
#                 "per_sample_transform": T.Compose([
#                     T.RandomResizedCrop(224),
#                     T.RandomHorizontalFlip(),
#                     T.ToTensor(),
#                     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#                 ]),
#                 "per_batch_transform_on_device": nn.Sequential(K.RandomAffine(360), K.ColorJitter(0.2, 0.3, 0.2, 0.3))
#             }
#         """
#         train_transform, val_transform, test_transform, predict_transform = cls._resolve_transforms(
#             train_transform, val_transform, test_transform, predict_transform
#         )

#         preprocess_cls = preprocess_cls or cls.preprocess_cls
#         preprocess = preprocess_cls(train_transform, val_transform, test_transform, predict_transform)
#         return preprocess

#     @classmethod
#     def _resolve_transforms(
#         cls,
#         train_transform: Optional[Union[str, Dict]] = 'default',
#         val_transform: Optional[Union[str, Dict]] = 'default',
#         test_transform: Optional[Union[str, Dict]] = 'default',
#         predict_transform: Optional[Union[str, Dict]] = 'default',
#     ):

#         if not train_transform or train_transform == 'default':
#             train_transform = cls.default_train_transforms()

#         if not val_transform or val_transform == 'default':
#             val_transform = cls.default_val_transforms()

#         if not test_transform or test_transform == 'default':
#             test_transform = cls.default_val_transforms()

#         if not predict_transform or predict_transform == 'default':
#             predict_transform = cls.default_val_transforms()

#         return (
#             cls._check_transforms(train_transform), cls._check_transforms(val_transform),
#             cls._check_transforms(test_transform), cls._check_transforms(predict_transform)
#         )
    
#     @staticmethod
#     def _check_transforms(transform: Dict[str, Union[nn.Module, Callable]]) -> Dict[str, Union[nn.Module, Callable]]:
#         if transform and not isinstance(transform, Dict):
#             raise MisconfigurationException(
#                 "Transform should be a dict. "
#                 f"Here are the available keys for your transforms: {DataPipeline.PREPROCESS_FUNCS}."
#             )
#         if "per_batch_transform" in transform and "per_sample_transform_on_device" in transform:
#             raise MisconfigurationException(
#                 f'{transform}: `per_batch_transform` and `per_sample_transform_on_device` '
#                 f'are mutual exclusive.'
#             )
#         return transform