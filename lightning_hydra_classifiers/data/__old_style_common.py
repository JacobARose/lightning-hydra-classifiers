"""
DEPRECATED -- July 14th, 2021
old style of common Datasets based on ImageFolder. See common.py for newer torchdata.datasets.Files - based common classes

contrastive_learning.data.pytorch.common


Common classes & functions for simplifying the boilerplate code in definitions of custom datasets in this repo.

Created by: Sunday April 25th, 2021
Author: Jacob A Rose





"""
from torchvision.datasets import folder, vision, ImageFolder
from torch.utils.data import Dataset, Subset, random_split, DataLoader
from typing import Sequence
from typing import Any, Callable, List, Optional, Tuple, Union, Dict
import random
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import functional as F
import pytorch_lightning as pl
from pathlib import Path
from PIL import ImageOps
import wandb
import matplotlib.pyplot as plt
from PIL.Image import Image
import textwrap
import logging

__all__ = ['LeavesDataset', 'LeavesLightningDataModule', 'seed_worker', 'TrainValSplitDataset', 'SubsetImageDataset', 'plot_class_distributions',
'plot_trainvaltest_splits']



def validate_dataset_dir(name: Optional[str]=None,
                         dataset_dir: Optional[str]=None,
                         available_datasets: Optional[Dict[str,str]] = None):
    
    if (name is None) and (dataset_dir is None):
        raise Exception('Either `name` or `dataset_dir` must be provided.')
    
    available_datasets = available_datasets or {}
    if os.path.exists(str(dataset_dir)):
        dataset_dir = Path(dataset_dir) #Path(kwargs['dataset_dir'])
        name = Path(dataset_dir).stem
        if name not in available_datasets:
            available_datasets[name] = dataset_dir
    elif isinstance(name, str):
        try:
            assert name in available_datasets
            dataset_dir = Path(available_datasets[name])
        except AssertionError as e:
            raise Exception(f"{name} is not in the set of available datasets. Please try one of the following: \n{available_datasets.keys()}")
        
    else:
        print(f'Warning, no values for name or dataset_dir provided to constructor.')
    
    return name, dataset_dir, available_datasets


available_datasets = {}

class LeavesDataset(ImageFolder):

    splits_on_disk : Tuple[str]= ("train", "val", "test")
    
    def __init__(
            self,
            name: str=None,
            split: str="train",
            dataset_dir: Optional[str]=None,
            return_items: List[str] = ["image","target","path"],
            image_return_type: str = "tensor",
            class2idx: Optional[Dict[str,int]]=None,
            data_df: pd.DataFrame=None,
            **kwargs: Any
            ) -> None:
        self.imgs, self.samples, self.targets = [], [], []
        self.name, self.dataset_dir = name, Path(self.available_datasets[name])
#         self.name, self.dataset_dir, _ = validate_dataset_dir(name, dataset_dir, self.available_datasets)

#         self.name = name
        self.split = split
        if isinstance(self.dataset_dir, (str, Path)):
            self.split_dir = self.dataset_dir / self.split
            logging.info(f'self.split_dir={self.split_dir}')
            super().__init__(root=self.split_dir,
                             **kwargs)
            logging.info(f'split={split}, num_samples={len(self)}')
            logging.info(f'num_classes={len(self.classes)}')
            


        if (not hasattr(self,'class2idx')) and class2idx:
            self.update_class2idx()
            
        self.return_items = return_items
        self.image_return_type = image_return_type
            
#         self.return_paths = return_paths
        
        
    def update_class2idx(self,
                         class2idx: Optional[Dict[str,int]] = None):
        
        if isinstance(class2idx, dict):
            self.classes = sorted(class2idx.keys())
            self.class2idx = class2idx
        else:
            self.classes = sorted(set(self.targets))
            self.class2idx = {name:idx for idx, name in enumerate(self.classes)}

        

    @classmethod
    def from_wandb_table(cls, table: wandb.data_types.Table) -> "LeavesDataset":
        data_df = pd.DataFrame(data=table.data, columns=table.columns)
        data_df = data_df.assign(image=data_df.image.apply(lambda x: x._image))
        return data_df
        
        
        
        
#     @property
#     def return_paths(self):
#         return self._return_paths
    
#     @return_paths.setter
#     def return_paths(self, return_paths: bool):
#         self._return_paths = return_paths
#         if return_paths:
#             self.get_item = lambda sample: (sample[0], sample[1], sample[2]) # (img, label, path)
#         else:
#             self.get_item = lambda sample: (sample[0], sample[1]) # (img, label)


    @property
    def return_items(self):
        return self._return_items
    
    @return_items.setter
    def return_items(self, return_items: bool):
        self._return_items = return_items
        if return_items:
            self.get_item = lambda sample: (sample[0], sample[1], sample[2]) # (img, label, path)
        else:
            self.get_item = lambda sample: (sample[0], sample[1]) # (img, label)


    @property
    def available_datasets(self):
        """
        Subclasses must define this property
        Must return a dict mapping dataset key names to their absolute paths on disk.
        """
        return available_datasets
        
    @available_datasets.setter
    def available_datasets(self, new: Dict[str,str]):
        """
        Subclasses must define this property
        Must return a dict mapping dataset key names to their absolute paths on disk.
        """
        try:
            available_datasets.update(new)
        except:
            raise Exception

        
        
    def __repr__(self):
        content = super().__repr__()
        content += f'\n    Name: {self.name}'
        content += f'\n    Split: {self.split}'
        return content
    
    

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index][:2]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return self.get_item((img, target, path))
    
    
    ############################
    

    def filter_samples_by_threshold(self,
                                    threshold: int=1,
                                    update_class2idx: bool=True,
                                    x_col = 'path',
                                    y_col = "family",
                                    in_place: bool=False) -> "FossilDataset":
        
        
        counts = collections.Counter(self.targets)
        if min(counts.values()) >= threshold:
            # Only filter if any classes are still below the threshold
            if in_place:
                return None
            else:
                return self
        
        if update_class2idx:
            class2idx=None
        else:
            class2idx=self.class2idx


        df = pd.DataFrame(self.samples)
        df = df.rename(columns={0:"path",
                                1:"family",
                                2:"genus",
                                3:"species",
                                4:"collection",
                                5:"catalog_number"})#.value_counts()

        df = filter_df_by_threshold(df=df,
                                    threshold=threshold,
                                    y_col=y_col)

        files = df[x_col].to_list()
        
        if in_place:
            self.__init__(files=files,
                          name=self.name,
                          return_items=self.return_items,
                          image_return_type=self.image_return_type,
                          class2idx=class2idx,
                          threshold=threshold)
            return None

        return type(self)(files=files,
                             name=self.name,
                             return_items=self.return_items,
                             image_return_type=self.image_return_type,
                             class2idx=class2idx)

    def select_from_indices(self,
                            indices: Sequence,
                            update_class2idx: bool=False,
                            x_col = 'path',
                            y_col = "family",
                            in_place: bool=False) -> Optional["FossilDataset"]:
        """
        Helper method to create a new FossilDataset containing only samples contained in `indices`
        Useful for producing train/val/test splits

        """
        if update_class2idx:
            class2idx=None
        else:
            class2idx=self.class2idx


        df = pd.DataFrame(self.samples)
        df = df.rename(columns={0:"path",
                                1:"family",
                                2:"genus",
                                3:"species",
                                4:"collection",
                                5:"catalog_number"})#.value_counts()

        df = df.iloc[indices,:]
        
        files = df[x_col].to_list()
        
        if in_place:
            self.__init__(files=files,
                          name=self.name,
                          return_items=self.return_items,
                          image_return_type=self.image_return_type,
                          class2idx=class2idx)
            return None

        return FossilDataset(files=files,
                             name=self.name,
                             return_items=self.return_items,
                             image_return_type=self.image_return_type,
                             class2idx=class2idx)    
    
    @classmethod
    def create_trainvaltest_splits(cls,
                                   dataset,
                                   test_split: float=0.3,
                                   val_train_split: float=0.2,
                                   shuffle: bool=True,
                                   seed: int=3654,
                                   plot_distributions: bool=False) -> Tuple["FossilDataset"]:

        dataset_size = len(dataset)
        indices = np.arange(dataset_size)
        samples = np.array(dataset.samples)
        targets = np.array(dataset.targets)

        train_val_idx, test_idx = train_test_split(indices,
                                                   test_size=test_split,
                                                   random_state=seed,
                                                   shuffle=shuffle,
                                                   stratify=targets)

        train_val_targets = targets[train_val_idx]
        trainval_indices = np.arange(len(train_val_targets))
        train_idx, val_idx = train_test_split(trainval_indices,
                                              test_size=val_train_split,
                                              random_state=seed,
                                              shuffle=shuffle,
                                              stratify=train_val_targets)

        train_data = dataset.select_from_indices(indices=train_idx,
                                                 update_class2idx=False,
                                                 x_col = 'path',
                                                 y_col = "family")

        val_data = dataset.select_from_indices(indices=val_idx,
                                               update_class2idx=False,
                                               x_col = 'path',
                                               y_col = "family")

        test_data = dataset.select_from_indices(indices=test_idx,
                                                update_class2idx=False,
                                                x_col = 'path',
                                                y_col = "family")
        
        if plot_distributions:
            cls.plot_trainvaltest_splits(train_data,
                                         val_data,
                                         test_data)

        return train_data, val_data, test_data
    
    @staticmethod
    def plot_trainvaltest_splits(train_data,
                                 val_data,
                                 test_data):
    
        fig, ax = plot_trainvaltest_splits(train_data,
                                           val_data,
                                           test_data)
        return fig, ax
    
    
    
    
    
    
    
    
    
    
    
    
#######################################################


class LeavesLightningDataModule(pl.LightningDataModule):
    
#     worker_init_fn=seed_worker
    
    image_size = 224
#     target_size = (224, 224)
    channels = 3
    image_buffer_size = 32
    mean = [0.5, 0.5, 0.5]
    std = [1.0, 1.0, 1.0]
    
    totensor: Callable = transforms.ToTensor()
    
    def __init__(self,
                 name: str=None,
                 batch_size: int=32,
                 val_split: float=0.0,
                 test_split: float=0.0,
                 normalize: bool=True,
                 image_size: Union[int,str] = None,
                 grayscale: bool = True,
                 channels: int=None,
                 dataset_dir: Optional[str]=None,
                 return_paths: bool=False,
                 num_workers=0,
                 pin_memory: bool=False,
                 seed: int=None,
                 debug: bool=False,
                 predict_on_split: str="val",
                 **kwargs
                 ):
        """ Abstract Base Class meant to be subclassed for each custom datamodule associated with the leavesdb database.
        
        Subclasses must override definitions for the following methods/properties:
        
        - available_datasets -> returns a dictionary mapping dataset names -> dataset absolute paths
        - get_dataset_split -> returns a LeavesDataset object for either the train, val, or test splits.

        Args:
            name (str, optional): Defaults to None.
                Subclasses should define a default name.
            batch_size (int, optional): Defaults to 32.
            val_split (float, optional): Defaults to 0.2.
                Must be within [0.0, 1.0]. If omitted, looks for val/ subdir on disk.
            test_split (float, optional): Defaults to 0.0.
                Must be within [0.0, 1.0]. If omitted, looks for test/ subdir on disk.
            normalize (bool, optional): Defaults to True.
                If True, applies mean and std normalization to images at the end of all sequences of transforms. Base class has default values of mean = [0.5, 0.5, 0.5] and std = [1.0, 1.0, 1.0], which results in normalization becoming an identity transform. Subclasses should provide custom overrides for these values. E.g. imagenet  for example
            image_size (Union[int,str], optional): Defaults to 224 if None.
                if 'auto', extract image_size from dataset name (e.g. "Exant_Leaves_family_10_1024 -> image_size=1024")
            grayscale (bool, optional): Defaults to True.
                
            channels (int, optional): Defaults to 3 if None.
                Options are either 1 or 3 channels. No-op if grayscale=False.
            return_paths (bool, optional): Defaults to False.
                If True, internal datasets return tuples of length 3 containing (img, label, path). If False, return tuples of length 2 containing (img, label).
            num_workers (int, optional): Defaults to 0.
            pin_memory (bool, optional): Defaults to False.
            seed (int, optional): Defaults to None.
                This must be set for reproducable train/val splits.
            debug (bool, optional): Defaults to False.
                Set to True in order to turn off (1) shuffling, and (2) image augmentations.
                Purpose is to allow removing as much randomness as possible during runtime. Augmentations are removed by applying the eval_transforms to all subsets.

        """
        super().__init__()

#         self.name, self.dataset_dir, available_datasets = validate_dataset_dir(name, dataset_dir, self.available_datasets)
        self.name = name
        self.dataset_dir = dataset_dir
        if name in self.available_datasets:
            self.dataset_dir = self.available_datasets[name]
            logging.info(f'self.dataset_dir={self.dataset_dir}')
        
        if val_split: # is not None:
            assert ((val_split >= 0) and (val_split <= 1)), "[!] val_split should be either None, or a float in the range [0, 1]."
        self.val_split = val_split
        if test_split: # is not None:
            assert ((test_split >= 0) and (test_split <= 1)), "[!] test_split should be either None, or a float in the range [0, 1]."
        self.test_split = test_split
        
        if image_size == 'auto':
            try:
                image_size = int(self.name.split('_')[-1])
            except ValueError:
                image_size = self.image_size
    
        self.batch_size = batch_size
        self.normalize = normalize
        self.image_size = image_size or self.image_size
        self.grayscale = grayscale
        self.channels = channels or self.channels
        self.return_paths = return_paths
        self.predict_on_split = predict_on_split
        
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        self.debug = debug
        self.augment = not debug
        self.shuffle = not debug

        
        self._is_fit_setup=False
        self._is_test_setup=False
        
        
    def setup(self,
              stage: str=None,
              train_transform: Optional[Callable] = None,
              eval_transform: Optional[Callable] = None,
              target_transform: Optional[Callable] = None
              ):
        logging.info(f"stage={stage}")
        if stage == 'fit' or stage is None:
            logging.info(f"Calling datamodule.setup({stage})")
            self.init_dataset_stage(stage='fit',
                                    train_transform=train_transform,
                                    eval_transform=eval_transform)
            self._is_fit_setup=True
        elif stage == 'test': # or stage is None:
            logging.info(f"Calling datamodule.setup({stage})")
            self.init_dataset_stage(stage='test',
                                    eval_transform=eval_transform)
            self._is_test_setup=True
        elif stage == 'predict': # or stage is None:
            logging.info(f"Calling datamodule.setup({stage})")
            self.init_dataset_stage(stage='predict',
                                    eval_transform=eval_transform)
        else:
            raise DataStageError(stage,
                                 valid_stages=("fit", "test", "predict", None))

        
    def init_dataset_stage(self,
                           stage: str='fit',
                           train_transform: Optional[Callable] = None,
                           eval_transform: Optional[Callable] = None):
    
        
        self.train_transform = train_transform or self.default_train_transforms(augment=self.augment,
                                                                                normalize=self.normalize,
                                                                                grayscale=self.grayscale,
                                                                                channels=self.channels)
        self.eval_transform = eval_transform or self.default_eval_transforms(normalize=self.normalize,
                                                                             resize_PIL=True,
                                                                             grayscale=self.grayscale,
                                                                             channels=self.channels)
        
        if stage == 'fit' or stage is None:
            logging.info(f'stage={stage}')
            self.train_dataset = self.get_dataset_split(split='train')
            self.val_dataset = self.get_dataset_split(split='val')
            self.classes = self.train_dataset.classes
            self.num_classes = len(self.classes)
            
            logging.info(f'stage={stage}, num_classes={self.num_classes}')
            self.train_dataset.transform = self.train_transform
            self.val_dataset.transform = self.eval_transform
            
        elif stage == 'test' or stage is None:
            self.test_dataset = self.get_dataset_split(split='test')
            self.test_dataset.transform = self.eval_transform
        elif stage == 'predict': # or stage is None:
            return_paths = bool(self.return_paths)
            self.return_paths = True
            self.predict_dataset = self.get_dataset_split(split='predict')
            self.predict_dataset.transform = self.eval_transform
            self.return_paths = return_paths
            
        else:
            raise DataStageError(stage,
                                 valid_stages=("fit", "test", "predict", None))

#             raise  ValueError(f"stage value ({stage}) is not in set of valid stages, must provide 'fit', 'test', or None.")
            
    def get_dataset_split(self, split: str) -> LeavesDataset:        
        """
        TODO: Move this splitting logic to the Dataset
        """
        if split in ("train","val"):
            train_dataset = self.DatasetConstructor(self.name,
                                                    split="train",
                                                    dataset_dir=self.dataset_dir,
                                                    return_paths=self.return_paths)
            if "val" in os.listdir(train_dataset.dataset_dir):
                val_dataset = self.DatasetConstructor(self.name,
                                                      split="val",
                                                      dataset_dir=self.dataset_dir,
                                                      return_paths=self.return_paths)
            elif self.val_split:
                logging.info(f'self.val_split={self.val_split}')
                train_dataset, val_dataset = TrainValSplitDataset.train_val_split(train_dataset,
                                                                                  val_split=self.val_split,
                                                                                  seed=self.seed)
                print('finished train_dataset, val_dataset = TrainValSplitDataset.train_val_split')
                logging.info(f'len(train_dataset)={len(train_dataset)}, len(val_dataset)={len(val_dataset)}')
            else:
                print(f'[ERROR] Must provide either a `val` subdir in self.dataset_dir = {self.dataset_dir}, or a non-zero value for `val_split`.')
                raise
                
            if split == "train":
                return train_dataset
            else:
                return val_dataset
        elif split == "test":
            test_dataset = self.DatasetConstructor(self.name,
                                                   split="test",
                                                   dataset_dir=self.dataset_dir,
                                                   return_paths=self.return_paths)
            return test_dataset
        
        elif split == "predict":
            if self.predict_on_split == "val":
#                 if not self._is_fit_setup:
                self.setup(stage="fit")
            elif self.predict_on_split == "test":
#                 if not self._is_test_setup:
                self.setup(stage="test")
                print('returning the test set for prediction~')
            predict_dataset = self.get_dataset_split(split=self.predict_on_split)
            return predict_dataset
        else:
            raise Exception(f"'split' argument must be a string pertaining to one of the following: {self.available_splits}")


        
    def get_dataset(self, stage: str='train'):
        if stage=='train': return self.train_dataset
        if stage=='val': return self.val_dataset
        if stage=='test': return self.test_dataset
        if stage=='predict': return self.predict_dataset

        
    def get_dataloader(self, stage: str='train'):
        if stage=='train': return self.train_dataloader()
        if stage=='val': return self.val_dataloader()
        if stage=='test': return self.test_dataloader()
        if stage=='predict': return self.predict_dataloader()

    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset,
                                  num_workers=self.num_workers,
                                  batch_size=self.batch_size,
                                  pin_memory=self.pin_memory,
                                  shuffle=self.shuffle,
                                  drop_last=True)
        return train_loader
        
    def val_dataloader(self):
        val_loader = DataLoader(self.val_dataset,
                                num_workers=self.num_workers,
                                batch_size=self.batch_size,
                                pin_memory=self.pin_memory)
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.test_dataset,
                                 num_workers=self.num_workers,
                                 batch_size=self.batch_size,
                                 pin_memory=self.pin_memory)
        return test_loader

    
    def predict_dataloader(self):
        predict_loader = DataLoader(self.predict_dataset,
                                num_workers=self.num_workers,
                                batch_size=self.batch_size,
                                pin_memory=self.pin_memory)
        return predict_loader


    @property
    def normalize_transform(self) -> Callable:
        return transforms.Normalize(mean=self.mean, 
                                    std=self.std)
    
#     def resized_crop(self, img: torch.Tensor) -> Callable: #torch.Tensor:
#         crop_size = min([self.image_size,
#                     self.image_size+self.image_buffer_size])
#         return transforms.Lambda(F.resized_crop(img,
#                                  top=0.0,
#                                  left=0.0,
#                                  height = crop_size,
#                                  width = crop_size,
#                                  size = self.image_size,
#                                  interpolation = Image.LANCZOS))
        
    def default_train_transforms(self,
                                 normalize: bool=True, 
                                 augment:bool=True,
                                 grayscale: bool=True,
                                 channels: Optional[int]=3):
        """Subclasses can override this or user can provide custom transforms at runtime"""
        transform_list = []
#         transform_jit_list = []        
        if augment:
            transform_list.extend([transforms.RandomResizedCrop(size=self.image_size,
                                                                scale=(0.25, 1.2),
                                                                ratio=(0.7, 1.3),
                                                                interpolation=2),
                                   self.totensor
                                 ])
        return self.default_eval_transforms(normalize=normalize,
                                            resize_PIL=False,
                                            grayscale=grayscale,
                                            channels=channels,
                                            transform_list=transform_list)
#         if normalize:
#             transform_jit_list.append(self.normalize_transform)
            
#         if grayscale:
#             transform_jit_list.append(transforms.Grayscale(num_output_channels=channels))
            
#         return transforms.Compose([*transform_list, *transform_jit_list])

    
    def default_eval_transforms(self, 
                                normalize: bool=True,
                                resize_PIL: bool=True,
                                grayscale: bool=True,
                                channels: Optional[int]=3,
                                transform_list: Optional[List[Callable]]=None):
        """Subclasses can override this or user can provide custom transforms at runtime"""
        transform_list = transform_list or []
        transform_jit_list = []
        
        if resize_PIL:
            # if True, assumes input images are PIL.Images (But need to check if this even matters.)
            # if False, expects input images to already be torch.Tensors
            transform_list.extend([transforms.Resize(self.image_size+self.image_buffer_size),
                                   transforms.CenterCrop(self.image_size),
                                   self.totensor])
        if normalize:
            transform_jit_list.append(self.normalize_transform)
            
        if grayscale:
            transform_jit_list.append(transforms.Grayscale(num_output_channels=channels))
            
        return transforms.Compose([*transform_list, *transform_jit_list])

    
    def get_batch(self, stage: str='train', batch_idx: int=0):
        """Useful utility function for selecting a specific batch by its index and stage."""
        data = self.get_dataloader(stage)
        for i, batch in enumerate(iter(data)):
            if i == batch_idx:
                return batch
        
    
    def show_batch(self, stage: str='train', batch_idx: int=0, cmap: str='cividis', grayscale=True, titlesize=30):
        """
        Useful utility function for plotting a single batch of images as a single grid.
        
        Good mild grayscale cmaps: ['magma', 'cividis']
        Good mild grayscale cmaps: ['plasma', 'viridis']
        """
        batch = self.get_batch(stage=stage, batch_idx=batch_idx)
        x, y = batch[:2]
        batch_size = x.shape[0]
        
        fig, ax = plt.subplots(1,1, figsize=(20,20))
        grid_img = torchvision.utils.make_grid(x, nrow=int(np.ceil(np.sqrt(batch_size))))
        
        img_min, img_max = grid_img.min(), grid_img.max()
#         grid_img = (grid_img - img_min)/(img_max - img_min)
        
        if np.argmin(grid_img.shape) == 0:
            grid_img = grid_img.permute(1,2,0)

        if grayscale and len(grid_img.shape)==3:
            grid_img = grid_img[:,:,0]

        img_ax = ax.imshow(grid_img, cmap=cmap, vmin = img_min, vmax = img_max)
        colorbar(img_ax)
        plt.axis('off')
        plt.suptitle(f'{stage} batch', fontsize=titlesize)
        plt.tight_layout()
        return fig, ax

    @property
    def dims(self):
        return (self.channels, self.image_size, self.image_size)
    
    def __repr__(self):
        content = str(type(self)) + '\n'
        content += f'Name: {self.name}' + '\n'
        content += f'Size: {self.size()}' + '\n'
        content += f'batch_size: {self.batch_size}' + '\n'
        content += f'seed: {self.seed}'
        return content

    
    @property
    def available_datasets(self):
        """
        Subclasses must define this property
        Each custom dataset must have its own custom subclass of LeavesDataset and LeavesLightningDataModule.
        Must return a dict mapping dataset key names to their absolute paths on disk.
        """
        return available_datasets
        
    @available_datasets.setter
    def available_datasets(self, new: Dict[str,str]):
        """
        Subclasses must define this property
        Must return a dict mapping dataset key names to their absolute paths on disk.
        """
        try:
            available_datasets.update(new)
        except:
            raise Exception

#######################################################



class CommonLeavesError(ValueError):
    
    def __init__(self, obj: str=None, msg: str=None):
        if msg is None:
            # Set some default useful error message
            msg = "An error occured with Leaves object:\n %s" % str(obj)
        super().__init__(msg)
        self.obj = obj

class DataStageError(CommonLeavesError):
    def __init__(self, requested_stage, valid_stages):
        msg = f"'{requested_stage}' is not in the set of valid stages, must provide one of the following:\n     "
        msg += str(valid_stages)
        super().__init__(msg)


#######################################################



def display_images(
                   images: List[Image],
                   labels: Optional[List[str]]=None,
                   max_images: int=32,
                   columns: int=5,
                   width: int=20,
                   height: int=12,
                   label_wrap_length: int=50,
                   label_font_size: int="medium") -> None:

    if len(images) > max_images:
        print(f"Showing {max_images} images of {len(images)}:")
        images=images[0:max_images]

    rows = int(len(images)/columns)
#     height = max(height, rows * height)
    plt.subplots(rows, columns, figsize=(width, height), sharex=True, sharey=True)
    for i, image in enumerate(images):

        plt.subplot(rows + 1, columns, i + 1)
        plt.imshow(image)
        title=None
        if isinstance(labels, list):
            title = labels[i]
        elif hasattr(image, 'filename'):
            title=image.filename
            
        if title:
#             if title.endswith("/"): title = title[:-1]
            title=Path(title).stem
            title=textwrap.wrap(title, label_wrap_length)
            title="\n".join(title)
            plt.title(title, fontsize=label_font_size); 

    plt.subplots_adjust(left=None, bottom=0.1, right=None, top=0.85, wspace=0.05, hspace=0.1)



#######################################################


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    

#######################################################
# TODO: Move the following elsewhere

def filter_df_by_threshold(df: pd.DataFrame,
                           threshold: int,
                           y_col: str='family'):
    """
    Filter rare classes from dataset in a pd.DataFrame
    
    Input:
        df (pd.DataFrame):
            Must contain at least 1 column with name given by `y_col`
        threshold (int):
            Exclude any rows from df that contain a `y_col` value with fewer than `threshold` members in all of df.
        y_col (str): default="family"
            The column in df to look for rare classes to exclude.
    Output:
        (pd.DataFrame):
            Returns a dataframe with the same number of columns as df, and an equal or lower number of rows.
    """
    return df.groupby(y_col).filter(lambda x: len(x) >= threshold)




import collections
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context(context='talk', font_scale=0.8)
# sns.set_style("whitegrid")

def plot_class_distributions(targets: List[Any], 
                             sort: Union[bool,Sequence]=True,
                             ax=None,
                             xticklabels: bool=True):
    """
    Example:
        counts = plot_class_distributions(targets=data.targets, sort=True)
    """
    counts = collections.Counter(targets)
    if hasattr(sort, "__len__"):
        counts = {k: counts[k] for k in sort}
    elif sort is True:
        counts = dict(sorted(counts.items(), key = lambda x:x[1], reverse=True))

    keys = list(counts.keys())
    values = list(counts.values())

    if ax is None:
        plt.figure(figsize=(16,12))
    ax = sns.histplot(x=keys, weights=values, discrete=True, ax=ax)
    plt.sca(ax)
    if xticklabels:
        plt.xticks(
            rotation=45, 
            horizontalalignment='right',
            fontweight='light',
            fontsize='medium'
        )
    else:
        ax.set_xticklabels([])
    
    return counts


def plot_trainvaltest_splits(train_data,
                             val_data,
                             test_data):
    """
    Create 3 vertically-stacked count plots of train, val, and test dataset class label distributions
    """
    fig, ax = plt.subplots(3, 1, figsize=(16,8*3))

    train_counts = plot_class_distributions(targets=train_data.targets, sort=True, ax = ax[0], xticklabels=False)
    plt.gca().set_title(f"train (n={len(train_data)})", fontsize='large')
    sort_classes = train_counts.keys()

    val_counts = plot_class_distributions(targets=val_data.targets, ax = ax[1], sort=sort_classes, xticklabels=False)
    plt.gca().set_title(f"val (n={len(val_data)})", fontsize='large')
    test_counts = plot_class_distributions(targets=test_data.targets, ax = ax[2], sort=sort_classes)
    plt.gca().set_title(f"test (n={len(test_data)})", fontsize='large')

    num_samples = len(train_data) + len(val_data) + len(test_data)
    
    plt.suptitle(f"Train-Val-Test_splits (total={num_samples})", fontsize='x-large')

#     plt.suptitle(f"Train-Val-Test_splits (total={len(data)})", fontsize='x-large')

    plt.subplots_adjust(bottom=0.1, top=0.95, wspace=None, hspace=0.07)
    
    return fig, ax



#######################################################













class TrainValSplitDataset(LeavesDataset): #ImageFolder):
    """
    Use this class as a Factory for creating train and val splits from a single dataset, 
    returning 2 separate datasets with no shared references (as opposed to the native SubsetDataset)
    
    Example
    =======
    >> train_dataset = ImageFolder(root=train_dir)
    >> train_data, val_data = TrainValSplitDataset.train_val_split(train_dataset, val_split=0.2, seed=0)
    
    or, identically,
    
    >> train_dataset = TrainValSplitDataset(root=train_dir)
    >> train_data, val_data = TrainValSplitDataset.train_val_split(train_dataset, val_split=0.2, seed=0)
    """
    
    all_params: List[str]= [
                            'class_to_idx',
                            'classes',
                            'extensions',
                            'imgs',
                            'loader',
                            'root',
                            'samples',
                            'target_transform',
                            'targets',
                            'transform',
                            'transforms'
                            ]
        
    non_sample_params: List[str]= [
                                   'class_to_idx',
                                   'classes',
                                   'extensions',
                                   'loader',
                                   'root',
                                   'target_transform',
                                   'transform',
                                   'transforms'
                                   ]
        
    sample_params: List[str] = ['imgs',
                                'samples',
                                'targets']
    
    @classmethod
    def train_val_split(cls, full_dataset, val_split: float=0.2, seed: float=None) -> Tuple[ImageFolder]:
        
        num_samples = len(full_dataset)
        split_idx = (int(np.ceil((1-val_split) * num_samples)),
                     int(np.floor(val_split * num_samples)))
        if seed is None:
            generator = None
        else:
            generator = torch.Generator().manual_seed(seed)

        train_data, val_data = random_split(full_dataset, 
                                            split_idx,
                                            generator=generator)
        
        train_dataset = cls.select_from_dataset(full_dataset, indices=train_data.indices)
        val_dataset = cls.select_from_dataset(full_dataset, indices=val_data.indices)
        
        return train_dataset, val_dataset
        
    @classmethod
    def from_dataset(cls, dataset):
        new_dataset = cls(root=dataset.root)
        
        for key in cls.all_params:
            if hasattr(dataset, key):
                setattr(new_dataset, key, getattr(dataset, key))
                         
        return new_dataset
    
    
    @classmethod
    def from_datasets(cls, *datasets: list, root: str='auto', remap_classes: Callable=None):
        """
        
        """
        skip_first_dataset = False
        if root=='auto':
            root = datasets[0].root
            skip_first_dataset = True
        new_dataset = cls(root=root)

        for key in cls.non_sample_params:
            if hasattr(datasets[0], key):
                setattr(new_dataset, key, getattr(datasets[0], key))

        for dataset in datasets:
            if skip_first_dataset:
                skip_first_dataset=False
                continue
            for key in cls.sample_params:
                if hasattr(dataset, key):
                    getattr(new_dataset, key).extend(getattr(dataset, key))
#             new_dataset.samples.extend(dataset.samples)
#             print(len(new_dataset.samples))
                         
        return new_dataset
    
    
    @classmethod
    def select_from_dataset(cls, dataset, indices=None):
        upgraded_dataset = cls.from_dataset(dataset)
        return upgraded_dataset.select(indices)
    
    
    def select(self, indices):
        new_subset = self.from_dataset(self)
        for key in self.sample_params:
            old_attr = getattr(self, key)
            new_attr = [old_attr[idx] for idx in indices]
            setattr(new_subset, key, new_attr)
        return new_subset

                
#     def __repr__(self) -> str:
#         head = "Dataset " + self.__class__.__name__
#         body = ["Number of datapoints: {}".format(self.__len__())]
#         if self.root is not None:
#             body.append("Root location: {}".format(self.root))
#         body += self.extra_repr().splitlines()
#         if hasattr(self, "transforms") and self.transforms is not None:
#             body += [repr(self.transforms)]
#         lines = [head] + [" " * self._repr_indent + line for line in body]
#         return '\n'.join(lines)

#     def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
#         lines = transform.__repr__().splitlines()
#         return (["{}{}".format(head, lines[0])] +
#                 ["{}{}".format(" " * len(head), line) for line in lines[1:]])


    
###############################
###############################
# class SubsetImageDataset(folder.ImageFolder):
class SubsetImageDataset(folder.DatasetFolder):
# class SubsetDataset(Dataset):
    """
    Custom class for creating a Subset of a Dataset while retaining the built-in methods/attributes/properties of Datasets.
    
    User provides a full dataset to be split, along with indices for inclusion in this subset.

    Arguments:
        data (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self,
                 data: Dataset,
                 indices: Sequence,
                 transforms: Optional[Callable] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 ):
        subset = Subset(data, indices)
        self.indices = subset.indices
        data = subset.dataset

        self.class_to_idx = data.class_to_idx
        self.classes = data.classes
        self.root = data.root
        self.loader = data.loader
        self.extensions = data.extensions
                
        self.samples = [data.samples[idx] for idx in self.indices]
        self.targets = [data.targets[idx] for idx in self.indices]

        
        #######################################
        transforms = transforms or data.transforms
        transform = transform or data.transform
        target_transform = target_transform or data.target_transform
        
        has_separate_transform = transform is not None or target_transform is not None

        self.transform = transform or data.transform
        self.target_transform = target_transform or data.target_transform
        if has_separate_transform:
            transforms = vision.StandardTransform(transform, target_transform)
        self.transforms = transforms
        
    def __len__(self):
        return len(self.targets)
    
    
    
    
def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar
    
    
def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')
    
    
    
#######################################
#######################################