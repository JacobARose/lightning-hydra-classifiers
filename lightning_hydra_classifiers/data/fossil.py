"""
Utility functions for loading Fossil Leaves images into TorchVision dataloaders

Created on: Thursday July 8th, 2021
Author: Jacob A Rose



"""


# TODO (Jacob): Hardcode the mean & std for PNAS, Extant Leaves, Imagenet, etc.. for standardization across lab


from .common import (display_images,
                     filter_df_by_threshold,
                     plot_class_distributions,
                     plot_trainvaltest_splits)
#                      LeavesDataset, 
# #                      LeavesLightningDataModule,
#                      TrainValSplitDataset, 
#                      SubsetImageDataset,
#                      seed_worker)


import logging
import collections
import numpy as np 
import random
import torch
import pytorch_lightning as pl
from typing import List, Callable, Dict, Union, Type, Optional, Any, Tuple, Sequence

import torchvision
import torchdata
from PIL import Image
import pandas as pd
from pathlib import Path
from itertools import repeat, chain
from more_itertools import collapse, flatten
from dataclasses import dataclass

from itertools import *

from sklearn.model_selection import train_test_split



available_datasets = {
    "Wilf_Fossil_512": "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/Fossil/ccrop_512/Wilf_Fossil",
    "Wilf_Fossil_1024": "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/Fossil/ccrop_1024/Wilf_Fossil",
    "Wilf_Fossil_1536": "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/Fossil/ccrop_1536/Wilf_Fossil",
    "Wilf_Fossil_2048": "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/Fossil/ccrop_2048/Wilf_Fossil",
    
    "Florissant_Fossil_512": "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/Fossil/ccrop_512/Florissant_Fossil",
    "Florissant_Fossil_1024": "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/Fossil/ccrop_1024/Florissant_Fossil",
    "Florissant_Fossil_1536": "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/Fossil/ccrop_1536/Florissant_Fossil",
    "Florissant_Fossil_2048": "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/Fossil/ccrop_2048/Florissant_Fossil"
}

available_datasets["Fossil_512"] = ["/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/Fossil/ccrop_512/Wilf_Fossil",
                                    "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/Fossil/ccrop_512/Florissant_Fossil"]
available_datasets["Fossil_1024"] = ["/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/Fossil/ccrop_1024/Wilf_Fossil",
                                     "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/Fossil/ccrop_1024/Florissant_Fossil"]
available_datasets["Fossil_1536"] = ["/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/Fossil/ccrop_1536/Wilf_Fossil",
                                     "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/Fossil/ccrop_1536/Florissant_Fossil"]
available_datasets["Fossil_2048"] = ["/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/Fossil/ccrop_2048/Wilf_Fossil",
                                     "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/Fossil/ccrop_2048/Florissant_Fossil"]



default_name = "Fossil_512"




__all__ = ["FossilDataset", "DatasetConfig"] #"ExtantLeavesDataset", "ExtantLightningDataModule"]

fossil_collections = {"Florissant":"florissant_fossil",
                      "Wilf":"wilf_fossil"}

@dataclass
class DatasetConfig:
    name: str
    dataset: str=None
    collection: str=None
    resolution: int=None
        
    num_files: Optional[int]=None
    num_classes: Optional[int]=None
    class_type: str="family"
    path_schema: str = "{family}_{genus}_{species}_{collection}_{catalog_number}"
                
        
    def __init__(self, name: str, **kwargs):
        self.name = name
        parts = self.name.split("_")
        self.resolution = int(parts[-1])
        if len(parts)==3:
            self.dataset = parts[1]
            self.collection = "_".join(parts[:2])
        elif len(parts)==2:    
            self.dataset = parts[0]
            self.collection = ["_".join([c, self.dataset]) for c in fossil_collections.keys()]
            
        self.__dict__.update(kwargs)

    def __repr__(self):
        disp = f"""<{str(type(self)).strip("'>").split('.')[1]}>:"""
        
        disp += "\nFields:\n"
        for k in self.__dataclass_fields__.keys():
            disp += f"    {k}: {getattr(self,k)}\n"
        return disp
    
####################################################################
####################################################################








class FossilDataset(torchdata.datasets.Files): #ImageDataset):
    """
    
    
    """
    
    splits_on_disk : Tuple[str] = tuple()
    
#     loader: Callable = Image.open
    transform = None
    target_transform = None
    
    class_type: str="family"
    totensor: Callable = torchvision.transforms.ToTensor()
    toPIL: Callable = torchvision.transforms.ToPILImage("RGB")

    def __init__(self,
                 files: List[Path]=None,
                 name: Optional[str]=None,
                 return_items: List[str] = ["image","target","path"],
                 image_return_type: str = "tensor",
                 class2idx: Optional[Dict[str,int]] = None,
                 threshold: int = 0,
                 *args, **kwargs):
        """
        If name is specified, files arg is ignored. Otherwise, creates torchdata.datasets.Files instance from files
        
        """
        
        if isinstance(name,str) and files is None:
            data = self.create_dataset(name=name)
            assert len(data.files) > 0
#             log.info(f"Creating FossilDataset with name {name} and {len(files)} files.")
            files = data.files
#             kwargs['name'] = name
        
        super().__init__(files=files, *args, **kwargs)
        
        self.name = name #kwargs.get("name","")
        self.return_items = return_items
        self.image_return_type = image_return_type
        
        self.samples = [self.parse_item(idx) for idx in range((len(self)))]
        self.targets = [sample[1] for sample in self.samples]
        self.threshold = threshold
        
        self.update_class2idx(class2idx=class2idx)
        
        self.filter_samples_by_threshold(threshold=threshold,
                                         update_class2idx=(class2idx is None),
                                         x_col = 'path',
                                         y_col = "family",
                                         in_place=True)
        
        self.config = DatasetConfig(self.name,
                                    class_type=self.class_type,
                                    num_files=len(self.files),
                                    num_classes=len(self.classes)
                                   )
    def update_class2idx(self,
                         class2idx: Optional[Dict[str,int]] = None):
        
        if isinstance(class2idx, dict):
            self.classes = sorted(class2idx.keys())
            self.class2idx = class2idx
        else:
            self.classes = sorted(set(self.targets))
            self.class2idx = {name:idx for idx, name in enumerate(self.classes)}

        
        
    def getitem(self, index: Union[int, Sequence]) -> Tuple[str]:
        
        
#         if isinstance(index, (Sequence, slice)):
#             path, family, genus, species, collection, catalog_number = self.samples[index]            
        
        path, family, genus, species, collection, catalog_number = self.samples[index]
        img = Image.open(path)
        return img, family, path

        
    def __getitem__(self, index: int):
        
#         print(index, type(index))
        
        img, family, path = self.getitem(index)
        target = self.class2idx[family]
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.image_return_type == "tensor":
            img = self.toPIL(img)
#             img = self.totensor(img)
            
        return img, target, str(path)
    
    
    def parse_item(self, index: int):
        path = self.files[index]
        family, genus, species, collection, catalog_number = path.stem.split("_", maxsplit=4)
        return path, family, genus, species, collection, catalog_number
    
    def __repr__(self):
        return self.config.__repr__()
        

    @classmethod
    def create_dataset(cls, name: str) -> "ImageDataset":
        dataset_dirs = available_datasets[name]
        if isinstance(available_datasets[name], str):
            dataset_dirs = [available_datasets[name]]
        assert isinstance(dataset_dirs, list)
        file_list = list(flatten(
                            [torchdata.datasets.Files.from_folder(Path(root),
                                                                  regex="*/*.jpg").files
                             for root in dataset_dirs]
                                                    ))
        data = FossilDataset(files=file_list,
                             name=name)

        return data
    
    @property
    def num_samples(self):
        return len(self)
    

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

        return FossilDataset(files=files,
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
    
    


            
    def display_grid(self,
                   indices: Union[int,List[int]]=16,
                   repeat_n: Optional[int]=1,
                   max_images: int=64,
                   columns: int=5,
                   width: int=20,
                   height: int=18,
                   label_wrap_length: int=50,
                   label_font_size: int=8) -> None:
        """
        idx = [0,1,2,3,4]
        data.display_grid(idx, repeat_n=5)
        idx = 10
        data.display_grid(idx, repeat_n=5)
        
        """
        
        if (int(repeat_n) == 0) or (repeat_n is None):
            repeat_n=1
        if isinstance(indices, int):
            indices = random.sample(range(self.num_samples), indices)

        indices = collapse((repeat(i,repeat_n) for i in indices))
        indices = list(indices)
        
        if self.image_return_type == 'tensor':
            images = [self.toPIL(self[idx][0]) for idx in indices]
        else:
            images = [self[idx][0] for idx in indices]
        labels = [self.classes[self[idx][1]] for idx in indices]
        # TODO: add optional pprediction
        
        display_images(images=images,
                       labels=labels,
                       max_images=max_images,
                       columns=columns,
                       width=width,
                       height=height,
                       label_wrap_length=label_wrap_length,
                       label_font_size=label_font_size)




#################################################




# class FossilLeavesDataset(LeavesDataset):
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



class FossilLightningDataModule(object):#LeavesLightningDataModule):
    
#     DatasetConstructor = FossilDataset
#     splits_on_disk : Tuple[str] = DatasetConstructor.splits_on_disk
    available_splits: Tuple[str] = ("train", "val", "test")
        
    image_size = 224
#     target_size = (224, 224)
    image_buffer_size = 32
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def __init__(self,
                 name: str=default_name,
                 test_split: float=0.3,
                 val_train_split: float=0.2,
                 threshold: Optional[int]=0,
                 batch_size: int=32,
                 seed: int=None,
                 num_workers=0,
                 debug: bool=False,
                 normalize: bool=True,
                 image_size: int = 'auto',
                 color_mode: str = 'grayscale',
                 channels: int=None,
                 dataset_dir: str=None,
                 predict_on_split: str="val",
                 **kwargs):
#         num_workers=4
        self.class2idx = None
        self.threshold = threshold
        self.grayscale = color_mode == 'grayscale'
        self.val_train_split = val_train_split
        self.test_split = test_split
                
        super().__init__(name=name,
                         batch_size=batch_size,
                         val_split=val_train_split,
                         test_split=test_split,
                         num_workers=num_workers,
                         seed=seed,
                         debug=debug,
                         normalize=normalize,
                         image_size=image_size,
                         channels=channels,
                         dataset_dir=dataset_dir,
                         return_paths=False,
                         predict_on_split=predict_on_split,
                         **kwargs)
        
        
    
    @property
    def available_datasets(self):
        return available_datasets

    
    
    def get_dataset_split(self, 
                          split: str,
                          ) -> "FossilDataset":
        """
        Overriding the superclass's default version of this method
        TODO: Move this splitting logic to the Dataset
        """

        self.data = self.DatasetConstructor(files=None,
                                       name=self.name,
                                       return_items = ["image","target","path"],
                                       image_return_type = "tensor",
                                       class2idx = self.class2idx,
                                       threshold = self.threshold)
        
        train_data, val_data, test_data = self.data.create_trainvaltest_splits(dataset=self.data,
                                                                               test_split=self.test_split,
                                                                               val_train_split=self.val_train_split,
                                                                               shuffle=self.shuffle,
                                                                               seed=self.seed,
                                                                               plot_distributions=False) # self.plot_distributions)
        
        if split == "train":
            logging.info(f'Retrieving the train split')
            return train_data
        if split == "val":
            logging.info(f'Retrieving the val split')
            return val_data
        if split == "test":
            logging.info(f'Retrieving the test split')
            return test_data
        elif split == "predict":
            if self.predict_on_split == "val":
                self.setup(stage="fit")
            elif self.predict_on_split == "test":
                self.setup(stage="test")
                
            logging.info(f'Retrieving the {self.predict_on_split} set for prediction~')
            predict_data = self.get_dataset_split(split=self.predict_on_split)
            return predict_data
        else:
            logging.error(f'[ERROR] `split` value {split} must be one of the following: (train, val, test)')
            raise