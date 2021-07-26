"""
lightning_hydra_classifiers.data.common


Common classes & functions for simplifying the boilerplate code in definitions of custom datasets in this repo.

Created on: Sunday April 25th, 2021
Updated on: Tuesday July 13th, 2021
    - Refactoring common datasets to have torchdata.datasets.Files as superclass by default, instead of torchvision ImageFolders. For greater flexibility.





"""
from rich import print as pp
import collections
from torchvision.datasets import folder, vision, ImageFolder
from torch.utils.data import Dataset, Subset, random_split, DataLoader
from typing import Sequence
from typing import Any, Callable, List, Optional, Tuple, Union, Dict
from omegaconf import DictConfig, OmegaConf
import random
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import torchdata
import torchvision
from torchvision import transforms
from torchvision.transforms import functional as F
import pytorch_lightning as pl
from pathlib import Path
from PIL import ImageOps
import wandb
import matplotlib.pyplot as plt
# from PIL.Image import Image
from PIL import Image
import textwrap
import logging
from itertools import repeat, chain
from more_itertools import collapse, flatten
from copy import deepcopy
# from sklearn.model_selection import train_test_split

from lightning_hydra_classifiers.utils.common_utils import LabelEncoder, trainvaltest_split, trainval_split
from lightning_hydra_classifiers.utils.registry import Registry
from lightning_hydra_classifiers.utils import template_utils

log = template_utils.get_logger(__name__)

__all__ = ['LeavesDataset', 'LeavesLightningDataModule', 'seed_worker', 'plot_class_distributions', 'plot_trainvaltest_splits', 
           'plot_split_distributions', "save_config", "load_config", "export_image_data_diagnostics", "export_dataset_to_csv", "import_dataset_from_csv"]


# __all__ = ['LeavesDataset', 'LeavesLightningDataModule', 'seed_worker', 'TrainValSplitDataset', 'SubsetImageDataset', 'plot_class_distributions',
# 'plot_trainvaltest_splits']
"testing";
#         if trainer.is_global_zero and trainer.logger:
#             trainer.logger.after_save_checkpoint(proxy(self))
#########################################################

# __all__ = ["FossilDataset", "DatasetConfig"] #"ExtantLeavesDataset", "ExtantLightningDataModule"]

# fossil_collections = {"Florissant":"florissant_fossil",
#                       "Wilf":"wilf_fossil"}
from dataclasses import dataclass

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





@dataclass 
class PathSchema:
    path_schema: str = Path("{family}_{genus}_{species}_{collection}_{catalog_number}")
        
    def __init__(self,
                 path_schema,
                 sep: str="_"):

        self.sep = sep
        self.schema_parts: List[str] = path_schema.split(sep)
        self.maxsplit: int = len(self.schema_parts) - 2
    
    def parse(self, path: Union[Path, str], sep: str="_"):
    
        parts = Path(path).stem.split(sep, maxsplit=self.maxsplit)
        if len(parts) == 5:
            family, genus, species, collection, catalog_number = parts
        elif len(parts) == 4:
            family, genus, species, catalog_number = parts
            collection = catalog_number.split("_")[0]
        else:
            print(f'len(parts)={len(parts)}, parts={parts}, path={path}')

        return family, genus, species, collection, catalog_number
    


totensor: Callable = torchvision.transforms.ToTensor()
toPIL: Callable = torchvision.transforms.ToPILImage("RGB")

    
class CommonDataSplitter:
        
    valid_splits: Tuple[str] = ("train", "val", "test")

    @classmethod
    def get_split_subdir_stems(cls, dataset_dir: str):
        
        subdirs = os.listdir(dataset_dir)
        stems = []
        for subdir in subdirs:
            if subdir in cls.valid_splits:
                stems.append(subdir)
        return stems
        
    @classmethod
    def locate_files(cls,
                     dataset_dir: str,
                     select_subset: Optional[str]=None):

        files = {}
        splits = cls.get_split_subdir_stems(dataset_dir=dataset_dir)
        
        if len(splits) > 1:
            # root
            #     /train
            #          /class_0
            #     /test
            #     ....
            for subdir in splits:
                files[subdir] = TorchFiles.from_folder(Path(dataset_dir, subdir), regex="*/*.jpg").files 
            files["all"] = list(flatten([files[subdir] for subdir in splits]))
            
        elif len(os.listdir(dataset_dir)) > 1:
            # root
            #     /class_0
            #     /class_1
            #     ....
            files["all"] = TorchFiles.from_folder(Path(dataset_dir), regex="*/*.jpg").files
        else:
            raise Exception(f"# of valid subdirs = {len(os.listdir(dataset_dir))} is invalid for locating files.")

        if isinstance(select_subset, str):
            files = {select_subset: files[select_subset]}

        return files


    @classmethod
    def create_trainvaltest_splits(cls,
                                   data: torchdata.Dataset,
                                   test_split: Union[str, float]=0.3,
                                   val_train_split: float=0.2,
                                   shuffle: bool=True,
                                   seed: int=3654,
                                   stratify: bool=True,
                                   plot_distributions: bool=False) -> Tuple["FossilDataset"]:
        if (test_split == "test") or (test_split is None):
            train_split = 1 - val_train_split
            if hasattr(data, f"test_dataset"):
                data = getattr(data, f"train_dataset")            
        elif isinstance(test_split, float):
            train_split = 1 - (test_split + val_train_split)
        else:
            raise ValueError(f"Invalid split arguments: val_train_split={val_train_split}, test_split={test_split}")
            

        splits=(train_split, val_train_split, test_split)
        splits = list(filter(lambda x: isinstance(x, float), splits))
        y = data.targets
        
        if len(splits)==2:
            data_splits = trainval_split(x=None,
                                         y=y,
                                         val_train_split=splits[-1],
                                         random_state=seed,
                                         stratify=stratify)
            
        else:
            data_splits = trainvaltest_split(x=None,
                                             y=y,
                                             splits=splits,
                                             random_state=seed,
                                             stratify=stratify)
        dataset_splits={}
        for split, (split_idx, split_y) in data_splits.items():
            print(split, len(split_idx))
            dataset_splits[split] = data.select_subset_from_indices(indices=split_idx,
                                                                    x_col = 'path',
                                                                    y_col = "family")
        
        label_encoder = LabelEncoder() # class2idx)
        label_encoder.fit(dataset_splits["train"].targets)
        
        for d in [*list(dataset_splits.values()), data]:
            d.label_encoder = label_encoder
#             d.config.num_classes = len(d.label_encoder)
#             d.config.num_samples = len(d)        
        
#         if plot_distributions:
#             cls.plot_trainvaltest_splits(train_data,
#                                          val_data,
#                                          test_data)

        log.debug(f"[RUNNING] [create_trainvaltest_splits()] splits={splits}")
        return dataset_splits
    
    @staticmethod
    def plot_trainvaltest_splits(train_data,
                                 val_data,
                                 test_data):
    
        fig, ax = plot_trainvaltest_splits(train_data,
                                           val_data,
                                           test_data)
        return fig, ax

    
    
TorchFiles = torchdata.datasets.Files
    
class CommonDataSelect(torchdata.datasets.Files):
    
    def __init__(self,
                 name: str=None,
                 files: List[Path]=None,
                 subset_key: str=None,
#                  class2idx: Optional[Dict[str,int]]=None,
                 **kwargs):
        super().__init__(files=files, **kwargs)
        if name:
            self.name = name
        if subset_key:
            self.subset_key = subset_key
#         if class2idx or not hasattr(self, "label_encoder"):
#             self.label_encoder = LabelEncoder(class2idx=class2idx)
    
    
    @classmethod
    def select_dataset_by_name(cls, name: str, config: DictConfig=None) -> "ImageDataset":
        config = config or DictConfig({})
        if not hasattr(cls, 'available_datasets'):
            raise ValueError("Must provide available dataset")
        dataset_dirs = cls.available_datasets[name]
        if isinstance(dataset_dirs, str):
            dataset_dirs = [dataset_dirs]
        assert isinstance(dataset_dirs, list)
        
        files = [cls.locate_files(dataset_dir) for dataset_dir in dataset_dirs]
        file_splits = {k:[] for k in files[0].keys()}

        for k in file_splits.keys():
            file_splits[k] = list(flatten(file[k] for file in files))

        log.info(f"[SELECT DATASET] (name={name}, num_files={len(file_splits[k])}), \ndataset_dirs=\n    " + '\n    '.join(dataset_dirs))

        config.dataset_dirs = dataset_dirs
        config.subset_dirs = [k for k in file_splits.keys() if k!="all"]
        config.name = name
        
        data = {}
        for k in file_splits.keys():
            data[k] = cls(config=config,
                       name=name,
                       subset_key=k if k!="all" else None,
                       files=file_splits[k])

        return data




    def select_subset_from_indices(self,
                                   indices: Sequence,
#                                    update_class2idx: bool=False,
                                   x_col = 'path',
                                   y_col = "family",
                                   subset_key: str = None,
                                   in_place: bool=False) -> Optional["FossilDataset"]:
        """
        Helper method to create a new FossilDataset containing only samples contained in `indices`
        Useful for producing train/val/test splits

        """
        config = self.config or DictConfig({})
        
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
            self.__init__(config = self.config,
                          files=files,
                          subset_key=subset_key)
#                           class2idx=class2idx)
            return None

        return type(self)(config = self.config,
                          files=files,
                          subset_key=subset_key)
#                           class2idx=class2idx)


    def filter_samples_by_threshold(self,
                                    threshold: int=1,
#                                     update_class2idx: bool=True,
                                    x_col = 'path',
                                    y_col = "family",
                                    subset_key: str = None,
                                    in_place: bool=False) -> "FossilDataset":
        
        config = self.config or DictConfig({})
        counts = collections.Counter(self.targets)
        if min(counts.values()) >= threshold:
            # Only filter if any classes are still below the threshold
            if in_place:
                return None
            else:
                return self

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
        
        self.config.threshold = threshold
        
        if in_place:
            self.__init__(config = self.config,
                          files=files,
                          subset_key=subset_key)
#                           class2idx=class2idx)
            return None

        return type(self)(config = self.config,
                          files=files,
                          subset_key=subset_key)







class CommonDataArithmetic: # (CommonDataset):
    
    
    @property
    def samples_df(self):        
        data_df = pd.DataFrame(self.samples)
        data_df = data_df.convert_dtypes()
        return data_df
    
    @classmethod
    def get_files_from_samples(cls,
                               samples: Union[pd.DataFrame, List],
                               x_col: Optional[str]="path"):
        if isinstance(samples, pd.DataFrame):
            if x_col in samples.columns:
                files = list(samples[x_col].values)
            else:
                files = list(samples.iloc[:,0].values)
        elif isinstance(samples, list):
            files = [s[0] for s in self.samples]
            
        return files
    
    def intersection(self, other):
        samples_df = self.samples_df
        other_df = other.samples_df
        
        intersection = samples_df.merge(other_df, how='inner', on=self.id_col)
        return intersection

    def __sub__(self, other):
    
        intersection = self.intersection(other)
        samples_df = self.samples_df
        
        remainder = samples_df[samples_df[self.id_col].apply(lambda x: x not in intersection[self.id_col])]
        
        return remainder







    
#     splits_on_disk : Tuple[str] = tuple()
#     transform = None
#     target_transform = None
#     totensor: Callable = torchvision.transforms.ToTensor()
#     toPIL: Callable = torchvision.transforms.ToPILImage("RGB")

############################################################
############################################################


class CommonDataset(CommonDataArithmetic, CommonDataSelect, CommonDataSplitter):
    """
    Meant to be a general custom class for handling the most common standard data formats.
    
    Specific datasets with any variations in foramat requirements should subclass and override as necessary.
    
    """
    available_datasets: Dict[str,Path] = {}
    transform = None
    target_transform = None
#     class_type: str="family"        
    SampleSchema: collections.namedtuple = collections.namedtuple("SampleSchema", ["path", "family", "genus", "species", "collection", "catalog_number"])


    def __init__(self,
                 config: DictConfig,
                 files: Union[Dict[str, List[Path]], List[Path]]=None,
                 return_signature: List[str] = ["image","target","path"],
                 subset_key: str=None,
                 *args, **kwargs):
        """
        If name is specified, files arg is ignored. Otherwise, creates torchdata.datasets.Files instance from files
        
        """
        self.root_config = self.parse_config(config)        
        self.config = self.root_config.dataset.config

#         pp(OmegaConf.to_container(self.config, resolve=True))
        
        self.return_signature = collections.namedtuple("return_signature", return_signature)
        
        
        if not isinstance(subset_key, str):
            subset_key = "all"            
        
        if files is None:
            assert isinstance(self.name, str), f"name must be a string if files is None."
            files = self.select_dataset_by_name(name=self.name,
                                                config=self.config)
#             assert len(files) > 0
        if isinstance(files, dict):
            for subset, data in files.items():
                if subset != "all":
                    log.debug(f'Adding {subset}_dataset')
                    setattr(self, f"{subset}_dataset", data)
            files = files[subset_key]
        if hasattr(files, "files"):
            files = files.files
        super().__init__(files=files, subset_key=subset_key, *args, **kwargs)
                
        self.samples = [self.parse_sample(idx) for idx in range((len(self)))]
        self.targets = [sample[1] for sample in self.samples]
        
        self.init_params = {
                 "config": self.root_config,
                 "files": self.files,
                 "return_signature": self.return_signature,
                 "subset_key": subset_key,
                 "args":args,
                 "kwargs":kwargs}
        
        
    @classmethod
    def default_config(cls) -> DictConfig:
        config = DictConfig({"dataset":
                                        {"config":{
                                                 "name": "",
                                                 "val_train_split": None,
                                                 "test_split": None,
                                                 "threshold": 0,
                                                 "seed": 956378485,
                                                 "class_type": "family",
                                                 "x_col": "path",
                                                 "y_col": "${.class_type}",
                                                 "id_col": "catalog_number",
                                                 "path_schema": "{family}_{genus}_{species}_{collection}_{catalog_number}",
                                                 "num_classes": None,
                                                 "num_samples": None,
                                                 "dataset_dirs": None,
                                                 "return_signature":["image","target","path"],
                                                 "subset_key":None
                                                }
                                        }
                        })
        return config

        
    @classmethod
    def from_dataframe(cls, 
                       sample_df: pd.DataFrame,
                       config: Optional[DictConfig] = None,
                       return_signature: Optional[List[str]] = ["image","target","path"],
                       subset_key: Optional[str]=None,
                       *args, **kwargs):
        
        config = config or cls.default_config()
        if "dataset" not in config:
            if "config" in config:                
                config = DictConfig({"dataset":config})
            else:
                config = DictConfig({"dataset":{
                                            "config":config
                                               }})


        pp(dir(config.dataset))
        pp(dir(config.dataset.config))
        files = cls.get_files_from_samples(samples=sample_df,
                                           x_col=config.dataset.config.x_col)
        config.dataset.config.num_samples = len(files)
        init_params = {
                 "config": config,
                 "files": files,
                 "return_signature": return_signature,
                 "subset_key": subset_key,
                 "args":args,
                 "kwargs":kwargs}
        return cls(**init_params)

        

    @classmethod
    def from_dataset(cls, other):
        init_params = other.init_params
        return cls(**init_params)

    @property
    def num_samples(self) -> int:
        return len(self.files)
    
    @property
    def classes(self):
        return self.label_encoder.classes

    @property
    def num_classes(self):
        return len(self.classes)
    
    
    def setup(self, 
              stage: str=None,
              train_transform: Optional[Callable] = None,
              eval_transform: Optional[Callable] = None,
              target_transform: Optional[Callable] = None
              ):
            
        self.filter_samples_by_threshold(threshold=self.threshold,
                                         x_col = self.x_col,
                                         y_col = self.y_col,
                                         in_place=True)

        config = self.config #.dataset.config
        if (config.test_split is None) and (config.val_train_split is None or config.val_train_split==0.0):
            dataset_splits = {"train": self}
        else:
            dataset_splits = self.create_trainvaltest_splits(data=self,
                                                             test_split=config.test_split,
                                                             val_train_split=config.val_train_split,
                                                             shuffle=True,
                                                             seed=config.seed,
                                                             plot_distributions=False)
        
        for split in dataset_splits.keys():
            if split == "train":
                dataset_splits[split].transform = train_transform
            elif split in ["val", "test"]:
                dataset_splits[split].transform = eval_transform
#             elif split == "predict":
        
        data = {}        
        for split in ["train", "val", "test"]:
            if stage == split or stage is None:
                if split in dataset_splits.keys():
                    data[split] = dataset_splits[split]
                elif hasattr(self, f"{split}_dataset"):
                    data[split] = getattr(self, f"{split}_dataset")
        
        label_encoder = dataset_splits["train"].label_encoder
        for k, d in data.items():
            d.label_encoder = label_encoder
            d.config.subset_key = k
            d.config.num_classes = len(d.label_encoder)
            d.config.num_samples = len(d)
        
        if stage == "predict": # or stage is None:
            data["predict"] = data["test"] # self.predict_on_split]

        if stage is None: stage = "all"
        log.debug(f"[RUNNING] [{self}.setup(stage={stage}) with {len(data)} splits: {data.keys()}")
        
        if len(data)==1:
            if isinstance(data, dict):
                data = list(data.values())[0]
            elif isinstance(data, list):
                data = data[0]
        return data
        
        
    def parse_config(self,
                     config: DictConfig) -> DictConfig:
#         config = deepcopy(config)
        base_config = config
        
        if "dataset" in config:
            config = config.dataset
        if "config" in config:
            config = config.config
            
#         pp(OmegaConf.to_container(config, resolve=True))
        
        if "name" in config:
            self.name = config.name
        if "threshold" in config:
            self.threshold = config.threshold
            
        if "dataset_dirs" in config:
            self.dataset_dirs = config.dataset_dirs
        if "trainvaltest_split" in config:
            self.trainvaltest_split = config.trainvaltest_split
        if "test_split" in config:
            self.test_split = config.test_split
        if "val_split" in config:
            self.val_split = config.val_split
        if "seed" in config:
            self.seed = config.seed
        if "subset_key" in config:
            self.subset_key = config.subset_key
        if hasattr(self, "label_encoder") :
            config.classes = self.classes
            config.num_classes = self.num_classes
        self.x_col = config.x_col
        self.y_col = config.y_col
        self.id_col = config.id_col
        
        
        if "return_signature" in config:
            self.return_signature = collections.namedtuple("return_signature", config.return_signature)
        
        self.path_schema = PathSchema(path_schema=config.path_schema)
        
#         config = DictConfig({"dataset":{
#                                     "config":base_config
#                                        }
#                             })
        if "dataset" not in config:
            if "config" in config:
                
                config = DictConfig({"dataset":config})
            else:
                config = DictConfig({"dataset":{
                                            "config":config
                                               }
                                    })
#         elif "config" not in config.get("dataset", ""):
#             config = DictConfig({"config":config})
        return config
    
        
    def parse_sample(self, index: int):
        path = self.files[index]        
        family, genus, species, collection, catalog_number = self.path_schema.parse(path)

        return self.SampleSchema(path=path,
                                 family=family,
                                 genus=genus,
                                 species=species,
                                 collection=collection,
                                 catalog_number=catalog_number)
    
    def fetch_item(self, index: Union[int, Sequence]) -> Tuple[str]:
        sample = self.samples[index]
        image = Image.open(sample.path)
        return self.return_signature(image=image, target=sample.family, path=sample.path)

        
    def __getitem__(self, index: int):
        
        item = self.fetch_item(index)
        image, target, path = item.image, item.target, str(item.path)
        
        target = self.label_encoder.class2idx[target]
        
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

#         if not self.image_return_type == "tensor":
#             image = self.toPIL(image)
        image = totensor(image)
            
        return tuple(self.return_signature(image=image, target=target, path=path))

    
    def __repr__(self):
        disp = f"""<{str(type(self)).strip("'>").split('.')[1]}>:"""
        
        disp += "\nFields:\n"
        for k in self.config.keys():
            disp += f"    {k}: {getattr(self.config,k)}\n"
        return disp
    
                
    def display_grid(self,
                   indices: Union[int,List[int]]=64,
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
        
        if not hasattr(self, "image_return_type"):
            self.image_return_type = "tensor" if isinstance(self[0][0], torch.Tensor) else "PIL.Image"
        if self.image_return_type == 'tensor':
#         if self.transform is not None:
            images = [toPIL(self[idx][0]) for idx in indices]
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


    def slice(self, indices: Union[List, np.array]) -> torch.Tensor:
        indices = np.array(indices)
        batch = [self[idx] for idx in indices]
        batch = (torch.stack([item[0] for item in batch]),
                 torch.Tensor(np.array([(item[1]) for item in batch])).to(int))
        return batch
        
    def show_batch(self,
                   indices: Union[int,List[int]]=64,
                   repeat_n: Optional[int]=1,
                   max_images: int=64,
                   cmap: str='cividis',
                   grayscale=True,
                   titlesize=30,
                   include_colorbar: bool=True,
                   suptitle: str="default"):
        """
        Useful utility function for plotting a single batch of images as a single grid.
        
        Good mild grayscale cmaps: ['magma', 'cividis']
        Good mild grayscale cmaps: ['plasma', 'viridis']
        """
        if (int(repeat_n) == 0) or (repeat_n is None):
            repeat_n=1
        if isinstance(indices, int):
            indices = random.sample(range(self.num_samples), indices)

        indices = collapse((repeat(i,repeat_n) for i in indices))
        indices = list(indices)


        batch = self.slice(indices)
        x = batch[0]
        y = [self.targets[idx] for idx in indices]
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
        if include_colorbar:
            colorbar(img_ax)

        if suptitle is not None:
            if suptitle == 'default':
                suptitle = f"{max_samples} random samples"
                if self.config.get("subset_key") is not None:
                    suptitle += f"from subset: {self.config.get('subset_key')}"
            plt.suptitle(suptitle, fontsize=titlesize)
                
        plt.subplots_adjust(left=None, bottom=0.0, right=None, top=0.96, wspace=None, hspace=None)
        plt.axis("off")
        plt.axis("tight")
        plt.tight_layout()
        return fig, img_ax

##############################################################

#         self.dataset_config = OmegaConf.create({"name": "Fossil_512",
#                                                 "val_split": 0.2,
#                                                 "test_split": 0.3,
#                                                 "threshold": 3,
#                                                 "seed": 987485,
#                                                 "class_type": "family",
#                                                 "x_col":"path",
#                                                 "y_col":"${.class_type}",
#                                                 "id_col":"catalog_number"
# })
        
#         datamodule_config = OmegaConf.create({"batch_size":128,
#                                                    "normalize":True,
#                                                    "image_size": 512,
#                                                    "grayscale" :True,
#                                                     "channels":3,
#                                                     "predict_on_split":"test",
#                                                     "num_workers":0,
#                                                     "pin_memory":True,
#                                                     "seed":9877,
#                                                     "debug":False,
#                                                     "augment":True,
#                                                     "shuffle":True})

##############################################################



class LoggingDataModule(pl.LightningDataModule):
    
    @classmethod
    def default_config(cls) -> DictConfig:
        dataset_config = CommonDataset.default_config()
        config = DictConfig({
            "datamodule":{
                        "config":{
                                    "name": '${.dataset.name}',
                                    "batch_size": 128,
                                    "normalize": True,
                                    "image_size": None,
                                    "grayscale": True,
                                    "channels": 3,
                                    "num_workers": 0,
                                    "pin_memory": True,
                                    "drop_last": False,
                                    "seed": 9877,
                                    "augment": True,
                                    "shuffle": True,
                                    "num_classes": '${.dataset.num_classes}',
                                    "predict_on_split": "test",
                                    "debug": True,
                                    "dataset": dataset_config
                                }
                        }
                            })
        return config
    
    
    def _setup_preconfigured_splits(self,
                                    data_splits: Dict[str, CommonDataset],
                                    train_transform: Optional[Callable] = None,
                                    eval_transform: Optional[Callable] = None,
                                    target_transform: Optional[Callable] = None
                                    ):
        label_encoder = data_splits["train"].label_encoder
        transforms = self.get_default_transforms()
        
        for k, d in data_splits.items():
            d.label_encoder = label_encoder
            d.config.subset_key = k
            d.config.num_classes = len(d.label_encoder)
            d.config.num_samples = len(d)
            
            if "train" in k:
                d.transform = train_transform or transforms[0]
            else:
                d.transform = eval_transform or transforms[1]
            d.target_transform = target_transform
            
        log.info(f"[RUNNING] datamodule._setup_preconfigured_splits({data_splits.keys()})")
        return data_splits

    
    @staticmethod
    def validate_data_dir(data_dir: str) -> bool:
        
        if os.path.exists(data_dir) and (len(os.listdir(data_dir))>0):
            return True
        else:
            return False
        
    
    def import_dataset_from_csv(self,
                                data_dir: str,
                                user_config: Optional[DictConfig]=None) -> None: #Dict[str, CommonDataset]:
        """
        data_dir -> Directory that contains 1 or more csv files representing a table of samples, where each row = 1 sample
        
        Returns:
            Dict[subset_keys: str, CommonDataset]
        
        """
        if not self.validate_data_dir(data_dir=data_dir):
            raise ValueError(f"data_dir: {data_dir} is invalid")
        
        data_splits, config = import_dataset_from_csv(data_dir)
        
        default_config = self.default_config()
        if config is None:
            config = default_config
        else:
            config = OmegaConf.merge(default_config, config)
        if user_config is not None:
            config = OmegaConf.merge(config, user_config)
            
        self.dataset_config = config.dataset
        self.datamodule_config = self.parse_config(config)
        
        self.data_splits = self._setup_preconfigured_splits(data_splits)
        
        for subset, data in self.data_splits.items():
            if "train" in subset:
                self.train_dataset = data
                self.dataset = data
#                 self.dataset.config = deepcopy(OmegaConf.to_container(data.config, resolve=True))
#                 self.dataset.config.subset_key = None
            elif "val" in subset:
                self.val_dataset = data
            if "test" in subset:
                self.test_dataset = data

        self.dataset.label_encoder = self.train_dataset.label_encoder
        
        
    
    def export_dataset_to_csv(self,
                               output_dir: str) -> Dict[str,str]:
        
#         output_dir = "/media/data/jacob/GitHub/prj_fossils_contrastive/notebooks/Extant_family_10_512_minus_PNAS_family_100_512"

        data_paths = export_dataset_to_csv(data_splits=self.data_splits,
                                           label_encoder = self.dataset.label_encoder,
                                           output_dir=output_dir)
        return data_paths










class LeavesLightningDataModule(LoggingDataModule): #pl.LightningDataModule):
    
#     worker_init_fn=seed_worker
    
    image_size = 224
#     target_size = (224, 224)
    channels = 3
    image_buffer_size = 32
    mean = [0.5, 0.5, 0.5]
    std = [1.0, 1.0, 1.0]
    
    DatasetConstructor: Callable = CommonDataset
    
    totensor: Callable = transforms.ToTensor()
    
    def __init__(self,
                 config: DictConfig,
                 data_dir: Optional[str]=None):
#                  name: str=None,
#                  batch_size: int=32,
#                  val_split: float=0.0,
#                  test_split: float=0.0,
#                  normalize: bool=True,
#                  image_size: Union[int,str] = None,
#                  grayscale: bool = True,
#                  channels: int=None,
#                  dataset_dir: Optional[str]=None,
#                  return_paths: bool=False,
#                  num_workers=0,
#                  pin_memory: bool=False,
#                  seed: int=None,
#                  debug: bool=False,
#                  predict_on_split: str="val",
#                  **kwargs
#                  ):
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
        try:
            self.import_dataset_from_csv(data_dir=data_dir,
                                         user_config=config)
            return
        except Exception as e:
            pass        
        self.data_splits = {}
        
        
        default_config = self.default_config()
        if config is None:
            config = default_config
        else:
            config = OmegaConf.merge(default_config, config)
        
#         import pdb;pdb.set_trace()
        
        

        config = self.parse_config(config)
        self.dataset_config = config.dataset
        self.datamodule_config = config

        self.dataset = self.DatasetConstructor(config=self.dataset_config)
        self.init_dataset_stage(stage=None)
        self.dataset.label_encoder = self.train_dataset.label_encoder
        
#         if image_size == 'auto':
#             try:
#                 image_size = int(self.name.split('_')[-1])
#             except ValueError:
#                 image_size = self.image_size
#         self._is_fit_setup=False
#         self._is_test_setup=False
        
        
    def parse_config(self,
                     config: DictConfig) -> DictConfig:

        dataset_config = None
        
        if "datamodule" in config:
            config = config.datamodule
        if "dataset" in config:
            print('setting dataset_config at top of datamodule.parse_config')
            dataset_config = config.dataset
            
        if "config" in config:
            config = config.config
            
        if "dataset" in config:
            dataset_config = config.dataset
#         pp(OmegaConf.to_container(config, resolve=True))
            
        if "batch_size" in config:
            self.batch_size = config.batch_size
        if "normalize" in config:
            self.normalize = config.normalize
        if "image_size" in config:
            if isinstance(config.image_size, int):
                self.image_size = config.image_size
            else:
#             if config.dataset.get('name') != "":
                try:
                    self.image_size = int(config.dataset.name.split("_")[-1])
                except:
                    self.image_size = None
#             self.image_size = config.image_size
        
        if "grayscale" in config:
            self.grayscale = config.grayscale
        if "channels" in config:
            self.channels = config.channels
#         if "image_size" in config:
#             self.image_size = config.image_size
            
        if "predict_on_split" in config:
            self.predict_on_split = config.predict_on_split
        if "num_workers" in config:
            self.num_workers = config.num_workers
        if "pin_memory" in config:
            self.pin_memory = config.pin_memory
        if "drop_last" in config:
            self.drop_last = config.drop_last

        if "seed" in config:
            self.seed = config.seed
        if "debug" in config:
            self.debug = config.debug
        if "augment" in config:
            self.augment = config.augment
        if "shuffle" in config:
            self.shuffle = config.shuffle
            
            
#         if "dataset" not in config:
#             if "config" in config:
                
#                 config = DictConfig({"dataset":config})
#             else:
#                 config = DictConfig({"dataset":{
#                                             "config":config
#                                                }
#                                     })
                
                
        config.dataset = dataset_config
            
            

        return config
        
        
        
    def setup(self,
              stage: str=None,
              train_transform: Optional[Callable] = None,
              eval_transform: Optional[Callable] = None,
              target_transform: Optional[Callable] = None
              ):
        

        log.info(f"[RUNNING] datamodule.setup({stage})")
        if stage == 'fit' or stage is None:
            self.train_dataset = self.dataset.setup(stage="train",
                                                    train_transform=train_transform)
            self.val_dataset = self.dataset.setup(stage="val",
                                                  eval_transform=eval_transform)
            self.data_splits.update({"train":self.train_dataset,
                                     "val":self.val_dataset})
        if stage == 'test' or stage is None:
            self.test_dataset = self.dataset.setup(stage="test",
                                                   eval_transform=eval_transform)
            self.data_splits.update({"test":self.test_dataset})
        if stage == 'predict' or stage is None:
            self.predict_dataset = self.dataset.setup(stage="test",
                                                      eval_transform=eval_transform)

    @property
    def classes(self):
        return self.dataset.label_encoder.classes

    @property
    def num_classes(self):
        return len(self.classes)
        
    def init_dataset_stage(self,
                           stage: str='fit',
                           train_transform: Optional[Callable] = None,
                           eval_transform: Optional[Callable] = None):
        
#         self.train_transform = train_transform or self.default_train_transforms(augment=self.augment,
#                                                                                 normalize=self.normalize,
#                                                                                 grayscale=self.grayscale,
#                                                                                 channels=self.channels)
#         self.eval_transform = eval_transform or self.default_eval_transforms(normalize=self.normalize,
#                                                                              resize_PIL=True,
#                                                                              grayscale=self.grayscale,
#                                                                              channels=self.channels)   
        default_transforms = self.get_default_transforms()
        self.train_transform = train_transform or default_transforms[0]
        self.eval_transform = eval_transform or default_transforms[1]
    
        self.setup(stage=stage,
                   train_transform=train_transform,
                   eval_transform=eval_transform,
                   target_transform=None)
        
        
        

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
                                  drop_last=self.drop_last)
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
            
    def default_train_transforms(self,
                                 normalize: bool=True, 
                                 augment:bool=True,
                                 grayscale: bool=True,
                                 channels: Optional[int]=3):
        """Subclasses can override this or user can provide custom transforms at runtime"""
        transform_list = []
#         transform_jit_list = []
        resize_PIL = not augment
        if augment:
            transform_list.extend([transforms.RandomResizedCrop(size=self.image_size,
                                                                scale=(0.25, 1.2),
                                                                ratio=(0.7, 1.3),
                                                                interpolation=2),
                                   totensor
                                 ])
        return self.default_eval_transforms(normalize=normalize,
                                            resize_PIL=resize_PIL,
                                            grayscale=grayscale,
                                            channels=channels,
                                            transform_list=transform_list)
    
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
                                   totensor])
        if normalize:
            transform_jit_list.append(self.normalize_transform)
            
        if grayscale:
            transform_jit_list.append(transforms.Grayscale(num_output_channels=channels))
            
        return transforms.Compose([*transform_list, *transform_jit_list])

    
    def get_default_transforms(self):
        train_transform = self.default_train_transforms(augment=self.augment,
                                                        normalize=self.normalize,
                                                        grayscale=self.grayscale,
                                                        channels=self.channels)
        eval_transform = self.default_eval_transforms(normalize=self.normalize,
                                                      resize_PIL=True,
                                                      grayscale=self.grayscale,
                                                      channels=self.channels)
        return train_transform, eval_transform
    
    
    
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
                   images: List[Image.Image],
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
        ax = plt.gca()
        for label in ax.xaxis.get_ticklabels():
            label.set_visible(False)
        for label in ax.yaxis.get_ticklabels():
            label.set_visible(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
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

    plt.subplots_adjust(left=None, bottom=0.05, right=None, top=0.9, wspace=0.0, hspace=0.2*rows) #wspace=0.05, hspace=0.1)



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
sns.set_style('darkgrid')
sns.set_palette('Set2')
# sns.set_style("whitegrid")

def compute_class_counts(targets: Sequence,
                         sort_by: Optional[Union[str, bool, Sequence]]="count"
                        ) -> Dict[str, int]:
    
    counts = collections.Counter(targets)
#     if hasattr(sort_by, "__len__"):
    if isinstance(sort_by, dict):
        counts = {k: counts[k] for k in sort_by.keys()}
    if isinstance(sort_by, list):
        counts = {k: counts[k] for k in sort_by}
    elif (sort_by == "count"):
        counts = dict(sorted(counts.items(), key = lambda x:x[1], reverse=True))
    elif (sort_by is True):
        counts = dict(sorted(counts.items(), key = lambda x:x[0], reverse=True))
        
    return counts

def plot_class_distributions(targets: List[Any], 
                             sort_by: Optional[Union[str, bool, Sequence]]="count",
                             ax=None,
                             xticklabels: bool=True):
    """
    Example:
        counts = plot_class_distributions(targets=data.targets, sort=True)
    """
    
    counts = compute_class_counts(targets,
                                  sort_by=sort_by)
                        
    keys = list(counts.keys())
    values = list(counts.values())

    if ax is None:
        plt.figure(figsize=(20,12))
    ax = sns.histplot(x=keys, weights=values, discrete=True, ax=ax)
    plt.sca(ax)
    if xticklabels:
        xtick_fontsize = "medium"
        if len(keys) > 100:
            xtick_fontsize = "x-small"
        elif len(keys) > 75:
            xtick_fontsize = "small"
        plt.xticks(
            rotation=90, #45, 
            horizontalalignment='right',
            fontweight='light',
            fontsize=xtick_fontsize
        )
        if len(keys) > 100:
            for label in ax.xaxis.get_ticklabels()[::2]:
                label.set_visible(False)
        
    else:
        ax.set_xticklabels([])
    
    return counts


def plot_split_distributions(data_splits: Dict[str, CommonDataset]):
    """
    Create 3 vertically-stacked count plots of train, val, and test dataset class label distributions
    """
    assert isinstance(data_splits, dict)
    num_splits = len(data_splits)
    
    if num_splits < 4:
        rows = num_splits
        cols = 1
    else:
        rows = int(num_splits // 2)
        cols = int(num_splits % 2)
    fig, ax = plt.subplots(rows, cols, figsize=(20*cols,10*rows))
    ax = ax.flatten()
    
    
    train_key = [k for k,v in data_splits.items() if "train" in k]
    sort_by = True
    if len(train_key)==1:
        sort_by = compute_class_counts(data_splits[train_key[0]].targets,
                                       sort_by="count")
        log.info(f'Sorting by count for {train_key} subset, and applying order to all other subsets')
#         log.info(f"len(sort_by)={len(sort_by)}")

    num_classes = len(set(list(data_splits.values())[0].targets))    
    xticklabels=False
    num_samples = 0
    counts = {}
    for i, (k, v) in enumerate(data_splits.items()):
        if i == len(data_splits)-1:
            xticklabels=True
        counts[k] = plot_class_distributions(targets=v.targets, 
                                             sort_by=sort_by,
                                             ax = ax[i],
                                             xticklabels=xticklabels)
        num_nonzero_classes = len([name for name, count_i in counts[k].items() if count_i > 0])
        
        title = f"{k} [n={len(v)}"
        if num_nonzero_classes < num_classes:
            title += f", num_classes@(count > 0) = {num_nonzero_classes}-out-of-{num_classes} classes in dataset"
        title += "]"
        plt.gca().set_title(title, fontsize='large')
        
        num_samples += len(v)
    

    
    suptitle = '-'.join(list(data_splits.keys())) + f"_splits (total samples={num_samples}, total classes = {num_classes})"
    
    plt.suptitle(suptitle, fontsize='x-large')
    plt.subplots_adjust(bottom=0.1, top=0.94, wspace=None, hspace=0.08)
    
    return fig, ax






def plot_trainvaltest_splits(train_data,
                             val_data,
                             test_data):
    """
    Create 3 vertically-stacked count plots of train, val, and test dataset class label distributions
    """
    fig, ax = plt.subplots(3, 1, figsize=(16,8*3))

    train_counts = plot_class_distributions(targets=train_data.targets, sort_by=True, ax = ax[0], xticklabels=False)
    plt.gca().set_title(f"train (n={len(train_data)})", fontsize='large')
    sort_classes = train_counts.keys()

    val_counts = plot_class_distributions(targets=val_data.targets, ax = ax[1], sort_by=sort_classes, xticklabels=False)
    plt.gca().set_title(f"val (n={len(val_data)})", fontsize='large')
    test_counts = plot_class_distributions(targets=test_data.targets, ax = ax[2], sort_by=sort_classes)
    plt.gca().set_title(f"test (n={len(test_data)})", fontsize='large')

    num_samples = len(train_data) + len(val_data) + len(test_data)
    
    plt.suptitle(f"Train-Val-Test_splits (total={num_samples})", fontsize='x-large')

    plt.subplots_adjust(bottom=0.1, top=0.95, wspace=None, hspace=0.07)
    
    return fig, ax



#######################################################


    
    
    
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


    

"save_config", "load_config", "export_image_data_diagnostics", "export_dataset_to_csv", "import_dataset_from_csv"







def save_config(config: DictConfig, config_path: str):
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(config, resolve=True))

def load_config(config_path: str) -> DictConfig:    
    with open(config_path, "r") as f:
        loaded = OmegaConf.load(f)
    return loaded


def export_image_data_diagnostics(data_splits: Dict[str,CommonDataset],
                                  output_dir: str='.',
                                  max_samples: int = 64,
                                  export_sample_images: bool=True,
                                  export_class_distribution_plots: bool=True) -> Dict[str,str]:
    image_paths = {"images": {},
                   "class_distribution_plots":{}}
    
    image_dir = os.path.join(output_dir, "images")
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(image_dir, exist_ok = True)
    os.makedirs(plot_dir, exist_ok = True)

    if export_sample_images:
#         subsets = ['train', 'val', 'test']
        for subset in data_splits.keys():
            fig, ax = data_splits[subset].show_batch(indices=max_samples, include_colorbar=False,
                                                     suptitle = f"subset: {subset}, {max_samples} random images")
            img_path = os.path.join(image_dir, f"subset: {subset}, {max_samples} random images.jpg")
            image_paths["images"][subset] = img_path
            plt.savefig(img_path)

    if export_class_distribution_plots:
        fig, ax = plot_split_distributions(data_splits=data_splits)
        class_distribution_plot_path = os.path.join(plot_dir, f"class_distribution_plots_{[subset for subset in data_splits.keys()]}")
        image_paths["class_distribution_plots"]["all"] = class_distribution_plot_path
        plt.savefig(class_distribution_plot_path)

    return image_paths



def export_dataset_to_csv(data_splits: Dict[str,CommonDataset],
                          label_encoder: Optional[LabelEncoder]=None,
                          datamodule_config: Optional[DictConfig]=None,
                          output_dir: str='.',
                          export_sample_images: bool=True,
                          export_class_distribution_plots: bool=True) -> Dict[str,str]:
    output_paths = {"tables":{},
                    "class_labels":{},
                    "configs":{}}
    os.makedirs(output_dir, exist_ok=True)
    for k, data in data_splits.items():
        subset_data_path = os.path.join(output_dir, f"{k}_data_table.csv")
        data.samples_df.to_csv(subset_data_path)
        output_paths["tables"][k] = subset_data_path
        
        if hasattr(data, "config"):
            subset_config_path = os.path.join(output_dir, f"{k}_dataset_config.yaml")
            save_config(config=data.config, config_path=subset_config_path)
            output_paths["configs"][k] = subset_config_path
        
        if hasattr(data, 'label_encoder') and (label_encoder is None):
            subset_label_path = os.path.join(output_dir, k + "_label_encoder.json")
            data.label_encoder.save(subset_label_path)
            output_paths["class_labels"][k] = subset_data_path
            
    if label_encoder is not None:
        full_label_encoder_path = os.path.join(output_dir, "label_encoder.json")
        label_encoder.save(full_label_encoder_path)
        output_paths["class_labels"]["full"] = full_label_encoder_path

    
    export_image_data_diagnostics(data_splits=data_splits,
                                  output_dir=output_dir,
                                  max_samples = 64,
                                  export_sample_images=export_sample_images,
                                  export_class_distribution_plots=export_class_distribution_plots)
    
    if isinstance(datamodule_config, DictConfig):
        datamodule_config_dir = os.path.join(output_dir, "datamodule") 
        os.makedirs(datamodule_config_dir, exist_ok=True)
        datamodule_config_path = os.path.join(datamodule_config_dir, f"datamodule_config.yaml")
        save_config(config=datamodule_config, config_path=datamodule_config_path)
        output_paths["configs"]["datamodule"] = datamodule_config_path
    
    
    return output_paths
    

def import_dataset_from_csv(data_catalog_dir: str) -> Tuple[Dict[str, CommonDataset], DictConfig]:
    
    data_paths = list(Path(data_catalog_dir).glob("*.csv"))
    config_paths = list(Path(data_catalog_dir).glob("*.yaml"))
    label_encoder_paths = list(Path(data_catalog_dir).glob("*.json"))
    assert len(data_paths) == len(config_paths)
    
    datamodule_config_path = list(Path(data_catalog_dir, "datamodule").glob("*.yaml"))
    input_paths = {"tables":{},
                   "class_labels":{},
                   "configs":{}}
    subsets = ["train", "val", "test"]
    for subset in subsets:
        input_paths["tables"][subset] = [p for p in data_paths if p.stem.startswith(subset)][0]
        input_paths["configs"][subset] = [p for p in config_paths if p.stem.startswith(subset)][0]
    
    if len(label_encoder_paths) == 1:
        label_encoder = LabelEncoder.load(label_encoder_paths[0])
    else:
        raise(f'Currently cannot distinguish between multiple label_encoders, please delete all but 1 in experiment directory. Contents: {label_encoder_paths}')
    
    data_splits = {}
    for subset in subsets:
        sample_df = pd.read_csv(input_paths["tables"][subset])
        config = load_config(input_paths["configs"][subset])
        data_splits[subset] = CommonDataset.from_dataframe(sample_df,
                                                           config=config)
        data_splits[subset].label_encoder = label_encoder
    
    datamodule_config = None
    if len(datamodule_config_path):
        datamodule_config_path = datamodule_config_path[0]
        datamodule_config = load_config(datamodule_config_path)
        
    return data_splits, datamodule_config
