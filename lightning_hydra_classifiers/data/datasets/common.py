"""
lightning_hydra_classifiers.data.datasets.common


Common classes & functions for simplifying the boilerplate code in definitions of custom datasets in this repo.

Created on: Sunday April 25th, 2021
Updated on: Tuesday July 13th, 2021
    - Refactoring common datasets to have torchdata.datasets.Files as superclass by default, instead of torchvision ImageFolders. For greater flexibility.
Updated on: Thursday Sept 2nd, 2021
    - Switching to CSVDatasets objects originally defined in make_catalogs.py






"""
import collections
from collections import namedtuple
import logging
import os
import random
import textwrap
from copy import deepcopy
from dataclasses import dataclass, asdict
from itertools import chain, repeat
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchdata
import torchvision
import wandb
from more_itertools import collapse, flatten
from omegaconf import DictConfig, OmegaConf
# from PIL.Image import Image
from PIL import Image, ImageOps
from rich import print as pp
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder, folder, vision
from torchvision.transforms import functional as F

# from lightning_hydra_classifiers.utils.dataset_management_utils import ETL as ETLBase
from lightning_hydra_classifiers.utils.etl_utils import ETL as ETLBase
from lightning_hydra_classifiers.utils import template_utils
from lightning_hydra_classifiers.utils.common_utils import filter_df_by_threshold, Batch
from lightning_hydra_classifiers.data import catalog_registry
from lightning_hydra_classifiers.utils.dataset_management_utils import (LabelEncoder,
                                                                        DataSplitter,
                                                                        export_image_data_diagnostics,
                                                                        export_dataset_to_csv,
                                                                        import_dataset_from_csv)
from lightning_hydra_classifiers.utils.plot_utils import colorbar, display_images

# from sklearn.model_selection import train_test_split


log = template_utils.get_logger(__name__)

__all__ = ['PathSchema', 'SampleSchema', 'Batch', 'totensor', 'toPIL', 'CSVDatasetConfig', 'CSVDataset', "ETL"]


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
    
    def split(self, sep):
        return self.schema_parts


@dataclass
class SampleSchema:
    path : Union[str, Path] = None
    family : str = None
    genus : str = None
    species : str = None
    collection : str = None
    catalog_number : str = None

    @classmethod
    def keys(cls):
        return list(cls.__dataclass_fields__.keys())
        
    def __getitem__(self, index: int):
        return getattr(self, self.keys()[index])



# Batch = namedtuple("Batch", ("image", "target", "path", "catalog_number"))
totensor: Callable = torchvision.transforms.ToTensor()
toPIL: Callable = torchvision.transforms.ToPILImage("RGB")



class ETL(ETLBase):
    
    @classmethod
    def export_dataset_state(cls,
                             output_dir: Union[str, Path],
                             df: pd.DataFrame=None,
                             encoder: LabelEncoder=None,
                             dataset_name: Optional[str]="dataset",
                             config: "CSVDatasetConfig"=None
                             ) -> None:
        
        paths = {ftype: str(output_dir / str(dataset_name + ext)) for ftype, ext in cls.data_file_ext_maps.items()}
        
        output_dir = Path(output_dir)
        if isinstance(df, pd.DataFrame):
            cls.df2csv(df = df,
                       path = paths["df"])
            if config:
                config.data_path = paths["df"]
        if isinstance(encoder, LabelEncoder):
            cls.labels2json(encoder=encoder,
                            path = paths["encoder"])
            if config:
                config.label_encoder_path = paths["encoder"]
        if isinstance(config, CSVDatasetConfig):
            config.save(path = paths["config"])
#             cls.config2yaml(config=config,
#                             path = paths["config"])
            
            
    @classmethod
    def import_dataset_state(cls,
                             data_dir: Optional[Union[str, Path]]=None,
                             config_path: Optional[Union[Path, str]]=None,
                            ) -> Tuple["CSVDataset", "CSVDatasetConfig"]:
        if (not os.path.exists(str(data_dir))) and (not os.path.exists(config_path)):
            raise ValueError("Either data_dir or config_path must be existing paths")
        
        if os.path.isdir(str(data_dir)):
            data_dir = Path(data_dir)
        paths = {}
        
        # import config yaml file
        if os.path.isfile(str(config_path)):
            paths['config'] = config_path
#             config = cls.config_from_yaml(path = paths["config"])
            config = CSVDatasetConfig.load(path = paths["config"])
            if hasattr(config, "data_path"):
                paths["df"] = str(config.data_path)
            if hasattr(config, "label_encoder_path"):
                paths["encoder"] = str(config.label_encoder_path)
            data_dir = Path(os.path.dirname(config_path))
            
        for ftype, ext in cls.data_file_ext_maps.items():
            if ftype not in paths:
                paths[ftype] = str(list(data_dir.glob("*" + ext))[0])
                
                
        config.data_path = str(paths["df"])
        config.label_encoder_path = str(paths["encoder"])
        if os.path.isfile(paths["encoder"]):
            # import label encodings json file if it exists
            label_encoder = cls.labels_from_json(path = paths["encoder"])
            
        # import dataset samples from a csv file as a CustomDataset/CSVDataset object
        dataset = CSVDataset.from_config(config,
                                         eager_encode_targets=True)
        dataset.setup(samples_df=dataset.samples_df,
                      label_encoder=label_encoder,
                      fit_targets=True)
        
        return dataset, config
                    
            





@dataclass
class BaseConfig:

    def save(self,
             path: Union[str, Path]) -> None:
        
        cfg = asdict(self)
#         cfg = DictConfig({k: getattr(self,k) for k in self.keys()})
        ETL.config2yaml(cfg, path)
    
    @classmethod
    def load(cls,
             path: Union[str, Path]) -> "DatasetConfig":

        cfg = ETL.config_from_yaml(path)

#         keys = cls.__dataclass_fields__.keys()
        cfg = cls(**{k: cfg[k] for k in cls.keys()})
        return cfg
    
    @classmethod
    def keys(cls):
        return cls.__dataclass_fields__.keys()
    
    def __repr__(self):
        out = f"{type(self)}" + "\n"
        out += "\n".join([f"{k}: {getattr(self, k)}" for k in self.keys()])
#         out += f"\nroot_dir: {self.root_dir}"
#         out += "\nsubset_dirs: \n\t" + '\n\t'.join(self.subset_dirs)
        return out

    
@dataclass
class DatasetConfig(BaseConfig):
    base_dataset_name: str = "" # "Extant_Leaves"
    class_type: str = "family"
    threshold: Optional[int] = 10
    resolution: int = 512
    version: str = "v1_0"
    path_schema: str = "{family}_{genus}_{species}_{collection}_{catalog_number}"
    
    def __post_init__(self):
        assert self.version in self.available_versions
    
    @property
    def available_versions(self) -> List[str]:
        return list(catalog_registry.available_datasets().versions.keys())

    @property
    def full_name(self) -> str:
        name = []
        if len(self.base_dataset_name):
            name.append(self.base_dataset_name)
        if self.threshold:
            name.extend([str(self.class_type), str(self.threshold)])
        name.append(str(self.resolution))
        return "_".join(name)
#         name  = self.base_dataset_name
#         if self.threshold:
#             name += f"_{self.class_type}_{self.threshold}"
#         name += f"_{self.resolution}"
#         return name

    
class ImageFileDatasetConfig(DatasetConfig):    
    @property
    def root_dir(self):
        return catalog_registry.available_datasets.get(self.full_name, version=self.version)
    
    def is_valid_subset(self, subset: str):
        for s in ("train", "val", "test"):
            if s in subset:
                return True
        return False
    
    @property
    def subsets(self):
        if isinstance(self.root_dir, list):
            return []
        return [s for s in os.listdir(self.root_dir) if self.is_valid_subset(s)]
    
    @property
    def subset_dirs(self):
        return [os.path.join(self.root_dir, subset) for subset in self.subsets]

    def locate_files(self) -> Dict[str, List[Path]]:
        return ETL.locate_files(self.root_dir)

    @cached_property
    def num_samples(self):
#         subset_dirs = {Path(subset_dir).stem: Path(subset_dir) for subset_dir in self.subset_dirs}
        files = {subset: f for subset, f in self.locate_files().items() if self.is_valid_subset(subset)}
        return {subset: len(list(f)) for subset, f in files.items()}
    
    def __repr__(self):
        out = super().__repr__()
        out += f"\nroot_dir: {self.root_dir}"
        out += "\nsubsets: "
        for i, subset in enumerate(self.subsets):
            out += '\n\t' + f"{subset}:"
            out += '\n\t\t' + f"subdir: {self.subset_dirs[i]}"
            out += '\n\t\t' + f"subset_num_samples: {self.num_samples[subset]}"
        return out


@dataclass
class CSVDatasetConfig(BaseConfig):
    """
    Represents a single data subset, or the set of "all" data.
    
    """
    full_name: str = None
    data_path: str = None
    label_encoder_path: Optional[str] = None
    subset_key: str = "all"
    
    def update(self, **kwargs) -> None:
        if "subset_key" in kwargs:
            self.subset_key = kwargs["subset_key"]
        if "num_samples" in kwargs:
            self.num_samples = {self.subset_key: kwargs["num_samples"]}
    
    @cached_property
    def num_samples(self) -> Dict[str,int]:
        return {self.subset_key: len(self.locate_files())}

    def __repr__(self):
        out = super().__repr__()
        out += '\n' + f"num_samples: {self.num_samples[self.subset_key]}"
        return out

    def locate_files(self) -> pd.DataFrame:
        return ETL.df_from_csv(self.data_path)
    
    def load_label_encoder(self) -> Union[None, LabelEncoder]:
        if os.path.exists(str(self.label_encoder_path)):
            return ETL.labels_from_json(str(self.label_encoder_path))
        return

    @classmethod
    def export_dataset_state(cls,
                             df: pd.DataFrame,
                             output_dir: Union[str, Path],
                             config: DictConfig=None,
                             encoder: LabelEncoder=None,
                             dataset_name: Optional[str]="dataset"
                             ) -> None:
        ETL.export_dataset_state(output_dir=output_dir,
                                     df=df,
                                     config=config,
                                     encoder=encoder,
                                     dataset_name=dataset_name)
            
    @classmethod
    def import_dataset_state(cls,
                             data_dir: Optional[Union[str, Path]]=None,
                             config_path: Optional[Union[Path, str]]=None,
                            ) -> Tuple["CSVDataset", "CSVDatasetConfig"]:

        return ETL.import_dataset_state(data_dir=data_dir,
                                            config_path=config_path)



####################################
####################################
####################################
####################################

####################################
####################################
####################################
####################################


class CustomDataset(torchdata.datasets.Files): # (CommonDataset):

    def __init__(self,
                 files: List[Path]=None,
                 samples_df: pd.DataFrame=None,
                 path_schema: Path = "{family}_{genus}_{species}_{collection}_{catalog_number}",
                 batch_fields: List[str] = ["image","target", "path", "catalog_number"],
                 eager_encode_targets: bool = False,
                 config: Optional[BaseConfig]=None,
                 transform=None,
                 target_transform=None):
        files = files or []
        super().__init__(files=files)
        self.path_schema = PathSchema(path_schema)
#         self.Batch = collections.namedtuple("Batch", batch_fields, defaults = (None,)*len(batch_fields))
        
        self.x_col = "path"
        self.y_col = "family"
        self.id_col = "catalog_number"
        self.eager_encode_targets = eager_encode_targets
        self.config = config or {}
        self.transform = transform
        self.target_transform = target_transform
        self.setup(samples_df=samples_df)
        
        
    def fetch_item(self, index: int) -> Tuple[str]:
        """
        Returns identically-structured namedtuple as __getitem__, with the following differences:
            - PIL Image w/o any transforms vs. torch.Tensor after all transforms
            - target text label vs, target int label
            - image path
            - image catalog_number
        
        """
        sample = self.parse_sample(index)
        image = Image.open(sample.path)
        metadata={
                  "path":getattr(sample, self.x_col),
                  "catalog_number":getattr(sample, self.id_col)
                 }
        return Batch(image=image,
                     target=getattr(sample, self.y_col),
                     metadata=metadata)
    

    def __getitem__(self, index: int):
        
        item = self.fetch_item(index)
        image, target, metadata = item.image, item.target, item.metadata
        target = self.label_encoder.class2idx[target]
        
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return Batch(image=image,
                     target=target,
                     metadata=metadata)
        
    def setup(self,
              samples_df: pd.DataFrame=None,
              label_encoder: LabelEncoder=None,
              fit_targets: bool=True):
        """
        Running setup() should result in the Dataset having assigned values for:
            self.samples
            self.targets
            self.samples_df
            self.label_encoder
        
        """
        if samples_df is not None:
            self.samples_df = samples_df.convert_dtypes()
        self.samples = [self.parse_sample(idx) for idx in range((len(self)))]
        self.targets = [sample[1] for sample in self.samples]
        self.samples_df = pd.DataFrame(self.samples).convert_dtypes()
        
        self.label_encoder = label_encoder or LabelEncoder()
        if fit_targets:
            self.label_encoder.fit(self.targets)
            
        if self.eager_encode_targets:
            self.targets = self.label_encoder.encode(self.targets).tolist()
        
    @classmethod
    def from_config(cls, config: DatasetConfig, subset_keys: List[str]=None) -> "CustomDataset":
        pass
        
    def parse_sample(self, index: int):
        pass
    
    @property
    def classes(self):
        return self.label_encoder.classes
    
    def __repr__(self):
        disp = f"""<{str(type(self)).strip("'>").split('.')[1]}>:"""
        disp += '\n\t' + self.config.__repr__().replace('\n','\n\t')
        return disp

    
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
    
    def intersection(self, other, suffixes=("_x","_y")):
        samples_df = self.samples_df
        other_df = other.samples_df
        
        intersection = samples_df.merge(other_df, how='inner', on=self.id_col, suffixes=suffixes)
        return intersection
    
    def __add__(self, other):
    
        intersection = self.intersection(other)[self.id_col].tolist()
        samples_df = self.samples_df
        
        left_union = samples_df[samples_df[self.id_col].apply(lambda x: x in intersection)]
        
        return left_union
    
    def __sub__(self, other):
    
        intersection = self.intersection(other)[self.id_col].tolist()
        samples_df = self.samples_df
        
        remainder = samples_df[samples_df[self.id_col].apply(lambda x: x not in intersection)]
        
        return remainder
    
    def filter(self, indices, subset_key: Optional[str]="all"):
        out = type(self)(samples_df = self.samples_df.iloc[indices,:],
                         config = deepcopy(self.config))
        out.config.update(subset_key=subset_key,
                          num_samples=len(out))
        return out
    
    def get_unsupervised(self):
        return UnsupervisedDatasetWrapper(self)


    
class ImageFileDataset(CustomDataset):
    
    @classmethod
    def from_config(cls, config: DatasetConfig, subset_keys: List[str]=None) -> "CustomDataset":
        files = config.locate_files()
        if isinstance(subset_keys, list):
            files = {k: files[k] for k in subset_keys}
        if len(files.keys())==1: 
            files = files[subset_keys[0]]
        new = cls(files=files,
                  path_schema=config.path_schema)
        new.config = config
        return new
    
    def parse_sample(self, index: int):
        path = self.files[index]
        family, genus, species, collection, catalog_number = self.path_schema.parse(path)

        return SampleSchema(path=path,
                            family=family,
                            genus=genus,
                            species=species,
                            collection=collection,
                            catalog_number=catalog_number)




class CSVDataset(CustomDataset):
    
    @classmethod
    def from_config(cls,
                    config: DatasetConfig, 
                    subset_keys: List[str]=None,
                    eager_encode_targets: bool=False) -> Union[Dict[str, "CSVDataset"], "CSVDataset"]:
        
        files_df = config.locate_files()
        if subset_keys is None:
            subset_keys = ['all']
        if isinstance(subset_keys, list) and isinstance(files_df, dict):
            files_df = {k: files_df[k] for k in subset_keys}
            new = {k: cls(samples_df=files_df[k],  \
                          eager_encode_targets=eager_encode_targets) for k in subset_keys}
            for k in subset_keys:
                new[k].config = deepcopy(config)
                new[k].config.subset_key = k

        if len(subset_keys)==1:
            if isinstance(files_df, dict):
                files_df = files_df[subset_keys[0]]
            new = cls(samples_df=files_df, 
                      eager_encode_targets=eager_encode_targets)
            new.config = config
            new.config.subset_key = subset_keys[0]
        return new
    
    def setup(self,
              samples_df: pd.DataFrame=None,
              label_encoder: LabelEncoder=None,
              fit_targets: bool=True):
        
        if samples_df is not None:
            self.samples_df = samples_df.convert_dtypes()
        self.files = self.samples_df[self.x_col].apply(lambda x: Path(x)).tolist()
        super().setup(samples_df=self.samples_df,
                      label_encoder=label_encoder,
                      fit_targets=fit_targets)


    def parse_sample(self, index: int):
        
        row = self.samples_df.iloc[index,:].tolist()
        path, family, genus, species, collection, catalog_number = row
        return SampleSchema(path=path,
                             family=family,
                             genus=genus,
                             species=species,
                             collection=collection,
                             catalog_number=catalog_number)



class UnsupervisedDatasetWrapper(torchdata.datasets.Files):#torchvision.datasets.ImageFolder):
    
    def __init__(self, dataset):
        super().__init__(files=dataset.files)
        self.dataset = dataset
        
    def __getitem__(self, index):
        return self.dataset[index][0]
    
    def __len__(self):
        return len(self.dataset)
    
    def __repr__(self):
        out = "<UnsupervisedDatasetWrapper>\n"
        out += self.dataset.__repr__()
        return out















































###########################################
###########################################
# class CommonDataSelect(torchdata.datasets.Files):
#     def __init__(self,
#                  name: str=None,
#                  files: List[Path]=None,
#                  subset_key: str=None,
#                  **kwargs):
#     @classmethod
#     def select_dataset_by_name(cls, name: str, config: DictConfig=None) -> "ImageDataset":
#     def select_subset_from_indices(self,
#                                    indices: Sequence,
#                                    x_col = 'path',
#                                    y_col = "family",
#                                    subset_key: str = None,
#                                    in_place: bool=False) -> Optional["FossilDataset"]:
#     def filter_samples_by_threshold(self,
#                                     threshold: int=1,
#                                     x_col = 'path',
#                                     y_col = "family",
#                                     subset_key: str = None,
#                                     in_place: bool=False) -> "FossilDataset":
# ###########################################
# ###########################################

# class CommonDataArithmetic: # (CommonDataset):
    
#     def __init__(self):
#     @property
#     def samples_df(self):        
#     @classmethod
#     def get_files_from_samples(cls,
#                                samples: Union[pd.DataFrame, List],
#                                x_col: Optional[str]="path"):
#     def intersection(self, other):
#     def __add__(self, other):
#     def __sub__(self, other):
# ############################################################
# ############################################################


# class CommonDataset(CommonDataArithmetic, CommonDataSelect):
#     def __init__(self,
#                  config: DictConfig,
#                  files: Union[Dict[str, List[Path]], List[Path]]=None,
#                  return_signature: List[str] = ["image","target","path"],
#                  subset_key: str=None,
#                  *args, **kwargs):
#     @classmethod
#     def default_config(cls) -> DictConfig:
#     @classmethod
#     def from_dataframe(cls, 
#                        sample_df: pd.DataFrame,
#                        config: Optional[DictConfig] = None,
#                        return_signature: Optional[List[str]] = ["image","target","path"],
#                        subset_key: Optional[str]=None,
#                        *args, **kwargs):
#     @classmethod
#     def from_dataset(cls, other):
#     @property
#     def num_samples(self) -> int:
#     @property
#     def classes(self):
#     @property
#     def num_classes(self):
#     def setup(self, 
#               stage: str=None,
#               train_transform: Optional[Callable] = None,
#               eval_transform: Optional[Callable] = None,
#               target_transform: Optional[Callable] = None
#               ):
#     def parse_config(self,
#                      config: DictConfig) -> DictConfig:
#     def parse_sample(self, index: int):
#     def fetch_item(self, index: Union[int, Sequence]) -> Tuple[str]:
#     def __getitem__(self, index: int):
#     def __repr__(self):
#     def display_grid(self,
#                    indices: Union[int,List[int]]=64,
#                    repeat_n: Optional[int]=1,
#                    max_images: int=64,
#                    columns: int=5,
#                    width: int=20,
#                    height: int=18,
#                    label_wrap_length: int=50,
#                    label_font_size: int=8) -> None:
#     def slice(self, indices: Union[List, np.array]) -> torch.Tensor:
#     def show_batch(self,
#                    indices: Union[int,List[int]]=64,
#                    repeat_n: Optional[int]=1,
#                    max_images: int=64,
#                    cmap: str='cividis',
#                    grayscale=True,
#                    titlesize=30,
#                    include_colorbar: bool=True,
#                    suptitle: str="default"):

# ##############################################################
# ##############################################################



# class LoggingDataModule(pl.LightningDataModule):
#     @classmethod
#     def default_config(cls) -> DictConfig:
#     def _setup_preconfigured_splits(self,
#                                     data_splits: Dict[str, CommonDataset],
#                                     train_transform: Optional[Callable] = None,
#                                     eval_transform: Optional[Callable] = None,
#                                     target_transform: Optional[Callable] = None
#                                     ):
#     @staticmethod
#     def validate_data_dir(data_dir: str) -> bool:
#     def import_dataset_from_csv(self,
#                                 data_dir: str,
#                                 user_config: Optional[DictConfig]=None) -> None:
#     def export_dataset_to_csv(self,
#                                output_dir: str) -> Dict[str,str]:
# ###########################################
# ###########################################

# class LeavesLightningDataModule(LoggingDataModule): #pl.LightningDataModule):
#     image_size = 224
#     channels = 3
#     image_buffer_size = 32
#     mean = [0.5, 0.5, 0.5]
#     std = [1.0, 1.0, 1.0]
#     DatasetConstructor: Callable = CommonDataset
#     totensor: Callable = transforms.ToTensor()
#     def __init__(self,
#                  config: DictConfig,
#                  data_dir: Optional[str]=None):
#         """ Abstract Base Class meant to be subclassed for each custom datamodule associated with the leavesdb database.
        
#         Subclasses must override definitions for the following methods/properties:
        
#         - available_datasets -> returns a dictionary mapping dataset names -> dataset absolute paths
#         - get_dataset_split -> returns a LeavesDataset object for either the train, val, or test splits.

#         Args:
#             name (str, optional): Defaults to None.
#                 Subclasses should define a default name.
#             batch_size (int, optional): Defaults to 32.
#             val_split (float, optional): Defaults to 0.2.
#                 Must be within [0.0, 1.0]. If omitted, looks for val/ subdir on disk.
#             test_split (float, optional): Defaults to 0.0.
#                 Must be within [0.0, 1.0]. If omitted, looks for test/ subdir on disk.
#             normalize (bool, optional): Defaults to True.
#                 If True, applies mean and std normalization to images at the end of all sequences of transforms. Base class has default values of mean = [0.5, 0.5, 0.5] and std = [1.0, 1.0, 1.0], which results in normalization becoming an identity transform. Subclasses should provide custom overrides for these values. E.g. imagenet  for example
#             image_size (Union[int,str], optional): Defaults to 224 if None.
#                 if 'auto', extract image_size from dataset name (e.g. "Exant_Leaves_family_10_1024 -> image_size=1024")
#             grayscale (bool, optional): Defaults to True.
                
#             channels (int, optional): Defaults to 3 if None.
#                 Options are either 1 or 3 channels. No-op if grayscale=False.
#             return_paths (bool, optional): Defaults to False.
#                 If True, internal datasets return tuples of length 3 containing (img, label, path). If False, return tuples of length 2 containing (img, label).
#             num_workers (int, optional): Defaults to 0.
#             pin_memory (bool, optional): Defaults to False.
#             seed (int, optional): Defaults to None.
#                 This must be set for reproducable train/val splits.
#             debug (bool, optional): Defaults to False.
#                 Set to True in order to turn off (1) shuffling, and (2) image augmentations.
#                 Purpose is to allow removing as much randomness as possible during runtime. Augmentations are removed by applying the eval_transforms to all subsets.

#         """
#     def parse_config(self,
#                      config: DictConfig) -> DictConfig:
#     def setup(self,
#               stage: str=None,
#               train_transform: Optional[Callable] = None,
#               eval_transform: Optional[Callable] = None,
#               target_transform: Optional[Callable] = None
#               ):
#     @property
#     def classes(self):
#     @property
#     def num_classes(self):
#     def init_dataset_stage(self,
#                            stage: str='fit',
#                            train_transform: Optional[Callable] = None,
#                            eval_transform: Optional[Callable] = None):
#     def get_dataset(self, stage: str='train'):
#     def get_dataloader(self, stage: str='train'):
#     def train_dataloader(self):
#     def val_dataloader(self):
#     def test_dataloader(self):
#     def predict_dataloader(self):
#     @property
#     def normalize_transform(self) -> Callable:
#     def default_train_transforms(self,
#                                  normalize: bool=True, 
#                                  augment:bool=True,
#                                  grayscale: bool=True,
#                                  channels: Optional[int]=3):
#     def default_eval_transforms(self, 
#                                 normalize: bool=True,
#                                 resize_PIL: bool=True,
#                                 grayscale: bool=True,
#                                 channels: Optional[int]=3,
#                                 transform_list: Optional[List[Callable]]=None):
#     def get_default_transforms(self):
#     def get_batch(self, stage: str='train', batch_idx: int=0):
#     def show_batch(self, stage: str='train', batch_idx: int=0, cmap: str='cividis', grayscale=True, titlesize=30):
#         """
#         Useful utility function for plotting a single batch of images as a single grid.
        
#         Good mild grayscale cmaps: ['magma', 'cividis']
#         Good mild grayscale cmaps: ['plasma', 'viridis']
#         """
#     @property
#     def dims(self):
#     def __repr__(self):
#     @property
#     def available_datasets(self):
#     @available_datasets.setter
#     def available_datasets(self, new: Dict[str,str]):
#######################################################
#######################################################
#######################################################