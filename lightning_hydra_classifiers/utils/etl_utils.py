"""
lightning_hydra_classifiers/utils/dataset_management_utils.py



Created on: Tuesday, July 27th, 2021
Author: Jacob A Rose


"""

import os
from pathlib import Path
import numpy as np
import numbers
from typing import Union, List, Any, Tuple, Dict, Optional, Callable
import collections
from sklearn.model_selection import train_test_split
import json
from typing import *

import pandas as pd
import torchdata
from omegaconf import DictConfig, OmegaConf
import collections
import matplotlib.pyplot as plt
import seaborn as sns
from more_itertools import collapse, flatten
import dataclasses
from dataclasses import is_dataclass

from torchvision.datasets import ImageFolder

from lightning_hydra_classifiers.utils import template_utils
from lightning_hydra_classifiers.utils.common_utils import LabelEncoder, DataSplitter

from hydra.experimental import compose, initialize
# from lightning_hydra_classifiers.experiments.configs.config import *
log = template_utils.get_logger(__name__)


__all__ = ["save_config", "load_config", 
           "Extract", "Transform", "Load", "ETL"]






def save_config(config: DictConfig, config_path: str):
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(config, resolve=True))

def load_config(config_path: str) -> DictConfig:    
    with open(config_path, "r") as f:
        loaded = OmegaConf.load(f)
    return loaded


class Extract:
    

    valid_splits: Tuple[str] = ("train", "val", "test")
        
    data_file_ext_maps = {"df":".csv",
                          "encoder":".json",
                          "config":".yaml"}
    
    @classmethod
    def get_split_subdir_stems(cls, dataset_dir: str) -> List[str]:
        
        subdirs = os.listdir(dataset_dir)
        stems = []
        for subdir in subdirs:
            if subdir in cls.valid_splits:
                stems.append(subdir)
        return stems
        
    @classmethod
    def locate_files(cls,
                     dataset_dir: Union[str, List[str]],
                     select_subset: Optional[str]=None) -> Dict[str, torchdata.datasets.Dataset]:

        files = {}
        
        if isinstance(dataset_dir, list):
            # If a list of data directories is provided, concatenate their results into the 'all' key. e.g. When constructing the combined Florissant & General Fossil datasets
            assert np.all([os.path.isdir(d) for d in dataset_dir])
            files["all"] = []
            for data_dir in dataset_dir:
                files["all"].extend(cls.locate_files(data_dir)['all'])
            return files
#         else:
        splits = cls.get_split_subdir_stems(dataset_dir=dataset_dir)
        
        if len(splits) > 1:
            # root
            #     /train
            #          /class_0
            #     /test
            #     ....
            for subdir in splits:
                files[subdir] = torchdata.datasets.Files.from_folder(Path(dataset_dir, subdir), regex="*/*.jpg").files 
            files["all"] = list(flatten([files[subdir] for subdir in splits]))
            
        elif len(os.listdir(dataset_dir)) > 1:
            # root
            #     /class_0
            #     /class_1
            #     ....
            files["all"] = torchdata.datasets.Files.from_folder(Path(dataset_dir), regex="*/*.jpg").files
        else:
            raise Exception(f"# of valid subdirs = {len(os.listdir(dataset_dir))} is invalid for locating files.")

        if isinstance(select_subset, str):
            files = {select_subset: files[select_subset]}
        return files
    
    @classmethod
    def df_from_dir(cls,
                    root_dir: Union[str, Path],
                    select_subset: Optional[str]=None) -> pd.DataFrame:
        files = cls.locate_files(dataset_dir=root_dir,
                                 select_subset=select_subset)
        for k in list(files.keys()):
            files[k] = pd.DataFrame(files[k]).rename(columns={0:"path"}) #.samples)
        return files
    
    @classmethod
    def df_from_csv(cls, path: Union[str, Path], index_col: int=None) -> pd.DataFrame:
        return pd.read_csv(path, index_col=index_col)

    @classmethod
    def config_from_yaml(cls, path: Union[str, Path]) -> DictConfig:
        return load_config(config_path=path)
    
    @classmethod
    def labels_from_json(cls, path: Union[str, Path]) -> LabelEncoder:
        return LabelEncoder.load(path)

    @classmethod
    def df2csv(cls,
               df: pd.DataFrame,
               path: Union[str, Path],
               index: bool=False) -> None:
        df.to_csv(path, index=index)

    @classmethod
    def config2yaml(cls, 
                    config: DictConfig,
                    path: Union[str, Path]) -> None:
        save_config(config=config, config_path=path)

    @classmethod
    def labels2json(cls,
                    encoder: LabelEncoder,
                    path: Union[str, Path]) -> None:
        encoder.save(path)

        
    
    
class Transform:
    
    @classmethod
    def config2dataclass(cls, config, dataclass_type: Callable):
        """
        Recursively convert DictConfig to a Structured Config specified by the provided dataclass.
        
        Look at scripts in "lightning_hydra_classifiers/experiments/configs"
        """
        kwargs = {}
        for k, field in dataclass_type.__dataclass_fields__.items():
            if is_dataclass(field.type):
                kwargs[k] = cls.config2dataclass(config=config[k], dataclass_type=field.type)
            else:
#                 try:
                if isinstance(config[k], Container):
                    try:
                        kwargs[k] = field.type(**OmegaConf.to_container(config[k]))
                    except TypeError:
                        
                        kwargs[k] = tuple(OmegaConf.to_container(config[k]))
                    except ValueError:
                        if not isinstance(config[k], str):
                            kwargs[k] = field.type(**config[k])
#                     except TypeError as e:
                        else:
                            kwargs[k] = config[k]
        return dataclass_type(**kwargs)

    
class Load:
    
    @staticmethod
    def load_hydra_config(config_name: str = "multitask_experiment_config",
                          config_path: Union[str, None] = None,
                          job_name: Union[str, None] = None,
                          overrides: Optional[List[str]] = None
                         ) -> DictConfig:
        """
        This is a helper unction for instantiating DictConfigs from on-disk yaml files, mostly during unit testing.
        
        User should opt for using ETL.init_structured_config instead.
        """
        overrides = overrides or []
        with initialize(config_path=config_path,
                        job_name=job_name):
            cfg = compose(config_name=config_name, overrides=overrides)
            
        return cfg

            
class ETL(Load, Transform, Extract):
    
    @classmethod
    def init_structured_config(cls,
                               config_name: str = "multitask_experiment_config",
                               config_path: Union[str, None] = None,
                               job_name: Union[str, None] = None,
                               dataclass_type: Optional = None,
                               overrides: Optional[List[str]] = None,
                               cfg: Optional["Config"] = None
                              ) -> Union[DictConfig, "BaseExperimentConfig"]:
        """
        This is a helper function for instantiating StructuredConfigs as either an instance of a dataclass or a 
        DictConfig from on-disk yaml files, mostly during unit testing.
        
        User should opt for using ETL.init_structured_config.
        """
        if cfg is None:
            cfg = cls.load_hydra_config(config_name=config_name,
                                        config_path=config_path,
                                        job_name=job_name,
                                        overrides=overrides)
    
        if dataclass_type is not None:
            cfg = cls.config2dataclass(config=cfg, dataclass_type=dataclass_type)
            
        return cfg
    

#################################################
#################################################

#################################################
#################################################