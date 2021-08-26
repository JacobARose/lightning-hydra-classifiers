"""
lightning_hydra_classifiers/utils/dataset_management_utils.py



Created on: Tuesday, July 27th, 2021
Author: Jacob A Rose


"""

import os
from pathlib import Path
import numpy as np
import numbers
from typing import Union, List, Any, Tuple, Dict, Optional
import collections
from sklearn.model_selection import train_test_split
import json

import pandas as pd
import torchdata
from omegaconf import DictConfig, OmegaConf
import collections
import matplotlib.pyplot as plt
import seaborn as sns
from more_itertools import collapse, flatten
import dataclasses

from lightning_hydra_classifiers.utils import template_utils
from lightning_hydra_classifiers.utils.common_utils import LabelEncoder

log = template_utils.get_logger(__name__)


__all__ = ["save_config", "load_config", 
           "Extract", "DataSplitter",
           "export_image_data_diagnostics", 
           "export_dataset_to_csv",
           "import_dataset_from_csv",
           "DatasetFilePathParser",
           "parse_df_catalog_from_image_directory"
]



sns.set_context(context='talk', font_scale=0.8)
sns.set_style('darkgrid')
sns.set_palette('Set2')











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

        


        
        

#################################################
#################################################


class DataSplitter:

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

        log.debug(f"[RUNNING] [create_trainvaltest_splits()] splits={splits}")
        return dataset_splits






#################################################
#################################################



def export_image_data_diagnostics(data_splits: Dict[str,"CommonDataset"],
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



def export_dataset_to_csv(data_splits: Dict[str,"CommonDataset"],
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
    

def import_dataset_from_csv(data_catalog_dir: str) -> Tuple[Dict[str, "CommonDataset"], DictConfig]:
    from lightning_hydra_classifiers.data.common import CommonDataset
    
    
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









from torchvision.datasets import ImageFolder
# def get_image_dataset(root_dir):
#     """
#     Simple wrapper around torchvision.datasets.ImageFolder
#     """
#     dataset = ImageFolder(root_dir)
#     return dataset

##################
##################


from typing import *
import sys
import argparse


class DatasetFilePathParser:
    
#     def __init__(self, root_dir: str="/"):
#         self.root_dir = root_dir
    
    @classmethod
    def get_parser(cls, dataset_name: str) -> Dict[str, Callable]:
        if "Extant_Leaves" in dataset_name:
            return cls().ExtantLeavesParser
        if "Fossil" in dataset_name:
            return cls().FossilParser
        if "PNAS" in dataset_name:
            return cls().PNASParser
    
#     @property
    @classmethod
    def parse_dtypes(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.astype({
                            "path": pd.StringDtype(),
                            "family": pd.CategoricalDtype(),
                            "genus": pd.CategoricalDtype(),
                            "species": pd.CategoricalDtype(),
                            "catalog_number": pd.StringDtype(),
                            "relative_path": pd.StringDtype(),
                            "root_dir": pd.CategoricalDtype()
                           })
    
    @property
    def ExtantLeavesParser(self):
        return {
                "family": lambda x, col: Path(x[col]).stem.split('_')[0],
                "genus": lambda x, col: Path(x[col]).stem.split('_')[1],
                "species": lambda x, col: Path(x[col]).stem.split('_')[2],
                "catalog_number": lambda x, col: Path(x[col]).stem.split('_', maxsplit=4)[-1],
                "relative_path": lambda x, col: str(Path(x[col]).relative_to(Path(x[col]).parent.parent)),
                "root_dir": lambda x, col:  str(Path(x[col]).parent.parent)
               }

    @property
    def FossilParser(self):
        return {
                "family": lambda x, col: Path(x[col]).stem.split('_')[0],
                "genus": lambda x, col: Path(x[col]).stem.split('_')[1],
                "species": lambda x, col: Path(x[col]).stem.split('_')[2],
                "catalog_number": lambda x, col: Path(x[col]).stem.split('_', maxsplit=4)[-1],
                "relative_path": lambda x, col: str(Path(x[col]).relative_to(Path(x[col]).parent.parent)),
                "root_dir": lambda x, col:  str(Path(x[col]).parent.parent)
               }


    @property
    def PNASParser(self):
        return {
                "family": lambda x, col: Path(x[col]).stem.split('_')[0],
                "genus": lambda x, col: Path(x[col]).stem.split('_')[1],
                "species": lambda x, col: Path(x[col]).stem.split('_')[2],
                "catalog_number": lambda x, col: Path(x[col]).stem.split('_', maxsplit=3)[-1],
                "relative_path": lambda x, col: str(Path(x[col]).relative_to(Path(x[col]).parent.parent)),
                "root_dir": lambda x, col:  str(Path(x[col]).parent.parent)
               }

    
    
def parse_df_catalog_from_image_directory(root_dir: str, dataset_name: str="Extant_Leaves") -> pd.DataFrame:
    """
    Crawls root_dir and collects absolute paths of any images into a dataframe. Then, extracts
    maximum available metadata from file paths (e.g. family, species labels in file name).
    
    Arguments:
    
        root_dir (str):
            Location of the Imagenet-format organized image data on disk
    Returns:
        data_df (pd.DataFrame):
            
            
    """
    
    parser = DatasetFilePathParser().get_parser(dataset_name)
    data_df = Extract.df_from_dir(root_dir)['all']
    for col, func in parser.items():
        print(col)
        data_df = data_df.assign(**{col:data_df.apply(lambda x: func(x, "path"), axis=1)})
        
    data_df = DatasetFilePathParser.parse_dtypes(data_df)
    return data_df