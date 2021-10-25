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

import pandas as pd
import torchdata
from omegaconf import DictConfig, OmegaConf
import collections
import matplotlib.pyplot as plt
import seaborn as sns
from more_itertools import collapse, flatten
import dataclasses

from torchvision.datasets import ImageFolder

from lightning_hydra_classifiers.utils import template_utils
from lightning_hydra_classifiers.utils.etl_utils import Extract, Transform, Load, ETL
from lightning_hydra_classifiers.utils.common_utils import LabelEncoder, DataSplitter
# from lightning_hydra_classifiers.experiments.configs.config import *
log = template_utils.get_logger(__name__)


__all__ = ["save_config", "load_config", 
           "Extract", "Transform", "Load", "ETL",
           "export_image_data_diagnostics", 
           "export_dataset_to_csv",
           "import_dataset_from_csv",
           "DatasetFilePathParser",
           "parse_df_catalog_from_image_directory",
           "dataframe_difference",
           "diff_dataset_catalogs"
]



# sns.set_context(context='talk', font_scale=0.8)
# sns.set_style('darkgrid')
# sns.set_palette('Set2')
    

#################################################
#################################################

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
    
    Metadata fields in each file path are specified by using a DatasetFilePathParser object.
    
    Arguments:
    
        root_dir (str):
            Location of the Imagenet-format organized image data on disk
    Returns:
        data_df (pd.DataFrame):
            
            
    """
    
    parser = DatasetFilePathParser().get_parser(dataset_name)
    data_df = Extract.df_from_dir(root_dir)['all']
    if data_df.shape[0] == 0:
        print('Empty data catalog, skipping parsing step.')
        return data_df
    for col, func in parser.items():
        print(col)
        data_df = data_df.assign(**{col:data_df.apply(lambda x: func(x, "path"), axis=1)})
        
    data_df = DatasetFilePathParser.parse_dtypes(data_df)
    return data_df




def dataframe_difference(source_df: pd.DataFrame,
                         target_df: pd.DataFrame,
                         id_col: str="relative_path",
                         keep_cols: Optional[List[str]]=None):
    """
    Find rows which are different between two DataFrames.
    
    Example:
    
        shared, diff, source_only, target_only = dataframe_difference(source_df=data_df,
                                                                      target_df=target_data_df,
                                                                      id_col="relative_path",
                                                                      keep_cols=["path"])
    """
    keep_cols = keep_cols or []
#     import pdb;pdb.set_trace()
    
    comparison_df = source_df.merge(target_df.loc[:, [id_col, *keep_cols]], how="outer", on=id_col, indicator=True)
    
    comparison_df = comparison_df.replace({"left_only":"source_only",
                                           "right_only":"target_only"})

    shared = comparison_df[comparison_df["_merge"]=="both"]
    diff = comparison_df[comparison_df["_merge"]!="both"]
    source_only = comparison_df[comparison_df["_merge"]=="source_only"].rename(columns={"path_x":"path"})
    target_only = comparison_df[comparison_df["_merge"]=="target_only"].rename(columns={"path_y":"path"})
    
#     import pdb;pdb.set_trace()
    
    return shared, diff, source_only, target_only


def diff_dataset_catalogs(source_catalog: pd.DataFrame,
                          target_catalog: pd.DataFrame) -> Tuple[pd.DataFrame]:
    """
    Find the shared and unique rows between 2 dataframes based on the "relative_path" column.
    """
    
    shared, diff, source_only, target_only = dataframe_difference(source_df=source_catalog,
                                                                  target_df=target_catalog,
                                                                  id_col="relative_path",
                                                                  keep_cols=["path", "catalog_number"])    
    

    num_preexisting = sum([shared.shape[0] + target_only.shape[0]])
    if num_preexisting > 0:
        print(f"Found {num_preexisting} previously generated files in target location.")
        print(f"""
        shared: {shared.shape[0]}
        diff: {diff.shape[0]}
        source_only: {source_only.shape[0]}
        target_only: {target_only.shape[0]}
        """)
    else:
        print(f"No previously generated files found in target location.")
        
    return shared, diff, source_only, target_only