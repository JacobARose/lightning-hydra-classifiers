#!/usr/bin/env python
# coding: utf-8

"""

The following command:

python "/media/data/jacob/GitHub/lightning-hydra-classifiers/lightning_hydra_classifiers/data/utils/generate_multires_images.py" --run-all



# Run all 3 datasets for all 4 resolutions in serial

python "/media/data/jacob/GitHub/lightning-hydra-classifiers/lightning_hydra_classifiers/data/utils/generate_multires_images.py" --dataset_name "all" --run-all


# Run 1 dataset for all 4 resolutions in serial

python "/media/data/jacob/GitHub/lightning-hydra-classifiers/lightning_hydra_classifiers/data/utils/generate_multires_images.py" --dataset_name Extant_Leaves --run-all -skip 512

is equivalent to the following 4 cmdline statements (Run in parallel):

python "/media/data/jacob/GitHub/lightning-hydra-classifiers/lightning_hydra_classifiers/data/utils/generate_multires_images.py" --resolution=512 --dataset_name Extant_Leaves
python "/media/data/jacob/GitHub/lightning-hydra-classifiers/lightning_hydra_classifiers/data/utils/generate_multires_images.py" --resolution=1024 --dataset_name Extant_Leaves
python "/media/data/jacob/GitHub/lightning-hydra-classifiers/lightning_hydra_classifiers/data/utils/generate_multires_images.py" --resolution=1536 --dataset_name Extant_Leaves
python "/media/data/jacob/GitHub/lightning-hydra-classifiers/lightning_hydra_classifiers/data/utils/generate_multires_images.py" --resolution=2048 --dataset_name Extant_Leaves

Note: When launching in parallel from the cmdline, be mindful of specifying low enough num_workers.
"""


# # generate_images_multires_leavesdbv1_0-prerelease
# 
# Created by: Jacob A Rose  
# Created on: Tuesday June 29th, 2021

import logging
from tqdm.auto import tqdm, trange
from typing import Callable, Optional, Union, List, Tuple
from torchvision.datasets import ImageFolder
import torch
import os
import pandas as pd
import numpy as np
from PIL import Image, ImageStat
import PIL

from pathlib import Path
import random
from rich import print as pp
import matplotlib.pyplot as plt
import time
seed = 334455


random.seed(seed)
np.random.seed(seed)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 200)

from dataclasses import dataclass, asdict
from torchvision import transforms
from torchvision import utils
import torchvision
from torch import nn
from functools import partial
import json
from munch import Munch
from pandarallel import pandarallel
from lightning_hydra_classifiers.utils.ResizeRight.resize_right import resize_right, interp_methods
from lightning_hydra_classifiers.utils.dataset_management_utils import DatasetFilePathParser, parse_df_catalog_from_image_directory, diff_dataset_catalogs
from typing import *

tqdm.pandas()

plt.style.available
plt.style.use('seaborn-pastel')


# ### Image clever crop

def plot_img_hist(img: torch.Tensor):
    plt.hist(np.array(img).ravel(), bins=50, density=True);
    plt.xlabel("pixel values")
    plt.ylabel("relative frequency")
    plt.title("distribution of pixels");

def stats(img: torch.Tensor):
    print(f'img.min(): {img.min():.3f}')
    print(f'img.max(): {img.max():.3f}')
    print(f'img.mean(): {img.mean():.3f}')
    print(f'img.std(): {img.std():.3f}')
    
    print(f"% pixels < 0.0: {(img[img<0.0].shape[0] / np.prod(img[:].shape)):.2%}")
    print(f"% pixels >= 0.0: {(img[img>=0.0].shape[0] / np.prod(img[:].shape)):.2%}")

#####################################




class NormalizeImage(nn.Module):
    
    @staticmethod
    def channelwise_min(img: torch.Tensor) -> torch.Tensor:
        return img.min(dim=1).values.min(dim=1).values

    @staticmethod
    def channelwise_max(img: torch.Tensor) -> torch.Tensor:
        return img.max(dim=1).values.max(dim=1).values

    @staticmethod
    def normalize_image(img: torch.Tensor) -> torch.Tensor:
        """Enforce pixel bounds to range [0.0, 1.0]"""
        img_min = NormalizeImage.channelwise_min(img).view(-1,1,1)
        img_max = NormalizeImage.channelwise_max(img).view(-1,1,1)
        return ((img - img_min) / (img_max - img_min))

########################
########################

@dataclass
class CleverCropConfig:
    
    interp_method: Callable=interp_methods.cubic
    antialiasing: bool=True
    target_shape: Tuple[int]=(3, 128, 128)
    max_aspect_ratio: float=1.2
    grayscale: bool=False
    normalize: bool=True
    

class CleverCrop:
    
    def __init__(self,
                 interp_method=interp_methods.cubic,
                 antialiasing=True,
                 target_shape: Tuple[int]=(3, 128, 128),
                 max_aspect_ratio: float=1.2,
                 grayscale: bool=False,
                 normalize: bool=True
                ):
        """[summary]

        Args:
            interp_method: Defaults to interp_methods.cubic.
            antialiasing (bool): Defaults to True.
            target_shape (Tuple[int], optional): [description]. Defaults to (3,128,128).
            grayscale (bool, optional): [description]. Defaults to False.
            max_aspect_ratio (float): Defaults to 1.2
            normalize (bool): Defaults to True
                If True, normalize final image tensor to the range [0.0, 1.0]

        """

        self.interp_method=interp_method
        self.antialiasing=antialiasing
        self.target_shape=target_shape
        self.num_output_channels = self.target_shape[0]
        self.max_aspect_ratio=max_aspect_ratio
        self.grayscale=grayscale
        self.normalize=normalize
        
        self.resize = partial(resize_right.resize,
#                               out_shape=self.target_shape,
                              interp_method=self.interp_method,
                              antialiasing=self.antialiasing)
        self.normalize_image = NormalizeImage.normalize_image
        
#         print('CleverCrop.__init__() ->', repr(self))

    
    @staticmethod
    def aspect_ratio(img: torch.Tensor):
        
        minside = np.min(img.shape[1:])
        maxside = np.max(img.shape[1:])

        aspect_ratio = (maxside / minside)
        return aspect_ratio
        
        
    def __call__(self,
                 img: torch.Tensor,
                 target_shape: Optional[Tuple[int]]=None
                ) -> torch.Tensor:
        """[summary]

        Args:
            img (torch.Tensor): [description]
            target_shape: (Optional[Tuple[int]]): Optionally override this CleverCrop class instance's init value.

        Returns:
            torch.Tensor: [description]
        """
        target_shape = target_shape or self.target_shape

        minside = np.min(img.shape[1:])# + 1
        maxside = np.max(img.shape[1:])# + 1
        new_img = img

        aspect_ratio = (maxside / minside)
        if aspect_ratio > self.max_aspect_ratio:
#             print(f'Aspect Ratio = {aspect_ratio:.2f} > max_aspect_ratio = {self.max_aspect_ratio}, stacking.')
            num_repeats = np.floor((maxside / minside)) 
            min_dim = np.argmin(img.shape[1:]) + 1
            for _ in range(int(num_repeats)):
                new_img = torch.cat([new_img, img], dim=min_dim)
        if maxside == img.shape[2]:
#             print(f'maxside = {maxside} is width, rotating.')
            new_img = torch.rot90(new_img, k=1, dims=[1,2])

        new_img = self.resize(new_img, out_shape=target_shape)

        if self.grayscale:
            num_output_channels = self.target_shape[0]
            new_img = torchvision.transforms.functional.rgb_to_grayscale(img=new_img,
                                                                         num_output_channels=self.num_output_channels)

        if self.normalize:
            new_img = self.normalize_image(new_img)

        return new_img

    def __repr__(self):
        return json.dumps({
        "interp_method":str(self.interp_method),
        "antialiasing":self.antialiasing,
        "target_shape":self.target_shape,
        "num_output_channels":self.num_output_channels,
        "max_aspect_ratio":self.max_aspect_ratio,
        "grayscale":self.grayscale,
        "normalize":self.normalize})


clever_crop = CleverCrop()



def resize_and_resave_dataset(data,
                              res: int=512
                              ):
    num_samples = len(data)
    target_shape = (3, res, res)

    for sample_idx in trange(num_samples, desc=f"Smart Crop {dataset_name}"):

        row = data.iloc[sample_idx]
        source_path = row.path
        target_path = row.target_path

        resize_and_save_img(img=source_path,
                            target_path=target_path,
                            target_shape=(3,res,res))
        
        
        
##############################




def resize_and_save_img(img: Union[str, Path, PIL.Image.Image],
                        target_path: str,
                        target_shape=None
                        ):
    if isinstance(img, (str, Path)):
        img = Image.open(img)
    if os.path.isfile(target_path):
        return
    img = transforms.ToTensor()(img)
    target_img = clever_crop(img, target_shape=target_shape)
    target_img = (target_img * 255.0).to(torch.uint8)
    try:
        torchvision.io.write_jpeg(target_img,
                                  target_path,
                                  quality=100)
    except Exception as e:
        print(target_path, e)
    assert os.path.isfile(target_path)





    
def resize_and_resave_dataset_parallel(data,
                                       resolution: int=512,
                                       parallel=True
                                       ):
    target_shape = (3, resolution, resolution)
    if parallel:
        data.parallel_apply(lambda x: resize_and_save_img(img = x['path'], target_path = x['target_path'], target_shape=target_shape), axis=1)
    else:
        data.apply(lambda x: resize_and_save_img(img = x['path'], target_path = x['target_path'], target_shape=target_shape), axis=1)
    
    return data
    
    
#         resize_and_save_img(img=img,
#                             target_path=target_path)


##################

def get_image_dataset(root_dir):
    """
    Simple wrapper around torchvision.datasets.ImageFolder
    """
    dataset = ImageFolder(root_dir)
    return dataset


##################
##################



import sys
import argparse
from lightning_hydra_classifiers.data.utils.catalog_registry import *

# def query_and_preprocess_source_and_target_catalog(config,
#                                                    target_config):
#     tag = available_datasets.query_tags(dataset_name=config.dataset_name,
#                                         y_col=config.y_col,
#                                         threshold=config.threshold,
#                                         resolution=config.resolution)
#     root_dir = available_datasets.get_latest(tag)
#     print(f"Tag: {tag}")
#     print(f"Root Dir: {root_dir}")
#     data_df = parse_df_catalog_from_image_directory(root_dir=root_dir, dataset_name=config.dataset_name)

#     target_dir = root_dir.replace("original", f"{target_config.resolution}")
#     data_df = data_df.assign(target_path = data_df.apply(lambda x: str(Path(target_dir, x.family, Path(x.path).name)), axis=1))
#     family_dirs = list(set(data_df.target_path.apply(lambda x: str(Path(x).parent))))
#     [os.makedirs(subdir, exist_ok=True) for subdir in family_dirs];
    
#     return data_df





def query_and_preprocess_catalog(config):
    """
    
    Arguments:
        config (NameSpace):
            ::dataset_name
            ::y_col
            ::threshold
            ::resolution
    
    """
    tag = available_datasets.query_tags(dataset_name=config.dataset_name,
                                        y_col=config.y_col,
                                        threshold=config.threshold,
                                        resolution=config.resolution)
    root_dir = available_datasets.get_latest(tag)
    print(f"Tag: {tag}")
    print(f"Root Dir: {root_dir}")
    if not os.path.isdir(root_dir):
        print("[INFO] There is not yet any directory located at: {root_dir}. Skipping dataset validation.")
        return pd.DataFrame(), root_dir
    data_df = parse_df_catalog_from_image_directory(root_dir=root_dir, dataset_name=config.dataset_name)

    return data_df, root_dir


def preprocess_target_catalog(data_df: pd.DataFrame, 
                              config,
                              source_dir: str) -> pd.DataFrame:
    """
    
    Arguments:
        config (NameSpace):
            ::dataset_name
            ::y_col
            ::threshold
            ::resolution
    
    """
#     import pdb;pdb.set_trace()
    target_dir = source_dir.replace("original", f"{config.resolution}")
    
    target_path = None    
    if data_df.shape[0] > 0:
        target_path = data_df.apply(lambda x: str(Path(target_dir, x.family, Path(x.path).name)), axis=1)
    data_df = data_df.assign(target_path=target_path)
    family_dirs = list(set(data_df.target_path.apply(lambda x: str(Path(x).parent))))
    [os.makedirs(subdir, exist_ok=True) for subdir in family_dirs];

    return data_df



# def query_and_preprocess_source_and_target_catalog(config,
#                                                    target_config):
    
#     data_df, root_dir = query_and_preprocess_catalog(config)
#     data_df = preprocess_target_catalog(data_df=data_df, config=target_config, source_dir=root_dir)
    
#     return data_df


def warm_start_catalogs(config,
                        target_config,
                        save_report: bool=False) -> pd.DataFrame:
    """ Save time by skipping previously processed files and keeping only the yet-to-be processed catalog.
    
    Query and Preprocess catalogs from source and target configs,
    return only the rows that are unique to the source location,
    then format the remainder in preparation to run the multi-res image generation script (i.e. source_path -> "path", target_path -> "target_path" columns)
    
    """
    source_catalog, source_dir = query_and_preprocess_catalog(config)
    target_catalog, target_dir = query_and_preprocess_catalog(target_config)
#     import pdb; pdb.set_trace()
    if target_catalog.shape[0] == 0:
        data_df = source_catalog
    else:
        shared, diff, source_only, target_only = diff_dataset_catalogs(source_catalog=source_catalog,
                                                                       target_catalog=target_catalog)
        data_df = source_only

        if save_report is not None:
            report_dir = Path(target_dir).parent / "audit_logs"
            os.makedirs(report_dir, exist_ok=True)
            shared.to_csv(Path(report_dir, "shared_catalog.csv"))
            diff.to_csv(Path(report_dir, "diff_catalog.csv"))
            print(f"Exported catalog validation logs to directory: {report_dir}")
            print(f"Contents:")
            pp(os.listdir(report_dir))

    
    data_df = preprocess_target_catalog(data_df=data_df, config=target_config, source_dir=source_dir)
    
    return data_df





def setup_configs(args):
    """
    args:
        ::dataset_name
        ::resolution
    
    """
    print('args:', vars(args))
    print(args.resolution)
    config = Munch(dataset_name = args.dataset_name,
                   y_col="family",
                   threshold=0,
                   resolution="original")
    target_config = Munch(dataset_name = args.dataset_name,
                          resolution=args.resolution,
                          y_col="family",
                          threshold=0)
    config.target_shape = (3, config.resolution, config.resolution)
    target_config.target_shape = (3, target_config.resolution, target_config.resolution)
    
    return config, target_config





def validate_dataset(args, save_report: bool=False) -> bool:
    """
    Compare source & target datasets, optionally save comparison report to csv files, and return True only if they are identical.
    
    Return:
        (bool)
    """
    
    config, target_config = setup_configs(args)
    data_df = warm_start_catalogs(config,
                                  target_config,
                                  save_report=save_report)
    return data_df.shape[0]==0


def process(args):

    config, target_config = setup_configs(args)
    data_df = warm_start_catalogs(config,
                                  target_config,
                                  save_report=args.save_report)
    if data_df.shape[0] == 0:
        print(f'[SKIPPING PROCESS] Warm start skipping previously generated dataset: {args.dataset_name} at resolution: {args.resolution} ')
        return


    pp(target_config)
    resize_and_resave_dataset_parallel(data=data_df,
                                       resolution=target_config.resolution,
                                       parallel=True)

            
#########################################
#########################################


def main(args):
    
    dataset_names = args.dataset_name
    resolutions = args.resolution

    print(f'[INITIATING] Multi-resolution dataset creation')
    print(f'Datasets: {dataset_names}')
    print(f"Resolutions: {resolutions}")
    
    num_trials = len(dataset_names)*len(resolutions)
    i=0
    for dataset_name in dataset_names:
        for res in resolutions:
            args.dataset_name = dataset_name
            args.resolution = res
            print(f"Trial #{i}/{num_trials-1}")
            
            if args.validate_only:
                print("[RUNNING] Dataset validation only.")
                validate_dataset(args, save_report=args.save_report)
                i+=1
                continue
            print("[RUNNING DATASET PROCESSING].", f": ({time.strftime('%H:%M%p %Z on %b %d, %Y')})")
            print(f"[ARGS] dataset_name: {args.dataset_name}, resolution: {args.resolution}")
            
            process(args)
            print(f"[FINISHED PROCESSING]", f": ({time.strftime('%H:%M%p %Z on %b %d, %Y')})")
            print("[RUNNING POST-PROCESSING DATA VALIDATION]")
            validate_dataset(args, save_report=args.save_report)
            i+=1
            



#########################################
#########################################


def cmdline_args():
    p = argparse.ArgumentParser(description="Resize datasets to a new resolution, using the clever crop resize function. Requires source images to exist in {root_dir}/original/jpg and be organized into 1 subdir per-class.")
    p.add_argument("-r", "--resolution", dest="resolution", type=int, nargs="*",
                   help="target resolution, images resized to (3, res, res).")
    p.add_argument("-n", "--dataset_name", dest="dataset_name", type=str, nargs="*", default="Extant_Leaves",
                   help="""Base dataset_name to be used to query the source root_dir.""")
    p.add_argument("-a", "--run-all", dest="run_all", action="store_true",
                   help="Flag for when user would like to run through all default resolution arguments on a given dataset. Currently includes resolutions = [512, 1024, 1536, 2048].")
    p.add_argument("-skip", dest="skip", nargs="*", type=int, default=[],
                   help="Explicitly skip any provided dataset resolutions.")
    p.add_argument("--validate_only", dest="validate_only", action="store_true", default=False,
                   help="Flag for when user would like to only validate datasets without launching any of the multi-res processing stages..")
    p.add_argument("--save_report", dest="save_report", action="store_true", default=False,
                   help="Flag for when user would like to save csv files from the validation report.")
    p.add_argument("--num_workers", dest="num_workers", type=int, default=15)    
    args = p.parse_args()
    

    print(args)
    if args.dataset_name == "all":
        args.dataset_name = ["Extant_Leaves", "Florissant_Fossil", "General_Fossil"]
    if args.run_all:
        args.resolution = [512, 1024, 1536, 2048]
    args.skip = list(set(args.skip).intersection(set(args.resolution)))
    if len(args.skip) > 0:
        print(f'Skipping resolutions {args.skip}')
        args.resolution = sorted(list(set(args.resolution) - set(args.skip)))
    
    return args





if __name__ == "__main__":

    args = cmdline_args()
    
    num_workers = args.num_workers    
    pandarallel.initialize(nb_workers=num_workers, progress_bar=True)    

    main(args)

#     if args.run_all:
#         logging.info(f'[INITIATING] Creation of multiple dataset versions using smart-crop at the following resolutions: {[512, 1024, 1536, 2048]}.')
#         for res in [512, 1024, 1536, 2048]:
#             args.res = res
#             create_image_dataset_at_new_resolution(args)
            
#     elif isinstance(args.res, int):
#         create_image_dataset_at_new_resolution(args)
        
#     else:
#         logging.warning('User must provide either an int value for --resolution or the flag --run-all. Current args are:\n' + f"{args}")