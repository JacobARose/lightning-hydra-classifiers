#!/usr/bin/env python
# coding: utf-8

"""

The following command:

python "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/scripts/generate_images_multires_leavesdbv1_0.py" --run-all




python "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/scripts/generate_images_multires_leavesdbv1_0.py" -d "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images" --run-all



is equivalent to the following 4 cmdline statements:

python "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/scripts/generate_images_multires_leavesdbv1_0.py" --resolution=512
python "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/scripts/generate_images_multires_leavesdbv1_0.py" --resolution=1024
python "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/scripts/generate_images_multires_leavesdbv1_0.py" --resolution=1536
python "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/scripts/generate_images_multires_leavesdbv1_0.py" --resolution=2048

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
seed = 334455


random.seed(seed)
np.random.seed(seed)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 200)

from torchvision import transforms
from torchvision import utils
import torchvision
from ResizeRight.resize_right import resize_right, interp_methods
from functools import partial

from pandarallel import pandarallel
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

def channelwise_min(img: torch.Tensor) -> torch.Tensor:
    return img.min(dim=1).values.min(dim=1).values

def channelwise_max(img: torch.Tensor) -> torch.Tensor:
    return img.max(dim=1).values.max(dim=1).values

def normalize_image(img: torch.Tensor) -> torch.Tensor:
    """Enforce pixel bounds to range [0.0, 1.0]"""
    img_min = channelwise_min(img).view(-1,1,1)
    img_max = channelwise_max(img).view(-1,1,1)
    return ((img - img_min) / (img_max - img_min))


#####################################
resize = partial(resize_right.resize,
                 interp_method=interp_methods.cubic,
                 antialiasing=True)

def _clever_crop(img: torch.Tensor,
                 target_shape: Tuple[int]=(3, 128,128),
                 grayscale: bool=False,
                 max_aspect_ratio: float=1.2,
                 normalize: bool=True
                 ) -> torch.Tensor:
    """[summary]

    Args:
        img (torch.Tensor): [description]
        target_shape (Tuple[int], optional): [description]. Defaults to (3,128,128).
        grayscale (bool, optional): [description]. Defaults to False.
        max_aspect_ratio (float): Defaults to 1.2
        normalize (bool): Defaults to True
            If True, normalize final image tensor to the range [0.0, 1.0]

    Returns:
        torch.Tensor: [description]
    """
    
    minside = np.min(img.shape[1:])# + 1
    maxside = np.max(img.shape[1:])# + 1
    new_img = img
    
    if (maxside / minside) > max_aspect_ratio:
        repeating = np.floor((maxside / minside)) 
        min_dim = np.argmin(img.shape[1:]) + 1
        for _ in range(int(repeating)):
            new_img = torch.cat([new_img, img], dim=min_dim)
        new_img = torch.rot90(new_img, k=1, dims=[1,2])
        
    new_img = resize(new_img, out_shape=target_shape)
    
    if grayscale:
        num_output_channels = target_shape[0]
        new_img = torchvision.transforms.functional.rgb_to_grayscale(img=new_img, num_output_channels=num_output_channels)
    
    if normalize:
        new_img = normalize_image(new_img)
        
    return new_img


def setup_clever_crop(target_shape: Tuple[int]=(3,128,128),
                      grayscale: bool=False,
                      max_aspect_ratio: float=1.2,
                      normalize: bool=True
                      ) -> Callable:
    """
    Decorates the _clever_crop function with optional kwargs and returns new function that only expects a torch.Tensor as input.
    The returned function can then be used in an image processing pipeline compatible with torchvision.

    Args:
        target_shape (Tuple[int], optional): [description]. Defaults to (3,128,128).
        grayscale (bool, optional): [description]. Defaults to False.
        max_aspect_ratio (float): Defaults to 1.2
    Returns:
        Callable: [description]
    """
    return partial(_clever_crop, target_shape=target_shape, grayscale=grayscale, max_aspect_ratio=max_aspect_ratio, normalize=normalize)



##############################

def resize_and_resave_dataset(data,
                              target_root_dir: str,
                              res: int=512
                              ):
    dataset_name = Path(target_root_dir).parts[-1]
    for fam in data.classes:
        os.makedirs(Path(target_root_dir, fam), exist_ok=True)
    
    num_samples = len(data)
    sample_idx = random.sample(range(num_samples), k=1)[0]


    target_shape = (3, res, res)
    clever_crop = setup_clever_crop(target_shape=target_shape,
                                    grayscale=False,
                                    max_aspect_ratio=1.2,
                                    normalize=True
                                    )

#     def resize_and_save_img(img: PIL.Image,
#                             target_path: str
#                             ):
#         if os.path.isfile(target_path):
#             return
#         img = transforms.ToTensor()(img)
#         target_img = clever_crop(img)
#         target_img = (target_img * 255.0).to(torch.uint8)
#         try:
#             torchvision.io.write_jpeg(target_img,
#                                       target_path,
#                                       quality=100)
#         except Exception as e:
#             print(target_path, e)
#         assert os.path.isfile(target_path)

    for sample_idx in trange(num_samples, desc=f"Smart Crop {dataset_name}"):

        source_path = data.samples[sample_idx][0]
        img, label_idx = data[sample_idx]
        family = data.classes[label_idx]

        source_root_dir = Path(source_path).parent.parent
    #     family = Path(source_path).parts[-2]
        rel_path = Path(source_path).relative_to(source_root_dir)
        target_path = str(Path(target_root_dir, rel_path))

        resize_and_save_img(img=img,
                            target_path=target_path)
        
        
        
##############################




def resize_and_save_img(img: Union[str, Path, PIL.Image.Image],
                        target_path: str,
                        clever_crop: Callable
                        ):
    if isinstance(img, (str, Path)):
        img = Image.open(img)
    if os.path.isfile(target_path):
        return
    img = transforms.ToTensor()(img)
    target_img = clever_crop(img)
    target_img = (target_img * 255.0).to(torch.uint8)
    try:
        torchvision.io.write_jpeg(target_img,
                                  target_path,
                                  quality=100)
    except Exception as e:
        print(target_path, e)
    assert os.path.isfile(target_path)






def resize_and_resave_dataset_parallel(data,
                                       target_root_dir: str,
                                       res: int=512,
                                       parallel=True
                                       ):
    dataset_name = Path(target_root_dir).parts[-1]
    for fam in data.classes:
        os.makedirs(Path(target_root_dir, fam), exist_ok=True)
    
    num_samples = len(data)
    sample_idx = random.sample(range(num_samples), k=1)[0]


    target_shape = (3, res, res)
    clever_crop = setup_clever_crop(target_shape=target_shape,
                                    grayscale=False,
                                    max_aspect_ratio=1.2,
                                    normalize=True
                                    )

#     resize_and_save_img = partial(resize_and_save_img, clever_crop=clever_crop)
    data_df = []

    for sample_idx in trange(num_samples, desc=f"Smart Crop {dataset_name}"):

        source_path = data.samples[sample_idx][0]
        label_idx = data.targets[sample_idx]
        family = data.classes[label_idx]

        source_root_dir = Path(source_path).parent.parent
    #     family = Path(source_path).parts[-2]
        rel_path = Path(source_path).relative_to(source_root_dir)
        target_path = str(Path(target_root_dir, rel_path))

        data_df.append((source_path, target_path))
    data_df = pd.DataFrame(data_df, columns=['source','target'])
    
    if parallel:
        data_df.parallel_apply(lambda x: resize_and_save_img(img = x['source'], target_path = x['target'], clever_crop=clever_crop), axis=1)
    else:
        data_df.apply(lambda x: resize_and_save_img(img = x['source'], target_path = x['target'], clever_crop=clever_crop), axis=1)
    
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




def create_image_dataset_at_new_resolution(args):
    
    res = args.res
    root_dir = args.root_dir

    print(f'Initiating conversion of images to new image shape = {(3, res, res)}')
    # Instantiate main-reference datasets at torchvision Image datasets
    
    data_db_root_dir = Path(root_dir) #, "original/full/jpg")
    source_dataset_root_dirs = {
                                "Extant_Leaves": str(data_db_root_dir / "Extant_Leaves" / "original" / "full" / "jpg"),
                                "Florissant_Fossil": str(data_db_root_dir / "Fossil" / "Florissant_Fossil" / "original" / "full" / "jpg"),
                                "General_Fossil": str(data_db_root_dir / "Fossil" / "General_Fossil" / "original" / "full" / "jpg")
                               }

    # # Generate all datasets at new resolution
    target_db_root_dir = Path(root_dir)#, f"{res}/full/jpg")
    resolution_target_root_dirs = {
                                   "Extant_Leaves": str(target_db_root_dir / "Extant_Leaves" / str(res) / "full" / "jpg"),
                                   "Florissant_Fossil": str(target_db_root_dir / "Fossil" / "Florissant_Fossil" / str(res) / "full" / "jpg"),
                                   "General_Fossil": str(target_db_root_dir / "Fossil" / "General_Fossil" / str(res) / "full" / "jpg")
                                  }
    
    source_datasets = {}
    for k, v in source_dataset_root_dirs.items():
        source_datasets[k] = get_image_dataset(root_dir=v)
    
    
    for dataset_name in source_datasets.keys():
        data=source_datasets[dataset_name]
        print(dataset_name, len(data))
        
        resize_and_resave_dataset_parallel(data,
                                           target_root_dir=resolution_target_root_dirs[dataset_name],
                                           res=res)
        
#         resize_and_resave_dataset(data=data,
#                                   target_root_dir=resolution_target_root_dirs[dataset_name],
#                                   res=res)





        
        
def cmdline_args():
    p = argparse.ArgumentParser(description="Resize datasets to a new resolution, using the clever crop resize function. Requires source images to exist in {root_dir}/original/jpg and be organized into 1 subdir per-class.")
    p.add_argument("-r", "--resolution", dest="res", type=int,
                   help="target resolution, images resized to (3, res, res).")
    p.add_argument("-d", "--root_dir", dest="root_dir", type=str,
                   default="/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images",
                   help="""Destination image root dir. Script will expect source images to exist in class-wise subdirs in "./{dataset}/original/full/jpg". Then, for creating the target images
                   it will create subdirs "./{dataset}/{res}/full/jpg" for user-input resolution value.""")
    p.add_argument("-a", "--run-all", dest="run_all", action="store_true",
                   help="Flag for when user would like to run through all default resolution arguments on a given dataset. Currently includes resolutions = [512, 1024, 1536, 2048].")
    p.add_argument("--num_workers", dest="num_workers", type=int, default=16,
                   help="Number of parallel processes to be used by pandas to efficiently construct symlinks.")
    return p.parse_args()



if __name__ == "__main__":

    args = cmdline_args()
    
    num_workers = args.num_workers    
    pandarallel.initialize(nb_workers=num_workers, progress_bar=True)

    

    if args.run_all:
        logging.info(f'[INITIATING] Creation of multiple dataset versions using smart-crop at the following resolutions: {[512, 1024, 1536, 2048]}.')
        for res in [512, 1024, 1536, 2048]:
            args.res = res
            create_image_dataset_at_new_resolution(args)
            
    elif isinstance(args.res, int):
        create_image_dataset_at_new_resolution(args)
        
    else:
        logging.warning('User must provide either an int value for --resolution or the flag --run-all. Current args are:\n' + f"{args}")