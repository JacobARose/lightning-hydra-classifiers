"""

lightning_hydra_classifiers/transfer_sweep.py

Author: Jacob A Rose
Created: Wednesday June 23rd, 2021



export CUDA_VISIBLE_DEVICES=7;python "/media/data/jacob/GitHub/lightning-hydra-classifiers/lightning_hydra_classifiers/transfer_sweep.py" --trial_number 3
"""

import logging
import os
from pathlib import Path
from typing import Union, Callable, Tuple

import torch
import torch.nn.functional as F
# from torch import nn, optim
# from lightning_hydra_classifiers.models.transfer import cli_main
# from rich import print as pp

import argparse
import multiprocessing
from PIL import Image

from munch import Munch
import torchdata
import torchvision
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset

from byol_pytorch import BYOL
import pytorch_lightning as pl
from typing import *
# test model, a resnet 50

from lightning_hydra_classifiers.data.utils.make_catalogs import CSV_CATALOG_DIR_V1_0, CSVDatasetConfig, CSVDataset, DataSplitter
from lightning_hydra_classifiers.experiments.transfer_experiment import TransferExperiment

totensor: Callable = torchvision.transforms.ToTensor()

def toPIL(img: torch.Tensor, mode="RGB") -> Callable:
    return torchvision.transforms.ToPILImage(mode)(img)




# resnet = models.resnet50(pretrained=True)
# arguments

# parser = argparse.ArgumentParser(description='byol-lightning-test')

# parser.add_argument('--image_folder', type=str, required = True,
#                        help='path to your folder of images for self-supervised learning')

# args = parser.parse_args()

# constants

BATCH_SIZE = 32
EPOCHS     = 1000
LR         = 3e-4
NUM_GPUS   = 2
IMAGE_SIZE = 256
IMAGE_EXTS = ['.jpg', '.png', '.jpeg']
NUM_WORKERS = multiprocessing.cpu_count()

# pytorch lightning module

class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, net, **kwargs):
        super().__init__()
        self.learner = BYOL(net, **kwargs)

    def forward(self, images):
        return self.learner(images)

    def training_step(self, images, _):
        loss = self.forward(images)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)

    def on_before_zero_grad(self, _):
        if self.learner.use_momentum:
            self.learner.update_moving_average()

# images dataset

def expand_greyscale(t):
    return t.expand(3, -1, -1)

class ImagesDataset(Dataset):
    def __init__(self, folder, image_size):
        super().__init__()
        self.folder = folder
        self.paths = []

        for path in Path(f'{folder}').glob('**/*'):
            _, ext = os.path.splitext(path)
            if ext.lower() in IMAGE_EXTS:
                self.paths.append(path)

        print(f'{len(self.paths)} images found')

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(expand_greyscale)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        img = img.convert('RGB')
        return self.transform(img)


######################################
######################################
    
    

totensor: Callable = torchvision.transforms.ToTensor()

def toPIL(img: torch.Tensor, mode="RGB") -> Callable:
    return torchvision.transforms.ToPILImage(mode)(img)


def normalize_transform(mean = [0.485, 0.456, 0.406],
                        std = [0.229, 0.224, 0.225]) -> Callable:
    return transforms.Normalize(mean=mean,
                                std=std)

def default_train_transforms(image_size: int=224,
                             normalize: bool=True, 
                             augment:bool=True,
                             grayscale: bool=True,
                             channels: Optional[int]=3,
                             mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225]):
    """Subclasses can override this or user can provide custom transforms at runtime"""
    transform_list = []
#         transform_jit_list = []
    resize_PIL = not augment
    if augment:
        transform_list.extend([transforms.RandomResizedCrop(size=image_size,
                                                            scale=(0.25, 1.2),
                                                            ratio=(0.7, 1.3),
                                                            interpolation=2),
                               totensor
                             ])
    return default_eval_transforms(image_size=image_size,
                                        normalize=normalize,
                                        resize_PIL=resize_PIL,
                                        grayscale=grayscale,
                                        channels=channels,
                                        transform_list=transform_list,
                                        mean=mean,
                                        std=std)

def default_eval_transforms(image_size: int=224,
                            image_buffer_size: int=32,
                            normalize: bool=True,
                            resize_PIL: bool=True,
                            grayscale: bool=True,
                            channels: Optional[int]=3,
                            transform_list: Optional[List[Callable]]=None,
                            mean = [0.485, 0.456, 0.406],
                            std = [0.229, 0.224, 0.225]):
    """Subclasses can override this or user can provide custom transforms at runtime"""
    transform_list = transform_list or []
    transform_jit_list = []

    if resize_PIL:
        # if True, assumes input images are PIL.Images (But need to check if this even matters.)
        # if False, expects input images to already be torch.Tensors
        transform_list.extend([transforms.Resize(image_size+image_buffer_size),
                               transforms.CenterCrop(image_size),
                               totensor])
    if normalize:
        transform_jit_list.append(normalize_transform(mean, std))

    if grayscale:
        transform_jit_list.append(transforms.Grayscale(num_output_channels=channels))

    return transforms.Compose([*transform_list, *transform_jit_list])


def get_default_transforms(image_size: int=224,
                           normalize: bool=True,
                           augment:bool=True,
                           grayscale: bool=True,
                           channels: Optional[int]=3,
                           mean = [0.485, 0.456, 0.406],
                           std = [0.229, 0.224, 0.225]):

    
    train_transform = default_train_transforms(image_size=image_size,
                                               normalize=normalize,
                                               augment=augment,
                                               grayscale=grayscale,
                                               channels=channels,
                                               mean=mean,
                                               std=std)
    eval_transform = default_eval_transforms(image_size=image_size,
                                             image_buffer_size=32,
                                             normalize=normalize,
                                             resize_PIL=True, #not augment,
                                             grayscale=grayscale,
                                             channels=channels,
                                             transform_list=None,
                                             mean=mean,
                                             std=std)
    
    
    
    return train_transform, eval_transform







# train_transform, val_transform = get_default_transforms(image_size=224,
#                                                          normalize=True,
#                                                          augment=True,
#                                                          grayscale=True,
#                                                          channels=3,
#                                                          mean = [0.485, 0.456, 0.406],
#                                                          std = [0.229, 0.224, 0.225])    
    

# class UnsupervisedDatasetWrapper(torchdata.datasets.Files):
    
#     def __init__(self, dataset):
        
#         self.dataset = dataset
        
#     def __getitem__(self, index):
#         return self.dataset[index][0]
    
#     def __len__(self):
#         return len(self.dataset)

# class UnsupervisedDatasetWrapper(torchdata.datasets.Files):
# class UnsupervisedDatasetWrapper(torchvision.datasets.ImageFolder):
class UnsupervisedDatasetWrapper(CSVDataset):
    
    def __init__(self, dataset):
        self.dataset = dataset
        super().__init__(samples_df=samples_df.samples)
        
        
    def __getitem__(self, index):
        return self.dataset[index][0]
    
    def __len__(self):
        return len(self.dataset)
    
    def __repr__(self):
        out = "<UnsupervisedDatasetWrapper>\n"
        out += self.dataset.__repr__()
        return out

#     if "source_root_dir" not in config:
#         config.source_root_dir = CSV_CATALOG_DIR_V1_0
#     if "experiment_dir" not in config:
#         config.experiment_root_dir = "/media/data/jacob/GitHub/lightning-hydra-classifiers/notebooks/experiments_August_2021"
#     if "experiment_name" not in config:
#         config.experiment_name = "Extant-to-PNAS-512-transfer_benchmark"


# main

def main(args):
    
    exp = TransferExperiment()
    task_0, task_1 = exp.get_multitask_datasets()
    
    for subset in ["train","val","test"]:
        task_0[subset] = UnsupervisedDatasetWrapper(task_0[subset])
        task_1[subset] = UnsupervisedDatasetWrapper(task_1[subset])
    
#     ds = ImagesDataset(args.image_folder, IMAGE_SIZE)
    train_loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

    model = SelfSupervisedLearner(
        resnet,
        image_size = IMAGE_SIZE,
        hidden_layer = 'avgpool',
        projection_size = 256,
        projection_hidden_size = 4096,
        moving_average_decay = 0.99
    )

    trainer = pl.Trainer(
        gpus = NUM_GPUS,
        max_epochs = EPOCHS,
        accumulate_grad_batches = 1,
        sync_batchnorm = True
    )

    trainer.fit(model, train_loader)










# if __name__ == '__main__':




















# def run_sweep(trial_number: int=0):
    
#     trials = [dict(DATASET_NAME="Extant_family_10_512",
#                                      MODEL_NAME="resnet50",
#                                      batch_size=20,
#                                      image_size=(512,512),
#                                      channels=3),
#              dict(DATASET_NAME="PNAS_family_100_512",
#                                      MODEL_NAME="resnet50",
#                                      batch_size=16,
#                                      image_size=(512,512),
#                                      channels=3),
#              dict(DATASET_NAME="Extant_family_10_512",
#                                      MODEL_NAME="wide_resnet50_2",
#                                      batch_size=12,
#                                      image_size=(512,512),
#                                      channels=3),
#              dict(DATASET_NAME="PNAS_family_100_512",
#                                      MODEL_NAME="wide_resnet50_2",
#                                      batch_size=12,
#                                      image_size=(512,512),
#                                      channels=3)]
    
#     config_overrides = trials[trial_number]
    
#     print(f'Using trial #{trial_number} with config:\n{config_overrides}')
    
#     cli_main(config_overrides = config_overrides)
    
    
    
    
# if __name__ == "__main__":
    
#     import argparse
#     parser = argparse.ArgumentParser(description="Run 1 of 4 trials")
#     parser.add_argument("--trial_number", dest="trial_number", type=int, help="which config")
    
#     args = parser.parse_args()
    
#     run_sweep(trial_number=args.trial_number)
    