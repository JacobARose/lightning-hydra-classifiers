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
from torch import nn, optim


from lightning_hydra_classifiers.models.transfer import cli_main


def run_sweep(trial_number: int=0):
    
    trials = [dict(DATASET_NAME="Extant_family_10_512",
                                     MODEL_NAME="resnet50",
                                     batch_size=20,
                                     image_size=(512,512),
                                     channels=3),
             dict(DATASET_NAME="PNAS_family_100_512",
                                     MODEL_NAME="resnet50",
                                     batch_size=16,
                                     image_size=(512,512),
                                     channels=3),
             dict(DATASET_NAME="Extant_family_10_512",
                                     MODEL_NAME="wide_resnet50_2",
                                     batch_size=12,
                                     image_size=(512,512),
                                     channels=3),
             dict(DATASET_NAME="PNAS_family_100_512",
                                     MODEL_NAME="wide_resnet50_2",
                                     batch_size=12,
                                     image_size=(512,512),
                                     channels=3)]
    
    config_overrides = trials[trial_number]
    
    print(f'Using trial #{trial_number} with config:\n{config_overrides}')
    
    cli_main(config_overrides = config_overrides)
    
    
    
    
if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description="Run 1 of 4 trials")
    parser.add_argument("--trial_number", dest="trial_number", type=int, help="which config")
    
    args = parser.parse_args()
    
    run_sweep(trial_number=args.trial_number)
    