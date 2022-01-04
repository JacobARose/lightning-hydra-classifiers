"""
lightning_hydra_classifiers/utils/ckpt_utils.py



Created on: Monday, September 13th, 2021
Author: Jacob A Rose


"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import numbers
from typing import Union, List, Any, Tuple, Dict, Optional, Sequence
import collections
from sklearn.model_selection import train_test_split
import json
import torchdata

log = template_utils.get_logger(__name__)


__all__ = []

from collections import namedtuple













from rich import print as pp
import pandas as pd
import numpy as np
import os
from pathlib import Path

# import logging
from lightning_hydra_classifiers.utils.template_utils import get_logger
# logger = logging.Logger(__name__)
logger = get_logger(__name__)
logger.setLevel("DEBUG") # ('INFO')
from tqdm.auto import tqdm, trange
import torch
import torch.nn as nn
import timm
import glob
import hydra
from omegaconf import OmegaConf, DictConfig
from collections import OrderedDict
from typing import *

# from lightning_hydra_classifiers.models.transfer import *
from rich import print as pp
import pytorch_lightning as pl
from lightning_hydra_classifiers.scripts.multitask.train import load_data, resolve_config, configure_callbacks, configure_loggers
from lightning_hydra_classifiers.utils.etl_utils import ETL
from lightning_hydra_classifiers.scripts.pretrain import lr_tuner
# source: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial5/Inception_ResNet_DenseNet.html
from lightning_hydra_classifiers.callbacks.finetuning_callbacks import FinetuningLightningCallback
from lightning_hydra_classifiers.models.transfer import LightningClassifier



__all__ = ["scan_ckpt_dir", "load_results_if_previously_completed", "build_model_or_load_from_checkpoint",
           "lightning_checkpoint_connector", "pretrained_model_checkpoint_connector",
           "initialize_model_from_scratch_connector", "pretrained_model_from_imagenet_connector",
           "pretrained_backbone_w_new_classifier_connector"]



def lightning_checkpoint_connector(ckpt_path: Optional[str]=None,
                                   **kwargs) -> Optional[LightningClassifier]:
    # pretrained_filename = config.trainer.resume_from_checkpoint #config.checkpoint_dir
    if os.path.isfile(str(ckpt_path)):
        print(f"Found pretrained lightning checkpoint model at {ckpt_path}, loading...")
        return LightningClassifier.load_from_checkpoint(ckpt_path, **kwargs) # Automatically loads the model with the saved hyperparameters
    

def pretrained_model_checkpoint_connector(ckpt_path: Optional[str]=None,
                                          **kwargs) -> Optional[LightningClassifier]:
    if os.path.isfile(str(ckpt_path)):
        print(f"Found pretrained custom model checkpoint at {ckpt_path}, loading...")
        return LightningClassifier.load_model_from_checkpoint(ckpt_path, **kwargs)


def initialize_model_from_scratch_connector(**kwargs) -> Optional[LightningClassifier]:
    return LightningClassifier(**kwargs)
        # model.label_encoder = datamodule.label_encoder

def pretrained_model_from_imagenet_connector(**kwargs) -> Optional[LightningClassifier]:
    return LightningClassifier(**kwargs)


def pretrained_backbone_w_new_classifier_connector(ckpt_path: Optional[str]=None,
                                                   new_num_classes: Optional[int]=None,
                                                  **kwargs) -> Optional[LightningClassifier]:
    return LightningClassifier.init_pretrained_backbone_w_new_classifier(ckpt_path,
                                                                         new_num_classes=new_num_classes,
                                                                         **kwargs)
    
    
CKPT_MODES = {"lightning_checkpoint":lightning_checkpoint_connector,
              "pretrained_model_checkpoint":pretrained_model_checkpoint_connector,
              "pretrained_backbone_w_new_classifier":pretrained_backbone_w_new_classifier_connector,
              "initialize_model_from_scratch":initialize_model_from_scratch_connector,
              "pretrained_model_from_imagenet":pretrained_model_from_imagenet_connector}
    
    


def build_model_or_load_from_checkpoint(ckpt_path: Optional[str]=None,
                                        ckpt_dir: Optional[str]=None,
                                        ckpt_mode: Optional[str]=None,
                                        config=None) -> Tuple[LightningClassifier, argparse.Namespace]:
    """
    Build a LightningClassifier model or load from a checkpoint.
    
    Arguments:
        ckpt_path: Optional[str]=None
            If ckpt_path exists, attempts to load it first. If successful, ignores ckpt_dir.
        ckpt_dir: Optional[str]=None
            If ckpt_path is None, scans ckpt_dir and attempts to load last checkpoint.
        ckpt_mode: Optional[str]=None
            Specify preferred ckpt_mode, if this fails then cycles through all other ckpt_modes and uses the first one that works.
        config
            Should be the model config (pass in config.model from main experiment/script config)
    
    """
    if hasattr(config, "model"):
        config = config.model
    if os.path.isdir(str(ckpt_dir)) and (not os.path.isfile(str(ckpt_path))):
        ckpt_paths, ckpt_path = scan_ckpt_dir(ckpt_dir)
    config.update({"ckpt_path":ckpt_path})
    
    if ckpt_mode in CKPT_MODES:
        try:
            model = CKPT_MODES[ckpt_mode](**config)
        except Exception as e:
            print(e, f"Chosen ckpt_mode={ckpt_mode} did not work, cycling through other options.")
            
    else:
        for ckpt_mode in CKPT_MODES:
            try:
                model = CKPT_MODES[ckpt_mode](**config)
            except Exception as e:
                print(e, f"Chosen ckpt_mode={ckpt_mode} did not work, cycling through remaining options.")
                
    return model, config


def load_results_if_previously_completed(config) -> Optional[DictConfig]:
    """
    Checks if a results.yaml file has previously been saved in order to circumvent previously completed trials.
    
    1. First looks in config.results_dir for a file named "results.yaml"
    2. If none exists, looks for a file optionally specified in config.source.results_filepath
    3. If both 1 and 2 fail, returns nothing.
    """
    if os.path.isfile(os.path.join(config.results_dir, "results.yaml")):
        results_file_path = os.path.join(config.results_dir, "results.yaml")
        results = OmegaConf.load(results_file_path)
        print(f"Found pre-existing results saved to file: {results_file_path}")
        print(f"Results:"); pp(results)
        return results
    
    results_file_path = str(config.source.get("results_filepath"))
    if os.path.isfile(results_file_path):
        results = OmegaConf.load(results_file_path)
        print(f"Found results from source training stage saved to file: {results_file_path}")
        print(f"Results:"); pp(results)
        return results
    
        
def scan_ckpt_dir(ckpt_dir):
    ckpt_paths = [os.path.join(ckpt_dir, ckpt) for ckpt in os.listdir(ckpt_dir)]
    if len(ckpt_paths):
        print(f"Found {len(ckpt_paths)} ckpts:" + "\n" + f"{ckpt_paths}")
        last_ckpt = ckpt_paths[-1]
        return ckpt_paths, last_ckpt
    return ckpt_paths, None






###################################
###################################


# _STATE_DICT_KEY = "state_dict"
# def _is_lightning_checkpoint(checkpoint: Dict[str, Any]) -> bool:
#     """Returns true if we believe this checkpoint to be a Lightning checkpoint."""
#     return _STATE_DICT_KEY in checkpoint
# def _is_d2go_checkpoint(checkpoint: Dict[str, Any]) -> bool:
#     """Returns true if we believe this to be a D2Go checkpoint."""
#     d2_go_keys = [_OLD_STATE_DICT_KEY, "iteration"]
#     for key in d2_go_keys:
#         if key not in checkpoint:
#             return False
#     return True

# def set_requires_grad(model, reg_exps, value):
#     """
#     source: https://github.com/facebookresearch/d2go/blob/main/d2go/modeling/model_freezing_utils.py
#     """
#     total_num_parameters = 0
#     unmatched_parameters = []
#     unmatched_parameter_names = []
#     matched_parameters = []
#     matched_parameter_names = []
#     for name, parameter in model.named_parameters():
#         total_num_parameters += 1
#         matched = False
#         for frozen_layers_regex in reg_exps:
#             if re.match(frozen_layers_regex, name):
#                 matched = True
#                 parameter.requires_grad = value
#                 matched_parameter_names.append(name)
#                 matched_parameters.append(parameter)
#                 break
#         if not matched:
#             unmatched_parameter_names.append(name)
#             unmatched_parameters.append(parameter)
#     logger.info(
#         "Matched layers (require_grad={}): {}".format(value, matched_parameter_names)
#     )
#     logger.info("Unmatched layers: {}".format(unmatched_parameter_names))
#     return matched_parameter_names, unmatched_parameter_names