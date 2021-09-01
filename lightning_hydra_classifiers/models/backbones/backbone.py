"""
lightning_hydra_classifiers/models/backbone.py

Author: Jacob A Rose
Created: Saturday May 29th, 2021

"""

from typing import Any, List, Optional, Dict, Tuple

import torch
from torch import nn, functional as F
import torchvision
from pytorch_lightning import LightningModule

import torchmetrics as metrics
# from pytorch_lightning import metrics
#.classification import Accuracy
import timm
from torchsummary import summary
import pandas as pd
from pathlib import Path
from stuf import stuf
import wandb
import numpy as np
from rich import print as pp
from . import resnet, senet, efficientnet


AVAILABLE_MODELS = {"resnet":resnet.AVAILABLE_MODELS,
                    "senet":senet.AVAILABLE_MODELS,
                    "efficientnet":efficientnet.AVAILABLE_MODELS}





def build_model(model_name: str,
                pretrained: bool=False,
                progress: bool=True,
                num_classes: int=1000,
                **kwargs) -> nn.Module:

#     if 'resnet' in model_name:
    if model_name in resnet.AVAILABLE_MODELS:
        model = resnet.build_model(model_name=model_name,
                                  pretrained=pretrained,
                                  progress=progress,
                                  num_classes=num_classes,
                                  **kwargs)

#     elif 'senet' in model_name:
    elif model_name in senet.AVAILABLE_MODELS:
        model = senet.build_model(model_name=model_name,
                                  pretrained=pretrained,
                                  progress=progress,
                                  num_classes=num_classes,
                                  **kwargs)

#     elif 'efficientnet' in model_name:
    elif model_name in efficientnet.AVAILABLE_MODELS:
        model = efficientnet.build_model(model_name=model_name,
                                         pretrained=pretrained,
                                         progress=progress,
                                         num_classes=num_classes,
                                         **kwargs)
    
    else:
        print(f"model with name {model_name} has not be implemented yet.")
        print("Available Models:")
        pp(AVAILABLE_MODELS)
        return None
    
    
    model.name = model_name
    model.pretrained = pretrained
    
    print(f"[BUILDING MODEL] build_model({model_name}, pretrained={pretrained})")
    
    return model
    
#     model.name = model_name
#     model.pretrained = pretrained
#     return model










# 3.a Optional: Register a custom backbone
# This is useful to create new backbone and make them accessible from `ImageClassifier`
# @ImageClassifier.backbones(name="resnet18")
def fn_resnet(pretrained: bool = True):
    model = torchvision.models.resnet18(pretrained)
    # remove the last two layers & turn it into a Sequential model
    backbone = nn.Sequential(*list(model.children())[:-2])
    num_features = model.fc.in_features
    # backbones need to return the num_features to build the head
    return backbone, num_features


def create_classifier(num_features: int, num_classes: int, pool_type='avg', bias: bool=True):
    global_pool = nn.AdaptiveAvgPool2d(1)
    flatten_layer = nn.Flatten()
    linear_layer = nn.Linear(num_features, num_classes, bias=bias)
    return [global_pool, flatten_layer, linear_layer]

