"""
lightning_hydra_classifiers/utils/experiment_utils.py


Contains common experiment utils for use in multiple scripts, including:




Created on: Wednesday, November 3rd, 2021
Author: Jacob A Rose


"""

import argparse
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numbers
from typing import Union, List, Any, Tuple, Dict, Optional, Sequence
import collections
import seaborn as sns
from sklearn.model_selection import train_test_split
import json
import torchdata
import hydra
from omegaconf import OmegaConf, DictConfig

import pytorch_lightning as pl
from lightning_hydra_classifiers.utils import template_utils
from lightning_hydra_classifiers.utils.plot_utils import colorbar
from lightning_hydra_classifiers.utils.dataset_management_utils import LabelEncoder
from lightning_hydra_classifiers.models.transfer import LightningClassifier

log = template_utils.get_logger(__name__)
log.setLevel("DEBUG")

__all__ = ["load_data", "resolve_config", "configure_callbacks", "configure_loggers", "configure_trainer", "configure_model"]




def resolve_config(cfg: DictConfig) -> Dict[str, Any]:
    """
    
    """
    try:
        config = cfg
        config = asdict(config)
    except TypeError:
        config = OmegaConf.to_container(config, resolve=True)
    finally:
        config = dict(config)
    return config



def configure_callbacks(config) -> List[pl.Callback]:
    callbacks: List[pl.Callback] = []
    for k, cb_conf in config.callbacks.items():
        if "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))
    return callbacks


def configure_loggers(config) -> List[pl.loggers.LightningLoggerBase]:
    logger: List[pl.loggers.LightningLoggerBase] = []
    for _, lg_conf in config["logger"].items():
        if "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


def configure_trainer(config,
                      callbacks=None,
                      logger=None) -> pl.Trainer:
    
    ckpt_paths = [os.path.join(config.checkpoint_dir, ckpt) for ckpt in os.listdir(config.checkpoint_dir)]
    if len(ckpt_paths) and os.path.exists(ckpt_paths[-1]):
        print(f"Found {ckpt_paths[-1]}")
        config.resume_from_checkpoint = ckpt_paths[-1]
    
    trainer_config = resolve_config(config.trainer)
    trainer_config['callbacks'] = callbacks
    trainer_config['logger'] = logger
    trainer: pl.Trainer = hydra.utils.instantiate(trainer_config)
    return trainer


def configure_model(config: argparse.Namespace,
                    label_encoder: Optional[LabelEncoder]=None
                   ) -> Tuple[LightningClassifier, argparse.Namespace]:

    model, config.model = build_model_or_load_from_checkpoint(ckpt_path=config.model.ckpt_path,
                                                              ckpt_dir=config.model.ckpt_dir,
                                                              ckpt_mode=config.model.ckpt_mode,
                                                              config=config.model)
    if hasattr(model, "label_encoder"):
        label_encoder = model.label_encoder
    if label_encoder is not None:
        model.label_encoder = label_encoder

    return model, config


def load_data(config: argparse.Namespace,
              task_id: int=0
             ) -> "DataModule":
    
    if not getattr(config.data, "experiment", [None]):
        config.data.experiment = None

    if config.debug == True:
        log.warning(f"Debug mode activated, loading CIFAR10 datamodule")
        datamodule = CIFAR10DataModule(task_id=task_id,
                                       batch_size=config.data.batch_size,
                                       image_size=config.data.image_size,
                                       image_buffer_size=config.data.image_buffer_size,
                                       num_workers=config.data.num_workers,
                                       pin_memory=config.data.pin_memory,
                                       experiment_config=config.data.experiment)
    else:
#         print(f"Creating datamodule: config=")
#         pp(OmegaConf.to_container(config, resolve=True))
        datamodule = MultiTaskDataModule(task_id=task_id,
                                         batch_size=config.data.batch_size,
                                         image_size=config.data.image_size,
                                         image_buffer_size=config.data.image_buffer_size,
                                         num_workers=config.data.num_workers,
                                         pin_memory=config.data.pin_memory,
                                         experiment_config=config.data.experiment)
    datamodule.setup()
    return datamodule







# def load_data_and_model(config: argparse.Namespace, 
#                         task_id: int=0,
#                         ckpt_path: Optional[str]="") -> Tuple["DataModule", LitMultiTaskModule]:

#     datamodule = load_data(config=config,
#                            task_id=task_id)
    

#     config.model.num_classes = datamodule.num_classes
#     for task_id_idx in range(len(config.model.multitask)):
#         task_name = config.system.task_ids[task_id_idx]
#         datamodule.set_task(task_id_idx)
#         datamodule.setup("fit")
#         config.model.multitask[task_name].num_classes = datamodule.num_classes
        
#     datamodule.set_task(task_id)
#     datamodule.setup()
        
#     task_name = config.system.task_ids[task_id]
#     if ckpt_path in [None, ""]:
#         config.model.ckpt_path = os.path.join(config.system.tasks[task_name].model_ckpt_dir, "model.ckpt")
#     else:
#         config.model.ckpt_path = ckpt_path
#     model = load_model(config=config,
#                        task_id=task_id,
#                        ckpt_path=ckpt_path)
#     model.label_encoder = datamodule.label_encoder

#     return datamodule, model, config






#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################


def plot_class_distributions(targets: List[Any], 
                             sort_by: Optional[Union[str, bool, Sequence]]="count",
                             ax=None,
                             xticklabels: bool=True,
                             hist_kwargs: Optional[Dict]=None):
    """
    Example:
        counts = plot_class_distributions(targets=data.targets, sort=True)
    """
    
    counts = compute_class_counts(targets,
                                  sort_by=sort_by)
                        
    keys = list(counts.keys())
    values = list(counts.values())

    if ax is None:
        plt.figure(figsize=(20,12))
    ax = sns.histplot(x=keys, weights=values, discrete=True, ax=ax, **hist_kwargs)
    plt.sca(ax)
    if xticklabels:
        xtick_fontsize = "medium"
        if len(keys) > 100:
            xtick_fontsize = "x-small"
        elif len(keys) > 75:
            xtick_fontsize = "small"
        plt.xticks(
            rotation=90, #45, 
            horizontalalignment='right',
            fontweight='light',
            fontsize=xtick_fontsize
        )
        if len(keys) > 100:
            for label in ax.xaxis.get_ticklabels()[::2]:
                label.set_visible(False)
        
    else:
        ax.set_xticklabels([])
    
    return counts


#############################################################
#############################################################


def plot_split_distributions(data_splits: Dict[str, "CommonDataset"],
                             use_one_axis: bool=False,
                             hist_kwargs: Optional[Dict]=None):
    """
    Create 3 vertically-stacked count plots of train, val, and test dataset class label distributions
    
    Arguments:
        data_splits: Dict[str, "CommonDataset"],
            Dictionary mapping str split names to Dataset objects that at least have a Dataset.targets attribute for labels.
        use_one_axis: bool=False
            If true, Plot all subsets to the same axis overlayed on top of each other. If False, plot them in individual subplots in the same figure.
        hist_kwargs: Optional[Dict]=None
            Optional additional kwargs to be passed to sns.histplot(**hist_kwargs)
    
    """
    assert isinstance(data_splits, dict)
    num_splits = len(data_splits)
    
    if use_one_axis:
        rows, cols = 1, 1
        fig, ax = plt.subplots(rows, cols, figsize=(20*cols,10*rows))
        ax = [ax]*num_splits
    else:
        if num_splits <= 3:
            rows = num_splits
            cols = 1
        else:
            rows = int(num_splits // 2)
            cols = int(num_splits % 2)
            
        fig, ax = plt.subplots(rows, cols, figsize=(20*cols,10*rows))    
        if hasattr(ax, "flatten"):
            ax = ax.flatten()
    
    train_key = [k for k,v in data_splits.items() if "train" in k]
    sort_by = True
    if len(train_key)==1:
        sort_by = compute_class_counts(data_splits[train_key[0]].targets,
                                       sort_by="count")
        log.info(f'Sorting by count for {train_key} subset, and applying order to all other subsets')
#         log.info(f"len(sort_by)={len(sort_by)}")

    num_classes = len(set(list(data_splits.values())[0].targets))    
    xticklabels=False
    num_samples = 0
    counts = {}
    for i, (k, v) in enumerate(data_splits.items()):
        if i == num_splits-1:
            xticklabels=True
        counts[k] = plot_class_distributions(targets=v.targets, 
                                             sort_by=sort_by,
                                             ax = ax[i],
                                             xticklabels=xticklabels,
                                             hist_kwargs=hist_kwargs)
        num_nonzero_classes = len([name for name, count_i in counts[k].items() if count_i > 0])
        
        title = f"{k} [n={len(v)}"
        if num_nonzero_classes < num_classes:
            title += f", num_classes@(count > 0) = {num_nonzero_classes}-out-of-{num_classes} classes in dataset"
        title += "]"
        plt.gca().set_title(title, fontsize='large')
        
        num_samples += len(v)
    
    suptitle = '-'.join(list(data_splits.keys())) + f"_splits (total samples={num_samples}, total classes = {num_classes})"
    
    plt.suptitle(suptitle, fontsize='x-large')
    plt.subplots_adjust(bottom=0.1, top=0.94, wspace=None, hspace=0.08)
    
    return fig, ax


#############################################################
#############################################################



#############################################################
#############################################################



def compute_class_counts(targets: Sequence,
                         sort_by: Optional[Union[str, bool, Sequence]]="count"
                        ) -> Dict[str, int]:
    
    counts = collections.Counter(targets)
#     if hasattr(sort_by, "__len__"):
    if isinstance(sort_by, dict):
        counts = {k: counts[k] for k in sort_by.keys()}
    if isinstance(sort_by, list):
        counts = {k: counts[k] for k in sort_by}
    elif (sort_by == "count"):
        counts = dict(sorted(counts.items(), key = lambda x:x[1], reverse=True))
    elif (sort_by is True):
        counts = dict(sorted(counts.items(), key = lambda x:x[0], reverse=True))
        
    return counts

#############################################################
#############################################################


