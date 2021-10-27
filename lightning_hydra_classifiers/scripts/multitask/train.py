#!/usr/bin/env python
# coding: utf-8
# Script: "scripts/multitask/train.py"
# Previously: train_multi-task.py
# 
# Based on the notebook: `multi-task_model-train.ipynb`

# 
# End of August attempts to create good model training workflows for multi-task experiments
# 
# Author: Jacob A Rose  
# Created on: Monday August 29th, 2021
# Updated on: Friday September 4th, 2021

"""

Run default experiment:

>> python "/media/data/jacob/GitHub/lightning-hydra-classifiers/lightning_hydra_classifiers/scripts/multitask/train.py" --info


>> python "/media/data/jacob/GitHub/lightning-hydra-classifiers/lightning_hydra_classifiers/scripts/multitask/train.py" --list_available_backbones


>> python "/media/data/jacob/GitHub/lightning-hydra-classifiers/lightning_hydra_classifiers/scripts/multitask/train.py" --gpus "7" -e 120 -res 512 -buffer 32 -nproc 4 -model efficientnet_b3 -bsz 64 -init_freeze "layer4" -pre "imagenet"


python "/media/data/jacob/GitHub/lightning-hydra-classifiers/lightning_hydra_classifiers/scripts/multitask/train.py" --gpus "7" -e 10 -res 512 -buffer 32 -nproc 4 -model efficientnet_b3 -bsz 64 -init_freeze "layer4" -pre "imagenet" --experiment="Extant-to-Fossil-512-transfer_benchmark"


    })
    --experiment="Extant-to-PNAS-512-transfer_benchmark"
    --experiment="Extant-to-Fossil-512-transfer_benchmark"

"""



import numpy as np
import collections
import os
import sys
if 'TOY_DATA_DIR' not in os.environ:
    os.environ['TOY_DATA_DIR'] = "/media/data_cifs/projects/prj_fossils/data/toy_data"
default_root_dir = os.environ['TOY_DATA_DIR']
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 200)

import torch
from torch import nn
import torchvision
from torchvision import transforms
import pytorch_lightning as pl
import timm
# torch.backends.cudnn.benchmark = True

from dataclasses import asdict
from rich import print as pp
import matplotlib.pyplot as plt
from munch import Munch
import argparse
import json
from omegaconf import OmegaConf
from typing import Tuple, Union, List, Optional
# from lightning_hydra_classifiers.data.utils.make_catalogs import *
from lightning_hydra_classifiers.utils.dataset_management_utils import Extract
from lightning_hydra_classifiers.utils.metric_utils import get_per_class_metrics, get_scalar_metrics
# from lightning_hydra_classifiers.utils.logging_utils import get_wandb_logger
from lightning_hydra_classifiers.utils.callback_utils import get_wandb_logger
import wandb
from lightning_hydra_classifiers.experiments.transfer_experiment import TransferExperiment, TransferExperimentConfig, Extant_to_PNAS_ExperimentConfig, Extant_to_Fossil_ExperimentConfig
# from lightning_hydra_classifiers.models.backbones import backbone
# from torchinfo import summary
# model_stats = summary(your_model, (1, 3, 28, 28), verbose=0)
# from lightning_hydra_classifiers.utils.common_utils import LabelEncoder

from lightning_hydra_classifiers.experiments.multitask.datamodules import MultiTaskDataModule
from lightning_hydra_classifiers.experiments.multitask.modules import LitMultiTaskModule#, AdamWOptimizerConfig
from lightning_hydra_classifiers.experiments.reference_transfer_experiment import CIFAR10DataModule
from lightning_hydra_classifiers.utils.callback_utils import ImagePredictionLogger
from lightning_hydra_classifiers.utils.template_utils import get_logger

from lightning_hydra_classifiers.experiments.configs.config import MultiTaskExperimentConfig
from lightning_hydra_classifiers.experiments.configs.model import *
from lightning_hydra_classifiers.experiments.configs.trainer import *
from dataclasses import dataclass
from lightning_hydra_classifiers.scripts.pretrain import lr_tuner
from lightning_hydra_classifiers.utils.dataset_management_utils import ETL
# from pl_bolts.callbacks import TrainingDataMonitor
import pl_bolts

import hydra
from hydra.core.config_store import ConfigStore
############################################
logging = get_logger(name=__name__)
########################################################
########################################################
########################################################




def load_data(config: argparse.Namespace,
              task_id: int=0
             ) -> "DataModule":
    
    if not getattr(config.data, "experiment", [None]):
        config.data.experiment = None

    if config.debug == True:
        logging.warning(f"Debug mode activated, loading CIFAR10 datamodule")
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



def load_model(config: argparse.Namespace,
               task_id: int=0,
               ckpt_path: Optional[str]="") -> LitMultiTaskModule:

#     task_id = config.system.task_ids[task_id]

#     config.model.ckpt_path = str(config.system.tasks[task_id].model_ckpt_path)
#     ckpt_path = config.model.ckpt_path
    if os.path.isfile(ckpt_path):
        logging.info(f"Loading from model checkpoint: {str(ckpt_path)}")
        model = LitMultiTaskModule.load_from_checkpoint(ckpt_path, config=config.model, strict=False)
    else:
        if isinstance(ckpt_path, (str, Path)):
            logging.warning(f"User specified checkpoint path doesn't exist. Best checkpoint produced during training will be copied to that location: {ckpt_path}")
        logging.info(f"Instantiating model with hparams:")
        logging.info(OmegaConf.to_container(config.model, resolve=True))
        model = LitMultiTaskModule(config.model)

    return model



def load_data_and_model(config: argparse.Namespace, 
                        task_id: int=0,
                        ckpt_path: Optional[str]="") -> Tuple["DataModule", LitMultiTaskModule]:

    datamodule = load_data(config=config,
                           task_id=task_id)
    

    config.model.num_classes = datamodule.num_classes
    for task_id_idx in range(len(config.model.multitask)):
        task_name = config.system.task_ids[task_id_idx]
        datamodule.set_task(task_id_idx)
        datamodule.setup("fit")
        config.model.multitask[task_name].num_classes = datamodule.num_classes
        
    datamodule.set_task(task_id)
    datamodule.setup()
        
    task_name = config.system.task_ids[task_id]
    if ckpt_path in [None, ""]:
        config.model.ckpt_path = os.path.join(config.system.tasks[task_name].model_ckpt_dir, "model.ckpt")
    else:
        config.model.ckpt_path = ckpt_path
    model = load_model(config=config,
                       task_id=task_id,
                       ckpt_path=ckpt_path)
    model.label_encoder = datamodule.label_encoder

    return datamodule, model, config

######################################


def configure_trainer(config, callbacks=None, logger=None) -> pl.Trainer:
    trainer_config = resolve_config(config.trainer)

    trainer_config['callbacks'] = callbacks
    trainer_config['logger'] = logger
    print(f"trainer_config.gpus={trainer_config['gpus']}")
    print(f"config.data.batch_size={config.data.batch_size}")
    trainer: pl.Trainer = hydra.utils.instantiate(trainer_config)
    return trainer






def resolve_config(cfg):
    try:
        config = cfg
        config = asdict(config)
    except TypeError:
        config = OmegaConf.to_container(config, resolve=True)
    finally:
        config = dict(config)
    return config


######################################


def configure_callbacks(config) -> List[pl.Callback]:
    callbacks: List[pl.Callback] = []
    for k, cb_conf in config.callbacks.items():
        if "_target_" in cb_conf:
            logging.info(f"Instantiating callback <{cb_conf._target_}>")
#             if k == "image_prediction_logger":
#                 callbacks.append(hydra.utils.instantiate(cb_conf, datamodule=datamodule))
#             else:
            callbacks.append(hydra.utils.instantiate(cb_conf))
    return callbacks


def configure_loggers(config) -> List[pl.loggers.LightningLoggerBase]:
    logger: List[pl.loggers.LightningLoggerBase] = []
    for _, lg_conf in config["logger"].items():
        if "_target_" in lg_conf:
            logging.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


def train_task(config: argparse.Namespace, task_id: int=0):
    pl.seed_everything(config.seed)

    datamodule, model, config = load_data_and_model(config=config, task_id=task_id)
    

    group=f'{config.model.backbone.backbone_name}_{config.data.experiment.experiment_name}_task_{task_id}'
    config.logger.wandb.group = group
    config.callbacks.log_per_class_metrics_to_wandb.class_names = datamodule.classes

    callbacks = configure_callbacks(config)
    logger = configure_logger(config)

    trainer: pl.Trainer = configure_trainer(config, callbacks=callbacks, logger=logger)

    if config.debug == True:
        import pdb; pdb.set_trace()

    if config.pretrain is not None:

        logging.info(f"[Initiating Stage] lr_tuner")
        suggestion, lr_tuner_results, config = lr_tuner.run_lr_tuner(trainer=trainer,
                                                                     model=model,
                                                                     datamodule=datamodule,
                                                                     config=config,
                                                                     results_dir = config.system.tasks[f"task_{task_id}"].lr_tuner_dir,
                                                                     group=group)
    
        print(f"model.lr={model.lr}")
        model.lr = suggestion['lr']
        print(f"model.lr={model.lr}")
        model.config.optimizer.lr = suggestion['lr']
    
    print(f"model.config.lr={model.config.lr}")
    with wandb.init(job_type = "supervised_train",
                    config=resolve_config(config),
                    group=group, #f'{config.model.model_name}_task_{task_id}',
                    reinit=True) as run:

    #         wandb.watch(model, log='all')
#         trainer.logger = wandb_logger
#         wandb.watch(model.backbone, log='all')
#         wandb.watch(model.classifier, log='all')

        try:
            trainer.fit(model, datamodule)
        except KeyboardInterrupt as e:
            logging.warning("Interruption:", e)
        finally:
            checkpoint_callback = [c for c in callbacks if isinstance(c, pl.callbacks.ModelCheckpoint)][0]
            logging.info(f"checkpoint_callback.best_model_path: {str(checkpoint_callback.best_model_path)}")
            config.system.tasks[f"task_{task_id}"].ckpt_path = checkpoint_callback.best_model_path
            checkpoint_callback.best_model_score = checkpoint_callback.best_model_score or 0.0
            logging.info(f"checkpoint_callback.best_model_score: {checkpoint_callback.best_model_score:.3f}")
        logging.info(f"[Initiating TESTING on task_{task_id}]")


        test_results = run_multitask_test(trainer=trainer,
                                          model=model,
                                          datamodule=datamodule,
                                          config=config,
                                          tasks="all")#,
    #                                       results_path=results_path)


    logging.info(f"[FINISHED] TESTING on task_{task_id}")
    logging.info(f"Results: {test_results}")
    return test_results, config



def run_multitask_test(trainer: pl.Trainer,
                       model: pl.LightningModule,
                       datamodule: pl.LightningDataModule,
                       config: argparse.Namespace=None,
                       tasks: Union[str, List[int]]="all",
                       results_path: str=None,
                       run=None):
    if tasks == "all":
        tasks = list(range(len(datamodule.tasks)))
    
    test_results = {}
#     fig, ax = plt.subplots(1,len(tasks))
#     if not isinstance(ax, list):
#         ax = [ax]
    for task_id in tasks:
#         datamodule.set_task(task_id)
        tag=config.system.tasks[f"task_{task_id}"].task_id
#         model.init_metrics(stage='test', tag=tag)
        trainer.logger = pl.loggers.CSVLogger("logs", name=tag)
        datamodule.setup(task_id=task_id)
        logging.info(f"[TESTING] {tag}")

        test_results[task_id] = trainer.test(model, datamodule=datamodule)
        
    return test_results
    

import hydra

@hydra.main(config_name="config", config_path="../../../configs") #"multitask_experiment_config")
def main(config):

#     config.root_dir = "/media/data_cifs/projects/prj_fossils/users/jacob/experiments/July2021-Nov2021/experiment_logs/Transfer_Experiments"
#     pp(f"BEFORE")

    pl.seed_everything(config.seed)
    OmegaConf.set_struct(config, False)
#     config = ETL.init_structured_config(cfg=config,
#                                      dataclass_type = MultiTaskExperimentConfig)
    pp(OmegaConf.to_yaml(config, resolve=True))
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.callbacks.model_checkpoint.dirpath, exist_ok=True)
    
    os.environ["WANDB_ENTITY"] = "jrose"
    os.environ["WANDB_PROJECT"] = "image_classification_train"
    os.environ["WANDB_DIR"] = config.experiment_dir
        
#     torch.backends.cudnn.benchmark = True
#     torch.backends.cudnn.enabled = True
    results = {}
    results["task_0"], config = train_task(config=config, task_id=0)
    config.model.ckpt_path = config.system.tasks.task_0.ckpt_path
    
    if "task_1" in config.system.tasks.keys():
        print(f"[Initiating] Transfer to Task_1: {config.system.tasks.task_1.task_id}")
        print(f"[Re-loading model from checkpoint path] : {config.model.ckpt_path}")
        #TODO Log/Cache experiment artifacts here.    
        results["task_1"], config = train_task(config=config, task_id=1)
        
    print(f"[SUCCESSFULLY FINISHED TRAIN.PY]")
    import pdb; pdb.set_trace()
    print(type(results))
    
    print(type(results['task_1']))
    
    torch.save(results, str(Path(config.experiment_dir, "test_results.pth")))
    logging.info(json.dumps(results))
    if os.path.isfile(str(Path(config.experiment_dir, "test_results.pth"))):
        logging.info(f'Congratulations, your test results are pickled using torch.save to: {str(Path(config.experiment_dir, "test_results.pth"))}')
    else:
        logging.warning(f'[Warning] Saving test results to {str(Path(config.experiment_dir, "test_results.pth"))} failed')







        
if __name__ == '__main__':
    
    main()





#####################################################################
#####################################################################


# END

#####################################################################
#####################################################################
