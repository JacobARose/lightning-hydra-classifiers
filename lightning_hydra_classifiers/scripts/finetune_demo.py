"""
finetune_demo.py


"""


from rich import print as pp
import pandas as pd
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import time
import os
from pathlib import Path

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import logging

logger = logging.Logger(__name__)
logger.setLevel('INFO')

from tqdm.auto import tqdm, trange

import torch
import torch.nn as nn
import timm
import glob
import hydra
from collections import OrderedDict
from typing import *

from lightning_hydra_classifiers.models.transfer import *
from rich import print as pp
from lightning_hydra_classifiers.utils.model_utils import count_parameters, collect_results
from lightning_hydra_classifiers.utils.metric_utils import get_per_class_metrics, get_scalar_metrics
from lightning_hydra_classifiers.models.backbones.backbone import build_model
import pytorch_lightning as pl
pl.seed_everything(42)

from lightning_hydra_classifiers.scripts.multitask.train import MultiTaskDataModule, LitMultiTaskModule, ImagePredictionLogger, train_task, CIFAR10DataModule, run_multitask_test, load_data_and_model, load_data, resolve_config, configure_callbacks, configure_loggers, configure_trainer
from lightning_hydra_classifiers.data.datasets.common import toPIL
from lightning_hydra_classifiers.utils.etl_utils import ETL
from omegaconf import OmegaConf

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning_hydra_classifiers.scripts.pretrain import lr_tuner
from lightning_hydra_classifiers.scripts.multitask.train import configure_callbacks, configure_loggers#, configure_trainer
# source: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial5/Inception_ResNet_DenseNet.html
from lightning_hydra_classifiers.callbacks.finetuning_callbacks import FinetuningLightningCallback


from lightning_hydra_classifiers.models.transfer import *


def test_model_freeze_strategy(config, datamodule, **kwargs):
    """
    Inputs:
        model_name - Name of the model you want to run. Is used to look up the class in "model_dict"
        save_name (optional) - If specified, this name will be used for creating the checkpoint and logging directory.
    """
    
    config.model.finetuning_strategy = "finetuning_unfreeze_layers_on_plateau"
    
    group=f'{config.model.backbone.backbone_name}_{config.data.experiment.experiment_name}'#_task_{task_id}'
    config.logger.wandb.group = group
    config.callbacks.log_per_class_metrics_to_wandb.class_names = datamodule.classes

    callbacks = configure_callbacks(config)
    
#     callbacks.append(FinetuningLightningCallback(monitor="val_acc",
#                                                  mode="max",
#                                                  patience=1))
#     config.callbacks.update({"finetuning_lightning_callback":{
#             "monitor":"val_acc",
#             "mode":"max",
#             "patience":1
#     }})
    
    for cb in callbacks:
        if isinstance(cb, FinetuningLightningCallback):
            print(f"Confirmed found FinetuningLightningCallback.")
            print(f"cb.monitor={cb.monitor}, cb.mode={cb.mode}, cb.patience={cb.patience}")
    
    logger = configure_loggers(config)
    

    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(default_root_dir=config.checkpoint_dir, #config.experiment_dir,
                         gpus=config.trainer.gpus,
                         max_epochs=config.trainer.max_epochs,
                         callbacks=callbacks,
#                                     ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
#                                     LearningRateMonitor("epoch")],
                         logger=logger,
                         resume_from_checkpoint=config.trainer.resume_from_checkpoint,
                         progress_bar_refresh_rate=10)
#     trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
#     trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = config.trainer.resume_from_checkpoint #config.checkpoint_dir
    if os.path.isfile(str(pretrained_filename)):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        try:
            model = LightningClassifier.load_from_checkpoint(pretrained_filename) # Automatically loads the model with the saved hyperparameters
        except KeyError as e:
            print(e, "Trying to load model weights directly")
            pl.seed_everything(config.model.seed)
            model = LightningClassifier(**config.model, **kwargs)
            
            model.load_state_dict(torch.load(pretrained_filename))
            model.label_encoder = datamodule.label_encoder
            
            
            
    else:
        pl.seed_everything(config.model.seed)
        model = LightningClassifier(**config.model, **kwargs)
        model.label_encoder = datamodule.label_encoder
        
        if config.trainer.auto_lr_find:

            lr_tune_output = lr_tuner.run_lr_tuner(trainer=trainer,
                                                   model=model,
                                                   datamodule=datamodule,
                                                   config=config,
                                                   results_dir=config.lr_tuner_dir,
                                                   group="finetuning_trials")
        
        
        
    trainer.fit(model, datamodule=datamodule)

    model = LightningClassifier.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training
    print(f"Best checkpoint saved to: {trainer.checkpoint_callback.best_model_path}")

    # Test best model on validation and test set
    val_result = trainer.test(model, test_dataloaders=datamodule.val_dataloader(), verbose=False)
    test_result = trainer.test(model, test_dataloaders=datamodule.test_dataloader(), verbose=False)
    
    try:
        result = {"test_acc": test_result[0]["test_acc"], "val_acc": val_result[0]["test_acc"]}
    except Exception as e:
        print(e)
        result = {"test_acc": test_result, "val_acc": val_result}
        
    result["ckpt_path"] = trainer.checkpoint_callback.best_model_path

    return model, result





def get_config_and_load_data(overrides = None,
                             task_id: int = 1,
                             pool_type='avgdrop',
                             finetuning_strategy="feature_extractor_+_bn.eval()",
                             lr=2e-03,
                             dropout_p: float=0.3,
                             max_epochs: int=5):
    overrides = overrides or []    
    config = ETL.load_hydra_config(config_name = "config",
                                   config_path = "/media/data/jacob/GitHub/lightning-hydra-classifiers/configs",
                                   overrides=overrides)
    OmegaConf.set_struct(config, False)
    
    datamodule = load_data(config,
                           task_id=task_id)
    
    
    config.dataset_name = datamodule.dataset_names[f"task_{task_id}"]
    config.classifier_dropout_p = dropout_p

    model_config = OmegaConf.create(dict(
                                    backbone={"backbone_name":config.model.backbone.backbone_name},
                                    backbone_name=config.model.backbone.backbone_name,
                                    pretrained=True,
                                    num_classes=datamodule.num_classes,
                                    pool_type=pool_type,
                                    head_type='linear',
                                    hidden_size=None,
                                    dropout_p=dropout_p,
                                    lr=2e-03,
                                    backbone_lr_mult=0.1,
                                    finetuning_strategy=finetuning_strategy,
                                    weight_decay=0.01,
                                    seed=98))
    config.model = model_config
#     config.trainer.max_epochs = max_epochs
#     config.trainer.auto_lr_find = False
    config.experiment_name = f"{config.model.finetuning_strategy}-{config.dataset_name}-{datamodule.num_classes}_classes-res_{config.data.image_size}-bsz_{config.data.batch_size}-{config.model.backbone_name}-pretrained_{config.model.pretrained}-pool_{config.model.pool_type}"
    
    config.root_dir = os.path.join(os.getcwd(), "bn_unit_test_logs", config.model.pool_type)
    config.lr_tuner_dir = os.path.join(config.results_dir, f"task_{task_id}", "lr_tuner")
    
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.lr_tuner_dir, exist_ok=True)
    return config, datamodule


# model = LightningClassifier(**config.model)
# model = TestLightningClassifier(**config.model)
# model.label_encoder = datamodule.label_encoder




# valid_strategies = ("finetuning_unfreeze_layers_on_plateau")
# pool_types = ("avg", "avgdrop")#, "avgmax", "max", "avgmaxdrop")

# finetuning_strategy="feature_extractor"
# finetuning_strategy="feature_extractor_+_bn.eval()"

# pool_type='avgdrop'
# pool_type='avgmaxdrop'

def train(task_id: int=1, strategy="finetuning_unfreeze_layers_on_plateau"):

    pool_type="avgdrop"
    dropout_p = 0.1

    all_results = {}

    print(f"BEGINNING STRATEGY: {strategy}")
    overrides = ['model/backbone=resnet50',
                 "callbacks.early_stopping.patience=10",
                 "data=extant_to_pnas",
                 "trainer.max_epochs=75",
                 "trainer.auto_lr_find=true",
                 "trainer.precision=16",
                 "trainer.gpus=[0]",
                 'trainer.resume_from_checkpoint=null', #"/home/jrose3/bn_unit_test_logs/avgdrop/finetuning_unfreeze_layers_on_plateau-PNAS_family_100-19_classes-res_512-bsz_16-resnet50-pretrained_True-pool_avgdrop/replicate_1/results/checkpoints/epoch=11-val_loss=0.764-val_acc=0.681.ckpt"',
                 "data.batch_size=16",
                 "logger.wandb.project=finetuning_on_plateau"]

    config, datamodule = get_config_and_load_data(overrides = overrides,
                                                  task_id=task_id,
                                                  pool_type=pool_type,
                                                  finetuning_strategy=strategy, #"feature_extractor_+_bn.eval()",
                                                  lr=2e-03,
                                                  dropout_p=dropout_p)#,
#                                                   max_epochs=config.trainer.max_epochs)
    
    print(f"task_{task_id}: dataset_name={datamodule.dataset_names[f'task_{task_id}']}")

    ckpt_paths = os.listdir(os.path.join(config.checkpoint_dir))
    if len(ckpt_paths) and os.path.exists(ckpt_paths[-1]):
        print(f"Found {ckpt_paths[-1]}")
        config.resume_from_checkpoint = ckpt_paths[-1]


    model, results = test_model_freeze_strategy(config, datamodule)
    model.cpu()
    del model

    results['model_config'] = OmegaConf.to_container(config.model, resolve=True)
    results['data_config'] = OmegaConf.to_container(config.data, resolve=True)
    
    ETL.config2yaml(results, os.path.join(config.results_dir, "results.yaml"))
    print(f"[SAVED TRIAL RESULTS] Location: {os.path.join(config.results_dir, 'results.yaml')}")
    pp(results)
    
    all_results[strategy] = results

    print(f"ALL FINISHED!!! RESULTS:")
    pp(all_results)


    ETL.config2yaml(all_results, os.path.join(config.root_dir, "results.yaml"))
    
    
    
    
if __name__ == "__main__":
    
    train(task_id=1, strategy="finetuning_unfreeze_layers_on_plateau")