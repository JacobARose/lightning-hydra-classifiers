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
from omegaconf import OmegaConf
from collections import OrderedDict
from typing import *

from lightning_hydra_classifiers.models.transfer import *
from rich import print as pp
from lightning_hydra_classifiers.utils.model_utils import count_parameters, collect_results
from lightning_hydra_classifiers.utils.metric_utils import get_per_class_metrics, get_scalar_metrics
from lightning_hydra_classifiers.models.backbones.backbone import build_model
import pytorch_lightning as pl
pl.seed_everything(42)

from lightning_hydra_classifiers.scripts.multitask.train import load_data, resolve_config, configure_callbacks, configure_loggers
from lightning_hydra_classifiers.utils.etl_utils import ETL
from omegaconf import OmegaConf

from lightning_hydra_classifiers.scripts.pretrain import lr_tuner
# from lightning_hydra_classifiers.scripts.multitask.train import configure_callbacks, configure_loggers#, configure_trainer
# source: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial5/Inception_ResNet_DenseNet.html
from lightning_hydra_classifiers.callbacks.finetuning_callbacks import FinetuningLightningCallback
from lightning_hydra_classifiers.models.transfer import LightningClassifier



def configure_trainer(config, callbacks=None, logger=None) -> pl.Trainer:
    trainer_config = resolve_config(config.trainer)

    trainer_config['callbacks'] = callbacks
    trainer_config['logger'] = logger
    print(f"trainer_config.gpus={trainer_config['gpus']}")
    print(f"config.data.batch_size={config.data.batch_size}")
    trainer: pl.Trainer = hydra.utils.instantiate(trainer_config)
    return trainer




def test_model_freeze_strategy(config, datamodule, **kwargs):
    """
    Inputs:
        model_name - Name of the model you want to run. Is used to look up the class in "model_dict"
        save_name (optional) - If specified, this name will be used for creating the checkpoint and logging directory.
    """
    
#     config.model.finetuning_strategy = config.get("finetuning_strategy", "finetuning_unfreeze_layers_on_plateau")
    
    group=f'{config.model.backbone.backbone_name}_{config.data.experiment.experiment_name}'#_task_{task_id}'
    config.logger.wandb.group = group
    config.callbacks.log_per_class_metrics_to_wandb.class_names = datamodule.classes

    callbacks = configure_callbacks(config)
    
#     for cb in callbacks:
#         if isinstance(cb, FinetuningLightningCallback):
#             print(f"Confirmed found FinetuningLightningCallback.")
#             print(f"cb.monitor={cb.monitor}, cb.mode={cb.mode}, cb.patience={cb.patience}")
    
    logger = configure_loggers(config)
    
    trainer = configure_trainer(config, callbacks=callbacks, logger=logger)
#     trainer = pl.Trainer(**config.trainer)
#                         default_root_dir=config.checkpoint_dir, #config.experiment_dir,
#                          gpus=config.trainer.gpus,
#                          max_epochs=config.trainer.max_epochs,
#                          callbacks=callbacks,
# #                                     ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
# #                                     LearningRateMonitor("epoch")],
#                          logger=logger,
#                          resume_from_checkpoint=config.trainer.resume_from_checkpoint,
#                          progress_bar_refresh_rate=10)
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

    model = LightningClassifier.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    print(f"Best checkpoint saved to: {trainer.checkpoint_callback.best_model_path}")

    results, model = test_model(model, trainer, datamodule)
    
    
#     val_result = trainer.test(model, test_dataloaders=datamodule.val_dataloader(), verbose=False)
#     test_result = trainer.test(model, test_dataloaders=datamodule.test_dataloader(), verbose=False)
    
#     try:
#         result = {"test_acc": test_result[0]["test_acc"], "val_acc": val_result[0]["test_acc"]}
#     except Exception as e:
#         print(e)
#         result = {"test_acc": test_result, "val_acc": val_result}
        
#     result["ckpt_path"] = trainer.checkpoint_callback.best_model_path
    
    pp(f"FINAL RESULTS: {results}")

    return results, model, trainer





def get_config(config=None,
               overrides = None):
    
    if config is None:
        overrides = overrides or []    
        config = ETL.load_hydra_config(config_name = "finetune_config",
                                       config_path = "/media/data/jacob/GitHub/lightning-hydra-classifiers/configs",
                                       overrides=overrides)
        
    OmegaConf.set_struct(config, False)
    
    return config


def get_config_and_load_data(config=None,
                             task_id: int = 1):
#                              pool_type='avgdrop',
#                              finetuning_strategy="feature_extractor_+_bn.eval()",
#                              lr=2e-03,
#                              dropout_p: float=0.3,
#                              max_epochs: int=5):

    config = get_config(config=config)
    
#     if os.path.isfile(os.path.join(config.results_dir, "results.yaml")):
#         results_file_path = os.path.join(config.results_dir, "results.yaml")
#         results = OmegaConf.load(results_file_path)
#         print(f"Found pre-existing results saved to file: {results_file_path}")
#         print(f"Results:"); pp(results)
        
#         return results, config
    
    
    
    pp(config)
    datamodule = load_data(config,
                           task_id=config.get("task_id", 0))
    print(f"datamodule.num_classes={datamodule.num_classes}")
    
    
    model_config = OmegaConf.create(dict(
                                    backbone=config.model.get("backbone"), #{"backbone_name":config.model.backbone.backbone_name},
                                    heads=config.model.get("heads"),
                                    scheduler_config=config.model.get("scheduler"),
                                    backbone_name=config.model.backbone.backbone_name,
                                    pretrained=config.model.backbone.pretrained,
                                    num_classes=datamodule.num_classes,
                                    pool_type=config.model.heads.get("pool_type", "avg"),
                                    head_type=config.model.heads.get("head_type", 'linear'),
                                    hidden_size=config.model.heads.get("hidden_size", None),
                                    dropout_p=config.model.heads.get("dropout_p", 0.0),
                                    lr=2e-03,
                                    backbone_lr_mult=config.model.get("backbone_lr_mult", 0.1),
                                    feature_extractor_strategy=config.get("feature_extractor_strategy"),
                                    finetuning_strategy=config.get("finetuning_strategy"),
                                    weight_decay=config.model.optimizer.get("weight_decay", 0.01),
                                    seed=config.get("seed")))
    config.model = model_config
#     config.experiment_name = f"{config.model.finetuning_strategy}-{config.dataset_name}-{datamodule.num_classes}_classes-res_{config.data.image_size}-bsz_{config.data.batch_size}-{config.model.backbone_name}-pretrained_{config.model.pretrained}-pool_{config.model.pool_type}"
#     config.root_dir = os.path.join(os.getcwd(), "bn_unit_test_logs", config.model.pool_type)
    config.lr_tuner_dir = os.path.join(config.results_dir, f"task_{task_id}", "lr_tuner")
    
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.lr_tuner_dir, exist_ok=True)
    return config, datamodule



# valid_strategies = ("finetuning_unfreeze_layers_on_plateau")
# pool_types = ("avg", "avgdrop")#, "avgmax", "max", "avgmaxdrop")

# finetuning_strategy="feature_extractor"
# finetuning_strategy="feature_extractor_+_bn.eval()"

# pool_type='avgdrop'
# pool_type='avgmaxdrop'

# def train(task_id: int=1, strategy="finetuning_unfreeze_layers_on_plateau"):

#     pool_type="avgdrop"
#     dropout_p = 0.1
#     all_results = {}
#     print(f"BEGINNING STRATEGY: {strategy}")
#     overrides = ['model/backbone=resnet50',
#                  "callbacks.early_stopping.patience=10",
#                  "data=extant_to_pnas",
#                  "trainer.max_epochs=75",
#                  "trainer.auto_lr_find=true",
#                  "trainer.precision=16",
#                  "trainer.gpus=[0]",
#                  'trainer.resume_from_checkpoint=null', #"/home/jrose3/bn_unit_test_logs/avgdrop/finetuning_unfreeze_layers_on_plateau-PNAS_family_100-19_classes-res_512-bsz_16-resnet50-pretrained_True-pool_avgdrop/replicate_1/results/checkpoints/epoch=11-val_loss=0.764-val_acc=0.681.ckpt"',
#                  "data.batch_size=16",
#                  "logger.wandb.project=finetuning_on_plateau"]





def test_model(model, trainer, datamodule):
    model = LightningClassifier.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    print(f"Best checkpoint saved to: {trainer.checkpoint_callback.best_model_path}")

    model.init_metrics(stage="test")
    train_result = trainer.test(model, test_dataloaders=datamodule.train_dataloader(), verbose=False)
    model.init_metrics(stage="test")
    val_result = trainer.test(model, test_dataloaders=datamodule.val_dataloader(), verbose=False)
    model.init_metrics(stage="test")
    test_result = trainer.test(model, test_dataloaders=datamodule.test_dataloader(), verbose=False)
    
    try:
        results = {"test_acc": test_result[0]["test_acc"], 
                  "val_acc": val_result[0]["test_acc"],
                  "train_acc": train_result[0]["test_acc"]}
    except Exception as e:
        print(e)
        results = {"test_acc": test_result, 
                  "val_acc": val_result,
                  "train_acc": train_result}
        
    results["ckpt_path"] = trainer.checkpoint_callback.best_model_path

    return results, model


##################
###################




def train(config,
          task_id: int=1):

    config, datamodule = get_config_and_load_data(config,
                                                  task_id=task_id)
    if os.path.isfile(os.path.join(config.results_dir, "results.yaml")):
        results_file_path = os.path.join(config.results_dir, "results.yaml")
        results = OmegaConf.load(results_file_path)
        print(f"Found pre-existing results saved to file: {results_file_path}")
        print(f"Results:"); pp(results)
        return results, config
    
        
    print(f"task_{task_id}: dataset_name={datamodule.dataset_names[f'task_{task_id}']}")

    ckpt_paths = [os.path.join(config.checkpoint_dir, ckpt) for ckpt in os.listdir(config.checkpoint_dir)]
    if len(ckpt_paths) and os.path.exists(ckpt_paths[-1]):
        print(f"Found {ckpt_paths[-1]}")
        config.resume_from_checkpoint = ckpt_paths[-1]


    results, model, trainer = test_model_freeze_strategy(config, datamodule)
#     model.cpu()
#     del model

    results['model_config'] = OmegaConf.to_container(config.model, resolve=True)
    results['data_config'] = OmegaConf.to_container(config.data, resolve=True)
    results['hparams_config'] = OmegaConf.to_container(config.get("hparams",{}), resolve=True)
    
    ETL.config2yaml(results, os.path.join(config.results_dir, "results.yaml"))
    
    if wandb.get("run",None):
        wandb.save(os.path.join(config.results_dir, "results.yaml"))
    
    print(f"[SAVED TRIAL RESULTS] Location: {os.path.join(config.results_dir, 'results.yaml')}")
    pp(results)
    return results, config



################################################
################################################


def finetune_new_classifier(config,
                            task_id: int=1):

    config, datamodule = get_config_and_load_data(config,
                                                  task_id=task_id)
    
    if os.path.isfile(os.path.join(config.results_dir, "results.yaml")):
        results_file_path = os.path.join(config.results_dir, "results.yaml")
        results = OmegaConf.load(results_file_path)
        print(f"Found pre-existing results saved to file: {results_file_path}")
        print(f"Results:"); pp(results)
        return results, config
    
    group=f'{config.model.backbone.backbone_name}_{config.data.experiment.experiment_name}'#_task_{task_id}'
    config.logger.wandb.group = group
    config.callbacks.log_per_class_metrics_to_wandb.class_names = datamodule.classes

    callbacks = configure_callbacks(config)
    logger = configure_loggers(config)
    trainer = configure_trainer(config, callbacks=callbacks, logger=logger)
    
    
    
    
    ckpt_path = config.source.model.get("backbone_ckpt_path", "")
    if os.path.isfile(config.source.get("results_filepath", "")):
        results_file_path = config.source.get("results_filepath")
        source_results = OmegaConf.load(results_file_path)
        print(f"Found pre-existing results saved to file: {results_file_path}")
        print(f"Source Results:"); pp(source_results)
#         backbone_ckpt_path = config.source.model.get("backbone_ckpt_path","")
        if "ckpt_path" in source_results and os.path.isfile(str(ckpt_path)):
            assert source_results["ckpt_path"] == os.path.abspath(ckpt_path), f'source_results["ckpt_path"] != os.path.abspath(ckpt_path): {source_results["ckpt_path"]} == {os.path.abspath(ckpt_path)}'
#         config.source.model.get("backbone_ckpt_path", "")
    
        if os.path.isfile(source_results.get(str("ckpt_path"))) and (not os.path.isfile(str(ckpt_path))):
            ckpt_path = source_results["ckpt_path"]
    
        model = LightningClassifier.init_pretrained_backbone_w_new_classifier(ckpt_path=ckpt_path,
                                                                              new_num_classes=config.hparams.num_classes,
                                                                              **config.model)
        model.label_encoder = datamodule.label_encoder
            
    else:
        print("Error loading pretrained backbone and new classifier")
        raise Exception
    pl.seed_everything(config.model.seed)

    if config.trainer.auto_lr_find:

        lr_tune_output = lr_tuner.run_lr_tuner(trainer=trainer,
                                               model=model,
                                               datamodule=datamodule,
                                               config=config,
                                               results_dir=config.lr_tuner_dir,
                                               group="target_data_lr_tuner")
        
    trainer.fit(model, datamodule=datamodule)
    
    results, model = test_model(model, trainer, datamodule)
    
    pp(f"FINAL RESULTS: {results}")

    model.cpu()
    del model

    results['model_config'] = OmegaConf.to_container(config.model, resolve=True)
    results['data_config'] = OmegaConf.to_container(config.data, resolve=True)
    results['hparams_config'] = OmegaConf.to_container(config.get("hparams",{}), resolve=True)
    
    ETL.config2yaml(results, os.path.join(config.results_dir, "results.yaml"))
    print(f"[SAVED TRIAL RESULTS] Location: {os.path.join(config.results_dir, 'results.yaml')}")
    pp(results)
    return results, config

    
    

    
#     model = LightningClassifier.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
#     print(f"Best checkpoint saved to: {trainer.checkpoint_callback.best_model_path}")

#     train_result = trainer.test(model, test_dataloaders=datamodule.train_dataloader(), verbose=False)
#     val_result = trainer.test(model, test_dataloaders=datamodule.val_dataloader(), verbose=False)
#     test_result = trainer.test(model, test_dataloaders=datamodule.test_dataloader(), verbose=False)
    
#     try:
#         results = {"test_acc": test_result[0]["test_acc"], 
#                   "val_acc": val_result[0]["test_acc"],
#                   "train_acc": train_result[0]["train_acc"]}
#     except Exception as e:
#         print(e)
#         results = {"test_acc": test_result, 
#                   "val_acc": val_result,
#                   "train_acc": train_result}
        
#     results["ckpt_path"] = trainer.checkpoint_callback.best_model_path
    
#     pp(f"FINAL RESULTS: {results}")

#     model.cpu()
#     del model

#     results['model_config'] = OmegaConf.to_container(config.model, resolve=True)
#     results['data_config'] = OmegaConf.to_container(config.data, resolve=True)
#     results['hparams_config'] = OmegaConf.to_container(config.get("hparams",{}), resolve=True)
    
#     ETL.config2yaml(result, os.path.join(config.results_dir, "results.yaml"))
#     print(f"[SAVED TRIAL RESULTS] Location: {os.path.join(config.results_dir, 'results.yaml')}")
#     pp(results)
#     return results, config
    
    
    
    
    
    
    
    
    
# if __name__ == "__main__":
    
#     train(task_id=2, strategy="finetuning_unfreeze_layers_on_plateau")
    
    
    
    
    


@hydra.main(config_name="finetune_config", config_path="../../configs")
def main(config):

    pl.seed_everything(config.seed)
    OmegaConf.set_struct(config, False)
#     config = ETL.init_structured_config(cfg=config,
#                                      dataclass_type = MultiTaskExperimentConfig)
    pp(OmegaConf.to_yaml(config, resolve=True))
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.callbacks.model_checkpoint.dirpath, exist_ok=True)    
    os.environ["WANDB_ENTITY"] = "jrose"
#     os.environ["WANDB_PROJECT"] = "image_classification_train"
    os.environ["WANDB_DIR"] = config.experiment_dir
        
#     torch.backends.cudnn.benchmark = True
#     torch.backends.cudnn.enabled = True
    
    if config.run_trial == "train":
        results, config = train(config, task_id=config.task_id)
        
    elif config.run_trial == "finetune_new_classifier":
        results, config = finetune_new_classifier(config,
                                                  task_id=config.task_id)

    print(f"Final checkpoint saved to: {results['ckpt_path']}")
    return results["test_acc"]





        
if __name__ == '__main__':
    
    main()
