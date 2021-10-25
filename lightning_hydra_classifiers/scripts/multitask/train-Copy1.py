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
    
    
#     callbacks: List[Callback] = []
#     for k, cb_conf in config.callbacks.items():
#         if "_target_" in cb_conf:
#             logging.info(f"Instantiating callback <{cb_conf._target_}>")
# #             if k == "image_prediction_logger":
# #                 callbacks.append(hydra.utils.instantiate(cb_conf, datamodule=datamodule))
# #             else:
#             callbacks.append(hydra.utils.instantiate(cb_conf))

#     logger: List[LightningLoggerBase] = []
#     for _, lg_conf in config["logger"].items():
#         if "_target_" in lg_conf:
#             logging.info(f"Instantiating logger <{lg_conf._target_}>")
#             logger.append(hydra.utils.instantiate(lg_conf))

    

#     wandb_logger = pl.loggers.WandbLogger(#entity = "jrose",
# #                                           project = "image_classification_train",
#                                           job_type = "train_supervised",
#                                           config=asdict(config),
#                                           group=group, #f'{config.model.model_name}_{config.data.experiment}_task_{task_id}',
#                                           reinit=True,
#                                           dir=config.experiment_dir)

#     try:
#         trainer_config = config.trainer
#         trainer_config = asdict(config.trainer)
#     except TypeError:
#         trainer_config = OmegaConf.to_container(config.trainer, resolve=True)
#     finally:
#         trainer_config = dict(config.trainer)


#     trainer_config = resolve_config(config.trainer)

#     trainer_config['callbacks'] = callbacks
#     trainer_config['logger'] = logger
#     print(f"trainer_config.gpus={trainer_config['gpus']}")
#     print(f"config.data.batch_size={config.data.batch_size}")
#     trainer: pl.Trainer = hydra.utils.instantiate(trainer_config)
    
    
    
    

#####################################################################


# from lightning_hydra_classifiers.experiments.transfer_experiment import TransferExperiment
# output_root_dir = "/media/data_cifs/projects/prj_fossils/users/jacob/experiments/July2021-Nov2021/csv_datasets/experimental_datasets"
# experiment = TransferExperiment()
# experiment.export_experiment_spec(output_root_dir=output_root_dir)




# def cmdline_args(arg_overrides=None):
    
# #     parser = argparse.ArgumentParser(description="")
# #     subparsers = parser.add_subparsers()
# #     p = subparsers.add_parser('', help='model args')
#     p = argparse.ArgumentParser(description="")
    
#     # subparser_one = parser_one.add_subparsers()
# #     p.add_argument("--tasks", dest="tasks", type=int, default=[0], nargs="*",
# #                    help="Run either task 0 or task 1.")
#     p.add_argument("-o", "--output_dir", dest="output_dir", type=str,
#                    default="/media/data_cifs/projects/prj_fossils/users/jacob/experiments/July2021-Nov2021/experiment_logs/Transfer_Experiments",
#                    help="Output root directory for experiment logs.")
#     p.add_argument("-ckpt", "--load_from_checkpoint", dest="load_from_checkpoint", type=str,
#                    default=None,
#                    help="Attempt to load model weights from checkpoint prior to training. If user desires to also load previous epoch info, use resume_from_checkpoint instead. If pointing to nonexistent file path, will attempt to copy best checkpoint produced by ModelCheckpoint callback to the specified location.")
#     p.add_argument("-resume", "--resume_from_checkpoint", dest="resume_from_checkpoint", type=str,
#                    default=None,
#                    help="Provide a path to checkpoint in order to resume previous training from beginning of the last epoch. This will override --load_from_checkpoint flag.")
#     p.add_argument("-e", "--num_epochs", dest="num_epochs", type=int, default=100,
#                    help="Number of training epochs")
#     p.add_argument("-res", "--image_size", dest="image_size", type=int, default=224,
#                    help="image_size/model input resolution")
#     p.add_argument("-buffer", "--image_buffer_size", dest="image_buffer_size", type=int, default=32,
#                    help="Additional resolution to resize images before either random (train) or center (test) crop to image_size.")
#     p.add_argument("-bsz", "--batch_size", dest="batch_size", type=int, default=32,
#                    help="batch_size")
#     p.add_argument("-nproc", "--num_workers", dest="num_workers", type=int, default=4,
#                    help="num_workers per dataloader")
#     p.add_argument("-model", "--model_name", dest="model_name", type=str, default="resnet50",
#                    help="model backbone architecture")
#     p.add_argument("-exp", "--experiment", dest="experiment_name", type=str, default="Extant-to-PNAS-512-transfer_benchmark",
#                    choices=["Extant-to-PNAS-512-transfer_benchmark", "Extant-to-Fossil-512-transfer_benchmark"],
#                    help="Transfer experiment tag/name.")
    
    
#     p.add_argument("-l", "--list_available_backbones", dest="list_available_backbones", action="store_true", default=False,
#                    help="List names of available model backbone architectures, then exit.")
#     p.add_argument("-init_freeze", "--init_freeze_up_to", dest="init_freeze_up_to", default=None, #"layer4",
#                    help="freeze up to and including layer name or index.")    
#     p.add_argument("-pre", "--pretrained", dest="pretrained", default="imagenet", choices=["imagenet", "True", "False"],
#                    help="Use pretrained imagenet weights or randomly initialize from scratch.")
#     p.add_argument("-lr", "--learning_rate", dest="learning_rate", type=float, default=3e-8,
#                    help="Optional Initial learning rate. Ignored for now, in the future will force skip any lr_tuner stage.")
#     p.add_argument("-l2", "--weight_decay", dest="weight_decay", type=float, default=0.0,
#                    help="Weight decay.")
#     p.add_argument("-pool", "--global_pool_type", dest="global_pool_type", default="avg", choices=["avg", "max"],
#                    help="Specify either an AdaptiveAveragePool or AdaptiveMaxPool for connecting the model base to its classifier head.")    
#     p.add_argument("--gpus", dest="gpus", type=int, default=1, nargs="*",
#                    help="Specify number of gpus or specific gpu ids.")
#     p.add_argument("-d", "--debug", dest="debug", action="store_true", default=False,
#                    help="Flag for activating debug-related settings. Currently limited to switching out datamodule to use CIFAR10")
#     args = p.parse_args(arg_overrides)

#     if args.pretrained == "True":
#         args.pretrained = True
#     elif args.pretrained == "False":
#         args.pretrained = False
    
#     logging.info("Args:")
#     pp(args)
    
#     config = OmegaConf.create({
#         "seed":42,
# #         "num_epochs": args.num_epochs,
# #         "precision": 16,
# #         "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
#         "device": 'cuda' if torch.cuda.is_available() else 'cpu',
#         "output_dir":args.output_dir,
#         "debug":args.debug
#     })

#     config.trainer = OmegaConf.create({"precision": 16,
#                                        "max_epochs": args.num_epochs,
#                                        "gpus": args.gpus,
#                                        "resume_from_checkpoint": args.resume_from_checkpoint
#                                       })
    
#     config.data = OmegaConf.create({
#         "image_size": args.image_size,
#         "image_buffer_size": args.image_buffer_size,
#         "batch_size": args.batch_size,
#         "num_workers": args.num_workers,
#         "pin_memory": True
#     })
#     if args.experiment_name == "Extant-to-PNAS-512-transfer_benchmark":
#         config.data.experiment = Extant_to_PNAS_ExperimentConfig
#     elif args.experiment_name == "Extant-to-Fossil-512-transfer_benchmark":
#         config.data.experiment = Extant_to_Fossil_ExperimentConfig
        
#     config.model = OmegaConf.create({"model_name": args.model_name,
#                                      "init_freeze_up_to":args.init_freeze_up_to,
#                                      "pretrained": args.pretrained,
#                                      "lr": args.learning_rate,
#                                      "weight_decay":args.weight_decay,
#                                      "num_classes": None,
#                                      "global_pool_type": args.global_pool_type,
#                                      "drop_rate": 0.0,
#                                      "image_size": args.image_size,
#                                      "ckpt_path": args.load_from_checkpoint,
#                                      "min_lr": 1e-6})
#     from lightning_hydra_classifiers.experiments.configs.optimizer import AdamWOptimizerConfig

#     config.model.optimizer = OmegaConf.structured(AdamWOptimizerConfig(lr=args.learning_rate,
#                                                                       weight_decay=args.weight_decay))
    

#     config.callbacks = OmegaConf.create({"monitor":
#                               OmegaConf.create({"metric":"val_acc",
#                                      "mode": "max"})
# #                               OmegaConf.create({"metric":"val_loss",
# #                                      "mode": "min"})
#                              })

#     config.stages = OmegaConf.create({#"lr_tuner":True,
#                            "task_0":None,
#                            "task_1":None})
    
    
#     if args.list_available_backbones:
#         print(f"--list_available_backbones = True.","\nAvailable Models:")
#         pp(LitMultiTaskModule.available_backbones())
#         sys.exit()
    
#     if config.debug:
#         config.stages.task_0 = OmegaConf.create({"name":"CIFAR10",
#                                       "task_id":0,
#                                       "skip_lr_tuner":False})
#         del config.stages.task_1
#         config.trainer.update(OmegaConf.create(max_epochs=2,
#                                     limit_train_batches=4,
#                                     limit_val_batches=4,
#                                     limit_test_batches=4,
#                                     auto_lr_find=bool(config.model.lr is not None)))
        
#     else:
#         config.stages.task_0 = OmegaConf.create({"name":"Extant-PNAS",
#                                       "task_id":0,
#                                       "skip_lr_tuner":False})
#         config.stages.task_1 = OmegaConf.create({"name":"PNAS",
#                                       "task_id":1,
#                                       "skip_lr_tuner":False})

#     if "task_1" in config.stages:
#         task_tags = config.stages.task_0.name + "-to-" + config.stages.task_1.name
#     else:
#         task_tags = config.stages.task_0.name
        
# #     import pdb; pdb.set_trace()
#     if config.model.pretrained in ("imagenet", True):
#         weights_name = "imagenet_weights"
#     else:
#         weights_name = "random_weights"
        
#     config.experiment_name = "_".join([task_tags, config.model.model_name, weights_name])
#     config.experiment_dir = os.path.join(config.output_dir, config.experiment_name)
    
#     for task in config.stages.keys():
#         if config.stages[task] is None: continue
#         config.stages[task].model_ckpt_dir = str(Path(config.experiment_dir, task, "checkpoints"))
#         config.stages[task].lr_tuner_dir = str(Path(config.experiment_dir, task, "lr_tuner"))
#         config.stages[task].lr_tuner_results_path = str(Path(config.stages[task].lr_tuner_dir, "results.csv"))
#         config.stages[task].lr_tuner_hparams_path = str(Path(config.stages[task].lr_tuner_dir, "hparams.yaml"))

        
#     config.lr_tuner = OmegaConf.structured(lr_tuner.LRTunerConfig(min_lr = 1e-07,
#                                                                   max_lr = 1.2,
#                                                                   num_training = 150,
#                                                                   mode = 'exponential',
#                                                                   early_stop_threshold = 8.0))
#     return args, config
    
# def main():
    
#     args, config = cmdline_args()
#     os.makedirs(config.experiment_dir, exist_ok=True)

#     os.environ["WANDB_ENTITY"] = "jrose"
#     os.environ["WANDB_PROJECT"] = "image_classification_train"
#     os.environ["WANDB_DIR"] = config.experiment_dir
        
# #     torch.backends.cudnn.benchmark = True
# #     torch.backends.cudnn.enabled = True
    
#     results = {}
    
#     results["task_0"], config = train_task(config=config, task_id=0)
#     config.model.ckpt_path = config.stages["task_0"].ckpt_path
    
    
#     if "task_1" in config.stages:
#         print(f"[Initiating] Transfer to Task_1: {config.stages.task_1.name}")
#         print(f"[Re-loading model from checkpoint path] : {config.model.ckpt_path}")
#         #TODO Log/Cache experiment artifacts here.    
#         results["task_1"], config = train_task(config=config, task_id=1)
        
    
#     print(f"[SUCCESSFULLY FINISHED TRAIN.PY]")
#     torch.save(results, str(Path(config.experiment_dir, "test_results.pth")))
#     logging.info(json.dumps(results))
#     if os.path.isfile(str(Path(config.experiment_dir, "test_results.pth"))):
#         logging.info(f'Congratulations, your test results are pickled using torch.save to: {str(Path(config.experiment_dir, "test_results.pth"))}')
#     else:
#         logging.warning(f'[Warning] Saving test results to {str(Path(config.experiment_dir, "test_results.pth"))} failed')


# import hydra
# from hydra.core.config_store import ConfigStore



# @dataclass
# class MultiTaskExperimentConfig:
#     resolution: int = 512
#     channels: int = 3
#     root_dir: str = "."
#     seed: int = 1234

#     model: LitMultiTaskModuleConfig = LitMultiTaskModuleConfig()
#     trainer: TrainerConfig = TrainerConfig()
        

# cs = ConfigStore.instance()
# cs.store(name="config", node=MultiTaskExperimentConfig)




































#     ckpt_path = str(config.system.tasks[task_id].model_ckpt_path)
#     config.model.ckpt_path = ckpt_path
#     if os.path.isfile(ckpt_path):
#         logging.info(f"Loading from model checkpoint: {str(ckpt_path)}")
#         model = LitMultiTaskModule.load_from_checkpoint(ckpt_path, config=config.model)
#     else:
#         if isinstance(ckpt_path, (str, Path)):
#             logging.warning(f"User specified checkpoint path doesn't exist. Best checkpoint produced during training will be copied to that location: {ckpt_path}")
#         logging.info(f"Instantiating model with hparams:")
#         logging.info(config.model)
#         model = LitMultiTaskModule(config.model)


#     img_prediction_callback = ImagePredictionLogger(top_k_per_batch=k,
#                                                     bottom_k_per_batch=k)
#     checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor=config.callbacks.monitor.metric,
#                                                        save_top_k=1,
#                                                        save_last=True,
#                                                        dirpath=str(Path(config.model.ckpt_path).parent),
#                                                        filename='{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}',
#                                                        verbose=True,
#                                                        mode=config.callbacks.monitor.mode)
    
#     lr_monitor_callback = pl.callbacks.LearningRateMonitor(logging_interval="step", log_momentum=True)
#     data_monitor_callback = pl_bolts.callbacks.ModuleDataMonitor(submodules=True)
#     #TrainingDataMonitor(log_every_n_steps=25)
#     earlystopping = pl.callbacks.EarlyStopping(monitor=config.callbacks.monitor.metric,
#                                                patience=3,
#                                                mode=config.callbacks.monitor.mode)
#     callbacks=[earlystopping,
#                checkpoint_callback,
#                img_prediction_callback]#,
# #                data_monitor_callback]


    
    
    
    
    
    
    
    
    




# class MyLightningCLI(LightningCLI):
#     def add_arguments_to_parser(self, parser):
#         parser.add_argument('--tune', action="store_true")
#         parser.add_argument('--tune_max_lr', type=float, default=1e-2)
#         parser.add_argument('--tune_save_results', action="store_true")
        
#     def before_fit(self):
#         if self.config["tune"]:
#             lr_finder  = self.trainer.tuner.lr_find(self.model, max_lr=self.config["tune_max_lr"])  # could add more?
#             suggested_lr =  lr_finder.suggestion()
#             print(f"Changing learning rate to {suggested_lr}\n")
#             self.model.hparams.learning_rate = suggested_lr
#             if self.config["tune_save_results"]:
#                 fig = lr_finder.plot(suggest=True)
#                 logdir = Path(self.trainer.log_dir)
#                 if not os.path.exists(logdir):
#                     os.makedirs(logdir)
#                 fig.savefig( logdir / "lr_results.png")











                    
        


# def train_source_task(config: argparse.Namespace):
#     pl.seed_everything(config.seed)
    # ## DataModule
#     if config.debug == True:
#         print(f"Debug mode activated, loading CIFAR10 datamodule")
#         datamodule = CIFAR10DataModule(batch_size=config.data.batch_size,
#                                        task_id=0,
#                                        image_size=config.data.image_size,
#                                        image_buffer_size=config.data.image_buffer_size,
#                                        num_workers=config.data.num_workers,
#                                        pin_memory=config.data.pin_memory)
#     else:
#         datamodule = MultiTaskDataModule(batch_size=config.data.batch_size,
#                                      task_id=0,
#                                      image_size=config.data.image_size,
#                                      image_buffer_size=config.data.image_buffer_size,
#                                      num_workers=config.data.num_workers,
#                                      pin_memory=config.data.pin_memory)
#     datamodule.setup("fit")
#     config.model.num_classes = datamodule.num_classes
#     pp(config)
    
#     if os.path.isfile(str(config.model.ckpt_path)):
#         print(f"Loading from model checkpoint: {str(config.model.ckpt_path)}")
#         model = LitMultiTaskModule.load_from_checkpoint(config.model.ckpt_path, config=config.model)

#     else:
#         if isinstance(config.model.ckpt_path, (str, Path)):
#             print(f"User specified checkpoint path doesn't exist. Best checkpoint produced during training will be copied to that location: {config.model.ckpt_path}")
#         print(f"Instantiating model from scratch with hparams:")
#         print(config.model)
#         model = LitMultiTaskModule(config.model)
#     model.label_encoder = datamodule.label_encoder




#     pipeline_stage = str(pipeline_stage)
#     stage_config = config.wandb[pipeline_stage]

#     if not config.tuner.trainer_kwargs.auto_lr_find:
#         print(f'config.trainer.auto_lr_find is set to False, Skipping `run_lr_tuner(config, pipeline_stage={pipeline_stage})`')
#         print(f'Proceeding with:\n')
#         print(f'Learning rate = {config.model.lr:.3e}')
#         print(f'Batch size = {config.model.batch_size}')
        
#         return config.model.lr, None, config


#     os.environ["WANDB_PROJECT"] = stage_config.init.project
#     run = wandb.init(entity=stage_config.init.entity,
#                      project=stage_config.init.project,
#                      job_type=stage_config.init.job_type,
#                      group=stage_config.init.group,
#                      dir=stage_config.init.run_dir,
#                      config=OmegaConf.to_container(stage_config.init.config, resolve=True))

#     datamodule, data_artifact = fetch_datamodule_from_dataset_artifact(dataset_config=config.datamodule,
#                                                                        artifact_config=config.artifacts.input_dataset_artifact,
#                                                                        run_or_api=run)
#     config.model.num_classes = len(datamodule.classes)
    # assert (num_classes == 19) | (num_classes == 179)
    ########################################
    ########################################
#     model, model_artifact = build_and_log_model_to_artifact(model_config=config.model,
#                                         artifact_config=config.artifacts.input_model_artifact,
#                                         run_or_api=run)

#     model, model_artifact = use_model_artifact(artifact_config=config.artifacts.input_model_artifact,
#                                                model_config=config.model,
#                                                run_or_api=run)    









































    
    
    
    
    
# class MultiTaskDataModule(pl.LightningDataModule):
# #     valid_tasks = (0, 1)
    
#     def __init__(self, 
#                  batch_size,
#                  task_id: int=0,
#                  image_size: int=224,
#                  image_buffer_size: int=32,
#                  num_workers: int=4,
#                  pin_memory: bool=True):
#         super().__init__()
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.pin_memory = pin_memory
        
        
#         self.experiment = TransferExperiment()
#         self.set_task(task_id)        
        
#         self.image_size = image_size
#         self.image_buffer_size = image_buffer_size
#         self.mean = [0.485, 0.456, 0.406]
#         self.std = [0.229, 0.224, 0.225]
#         # Train augmentation policy
#         self.__init_transforms()
#         self.tasks = self.experiment.get_multitask_datasets(train_transform=self.train_transform,
#                                                             val_transform=self.val_transform)

#     def __init_transforms(self):
        
#         self.train_transform = transforms.Compose([
#             transforms.RandomResizedCrop(size=self.image_size,
#                                          scale=(0.25, 1.2),
#                                          ratio=(0.7, 1.3),
#                                          interpolation=2),
#             transforms.ToTensor(),
#             transforms.RandomHorizontalFlip(),
#             transforms.Normalize(self.mean, self.std),
#             transforms.Grayscale(num_output_channels=3)
#         ])

#         self.val_transform = transforms.Compose([
#             transforms.Resize(self.image_size+self.image_buffer_size),
#             transforms.ToTensor(),
#             transforms.CenterCrop(self.image_size),
#             transforms.Normalize(self.mean, self.std),
#             transforms.Grayscale(num_output_channels=3)            
#         ])

#     def set_task(self, task_id: int):
#         assert task_id in self.experiment.valid_tasks
#         self.task_id = task_id
        

#     @property
#     def current_task(self):
#         return self.tasks[self.task_id]

#     def setup(self, stage=None):
#         task = self.current_task
#         # Assign train/val datasets for use in dataloaders
#         if stage == 'fit' or stage is None:
#             self.train_dataset = task['train']
#             self.val_dataset = task['val']
            
#             self.classes = self.train_dataset.classes
#             self.num_classes = len(self.train_dataset.label_encoder)
#             self.label_encoder = self.train_dataset.label_encoder
            
#         elif stage == 'test':
#             self.test_dataset = task['test']
                        
#     def train_dataloader(self):
#         return torch.utils.data.DataLoader(self.train_dataset,
#                           batch_size=self.batch_size,
#                           pin_memory=self.pin_memory,
#                           num_workers=self.num_workers,
#                           shuffle=True,
#                           drop_last=True)

#     def val_dataloader(self):
#         return torch.utils.data.DataLoader(self.val_dataset,
#                           batch_size=self.batch_size,
#                           pin_memory=self.pin_memory,
#                           num_workers=self.num_workers)
    
#     def test_dataloader(self):
#         return torch.utils.data.DataLoader(self.test_dataset,
#                           batch_size=self.batch_size,
#                           pin_memory=self.pin_memory,
#                           num_workers=self.num_workers)


# ## Model & LightningModules
# class LitMultiTaskModule(pl.LightningModule):
    
#     def __init__(self, config): #, ckpt_path: Optional[str]=None):
#         super().__init__()
#         config = OmegaConf.create(config)
#         self.config = config
#         self.lr = config.lr
#         self.num_classes = config.num_classes
#         self.save_hyperparameters({"config":dict(config)})
        
#         self.init_model(config)
#         self.metrics = self.init_metrics(stage='all')
#         self.criterion = nn.CrossEntropyLoss()        

#     def update_config(self, config):
#         self.config.update(OmegaConf.create(config))
#         self.hparams.config.update(OmegaConf.create(config))
        
#     def forward(self, x, *args, **kwargs):
#         return self.model(x)

#     def configure_optimizers(self):
#         print(f"self.hparams={self.hparams}")
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)#hparams.lr)
#         self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.t_max, eta_min=self.config.min_lr)

#         return {'optimizer': self.optimizer, 'lr_scheduler': self.scheduler}

#     def step(self, batch, batch_idx):
#         image, y_true = batch[0], batch[1]
#         y_logit = self.model(image)
#         y_pred = torch.argmax(y_logit, dim=-1)
#         return y_logit, y_true, y_pred

    
#     def training_step(self, batch, batch_idx):
#         y_logit, y_true, y_pred = self.step(batch, batch_idx)
#         loss = self.criterion(y_logit, y_true)
#         scores = self.metrics_train(y_logit, y_true)
#         self.log_dict({"train_loss": loss, 'lr': self.optimizer.param_groups[0]['lr']},
#                       on_step=True, on_epoch=True, prog_bar=True, logger=True)
#         self.log_dict(self.metrics_train,
#                  on_step=False, on_epoch=True, prog_bar=True, logger=True)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         y_logit, y_true, y_pred = self.step(batch, batch_idx)
#         loss = self.criterion(y_logit, y_true)
# #         y_pred = torch.argmax(y_logit, dim=-1)
#         scores = self.metrics_val(y_logit, y_true)
        
#         self.log("val_loss", loss,
#                   on_step=False, on_epoch=True, prog_bar=True, logger=True)
#         self.log("val_acc", self.metrics_val["val/acc_top1"],
#                  on_step=False, on_epoch=True, prog_bar=True, logger=True)
#         self.log_dict({k:v for k,v in self.metrics_val.items() if k != "val/acc_top1"},
#                  on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
#         return {"loss":loss,
#                 "y_logit":y_logit,
#                 "y_pred":y_pred}
    
    
#     def test_step(self, batch, batch_idx):
#         y_logit, y_true, y_pred = self.step(batch, batch_idx)
#         loss = self.criterion(y_logit, y_true)
#         scores = self.metrics_test(y_logit, y_true)
#         self.log_dict("test_loss", loss,
#                       on_step=True, on_epoch=True, prog_bar=True, logger=True)
#         self.log_dict(self.metrics_test,
#                  on_step=False, on_epoch=True, prog_bar=True, logger=True)
#         return loss

        
        
# #         y_logit, y_true, y_pred = self.step(batch, batch_idx)
# #         return {'test_loss': F.cross_entropy(y_hat, y)}

# #     def test_end(self, outputs):
# #         # OPTIONAL
# #         avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
# #         tensorboard_logs = {'test_loss': avg_loss}
# #         return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}
    
# #     def test_epoch_end(self, outputs):
# #         avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
# #         logs = {'test_loss': avg_loss}
# #         return {'test_loss': avg_loss, 'log': logs, 'progress_bar': logs}
    
    
    
    
#     def init_model(self, config):
#         self.model =  backbone.build_model(model_name=config.model_name,
#                                            pretrained=config.pretrained,
#                                            num_classes=config.num_classes)
        
#         self.freeze_up_to(layer=config.init_freeze_up_to)
        
        
        
#     def freeze_up_to(self, layer: Union[int, str]=None):
        
#         if isinstance(layer, int):
#             if layer < 0:
#                 layer = len(list(self.model.parameters())) + layer
            
# #         self.model.enable_grad = True
#         self.model.requires_grad = True
#         for i, (name, param) in enumerate(self.model.named_parameters()):
            
#             if isinstance(layer, int):
#                 if layer == i:
#                     break
#             elif isinstance(layer, str):
#                 if layer in name:
#                     break
#             param.requires_grad = False
    
    
#     def init_metrics(self, stage: str='train'):
        
#         if stage in ['train', 'all']:
#             self.metrics_train = get_scalar_metrics(num_classes=self.num_classes, average='macro', prefix='train')
# #             self.metrics_train_per_class = get_per_class_metrics(num_classes=self.num_classes, prefix='train')
            
#         if stage in ['val', 'all']:
#             self.metrics_val = get_scalar_metrics(num_classes=self.num_classes, average='macro', prefix='val')
# #             self.metrics_val_per_class = get_per_class_metrics(num_classes=self.num_classes, prefix='val')
            
#         if stage in ['test', 'all']:
#             self.metrics_test = get_scalar_metrics(num_classes=self.num_classes, average='macro', prefix='test')
# #             self.metrics_test_per_class = get_per_class_metrics(num_classes=self.num_classes, prefix='test')




# from lightning_hydra_classifiers.utils.callback_utils import LogImagePredictions, LogF1PrecRecHeatmap, LogConfusionMatrix, UploadCheckpointsAsArtifact, UploadCodeAsArtifact, WatchModel, get_wandb_logger
# from torch.nn import functional as F
# Custom Callback
# class ImagePredictionLogger(pl.Callback):
#     def __init__(self, top_k_per_batch: int=5, bottom_k_per_batch: int=5):
#         super().__init__()
#         self.top_k_per_batch = top_k_per_batch
#         self.bottom_k_per_batch = bottom_k_per_batch
# #         self.num_samples = num_samples
# #         self.val_imgs, self.val_labels = val_samples['image'], val_samples['target']

#     def on_sanity_check_start(self, trainer, pl_module):
#         self._sanity_check = True

#     def on_sanity_check_end(self, trainer, pl_module):
#         self._sanity_check = False
    
        
#     def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
#         if ("loss" not in outputs) or self._sanity_check:
#             return
        
# #         loss = outputs["loss"]
#         y_true = batch[1]
# #         loss = nn.CrossEntropyLoss(reduction="none")(outputs["y_logit"], y_true).cpu().numpy()
        
#         loss = nn.CrossEntropyLoss(reduction="none")(outputs["y_logit"], y_true).cpu().numpy()        
        
    
#         y_pred = [pl_module.label_encoder.idx2class[i] for i in outputs["y_pred"].cpu().numpy()]
#         imgs = np.transpose(batch[0].cpu().numpy(), (0,2,3,1))
#         labels = [pl_module.label_encoder.idx2class[i] for i in batch[1].cpu().numpy()]
#         probs = outputs["y_logit"].softmax(dim=1).cpu().numpy().tolist()

#         sorted_idx = np.argsort(loss)#.cpu().numpy())
#         top_k_idx = sorted_idx[:self.top_k_per_batch]
#         bottom_k_idx = sorted_idx[::-1][:self.bottom_k_per_batch]
#         top_k = len(top_k_idx)

#         trainer.logger.experiment.log({"epoch":trainer.current_epoch,
#                                        **{f"bottom_k_per_batch":
#                                     wandb.Image(imgs[k,:,:,:], caption=f"Pred:{y_pred[k]}, Label:{labels[k]}, prob: {np.max(probs[k]):.4f}, loss:{loss[k]:.4f}")
#                                     for k in bottom_k_idx}
#                                       }, commit=False)

#         trainer.logger.experiment.log({"epoch":trainer.current_epoch,
#                                        **{f"top_k_per_batch":
#                                     wandb.Image(imgs[k,:,:,:], caption=f"Pred:{y_pred[k]}, Label:{labels[k]}, prob: {np.max(probs[k]):.4f}, loss:{loss[k]:.4f}") 
#                                     for k in top_k_idx}
#                                       }, commit=False)
