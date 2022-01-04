"""
finetune_demo.py


"""


from rich import print as pp
import pandas as pd
import numpy as np
import os
from pathlib import Path

from tqdm.auto import tqdm, trange
import torch
import torch.nn as nn
import timm
import glob
import hydra
from omegaconf import OmegaConf, DictConfig
from collections import OrderedDict
from typing import *
import pytorch_lightning as pl
# from lightning_hydra_classifiers.scripts.multitask.train import load_data, resolve_config, configure_callbacks, configure_loggers
from lightning_hydra_classifiers.utils.experiment_utils import load_data, resolve_config, configure_callbacks, configure_loggers, configure_trainer
from lightning_hydra_classifiers.utils.etl_utils import ETL
from lightning_hydra_classifiers.scripts.pretrain import lr_tuner
from lightning_hydra_classifiers.utils.ckpt_utils import scan_ckpt_dir, load_results_if_previously_completed, build_model_or_load_from_checkpoint
from lightning_hydra_classifiers.utils.template_utils import get_logger
logger = get_logger(__name__)
logger.setLevel("DEBUG") # ('INFO')

# source: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial5/Inception_ResNet_DenseNet.html
# from lightning_hydra_classifiers.callbacks.finetuning_callbacks import FinetuningLightningCallback
from lightning_hydra_classifiers.models.transfer import LightningClassifier




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

    config = get_config(config=config)
    
    ckpt_path = config.model.get("ckpt_path")
    if not os.path.isfile(str(ckpt_path)):
        ckpt_paths = scan_ckpt_dir(config.checkpoint_dir)
        ckpt_path = ckpt_paths[-1]

    pl.seed_everything(config.seed)
    datamodule = load_data(config,
                           task_id=config.get("task_id", 0))
    print(f"datamodule.num_classes={datamodule.num_classes}")
    config.model.update({"num_classes":datamodule.num_classes})
    config.lr_tuner_dir = os.path.join(config.results_dir, f"task_{task_id}", "lr_tuner")
    
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.lr_tuner_dir, exist_ok=True)
    return config, datamodule


    
def pretrain_hook(trainer,
                  model,
                  datamodule,
                  config,
                  group: str="finetuning_trials",
                  run: Optional=None) -> Dict[str,Any]:
    logs = {}

    if config.trainer.auto_lr_find:
        logs["lr_tune_output"] = lr_tuner.run_lr_tuner(trainer=trainer,
                                                       model=model,
                                                       datamodule=datamodule,
                                                       config=config,
                                                       results_dir=config.lr_tuner_dir,
                                                       group=group)
        
    return logs


##################
###################


def train(config,
          task_id: int=1,
          logs = {}):
    """
    Inputs:
        model_name - Name of the model you want to run. Is used to look up the class in "model_dict"
        save_name (optional) - If specified, this name will be used for creating the checkpoint and logging directory.
    """

    config, datamodule = get_config_and_load_data(config,
                                                  task_id=task_id)
    
    results = load_results_if_previously_completed(config)
    if (results is not None):
        if (config.model.get("ckpt_mode") ==  "pretrained_backbone_w_new_classifier") and os.path.isfile(str(results.get("ckpt_path"))):
            config.model.ckpt_path = str(results.get("ckpt_path"))
        else:
            return results, config
    
    config.callbacks.log_per_class_metrics_to_wandb.class_names = datamodule.classes
    callbacks = configure_callbacks(config)
    logger = configure_loggers(config)    
    trainer = configure_trainer(config, callbacks=callbacks, logger=logger)
    pp(config)
    model, config = configure_model(config, label_encoder)
    # model = build_model_or_load_from_checkpoint(ckpt_path=config.model.ckpt_path,
    #                                             ckpt_dir=config.model.ckpt_dir,
    #                                             ckpt_mode=config.model.ckpt_mode,
    #                                             config=config)
    # if not getattr(model, "label_encoder", None):
    #     model.label_encoder = datamodule.label_encoder
    
    logs["pretrain"] = pretrain_hook(trainer,
                                     model,
                                     datamodule,
                                     config,
                                     # group="finetuning_trials",
                                     run=None)#run)

    trainer.fit(model, datamodule=datamodule)
    ckpt_path = trainer.checkpoint_callback.best_model_path
    model = build_model_or_load_from_checkpoint(ckpt_path=config.model.ckpt_path,
                                                ckpt_dir=config.model.ckpt_dir,
                                                ckpt_mode=config.model.ckpt_mode,
                                                config=config)
    logs["ckpt"] = {"ckpt_path":config.model.ckpt_path,
                    "ckpt_dir":config.model.ckpt_dir,
                    "ckpt_mode":config.model.ckpt_mode}
    logs["test_results"], model = test_model(model, trainer, datamodule)
    pp(f"FINAL RESULTS: {logs['test_results']}")

    return logs, model, trainer

################################################
################################################


def test_model(model,
               trainer,
               datamodule):
    if os.path.isfile(trainer.checkpoint_callback.best_model_path):
        model = LightningClassifier.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        print(f"Loading best checkpoint from ckpt_path: {trainer.checkpoint_callback.best_model_path}")

    model.init_metrics(stage="test", tag="train")
    train_result = trainer.test(model, test_dataloaders=datamodule.train_dataloader(), verbose=False)
    model.init_metrics(stage="test", tag="val")
    val_result = trainer.test(model, test_dataloaders=datamodule.val_dataloader(), verbose=False)
    model.init_metrics(stage="test", tag="test")
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





@hydra.main(config_name="finetune_config", config_path="../../configs")
def main(config):

    pl.seed_everything(config.seed)
    OmegaConf.set_struct(config, False)
    os.makedirs(config.callbacks.model_checkpoint.dirpath, exist_ok=True)    
    os.environ["WANDB_ENTITY"] = "jrose"
#     os.environ["WANDB_PROJECT"] = "image_classification_train"
    os.environ["WANDB_DIR"] = config.experiment_dir        
#     torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if config.run_trial in ["train_classifier"]:
        logs, model, trainer = train(config, task_id=config.task_id)
    results = logs["test_results"]
    print(f"Final checkpoint saved to: {results['ckpt_path']}")
    return ["test_acc"]


        
if __name__ == '__main__':
    
    main()

##################################################
##################################################

# def lightning_checkpoint_connector(ckpt_path: Optional[str]=None,
#                                    **kwargs) -> Optional[LightningClassifier]:
#     # pretrained_filename = config.trainer.resume_from_checkpoint #config.checkpoint_dir
#     if os.path.isfile(str(ckpt_path)):
#         print(f"Found pretrained lightning checkpoint model at {ckpt_path}, loading...")
#         return LightningClassifier.load_from_checkpoint(ckpt_path, **kwargs) # Automatically loads the model with the saved hyperparameters
    

# def pretrained_model_checkpoint_connector(ckpt_path: Optional[str]=None,
#                                           **kwargs) -> Optional[LightningClassifier]:
#     if os.path.isfile(str(ckpt_path)):
#         print(f"Found pretrained custom model checkpoint at {ckpt_path}, loading...")
#         return LightningClassifier.load_model_from_checkpoint(ckpt_path, **kwargs)


# def initialize_model_from_scratch_connector(**kwargs) -> Optional[LightningClassifier]:
#     return LightningClassifier(**kwargs)
#         # model.label_encoder = datamodule.label_encoder

# def pretrained_model_from_imagenet_connector(**kwargs) -> Optional[LightningClassifier]:
#     return LightningClassifier(**kwargs)


# def pretrained_backbone_w_new_classifier_connector(ckpt_path: Optional[str]=None,
#                                                    new_num_classes: Optional[int]=None,
#                                                   **kwargs) -> Optional[LightningClassifier]:
#     return LightningClassifier.init_pretrained_backbone_w_new_classifier(ckpt_path,
#                                                                          new_num_classes=new_num_classes,
#                                                                          **kwargs)
    
    
# CKPT_MODES = {"lightning_checkpoint":lightning_checkpoint_connector,
#               "pretrained_model_checkpoint":pretrained_model_checkpoint_connector,
#               "pretrained_backbone_w_new_classifier":pretrained_backbone_w_new_classifier_connector,
#               "initialize_model_from_scratch":initialize_model_from_scratch_connector,
#               "pretrained_model_from_imagenet":pretrained_model_from_imagenet_connector}
    
    
# def build_model_or_load_from_checkpoint(ckpt_path: Optional[str]=None,
#                                         ckpt_dir: Optional[str]=None,
#                                         ckpt_mode: Optional[str]=None,
#                                         config=None) -> Optional[LightningClassifier]:
    
#     if os.path.isdir(str(ckpt_dir)) and (not os.path.isfile(str(ckpt_path))):
#         ckpt_paths, ckpt_path = scan_ckpt_dir(ckpt_dir)
#     config.model.update({"ckpt_path":ckpt_path})
#     if ckpt_mode in CKPT_MODES:
#         try:
#             model = CKPT_MODES[ckpt_mode](#ckpt_path=ckpt_path,
#                                           **config.model)
#         except Exception as e:
#             print(e, f"Chosen ckpt_mode={ckpt_mode} did not work, cycling through other options.")
            
#     else:
#         for ckpt_mode in CKPT_MODES:
#             try:
#                 model = CKPT_MODES[ckpt_mode](#ckpt_path=ckpt_path,
#                                               **config.model)
#             except Exception as e:
#                 print(e, f"Chosen ckpt_mode={ckpt_mode} did not work, cycling through remaining options.")
                
#     return model


# def load_results_if_previously_completed(config) -> Optional[DictConfig]:
#     """
#     Checks if a results.yaml file has previously been saved in order to circumvent previously completed trials.
#     """
#     if os.path.isfile(os.path.join(config.results_dir, "results.yaml")):
#         results_file_path = os.path.join(config.results_dir, "results.yaml")
#         results = OmegaConf.load(results_file_path)
#         print(f"Found pre-existing results saved to file: {results_file_path}")
#         print(f"Results:"); pp(results)
#         return results
    
#     results_file_path = str(config.source.get("results_filepath"))
#     if os.path.isfile(results_file_path):
#         results = OmegaConf.load(results_file_path)
#         print(f"Found results from source training stage saved to file: {results_file_path}")
#         print(f"Results:"); pp(results)
#         return results
        
# def scan_ckpt_dir(ckpt_dir):
#     ckpt_paths = [os.path.join(ckpt_dir, ckpt) for ckpt in os.listdir(ckpt_dir)]
#     if len(ckpt_paths):
#         print(f"Found {len(ckpt_paths)} ckpts:" + "\n" + f"{ckpt_paths}")
#         last_ckpt = ckpt_paths[-1]
#         return ckpt_paths, last_ckpt
#     return ckpt_paths, None




    
    
    
    

##################################################
##################################################



    # if config.trainer.auto_lr_find:
    #     lr_tune_output = lr_tuner.run_lr_tuner(trainer=trainer,
    #                                            model=model,
    #                                            datamodule=datamodule,
    #                                            config=config,
    #                                            results_dir=config.lr_tuner_dir,
    #                                            group="target_data_lr_tuner")

#     trainer.fit(model, datamodule=datamodule)
    
#     results, model = test_model(model, trainer, datamodule)
    
#     pp(f"FINAL RESULTS: {results}")

#     model.cpu()
#     del model

#     results['model_config'] = OmegaConf.to_container(config.model, resolve=True)
#     results['data_config'] = OmegaConf.to_container(config.data, resolve=True)
#     results['hparams_config'] = OmegaConf.to_container(config.get("hparams",{}), resolve=True)
    
#     ETL.config2yaml(results, os.path.join(config.results_dir, "results.yaml"))
#     print(f"[SAVED TRIAL RESULTS] Location: {os.path.join(config.results_dir, 'results.yaml')}")
#     pp(results)
#     return results, config