#!/usr/bin/env python
# coding: utf-8

# # Model Training with WandB Versioned Data `[version 2]`
# 
# `model_training_w_wandb_versioned_data-[version 2]-refactor_input-and-output_artifact_config.ipynb`
# 
# Author: Jacob A Rose  
# Created on: Wednesday May 12th, 2021
# 
# -----
# Based on [this](https://colab.research.google.com/drive/1PRnwxttjPst6OmGiw9LDoVYRDApbcIjE) notebook (or see the original [report](https://wandb.ai/stacey/mendeleev/reports/DSViz-for-Image-Classification--VmlldzozNjE3NjA)), in which the [iNaturalist 2017](https://github.com/visipedia/inat_comp/tree/master/2017) dataset is re-sampled to several different versions, each with different #s of samples per class.
# * These are then used to train and validate a keras model, then log many image predictions to W&B so that the versioned input images are directly associated with each new log. 
# * This allows a flexible and scalable opportunity for researchers and even untrained members of the public to easily interact with the data and interrogate model decisions.
# 
# * Users can query subsets of the prediction results using basic logical queries w/ an effective autocomplete providing real-time suggestions for valid queries.

# ----------------------------
# 



from pathlib import Path
import os
import wandb
# os.environ['WANDB_CACHE_DIR'] = "/media/data/jacob/wandb_cache"
# os.makedirs("/media/data/jacob/wandb_cache", exist_ok=True)
# os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import numpy as np
import pytorch_lightning as pl
import torchvision
import torch
from torch import nn
import matplotlib.pyplot as plt
from contrastive_learning.data.pytorch.datamodules import fetch_datamodule_from_dataset_artifact
from contrastive_learning.data.pytorch.utils.file_utils  import ensure_dir_exists
from lightning_hydra_classifiers.utils.train_basic_utils import (build_model,
                                                                 build_and_log_model_to_artifact,
                                                                 use_model_artifact,
                                                                 log_model_artifact,
                                                                 run_test_model,
                                                                 configure_trainer)
import lightning_hydra_classifiers
from tqdm.auto import trange, tqdm
from typing import Tuple, List, Dict, Any, Type, Union
from enum import Enum
from stuf import stuf
from box import Box
from rich import print as pp
from omegaconf import DictConfig, ListConfig, OmegaConf
########################################
# from pytorch_lightning import LightningModule, LightningDataModule, Callback, Trainer
# from pytorch_lightning.loggers import LightningLoggerBase
# from pytorch_lightning import seed_everything
import hydra
from lightning_hydra_classifiers.utils import template_utils
from copy import deepcopy
log = template_utils.get_logger(__name__)
# os.environ['WANDB_CACHE_DIR'] = "/media/data/jacob/wandb_cache"
# os.makedirs("/media/data/jacob/wandb_cache", exist_ok=True)
# os.environ['CUDA_VISIBLE_DEVICES'] = "2"

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('Using device:', device)
# if device.type == 'cuda':
#     print(torch.cuda.get_device_name(0))
#     print('Memory Usage:')
#     print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
#     print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

########################################





def run_full_tuned_experiment(config: DictConfig):
    backup_config = OmegaConf.create(config)
    OmegaConf.set_struct(backup_config, False)
    
    config, best_hparams = run_batch_size_tuner(config=config, pipeline_stage="0")
    torch.cuda.empty_cache()
    
    backup_config.datamodule.batch_size = int(config.datamodule.batch_size)
    config = OmegaConf.create(backup_config)
    OmegaConf.set_struct(config, False)
    
    #### lr
    suggestion, results, config = run_lr_tuner(config=config, pipeline_stage="1")
    torch.cuda.empty_cache()
    
    backup_config.model.optimizer.lr = float(config.model.optimizer.lr)
    config = OmegaConf.create(backup_config)
    OmegaConf.set_struct(config, False)
    
    ### train
    config = run_train(config=config,
                       pipeline_stage="2")
    torch.cuda.empty_cache()
    ## Part III: Test

    config = OmegaConf.create(backup_config)
    OmegaConf.set_struct(config, False)
    fix_catalog_number = "PNAS" in config.datamodule.basename
    prediction_artifact, test_results = run_test_model(config,
                                                       fix_catalog_number=fix_catalog_number,
                                                       predictions_only=False,
                                                       pipeline_stage="3")
    
    

    
################################
################################




# torch.backends.cudnn.benchmark = True
def efficient_zero_grad(model):
    """
    [source] https://ai.plainenglish.io/best-performance-tuning-practices-for-pytorch-3ef06329d5fe
    """
    for param in model.parameters():
        param.grad = None

def run_batch_size_tuner(config: Box,
                         pipeline_stage: Union[str,int]="0"):
    """
    Batch size tuner
    """
    pipeline_stage = str(pipeline_stage)
    stage_config = config.wandb[pipeline_stage]
    
    if not config.tuner.trainer_kwargs.auto_scale_batch_size:
        print(f'config.trainer.auto_scale_batch_size is set to False, Skipping run_batch_size_tuner().')
        print(f'Proceeding with batch size = {config.model.batch_size}')
        return config, None
    
    if os.path.isfile(stage_config.init.config.results_path) and not config.tuner.options.batch_size.force_rerun:
        
        best_hparams = OmegaConf.load(stage_config.init.config.results_path)
        
        best_bsz = best_hparams['batch_size']
        config.datamodule.batch_size = best_bsz
        assert config.model.batch_size == best_bsz
        
        print(f'[FOUND] Previously completed trial. Results located in file:\n`{stage_config.init.config.results_path}`')
        print(f'[LOADING] Previous results + avoiding repetition of tuning procedure.')
        print(f'Proceeding with batch_size = {config.model.batch_size}')
        print('Model hparams =')
        pp(best_hparams)
        
        return config, best_hparams
        
#         with open(stage_config.init.config.results_path, 'r') as f:
#             best_hparams = yaml.safe_load(f)
#         best_bsz = best_hparams['batch_size']
#         config.datamodule.batch_size = best_bsz
#         assert config.model.batch_size == best_bsz
    
    os.environ["WANDB_PROJECT"] = stage_config.init.project
    run = wandb.init(entity=stage_config.init.entity,
                     project=stage_config.init.project,
                     job_type=stage_config.init.job_type,
                     group=stage_config.init.group,
                     dir=stage_config.init.run_dir,
                     config=OmegaConf.to_container(stage_config.init.config, resolve=True))

    datamodule, data_artifact = fetch_datamodule_from_dataset_artifact(dataset_config=config.datamodule,
                                                                       artifact_config=config.artifacts.input_dataset_artifact,
                                                                       run_or_api=run)
    config.model.num_classes = len(datamodule.classes)
    # assert (num_classes == 19) | (num_classes == 179)
    ########################################
    ########################################
    model=None
    source_num_classes = config.artifacts.input_model_artifact.reset_classifier.source_num_classes
    target_num_classes = config.artifacts.input_model_artifact.reset_classifier.target_num_classes
    if source_num_classes != target_num_classes:
#         model = build_model(config.model)

        
        model, model_artifact = build_and_log_model_to_artifact(model_config=config.model,
                                            artifact_config=config.artifacts.input_model_artifact,
                                            run_or_api=run)
    else:
        
        model, model_artifact = use_model_artifact(artifact_config=config.artifacts.input_model_artifact,
                                                   model_config=config.model,
                                                   run_or_api=run)

    
    trainer = configure_trainer(config)

    
    bsz_tuner = trainer.tune(model, datamodule=datamodule) #, **config.tuner.tuner_kwargs.batch_size)
    best_bsz = model.hparams.batch_size
        
    
#     ensure_dir_exists(os.path.dirname(stage_config.init.config.results_path))
#     best_hparams = DictConfig({"optimized_hparam_key": "batch_size",
#                                   "lr":best_lr,
#                                   "batch_size":config.model.batch_size,
#                                   "input_shape": model.hparams.input_shape,
#                                   "image_size":config.datamodule.image_size})
    best_hparams = OmegaConf.merge({"optimized_hparam_key": "batch_size"},
                                    DictConfig(model.hparams))

    results_dir = os.path.dirname(stage_config.init.config.results_path)
    ensure_dir_exists(results_dir)
    OmegaConf.save(best_hparams, stage_config.init.config.results_path, resolve=True)
    
    print(f'Saved best batch_size value == {best_bsz} (along w/ batch_size, image_size) to file located at: {stage_config.init.config.results_path}')
    print(f'File contents expected to contain: \n{best_hparams}')
#     with open(stage_config.init.config.results_path, 'w') as fp:
#         yaml.dump({"optimized_hparam_key": "batch_size",
#                    **dict(model.hparams)}, fp)
            
    config.datamodule.batch_size = best_bsz
    assert config.model.batch_size == best_bsz

#     run.config.update(OmegaConf.to_container(config), allow_val_change=True)
    run.summary['best_batch_size'] = config.model.batch_size
    run.summary['image_size'] = config.datamodule.image_size

    run.finish()
    
    print(f'FINISHED: `run_batch_size_tuner(config, pipeline_stage={pipeline_stage})`')
    print(f'Proceeding with batch size = {config.model.batch_size}')
    
    del datamodule
    del model
    del trainer
    
    return config, best_hparams # datamodule, model





def run_lr_tuner(config: Box,
                 pipeline_stage: Union[str,int]="0"):
    """
    Learning rate tuner
    """
    pipeline_stage = str(pipeline_stage)
    stage_config = config.wandb[pipeline_stage]

    if not config.tuner.trainer_kwargs.auto_lr_find:
        print(f'config.trainer.auto_lr_find is set to False, Skipping `run_lr_tuner(config, pipeline_stage={pipeline_stage})`')
        print(f'Proceeding with:\n')
        print(f'Learning rate = {config.model.lr:.3e}')
        print(f'Batch size = {config.model.batch_size}')
        
        return config.model.lr, None, config

    if os.path.isfile(stage_config.init.config.results_path) and not config.tuner.options.lr.force_rerun:
#         with open(stage_config.init.config.results_path, 'r') as f:
#             best_hparams = yaml.safe_load(f)
            
        best_hparams = OmegaConf.load(stage_config.init.config.results_path)
            
        best_lr = best_hparams['lr']
        config.model.optimizer.lr = best_lr
        assert config.model.lr == best_lr

        print(f'[FOUND] Previously completed trial. Results located in file:\n`{stage_config.init.config.results_path}`')
        print(f'[LOADING] Previous results + avoiding repetition of tuning procedure.')
        print(f'Proceeding with learning rate, lr = {config.model.optimizer.lr}')
        print('Model hparams =')
        pp(best_hparams)
        return config.model.lr, None, config
    
    
    os.environ["WANDB_PROJECT"] = stage_config.init.project
    run = wandb.init(entity=stage_config.init.entity,
                     project=stage_config.init.project,
                     job_type=stage_config.init.job_type,
                     group=stage_config.init.group,
                     dir=stage_config.init.run_dir,
                     config=OmegaConf.to_container(stage_config.init.config, resolve=True))

    datamodule, data_artifact = fetch_datamodule_from_dataset_artifact(dataset_config=config.datamodule,
                                                                       artifact_config=config.artifacts.input_dataset_artifact,
                                                                       run_or_api=run)
    config.model.num_classes = len(datamodule.classes)
    # assert (num_classes == 19) | (num_classes == 179)
    ########################################
    ########################################
#     model, model_artifact = build_and_log_model_to_artifact(model_config=config.model,
#                                         artifact_config=config.artifacts.input_model_artifact,
#                                         run_or_api=run)

    model, model_artifact = use_model_artifact(artifact_config=config.artifacts.input_model_artifact,
                                               model_config=config.model,
                                               run_or_api=run)


    trainer = configure_trainer(config)
    
    try:
        if model.hparams.batch_size is None:
            model.hparams.batch_size = config.model.batch_size
#         model.hparams = OmegaConf.create(model.hparams) #, resolve=True)
        print('Continuing with model.hparams:', model.hparams)
    except Exception as e:
        print(e)
        print('conversion from Omegaconf failed', model.hparams)
        print('continuing')    
    
    lr_tuner = trainer.tuner.lr_find(model, datamodule, **config.tuner.tuner_kwargs.lr)

    # TODO: pickle lr_tuner object
    lr_tuner_results = lr_tuner.results
    best_lr = lr_tuner.suggestion()
    
    suggestion = {"lr": best_lr,
                  "loss":lr_tuner_results['loss'][lr_tuner._optimal_idx]}

    model.hparams.lr = suggestion["lr"]
    config.model.optimizer.lr = model.hparams.lr
    config.model.lr = model.hparams.lr
#     run.config.update(config, allow_val_change=True)
    
        
    best_hparams = DictConfig({"optimized_hparam_key": "lr",
                                  "lr":best_lr,
                                  "batch_size":config.model.batch_size,
                                  "input_shape": model.hparams.input_shape,
                                  "image_size":config.datamodule.image_size})
    
    results_dir = os.path.dirname(stage_config.init.config.results_path)
    ensure_dir_exists(results_dir)
    OmegaConf.save(best_hparams, stage_config.init.config.results_path, resolve=True)
    print(f'Saved best lr value (along w/ batch_size, image_size) to file located at: {stage_config.init.config.results_path}')
    print(f'File contents expected to contain: \n{OmegaConf.to_yaml(best_hparams)}')
    
        
    fig = lr_tuner.plot(suggest=True)
    plot_fname = 'lr_tuner_results_loss-vs-lr.png'
    plot_path = os.path.join(results_dir, plot_fname)
    plt.suptitle(f"Suggested lr={best_lr} |\n| Searched {lr_tuner.num_training} lr values $\in$ [{lr_tuner.lr_min},{lr_tuner.lr_max}] |\n| bsz = {config.model.batch_size}")
    plt.savefig(plot_path)
    run.summary['results_plot'] = wandb.Image(fig, caption=plot_fname)
    
    
    run.summary['best/loss'] = suggestion["loss"]
    run.summary['best/lr'] = suggestion["lr"]
    run.summary['batch_size'] = config.model.batch_size
    run.summary['image_size'] = config.datamodule.image_size
    run.summary['results'] = OmegaConf.to_container(best_hparams, resolve=True)

    run.finish()
    
    
    print(f'FINISHED: `run_lr_tuner(config, pipeline_stage={pipeline_stage})`')
    print(f'Proceeding with:\n')
    print(f'Learning rate = {config.model.lr:.3e}')
    print(f'Batch size = {config.model.batch_size}')
    
    del datamodule
    del model
    del trainer
    
    return suggestion, lr_tuner_results, config # datamodule, model




def run_train(config: Box,
              pipeline_stage: Union[str,int]="0"):
    """
    
    """
    
    
    pipeline_stage = str(pipeline_stage)
    stage_config = config.wandb[pipeline_stage]
    
    os.environ["WANDB_PROJECT"] = stage_config.init.project
    run = wandb.init(entity=stage_config.init.entity,
                     project=stage_config.init.project,
                     job_type=stage_config.init.job_type,
                     group=stage_config.init.group,
                     dir=stage_config.init.run_dir,
                     config=OmegaConf.to_container(config, resolve=True))

    datamodule, data_artifact = fetch_datamodule_from_dataset_artifact(dataset_config=config.datamodule,
                                                                       artifact_config=config.artifacts.input_dataset_artifact,
                                                                       run_or_api=run)
    config.model.num_classes = len(datamodule.classes)
    # assert (num_classes == 19) | (num_classes == 179)

    model, model_artifact = use_model_artifact(model=None,
                                               artifact_config=config.artifacts.input_model_artifact,
                                               model_config=config.model,
                                               run_or_api=run)


    trainer = configure_trainer(config, log_gpu_memory=True)

    trainer.fit(model, datamodule=datamodule)

    best_model_ckpt = trainer.callbacks[-1].best_model_path

    trainer.model = trainer.model.load_from_checkpoint(best_model_ckpt)
    trainer.save_checkpoint(config.artifacts.output_model_artifact.model_path)
    wandb.save(config.artifacts.output_model_artifact.model_path)

    model_artifact = log_model_artifact(artifact_config=config.artifacts.output_model_artifact,
                                        model_config=config.model,
                                        run_or_api=run,
                                        finalize=False)

    csv_logger = trainer.logger.experiment[1]
    try:
        run.config.update(OmegaConf.to_container(config, resolve=True), allow_val_change=True)
        model_artifact.add_dir(csv_logger.log_dir)
    except ValueError:
        pass
    run.log_artifact(model_artifact)
    
    run.finish()
    print(f'Finished train!')
    
    del trainer
    del model
    del datamodule
    
    return config
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def train(config: Box):
    """
    
    """
    return None
    
#     os.environ["WANDB_PROJECT"] = config.wandb.init.project
#     run = wandb.init(entity='jrose',
#                      project=config.wandb.init.project,
#                      job_type=config.wandb.init.job_type,
#                      group=config.wandb.init.group,
#                      dir=config.wandb.init.run_dir,
#                      config=config)

#     ########################################
#     #### Instantiate model ####
#     ########################################

# #     datamodule, data_artifact = fetch_datamodule_from_dataset_artifact(dataset_config=config.dataset,
# #                                                                        artifact_config=config.
# #                                                                        run_or_api=None
#                                                                        # (config=config, run_or_api=run)

#     # assert (num_classes == 19) | (num_classes == 179)

#     model, model_artifact = build_or_restore_model_from_artifact(model_config=model_config,
#                                                                  artifact_config=artifact_config,
#                                                                  run_or_api=run)


def run_auto_scale_batch_size(model, datamodule, config: Box):
    
    if config.trainer.auto_scale_batch_size:
        trainer = configure_trainer(config)
        bsz_tuner = trainer.tune(model, datamodule=datamodule)

        config.model.batch_size = model.hparams.batch_size
    
    
        
    if config.trainer.auto_lr_find:
        trainer = configure_trainer(config)
        lr_finder = trainer.tuner.lr_find(model, datamodule, num_training=config.trainer.auto_lr_num_training)

        trainer.tuner.lr_find


    new_lr = lr_finder.suggestion()

    print(f'Suggested starting learning rate: {new_lr:.2e}')
    model.hparams.lr = new_lr
    config.model.optimizer.init_lr = new_lr
    fig = lr_finder.plot(suggest=True)
#     wandb.
    fig.show()

    
    
    
    

    trainer = configure_trainer(config, log_gpu_memory=True) # limit_train_batches=2, limit_val_batches=2)

    trainer.fit(model, datamodule=datamodule)
    best_ckpt = trainer.callbacks[-1].best_model_path


#     log_model_checkpoint_2_artifact(model, 
#                                     ckpt_path,
#                                     artifact_config=config.wandb.output_artifacts[0],
#                                     run=run)


    fix_catalog_number = "PNAS" in config.dataset.name
    trainer = configure_trainer(config) #, limit_test_batches=2)

    test_results = test_model(trainer,
                              model,
                              output_model_artifact,
                              datamodule,
                              config,
                              fix_catalog_number=fix_catalog_number)



    wandb.finish()
    print(trainer.current_epoch, test_results)





    ##########################################
    ##########################################

#     fix_catalog_number = "PNAS" in config.dataset.name
#     trainer = configure_trainer(config) #, limit_test_batches=2)

#     test_results = test_model(trainer,
#                               model,
#                               output_model_artifact,
#                               datamodule,
#                               config,
#                               fix_catalog_number=fix_catalog_number)



#     wandb.finish()
#     print(trainer.current_epoch, test_results)









############################################
############################################

# if __name__ == "__main__":








# from typing import List, Optional
# from pytorch_lightning import LightningModule, LightningDataModule, Callback, Trainer
# from pytorch_lightning.loggers import LightningLoggerBase
# from pytorch_lightning import seed_everything
# import hydra
# from omegaconf import DictConfig
# from lightning_hydra_classifiers.utils import template_utils


# log = template_utils.get_logger(__name__)

# def train(config: DictConfig) -> Optional[float]:
#     """Contains training pipeline.
#     Instantiates all PyTorch Lightning objects from config.

#     Args:
#         config (DictConfig): Configuration composed by Hydra.

#     Returns:
#         Optional[float]: Metric score for hyperparameter optimization.
#     """

#     # Set seed for random number generators in pytorch, numpy and python.random
#     if "seed" in config:
#         seed_everything(config.seed)

#     # Init Lightning datamodule
#     log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
#     datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

#     # Init Lightning model
#     log.info(f"Instantiating model <{config.model._target_}>")
#     model: LightningModule = hydra.utils.instantiate(config.model)

#     # Init Lightning callbacks
#     callbacks: List[Callback] = []
#     if "callbacks" in config:
#         for _, cb_conf in config["callbacks"].items():
#             if "_target_" in cb_conf:
#                 log.info(f"Instantiating callback <{cb_conf._target_}>")
#                 callbacks.append(hydra.utils.instantiate(cb_conf))

#     # Init Lightning loggers
#     logger: List[LightningLoggerBase] = []
#     if "logger" in config:
#         for _, lg_conf in config["logger"].items():
#             if "_target_" in lg_conf:
#                 log.info(f"Instantiating logger <{lg_conf._target_}>")
#                 logger.append(hydra.utils.instantiate(lg_conf))

#     # Init Lightning trainer
#     log.info(f"Instantiating trainer <{config.trainer._target_}>")
#     trainer: Trainer = hydra.utils.instantiate(
#         config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
#     )

#     # Send some parameters from config to all lightning loggers
#     log.info("Logging hyperparameters!")
#     template_utils.log_hyperparameters(
#         config=config,
#         model=model,
#         datamodule=datamodule,
#         trainer=trainer,
#         callbacks=callbacks,
#         logger=logger,
#     )

#     import pdb; pdb.set_trace()
    
    
#     # Train the model
#     log.info("Starting training!")
#     trainer.fit(model=model, datamodule=datamodule)

#     # Evaluate model on test set after training
#     if not config.trainer.get("fast_dev_run"):
#         log.info("Starting testing!")
#         trainer.test()

#     # Make sure everything closed properly
#     log.info("Finalizing!")
#     template_utils.finish(
#         config=config,
#         model=model,
#         datamodule=datamodule,
#         trainer=trainer,
#         callbacks=callbacks,
#         logger=logger,
#     )

#     # Print path to best checkpoint
#     log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

#     # Return metric score for Optuna optimization
#     optimized_metric = config.get("optimized_metric")
#     if optimized_metric:
#         return trainer.callback_metrics[optimized_metric]











########################################
########################################
########################################






########################################

# # Notes
# 

# ### Left to do 5-16-2021
# 
# 1. trainer.fit ***
# 2. trainer.test ***
# 
# 3. Refactor ValLog Callback to log PyTorch Lightning predictions + per-class scores *****
# 
# 4. Add data tests for WandB downloaded data \*\*\*
# 
#     4a. Implement Data Drift framework!!!
#     
# 5. Add Salient Map interpretability callback \*\*\*
#     * https://github.com/MisaOgura/flashtorch
#     
# priority: \*, \*\*,\*\*\*, ****, *****


# ## Friday, May 21st 2021
# 
# * TODO: Perform bootstrapping and/or k-val split for this ~50-100 sample learning rate search
# 
# * Idea: What other hyperparameters can we constrain/further optimize with these wide->narrow sweeps




########################################################

########################################################
########################################################

########################################################



# def build_model(config: Box) -> pl.LightningModule:
    
#     model = ResNet(model_name=config.name,
#                    num_classes=config.num_classes,
#                    input_shape=config.input_shape,
#                    optimizer=config.optimizer)
#     model.reset_classifier(config.num_classes,'avg')
#     model.unfreeze(getattr(model, config.unfreeze[0]))
#     return model


# def log_model_artifact(model,
#                        artifact_config: Box,
#                        model_config: Box=None,
#                        run_or_api=None):
#     run = run_or_api or wandb.Api()
#     model_config = model_config or {}
    
#     artifact_config.init_model_path = Path(artifact_config.init_model_dir,
#                                            artifact_config.version,
#                                            "imagenet")
#     os.makedirs(os.path.dirname(artifact_config.init_model_path), exist_ok=True)
#     model.save_model(artifact_config.init_model_path)

#     artifact = wandb.Artifact(artifact_config.init_name,
#                               type=artifact_config.input_type,
#                               description=artifact_config.description,
#                               metadata=dict({
#                                             "artifact":artifact_config,
#                                              "model":model_config
#                                            })
#                              )
#     artifact.add_dir(os.path.dirname(artifact_config.init_model_path))
#     run.log_artifact(model_config)
    
#     return artifact



# # def log_model_checkpoint_2_artifact(model,
# #                                     artifact_config: Box,
# #                                     model_config: Box=None,
# #                                     run_or_api=None):
    
# def load_model_from_checkpoint_artifact(model,
#                                     artifact_config: Box,
#                                     model_config: Box=None,
#                                     run_or_api=None):
    
#     run = run or wandb.run
#     os.makedirs(os.path.dirname(artifact_config.output_model_path), exist_ok=True)
    
#     model = model.load_from_checkpoint(ckpt_path)
#     model.save_model(artifact_config.output_model_path)

#     output_model_artifact = wandb.Artifact(
#                                     artifact_config.output_name,
#                                     type=artifact_config.output_type,
#                                     description=artifact_config.description,
#                                     metadata=dict(**artifact_config,
#                                                   **config.model)
#                                     )
#     output_model_artifact.add_dir(artifact_config.output_model_dir)
#     run.log_artifact(output_model_artifact)

#     print("Output Model Artifact Checkpoint path:\n", 
#           artifact_config.output_model_path)


########################################################

########################################################

# def build_and_log_model_to_artifact(model_config: Box,
#                                     artifact_config: Box,
#                                     run_or_api=None) -> Tuple[pl.LightningModule, wandb.Artifact]:
#     """
    
    
#     """
#     model = build_model(model_config)

#     model_artifact = log_model_artifact(model, artifact_config, model_config, run_or_api=run_or_api)

#     return model, model_artifact


# def get_labels_from_filepath(path: str, fix_catalog_number: bool = False) -> Dict[str,str]:
#     """
#     Splits a precisely-formatted filename with the expectation that it is constructed with the following fields separated by '_':
#     1. family
#     2. genus
#     3. species
#     4. collection
#     5. catalog_number
    
#     If fix_catalog_number is True, assume that the collection is not included and must separately be extracted from the first part of the catalog number.
    
#     """
#     family, genus, species, collection, catalog_number = Path(path).stem.split("_", maxsplit=4)
#     if fix_catalog_number:
#         catalog_number = '_'.join([collection, catalog_number])
#     return {"family":family,
#             "genus":genus,
#             "species":species,
#             "collection":collection,
#             "catalog_number":catalog_number}


# def log_predictions(trainer, model, datamodule, fix_catalog_number=True):
#     subset = datamodule.predict_on_split
#     datamodule.setup(stage="predict")
#     classes = datamodule.classes
    
#     columns = ['catalog_number',
#                'image',
#                'guess',
#                'truth',
#                'softmax_guess',
#                'softmax_truth']
#     for j, class_name in enumerate(classes):
#         columns.append(f'score_{class_name}')

    
#     loss_func = torch.nn.CrossEntropyLoss(reduction='none')
#     batches = trainer.predict(model, datamodule=datamodule)
# #         x, y_logit, y_true, img_paths = list(np.concatenate(i) for i in zip(*batches))    
#     prediction_rows = []
#     for x, y_logit, y_true, img_paths in tqdm(batches, position=0, desc="epoch->", unit='batches'):

#         y_true = torch.from_numpy(y_true).to(dtype=torch.long)
#         y_logit = torch.from_numpy(y_logit)

#         loss = loss_func(y_logit, y_true)

#         y_pred = torch.argmax(y_logit, -1)
#         y_score = y_logit.softmax(1)

#         x = torch.from_numpy(x).permute(0,2,3,1)
#         x = (255 * (x - x.min()) / (x.max() - x.min())).numpy().astype(np.uint8)
# #         for i in tqdm(range(len(y_pred)), position=1, desc="batch->", unit='sample'):
#         for i in trange(len(y_pred), position=1, leave=False, desc="batch->", unit='sample'):
#             labels = get_labels_from_filepath(path=img_paths[i],
#                                               fix_catalog_number=fix_catalog_number)
#             row = [
#                     labels['catalog_number'],
#                     wandb.Image(x[i,...]),
#                     classes[y_pred[i]],
#                     classes[y_true[i]],
#                     y_score[i,y_pred[i]],
#                     y_score[i,y_true[i]]
#             ]

#             for j, score_j in enumerate(y_score[i,:].tolist()):
#                 row.append(np.round(score_j, 4))
#             prediction_rows.append(row)


#     prediction_table = wandb.Table(data=prediction_rows, columns=columns)
#     artifact_name = f"{wandb.run.name}_{wandb.run.id}"
#     prediction_artifact = wandb.Artifact(artifact_name,
#                                          type=f"{subset}_predictions")
    
#     prediction_artifact.add(prediction_table, f"{subset}_predictions")
#     wandb.run.log_artifact(prediction_artifact)
#     return prediction_artifact



# ########################################
# ########################################

# def test_model(trainer,
#                model,
#                model_artifact,
#                datamodule,
#                config,
#                fix_catalog_number=False, predictions_only=False):

#     run = wandb.init(entity=config.wandb.init.entity,
#                      project=config.wandb.init.project,
#                      job_type="test",
#                      group=config.wandb.init.group,
#                      dir=config.wandb.init.run_dir,
#                      config=config)
    
# #     model_artifact.wait()
# #     model_input_artifact = run.use_artifact(model_artifact)
    
#     model_input_artifact = run.use_artifact(config.wandb.model_artifact.output_name + ":latest")
#     model_input_dir = model_input_artifact.download(config.wandb.model_artifact.output_model_dir)
    
#     assert Path(config.wandb.model_artifact.output_model_path).name in os.listdir(model_input_dir)
# #     model.load_from_checkpoint(config.wandb.model_artifact.output_model_path)
#     model.load_model(config.wandb.model_artifact.output_model_path)
    
    
#     if not predictions_only:
#         test_results = trainer.test(model, datamodule=datamodule)
#         if isinstance(test_results, list):
#             assert len(test_results) == 1
#             test_results = test_results[0]
            
#     prediction_artifact = log_predictions(trainer, model, datamodule, fix_catalog_number=fix_catalog_number)
    
#     run.finish()    
#     return prediction_artifact, test_results

# #### Callbacks + Trainer


# def configure_trainer(config, **kwargs):
#     """
#     Example:
#         trainer = configure_trainer(config)
#     """
#     monitor = ModuleDataMonitor()#log_every_n_steps=25)
#     per_class_metric_plots_cb = LogPerClassMetricsToWandb()
#     early_stop_callback = EarlyStopping(monitor='val_loss',
#                                         patience=3,
#                                         verbose=False,
#                                         mode='min')

#     logger=pl.loggers.wandb.WandbLogger(name=f"{config.dataset.name}-timm-{config.model.name}",
#                                         config=config)

#     trainer = pl.Trainer(gpus=config.trainer.gpus,
#                          logger=logger,
#                          max_epochs=config.trainer.max_epochs,
#                          weights_summary=config.trainer.weights_summary,
#                          profiler=config.trainer.profiler,
#                          log_every_n_steps=config.trainer.log_every_n_steps,
#                          auto_scale_batch_size=config.trainer.auto_scale_batch_size,
#                          callbacks=[per_class_metric_plots_cb,
#                                     monitor,
#                                     early_stop_callback],
#                         **kwargs)
#     return trainer




# def log_model_checkpoint_2_artifact(model, 
#                                     ckpt_path: str,
#                                     artifact_config,
#                                     run=None):
    
#     run = run or wandb.run
#     os.makedirs(os.path.dirname(artifact_config.output_model_path), exist_ok=True)
    
#     model = model.load_from_checkpoint(ckpt_path)
#     model.save_model(artifact_config.output_model_path)

#     output_model_artifact = wandb.Artifact(
#                                     artifact_config.output_name,
#                                     type=artifact_config.output_type,
#                                     description=artifact_config.description,
#                                     metadata=dict(**artifact_config,
#                                                   **config.model)
#                                     )
#     output_model_artifact.add_dir(artifact_config.output_model_dir)
#     run.log_artifact(output_model_artifact)

#     print("Output Model Artifact Checkpoint path:\n", 
#           artifact_config.output_model_path)




# def build_or_restore_model_from_artifact(model_config: Box,


