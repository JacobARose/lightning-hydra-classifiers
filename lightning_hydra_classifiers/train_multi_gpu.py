#!/usr/bin/env python
# coding: utf-8

"""

python "/media/data/jacob/GitHub/lightning-hydra-classifiers/notebooks/scaling model training/4_acc=ddp-16bit-precision-2-gpu.py"

"""



# ## Notebook 4. (Must be run as python script)
# 
# file: `train_multi-gpu.py`
# ### ddp - 16bit-precision -- 2-gpu -- tune-batchsize -- tune_lr -- include a batch_size tuning step, then a learning_rate tuning step (both using only 'dp'), before training (using 'ddp') then testing
# 
# Created by: Jacob A Rose  
# Created on: Wednesday July 7th, 2021

# ### Scaling model training series
# 
# A collection of notebooks meant to demonstrate minimal-complexity examples for:
# * Integrating new training methods for scaling up experiments to large numbers in parallel &
# * Making maximum use of hardware resources
# 
# 1. 16bit precision, single gpu, train -> test
# 2. 16bit precision, single gpu, batch_size tune -> train -> test
# 3. 16bit precision, single gpu, batch_size tune -> lr tune -> train -> test
# 4. ddp, 16bit precision, 2x gpus, batch_size tune -> lr tune -> train -> test

# In[ ]:


from typing import Any, List, Optional
import shutil
import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import gc
import rich

import sys
import logging
import hydra
from hydra.experimental import compose, initialize_config_dir
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from rich.logging import RichHandler

import dotenv
import hydra
from omegaconf import DictConfig, OmegaConf

from lightning_hydra_classifiers.utils import template_utils
from pytorch_lightning import LightningModule, LightningDataModule, Callback, Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning import seed_everything

log = template_utils.get_logger(__name__)
dotenv.load_dotenv(override=True)

@hydra.main(config_path="/media/data/jacob/GitHub/lightning-hydra-classifiers/configs/experiment/",
            config_name="4_acc=ddp-16bit-precision-2-gpu_tune-batchsize--tune_lr")
def main(config: DictConfig):

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
#     from lightning_hydra_classifiers.train_basic import run_full_tuned_experiment, train
    from lightning_hydra_classifiers.utils import template_utils
    
    # A couple of optional utilities:
    # - disabling python warnings
    # - easier access to debug mode
    # - forcing debug friendly configuration
    # - forcing multi-gpu friendly configuration
    # You can safely get rid of this line if you don't want those
    template_utils.extras(config)
#     OmegaConf.set_struct(config, False)

#     # Pretty print config using Rich library
#     if config.get("print_config"):
#         template_utils.print_config(config, resolve=True)

        
    return train(config)





def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed)

    os.makedirs(config.log_dir, exist_ok=True)

#     template_utils.init(config)
    
    datamodule, config = configure_datamodule(config)
    
    print(f'config.datamodule.num_classes={config.datamodule.num_classes}')
    print(f'config.hparams.num_classes={config.hparams.num_classes}')
    
#     template_utils.print_config(config, resolve=True)

    model = configure_model(config)

    trainer = configure_trainer(config)

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    template_utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=trainer.callbacks,
        logger=trainer.logger,
    )

    
    # Pretty print config using Rich library
    if config.get("print_config"):
        template_utils.print_config(config, resolve=True)

    
    
    if not config.test_only:
        log.info("Starting training!")
        trainer.fit(model, datamodule=datamodule)    
#     test_results = trainer.test(datamodule=datamodule)

    # Evaluate model on test set after training
    if not config.trainer.get("fast_dev_run"):
        log.info("Starting testing!")
        trainer.test(model, datamodule=datamodule)


    # Make sure everything closed properly
    log.info("Finalizing!")
    template_utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=trainer.callbacks,
        logger=trainer.logger,
    )

    try:
        log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")
        ckpt = trainer.checkpoint_callback.best_model_path
        if os.path.isfile(ckpt):
            model = model.load_from_checkpoint(ckpt)
        else:
            log.error(f"Proceeding without loading from checkpoint: {ckpt}")
        
    except Exception as e:
        print(e)
        print(f'[Error] Verify checkpointing code')
        
    try:
        ckpt = trainer.checkpoint_callback.best_model_path
    except:
        ckpt = None

        
        
    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        log.info(f'optimized_metric={optimized_metric}')
        log.info(f'value: {trainer.callback_metrics[optimized_metric]}')
        if ckpt:
            log.info(f'Best checkpoint: {ckpt}')
            log.info(f'isfile: {os.path.isfile(ckpt)}')
        return trainer.callback_metrics[optimized_metric]









#########################
#########################

# ### Function definitions
# 
# 1. Configure logger (using python's logging module)
# 2. Configure experiment Config (using hydra + omegaconf.DictConfig)
# 3. Configure datamodule (using custom LightningDataModule)
# 4. Configure model (using custom LightningModule)
# 5. Configure trainer (using pl.Trainer, as well as pytorch lightning loggers & callbacks)




def get_standard_python_logger(name: str='notebook',
                               log_path: str=None):
    """
    Set up the standard python logging module for command line debugging
    """
    if log_path is None:
        log_path = f"./{name}.log"
    else:
        os.makedirs(log_path, exist_ok=True)
        log_path = str(Path(log_path, name)) + '.log'

    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stdout
    )
    log = logging.getLogger(name)
    rh = RichHandler(rich_tracebacks=True)
    rh.setLevel(logging.INFO)
    log.addHandler(rh)
    return log


def read_hydra_config(config_dir: str,
                      job_name: str="test_app",
                      config_name: str="experiment") -> DictConfig:
    """
    Read a yaml config file from disk using hydra and return as a DictConfig.
    """
    os.chdir(config_dir)
    with initialize_config_dir(config_dir=config_dir, job_name=job_name):
        cfg = compose(config_name=config_name)
        
    if cfg.get("print_config"):
        template_utils.print_config(cfg, resolve=True)        
    return cfg


def configure_datamodule(config: DictConfig) -> pl.LightningDataModule:
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(config.datamodule)

        
#     import pdb; pdb.set_trace()

    try:
        datamodule.setup(stage="fit")
        config.hparams.classes = datamodule.classes
        config.hparams.num_classes = len(config.hparams.classes)
        
        print(f'config.hparams.num_classes={config.hparams.num_classes}')
    except Exception as e:
        print(e)
        pass
        
    return datamodule, config


def configure_model(config: DictConfig) -> pl.LightningModule:
    log.info(f"Instantiating model <{config.model._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(config.model)
    
    if hasattr(config.hparams.classes, "__len__"):
        model.classes = config.hparams.classes
        log.info(f'Stored {len(model.classes)} class names in the lightning module')
    else:
        model.classes = []
        print(f'proceeding with empty class list in model obj')
    return model


# def configure_tuner(config: DictConfig) -> pl.tuner.tuning.Tuner:
#     config = OmegaConf.create(config)
#     if 'ddp' in config.trainer.accelerator:
#         config.trainer.accelerator = 'dp'
        
#     trainer: pl.Trainer = configure_trainer(config) #hydra.utils.instantiate(trainer_config)
#     tuner: pl.tuner.tuning.Tuner = hydra.utils.instantiate(config.tuner.instantiate, trainer=trainer)
    
#     return tuner



def configure_trainer(config: DictConfig) -> pl.Trainer:

    callbacks: List[pl.Callback] = []
    if "callbacks" in config:
        for cb_name, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                if cb_name == "wandb":
                    callbacks.append(hydra.utils.instantiate(cb_conf, config=OmegaConf.to_container(config, resolve=True)))
                    
                else:
                    callbacks.append(hydra.utils.instantiate(cb_conf))

    logger: List[pl.loggers.LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))


    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: pl.Trainer = hydra.utils.instantiate(config.trainer,
                                                  callbacks=callbacks,
                                                  logger=logger,
                                                  _convert_="partial")
        
    return trainer


if __name__ == "__main__":


    config_path = "/media/data/jacob/GitHub/lightning-hydra-classifiers/configs/experiment/4_acc=ddp-16bit-precision-2-gpu_tune-batchsize--tune_lr.yaml"

    log = get_standard_python_logger(name=Path(config_path).stem,
                                     log_path=Path(config_path).parent / "experiment_logs") # 'notebook_experiment')


    main()
