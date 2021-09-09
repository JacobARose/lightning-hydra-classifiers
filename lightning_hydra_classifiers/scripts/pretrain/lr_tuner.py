"""
lightning_hydra_classifiers/scripts/pretrain/lr_tuner.py


Created on: Friday Sept 3rd, 2021
Author: Jacob A Rose


"""


import pytorch_lightning as pl
import argparse
from omegaconf import DictConfig, OmegaConf
import os
from pathlib import Path
import matplotlib.pyplot as plt
import wandb
from rich import print as pp

from lightning_hydra_classifiers.utils.dataset_management_utils import Extract
from lightning_hydra_classifiers.utils.template_utils import get_logger
############################################
logger = get_logger(name=__name__)


__all__ = ["run_lr_tuner"]



from dataclasses import dataclass

@dataclass
class LRTunerConfig:
    
    min_lr: float = 1e-08
    max_lr: float = 1.0
    num_training: int = 100
    mode: str = 'exponential'
    early_stop_threshold: float = 4.0

DEFAULT_CONFIG = OmegaConf.structured(LRTunerConfig())


def run_lr_tuner(trainer: pl.Trainer,
                 model: pl.LightningModule,
                 datamodule: pl.LightningDataModule,
                 config: argparse.Namespace,
                 results_path: str,
                 group: str=None):
#                  run=None):
    """
    Learning rate tuner
    
    Adapted and refactored from "lightning-hydra-classifiers/lightning_hydra_classifiers/scripts/train_basic.py"
    """
    tuner_config = OmegaConf.create(DEFAULT_CONFIG)
    if "lr_tuner" in config:
        tuner_config = OmegaConf.merge(DEFAULT_CONFIG, dict(config.lr_tuner))
    else:
        for k, v in DEFAULT_CONFIG:
            if k in config:
                tuner_config.update({k:config[k]})
    config = OmegaConf.create(tuner_config)



    if os.path.isfile(results_path): #Path(config.experiment_dir,"lr_finder","hparams.yaml")):# and not config.tuner.options.lr.force_rerun:
        best_hparams = Extract.config_from_yaml(results_path)
        best_lr = best_hparams['lr']
        config.model.lr = best_lr
        assert config.model.lr == best_lr

        logger.info(f'[FOUND] Previously completed trial. Results located in file:\n`{results_path}`')
        logger.info(f'[LOADING] Previous results + avoiding repetition of tuning procedure.')
        logger.info(f'Proceeding with learning rate, lr = {config.model.lr:.3e}')
        logger.info('Model hparams =')
        pp(best_hparams)
        suggestion = {"lr": config.model.lr,
                      "loss": None}
        return suggestion, None, config
    
    try:
        if ("batch_size" not in model.hparams) or (model.hparams.batch_size is None):
            model.hparams.batch_size = config.data.batch_size
#         model.hparams = OmegaConf.create(model.hparams) #, resolve=True)
        logger.info('Using model.hparams:', model.hparams)
    except Exception as e:
        logger.warning(e)
        logger.warning('conversion from Omegaconf failed', model.hparams)
        logger.warning('continuing')
        
    with wandb.init(job_type = "lr_tune",
                    config=dict(config),
                    group=group,
                    reinit=True) as run:
        logger.info(f"[Initiating Stage] lr_tuner")
    
        lr_tuner = trainer.tuner.lr_find(model,
                                         datamodule,
                                         **config)
        # TODO: pickle lr_tuner object
        lr_tuner_results = lr_tuner.results
        best_lr = lr_tuner.suggestion()
    
        suggestion = {"lr": best_lr,
                      "loss":lr_tuner_results['loss'][lr_tuner._optimal_idx]}
        
    model.config.lr = suggestion['lr']
    model.hparams.lr = model.config.lr
    config.model.lr = model.config.lr
    best_hparams = OmegaConf.create({"optimized_hparam_key": "lr",
                                     "lr":best_lr,
                                     "batch_size":config.data.batch_size,
                                     "image_size":config.data.image_size,
                                     "lr_tuner_config":config})
    results_dir = Path(results_path).parent
    os.makedirs(results_dir, exist_ok=True)
    Extract.config2yaml(best_hparams, results_path)
    logger.info(f'Saved best lr value (along w/ batch_size, image_size) to file located at: {str(results_path)}') # {str(results_dir / "hparams.yaml")}')
    logger.info(f'File contents expected to contain: \n{dict(best_hparams)}')    
        
    fig = lr_tuner.plot(suggest=True)
    plot_fname = 'lr_tuner_results_loss-vs-lr.png'
    plot_path = results_dir / plot_fname
    plt.suptitle(f"Suggested lr={best_lr:.4e} |\n| Searched {lr_tuner.num_training} lr values $\in$ [{lr_tuner.lr_min},{lr_tuner.lr_max}] |\n| bsz = {config.data.batch_size}")
    plt.tight_layout()
    plt.savefig(plot_path)
    if run is not None:
#         run.summary['lr_finder/plot'] = wandb.Image(fig, caption=plot_fname)
        run.log('lr_finder/plot', wandb.Image(str(plot_path), caption=plot_fname))
        run.summary['lr_finder/best/loss'] = suggestion["loss"]
        run.summary['lr_finder/best/lr'] = suggestion["lr"]
        run.summary['lr_finder/batch_size'] = config.data.batch_size
        run.summary['image_size'] = config.data.image_size
        run.summary['lr_finder/results'] = dict(best_hparams)

    logger.info(f'FINISHED: `run_lr_tuner(config)`')
    logger.info(f'Proceeding with:\n')
    logger.info(f'Learning rate = {config.model.lr:.3e}')
    logger.info(f'Batch size = {config.data.batch_size}')
    
    return suggestion, lr_tuner_results, config



