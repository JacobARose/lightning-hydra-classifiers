"""
lightning_hydra_classifiers/scripts/pretrain/lr_tuner.py


Created on: Friday Sept 3rd, 2021
Author: Jacob A Rose


"""


import pytorch_lightning as pl
import argparse
from omegaconf import DictConfig, OmegaConf
import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import wandb
from rich import print as pp
from typing import *


from lightning_hydra_classifiers.utils.dataset_management_utils import Extract
from lightning_hydra_classifiers.utils.template_utils import get_logger
############################################
logger = get_logger(name=__name__)


__all__ = ["run_lr_tuner"]



from dataclasses import dataclass, asdict

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
                 results_dir: str,
                 group: str=None,
                 run: Optional=None):
                 # strict_resume: bool=False):
#                  run=None):
    """
    Learning rate tuner
    
    Adapted and refactored from "lightning-hydra-classifiers/lightning_hydra_classifiers/scripts/train_basic.py"
    """
    tuner_config = OmegaConf.create(DEFAULT_CONFIG)

    try:
        cfg = asdict(config)
    except TypeError:
        cfg = OmegaConf.to_container(config, resolve=True)
    finally:
        cfg = dict(config)
    
    if "pretrain" in cfg:
        logger.info(f"Proceeding with overrides merged with default parameters")
#         logger.info(f"overrides: {config.lr_tuner}")
#         logger.info(f"defaults: {tuner_config}")
        tuner_config = OmegaConf.merge(DEFAULT_CONFIG, cfg["pretrain"])
    else:
        for k, v in DEFAULT_CONFIG.items():
            if k in cfg:
                tuner_config.update({k:config[k]})

        config.pretrain = OmegaConf.create(tuner_config)


    results_path = str(Path(results_dir, "results.csv"))
    hparams_path = str(Path(results_dir, "hparams.yaml"))
    if os.path.isfile(hparams_path):
        
        best_hparams = Extract.config_from_yaml(hparams_path)
        results = None
        if os.path.isfile(results_path):
            results = Extract.df_from_csv(results_path)
        
        best_lr = best_hparams['lr']
        if hasattr(model, "config"):
            model.config.lr = best_lr
        

        model.hparams.lr = best_lr
        config.model.lr = best_lr
#         config.model.optimizer.lr = model.config.lr
        
        assert config.model.lr == best_lr

        logger.info(f'[FOUND] Previously completed trial. Results located in file:\n`{results_path}`')
        logger.info(f'[LOADING] Previous results + avoiding repetition of tuning procedure.')
        logger.info(f'Proceeding with learning rate, lr = {config.model.lr:.3e}')
        logger.info('Model hparams =')
        pp(best_hparams)
        suggestion = {"lr": config.model.lr,
                      "loss": None}
        return suggestion, results, config
    
    if run is None:
        run = wandb.init(job_type = "lr_tune",
                         config=cfg,
                         group=group,
                         reinit=True)
        logger.info(f"[Initiating Stage] lr_tuner")
        lr_tuner = trainer.tuner.lr_find(model,
                                         datamodule,
                                         **cfg.get("pretrain", {}))
        lr_tuner_results = lr_tuner.results
        best_lr = lr_tuner.suggestion()
        suggestion = {"lr": best_lr,
                      "loss":lr_tuner_results['loss'][lr_tuner._optimal_idx]}
        
        if hasattr(model, "config"):
            model.config.lr = suggestion['lr']
        model.hparams.lr = suggestion['lr']
        config.model.lr = model.hparams.lr
#         config.model.optimizer.lr = model.hparams.lr
        model.hparams.update(config.model)
        best_hparams = OmegaConf.create({"optimized_hparam_key": "lr",
                                         "lr":best_lr,
                                         "batch_size":config.data.batch_size,
                                         "image_size":config.data.image_size,
                                         "lr_tuner_config":config.pretrain}) #.lr_tuner})
        results_dir = Path(results_path).parent
        os.makedirs(results_dir, exist_ok=True)
        Extract.config2yaml(best_hparams, hparams_path)
        logger.info(f'Saved best lr value (along w/ batch_size, image_size) to file located at: {str(hparams_path)}') # {str(results_dir / "hparams.yaml")}')
        logger.info(f'File contents expected to contain: \n{dict(best_hparams)}')    

        fig = lr_tuner.plot(suggest=True)
        plot_fname = 'lr_tuner_results_loss-vs-lr.png'
        plot_path = results_dir / plot_fname

        plt.suptitle(f"Suggested lr={best_lr:.4e} |\n| Searched {lr_tuner.num_training} lr values $\in$ [{lr_tuner.lr_min},{lr_tuner.lr_max}] |\n| bsz = {config.data.batch_size}", fontsize='small')
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2, top=0.8)
        plt.savefig(plot_path)
        
        if run is not None:
    #         run.summary['lr_finder/plot'] = wandb.Image(fig, caption=plot_fname)
            run.log({'lr_finder/plot': wandb.Image(str(plot_path), caption=plot_fname)})
            run.log({'lr_finder/best/loss': suggestion["loss"]})
            run.log({'lr_finder/best/lr': suggestion["lr"]})
            run.log({'lr_finder/batch_size': config.data.batch_size})
            run.log({'image_size': config.data.image_size})
            run.log({'lr_finder/hparams': OmegaConf.to_container(best_hparams)})
            
            df = pd.DataFrame(lr_tuner.results)
            try:
                Extract.df2csv(df, results_path)
                run.log({"lr_finder/results":wandb.Table(dataframe=df)})
            except Exception as e:
                if hasattr(df, "to_pandas"):
                    run.log({"lr_finder/results":wandb.Table(dataframe=df.to_pandas())})

    logger.info(f'FINISHED: `run_lr_tuner(config)`')
    logger.info(f'Proceeding with:\n')
    logger.info(f'Learning rate = {config.model.lr:.3e}')
    logger.info(f'Batch size = {config.data.batch_size}')
    
    return suggestion, lr_tuner_results, config



