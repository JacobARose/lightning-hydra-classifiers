"""

lightning_hydra_classifiers/callbacks/class_label_stats_callbacks.py


Pytorch Lightning Callbacks for analyzing class label statistics

Created on: Wednesday Oct 27th, 2021
Author: Jacob A Rose


"""



from torch import nn
import torch
from pathlib import Path
import os
from typing import *
import pytorch_lightning as pl
# from pytorch_lightning.utilities.distributed import rank_zero_only
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import time
from lightning_hydra_classifiers.callbacks.wandb_callbacks import get_wandb_logger
from lightning_hydra_classifiers.utils.template_utils import get_logger
logger = get_logger(name=__name__)


__all__ = ["plot_class_counts", "ClassLabelStatsCallback"]



def plot_class_counts(df,
                      ax=None,
                      figsize=(25,10),
                      alpha=0.8,
                      ticklabel_rotation=40,
                      title: str=None):
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=figsize)
    else:
        fig = plt.gcf()
    sns.barplot(x=df.index, y=df.values, alpha=alpha, ax=ax)
#     sns.barplot(df.index, df.values, alpha=alpha, ax=ax)

    ax.set_xticklabels(ax.get_xticklabels(), 
                       rotation = ticklabel_rotation,
                       ha="right", fontsize="xx-small")
    if isinstance(title, str):
        ax.set_title(title)
    return fig, ax
        
        

class ClassLabelStatsCallback(pl.callbacks.Callback):
    """
    Calculates per-subset class label statistics
    Currently:
        - per-subset class counts bar plot
    
    """
    
    def __init__(self, 
                 name: str="class_stats",
                 subsets: List[str]=["train", "val", "test"],
                 y_col: str="family"):
        super().__init__()
        self.name = name
        self.subsets=subsets
        self.y_col = y_col


    def _plot_all_subsets(self, datamodule):
        figs = []
        axes = []
        for subset in self.subsets:

            df = getattr(datamodule, f"{subset}_dataset").samples_df.value_counts(self.y_col)

            fig, ax = plot_class_counts(df,
                                        ax=None,
                                        figsize=(25,10),
                                        alpha=0.8,
                                        ticklabel_rotation=40,
                                        title=subset)
            figs.append(fig)
            axes.append(ax)
        return figs, axes
        
        
    def on_pretrain_routine_start(self, trainer, pl_module):
        self._start_time = time.time()
#         dataloader = trainer.test_dataloader()
        datamodule = trainer.datamodule
        logger = get_wandb_logger(trainer=trainer)
        figs, axes = self._plot_all_subsets(datamodule)
        
        for subset, fig in zip(self.subsets, figs):
            logger.experiment.log({f"dataset_class_counts/{subset}": wandb.Image(fig)})
            plt.clf()
            plt.close(fig)

#         dataloader = trainer.train_dataloader()
#         self.accumulator.update(dataloader)

        
#     def on_pretrain_routine_end(self, trainer, pl_module):
#         logs = {}
        
#         summary_stats = self.accumulator.compute()
#         logger.info(str(self.accumulator))
#         logger.info("Summary Stats:\n" + f"Global Mean: {summary_stats[0]}, Global Std: {summary_stats[1]}")
#         trainer.datamodule.update_stats(mean=summary_stats[0], std=summary_stats[1])
#         total_time = time.time() - self._start_time
#         trainer.logger.log_metrics(logs, step=trainer.current_epoch)

        

#     def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
#         if (batch_idx + 1) % trainer.log_every_n_steps == 0:
#             x, y = batch[:1]
#             #TBD: Add logic for multiple loggers
#             logger = trainer.logger
#             logger.experiment.add_histogram("input", x, global_step=trainer.global_step)
#             logger.experiment.add_histogram("target", y, global_step=trainer.global_step)
            
            
            
            
        

#     def on_train_epoch_end(self, trainer, pl_module) -> None:
#         logs = {}
#         epoch_time = time.time() - self._start_time
#         trainer.logger.log_metrics(logs, step=trainer.current_epoch)

#         if self._verbose:
#             rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
#             rank_zero_info(f"Average Peak memory: {peak_memory:.2f} MB")
#             rank_zero_info(f"Average Free memory: {free_memory:.2f} MB")