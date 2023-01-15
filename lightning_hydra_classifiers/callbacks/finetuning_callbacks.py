"""

finetuning_callbacks.py


Created by: Jacob A Rose
Created on: Sunday Oct 24th, 2021

"""

import pytorch_lightning as pl
import torch
import numpy as np
from typing import *

class FinetuningLightningCallback(pl.callbacks.Callback):
    
# class FinetuningLightningPlugin:
    mode_dict = {"min": torch.lt, "max": torch.gt}
    order_dict = {"min": "<", "max": ">"}
    
    
    def __init__(self,
                 monitor: str="val_loss",
                 mode: str="min",
                 patience: int=2,
                 init_patience: int=2):
        """
        Arguments:
            monitor: str="val_loss",
            mode: str="min",
            patience: int=2
                After unfreezing the first milestone, use patience
            init_patience: int=2
                Up until unfreezing the first milestone, use init_patience
                
        
        """
        
#         if pl_module.hparams.finetuning_strategy == "finetuning_unfreeze_layers_on_plateau":
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.init_patience = init_patience
#         self.best_metric = 0
        self.milestone_index = 0
        
#         self.min_delta *= 1 if self.monitor_op == torch.gt else -1
        torch_inf = torch.tensor(np.Inf)
        self.best_score = torch_inf if self.monitor_op == torch.lt else -torch_inf
    
        self.milestone_logs = []
        
    def on_fit_start(self,
                     trainer,
                     pl_module):
        self.milestones = pl_module.finetuning_milestones
        print(f"Setting milestones: {pl_module.finetuning_milestones}")
        self._finished = False    
        self._reinit_optimizer_and_schedule = False
    
    def finetuning_pretrained_strategy(self,
                                       trainer: "pl.Trainer",
                                       pl_module):
        """
        
        
        """
        epoch = trainer.current_epoch
        logs = trainer.callback_metrics
        current = logs.get(self.monitor)
        
        if self.mode == "min":
            new_best = current < self.best_score
        elif self.mode == "max":
            new_best = current > self.best_score
        
        if self._finished:
            return
        
        if new_best:
            self.best_score = current
            self.wait_epochs = 0
            print(f"New best score: {self.monitor}={self.best_score}.")
        elif self.wait_epochs >= self.get_patience():
            
            next_to_unfreeze = self.milestones[self.milestone_index]
            print(f"Patience of {self.get_patience()} surpassed at epoch: {epoch} unfreezing down to: {next_to_unfreeze}")
            
            pl_module.unfreeze_backbone_top_layers(unfreeze_down_to=next_to_unfreeze)
            self.wait_epochs = 0
            self.milestone_index += 1
            self.milestone_logs.append({"epoch":epoch,
                                        "unfreeze_at_layer":next_to_unfreeze,
                                        "trainable_params":pl_module.get_trainable_parameters(count_params=True),
                                        "nontrainable_params":pl_module.get_nontrainable_parameters(count_params=True)})
            if self.milestone_index >= len(self.milestones):
                self._finished = True
        else:
            self.wait_epochs += 1
    
    def get_patience(self):
        if self.milestone_index == 0:
            return self.init_patience
        return self.patience
    
    @property
    def monitor_op(self) -> Callable:
        return self.mode_dict[self.mode]
    
    def on_epoch_end(self, trainer, pl_module):
        """Called when the epoch ends."""
        if not trainer.training:
            return

        self.finetuning_pretrained_strategy(trainer=trainer, pl_module=pl_module)
        self._reinit_optimizer_and_schedule = True
        try:
            pl_module.log("nontrainable_params", pl_module.get_nontrainable_parameters(count_params=True))
            pl_module.log("trainable_params", pl_module.get_trainable_parameters(count_params=True))
#             pl_module.logger.summary["milestones"] = self.milestone_logs[-1]
        except Exception as e:
            print(e)
            print(f"logging to wandb didnt work bro")
            
            
    def on_train_epoch_start(self, trainer, pl_module):
        if self._reinit_optimizer_and_schedule:
#             print(f"Reinitializing optimizer and schedulers at epoch: {trainer.current_epoch}")
#             opts, scheds = pl_module.configure_optimizers()
#             trainer.optimizers = opts
#             trainer.lr_schedulers = trainer.configure_schedulers(scheds)
#             trainer.optimizer_frequencies = [] # or optimizers frequencies if you have any
#             print('skipping reinit optimizers and schedulers')
            self._reinit_optimizer_and_schedule = False
