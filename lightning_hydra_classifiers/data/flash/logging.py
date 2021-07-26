"""
contrastive_learning.data.pytorch.flash.logging.py


Added by: Jacob A Rose
Added on: Tuesday, April 14th 2021

"""

from contextlib import contextmanager
from typing import Any, List, Sequence

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.states import RunningStage
from torch import Tensor

import flash


import pytorch_lightning as pl
import torch
import wandb
# source: https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch-lightning/Supercharge_your_Training_with_Pytorch_Lightning_%2B_Weights_%26_Biases.ipynb
# mirrored: https://colab.research.google.com/drive/16bJwZs9DOJz5WE1VlpRGwBhqpuzfN-Ro
class ImagePredictionLogger(pl.Callback):
    """ 
    PyTorch-Lightning Callback for logging images with corresponding model predictions
    
    After every validation_epoch ends, logs input images and output predictions using W&B's Image logger.
    """
    def __init__(self, val_samples, num_samples=32):
        super().__init__()
        self.val_imgs, self.val_labels = val_samples
        self.val_imgs = self.val_imgs[:num_samples]
        self.val_labels = self.val_labels[:num_samples]
          
    def on_validation_epoch_end(self, trainer, pl_module):
        val_imgs = self.val_imgs.to(device=pl_module.device)

        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, 1)

        trainer.logger.experiment.log({
            "examples": [wandb.Image(x, caption=f"Pred:{pred}, Label:{y}") 
                            for x, pred, y in zip(val_imgs, preds, self.val_labels)],
            "global_step": trainer.global_step
            })