"""
logging_utils.py

Created by: Jacob A Rose
Created On: Monday, March 15th, 2021

Contains:

func class_counts(y: np.ndarray, as_dataframe: bool=False) -> Union[Dict[Union[str,int],int],pd.DataFrame]
func log_model_artifact(model, model_path, encoder, run=None, metadata=None):

"""

import datetime
import numpy as np
from boltons.dictutils import OneToOne
import os
import pandas as pd
from pathlib import Path
import wandb
from typing import List, Any, Dict, Sequence, Optional
import pytorch_lightning as pl

import glob
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import wandb
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import BasePredictionWriter
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score








class PredictionWriter(BasePredictionWriter):
    """
    Base class to implement how the predictions should be stored.
    Args:
        write_interval: When to write.
    Example::
        import torch
        from pytorch_lightning.callbacks import BasePredictionWriter
        class CustomWriter(BasePredictionWriter):
            def __init__(self, output_dir: str, write_interval: str):
                super().__init__(write_interval)
                self.output_dir
            def write_on_batch_end(
                self, trainer, pl_module: 'LightningModule', prediction: Any, batch_indices: List[int], batch: Any,
                batch_idx: int, dataloader_idx: int
            ):
                torch.save(prediction, os.path.join(self.output_dir, dataloader_idx, f"{batch_idx}.pt"))
            def write_on_epoch_end(
                self, trainer, pl_module: 'LightningModule', predictions: List[Any], batch_indices: List[Any]
            ):
                torch.save(predictions, os.path.join(self.output_dir, "predictions.pt"))
    """

    def __init__(self, write_interval: str = "epoch") -> None:
        super().__init__(write_interval=write_interval)
        

    def write_on_epoch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        predictions: Sequence[Any],
        batch_indices: Optional[Sequence[Any]],
    ) -> None:
        
        print(type(predictions))

#     def on_predict_epoch_end(
#         self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', outputs: Sequence[Any]
#     ) -> None:
#         if not self.interval.on_epoch:
#             return
#         is_distributed = trainer.accelerator_connector.is_distributed
#         epoch_batch_indices = trainer.predict_loop.epoch_batch_indices if is_distributed else None
#         self.write_on_epoch_end(trainer, pl_module, trainer.predict_loop.predictions, epoch_batch_indices)













def get_wandb_logger(trainer: pl.Trainer) -> pl.loggers.WandbLogger:
    """Safely get Weights&Biases logger from Trainer."""

    if isinstance(trainer.logger, pl.loggers.WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, pl.loggers.LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, pl.loggers.WandbLogger):
                return logger

    raise Exception(
        "You are using wandb related callback, but WandbLogger was not found for some reason..."
    )

    
    
    
    

class UploadCodeAsArtifact(pl.Callback):
    """Upload all *.py files to wandb as an artifact, at the beginning of the run."""

    def __init__(self, code_dir: str):
        self.code_dir = code_dir

    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        code = wandb.Artifact("project-source", type="code")
        for path in glob.glob(os.path.join(self.code_dir, "**/*.py"), recursive=True):
            code.add_file(path)

        experiment.use_artifact(code)


class UploadCheckpointsAsArtifact(pl.Callback):
    """Upload checkpoints to wandb as an artifact, at the end of run."""

    def __init__(self,
                 ckpt_dir: str = "checkpoints/",
                 upload_best_only: bool = False):
        self.ckpt_dir = ckpt_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.upload_best_only = upload_best_only

    def on_train_end(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        ckpts = wandb.Artifact("experiment-ckpts", type="checkpoints")

        if self.upload_best_only:
            ckpts.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for path in glob.glob(os.path.join(self.ckpt_dir, "**/*.ckpt"), recursive=True):
                ckpts.add_file(path)

        experiment.use_artifact(ckpts)


        
        
        
#####################################        


class LogConfusionMatrix(pl.Callback):
    """Generate confusion matrix every epoch and send it to wandb.
    Expects validation step to return predictions and targets.
    """

    def __init__(self, class_names: List[str]=None):
        self.y_prob = []
        self.y_true = []
        self.class_names = class_names
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module) -> None:
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True
        
#     def on_train_start(self, trainer, pl_module):

    def on_validation_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment
        
        class_names = list(range(pl_module.num_classes))
        if hasattr(self, "class_names"):
            class_names = self.class_names
        elif hasattr(pl_module, "classes"):
            class_names = pl_module.classes
        assert isinstance(list(class_names), list), f"class_names is wrong, not a list: type = {type(class_names)}"
        self.class_names = class_names
#         self.table = wandb.Table(columns=class_names)
#         self.__debug = True

#     def on_validation_batch_end(
#         self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
#     ):
#         """Gather data from single batch."""
#         if self.ready:
# #             self.y_prob.append(outputs["y_prob"])
# #             self.y_true.append(outputs["y_true"])
#             pass

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate confusion matrix."""
        if self.ready:
            logger = get_wandb_logger(trainer)
            experiment = logger.experiment
            
            
            f1 = pl_module.metrics_val_per_class['val/F1'].compute().cpu()
#             cm = pl_module.metrics_val_per_class['val/ConfusionMatrix'].compute().cpu()
                
            experiment.log({**{f"val_f1-macro/{name}": score for name, score in zip(self.class_names, list(f1.numpy()))},
                            **{"global_step": trainer.global_step}},
                           commit=False)
                
#             self.table.add_data(*list(f1.numpy()))
            
#             plt.figure(figsize=(14, 8))
#             sns.set(font_scale=1.4)
#             sns.heatmap(cm.numpy(), annot=True, annot_kws={"size": 8}, fmt="g")
#             experiment.log({f"confusion_matrix/{experiment.name}": wandb.Image(plt),
#                             "global_step": trainer.global_step}) #, commit=False)

            # reset plot
#             plt.clf()
            pl_module.metrics_val_per_class.reset()

#     def on_validation_end(self, trainer, pl_module):
        
#         if self.ready:
#             logger = get_wandb_logger(trainer)
#             experiment = logger.experiment
            
#             experiment.log({"val/per-class-F1-macro": self.table,
#                             "epoch": pl_module.current_epoch})
        

#             y_prob = torch.cat(self.y_prob).cpu().numpy()
#             y_true = torch.cat(self.y_true).cpu().numpy()


#             self.y_prob.clear()
#             self.y_true.clear()

            
            
            
            
            
            
            
            
            
            
            
            
            

# class LogF1PrecRecHeatmap(Callback):
#     """Generate f1, precision, recall heatmap every epoch and send it to wandb.
#     Expects validation step to return predictions and targets.
#     """

#     def __init__(self, class_names: List[str] = None):
#         self.preds = []
#         self.targets = []
#         self.ready = True

#     def on_sanity_check_start(self, trainer, pl_module):
#         self.ready = False

#     def on_sanity_check_end(self, trainer, pl_module):
#         """Start executing this callback only after all validation sanity checks end."""
#         self.ready = True

#     def on_validation_batch_end(
#         self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
#     ):
#         """Gather data from single batch."""
#         if self.ready:
#             self.preds.append(outputs["preds"])
#             self.targets.append(outputs["targets"])

#     def on_validation_epoch_end(self, trainer, pl_module):
#         """Generate f1, precision and recall heatmap."""
#         if self.ready:
#             logger = get_wandb_logger(trainer=trainer)
#             experiment = logger.experiment

#             preds = torch.cat(self.preds).cpu().numpy()
#             targets = torch.cat(self.targets).cpu().numpy()
#             f1 = f1_score(preds, targets, average=None)
#             r = recall_score(preds, targets, average=None)
#             p = precision_score(preds, targets, average=None)
#             data = [f1, p, r]

#             # set figure size
#             plt.figure(figsize=(14, 3))

#             # set labels size
#             sn.set(font_scale=1.2)

#             # set font size
#             sn.heatmap(
#                 data,
#                 annot=True,
#                 annot_kws={"size": 10},
#                 fmt=".3f",
#                 yticklabels=["F1", "Precision", "Recall"],
#             )

#             # names should be uniqe or else charts from different experiments in wandb will overlap
#             experiment.log({f"f1_p_r_heatmap/{experiment.name}": wandb.Image(plt)}, commit=False)

#             # reset plot
#             plt.clf()

#             self.preds.clear()
#             self.targets.clear()


# class LogImagePredictions(Callback):
#     """Logs a validation batch and their predictions to wandb.
#     Example adapted from:
#         https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY
#     """

#     def __init__(self, num_samples: int = 8):
#         super().__init__()
#         self.num_samples = num_samples
#         self.ready = True

#     def on_sanity_check_start(self, trainer, pl_module):
#         self.ready = False

#     def on_sanity_check_end(self, trainer, pl_module):
#         """Start executing this callback only after all validation sanity checks end."""
#         self.ready = True

#     def on_validation_epoch_end(self, trainer, pl_module):
#         if self.ready:
#             logger = get_wandb_logger(trainer=trainer)
#             experiment = logger.experiment

#             # get a validation batch from the validation dat loader
#             val_samples = next(iter(trainer.datamodule.val_dataloader()))
#             val_imgs, val_labels = val_samples

#             # run the batch through the network
#             val_imgs = val_imgs.to(device=pl_module.device)
#             logits = pl_module(val_imgs)
#             preds = torch.argmax(logits, axis=-1)

#             # log the images as wandb Image
#             experiment.log(
#                 {
#                     f"Images/{experiment.name}": [
#                         wandb.Image(x, caption=f"Pred:{pred}, Label:{y}")
#                         for x, pred, y in zip(
#                             val_imgs[: self.num_samples],
#                             preds[: self.num_samples],
#                             val_labels[: self.num_samples],
#                         )
#                     ]
#                 }
#             )