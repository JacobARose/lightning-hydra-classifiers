"""


"""


from typing import List



__all__ = ["ImagePredictionLogger", "LogF1PrecRecHeatmap", "LogConfusionMatrix",
           "UploadCheckpointsAsArtifact", "UploadCodeAsArtifact",
           "WatchModel", "get_wandb_logger"]

import pytorch_lightning as pl
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from pytorch_lightning.utilities import rank_zero_only



import wandb
from pathlib import Path
import subprocess
import torch
from torch import nn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np


def get_wandb_logger(trainer: pl.Trainer) -> WandbLogger:
    """Safely get Weights&Biases logger from Trainer."""

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    raise Exception("You are using wandb related callback, but WandbLogger was not found for some reason...")

    
    

class WatchModel(pl.Callback):
    """Make wandb watch model at the beginning of the run."""

    def __init__(self, log: str = "gradients", log_freq: int = 100):
        self.log = log
        self.log_freq = log_freq

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        logger.watch(model=trainer.model, log=self.log, log_freq=self.log_freq)

        
        
class UploadCodeAsArtifact(pl.Callback):
    """Upload all code files to wandb as an artifact, at the beginning of the run."""

    def __init__(self, code_dir: str, use_git: bool = True):
        """

        Args:
            code_dir: the code directory
            use_git: if using git, then upload all files that are not ignored by git.
            if not using git, then upload all '*.py' file
        """
        self.code_dir = code_dir
        self.use_git = use_git

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        code = wandb.Artifact("project-source", type="code")

        if self.use_git:
            # get .git folder
            # https://alexwlchan.net/2020/11/a-python-function-to-ignore-a-path-with-git-info-exclude/
            git_dir_path = Path(
                subprocess.check_output(["git", "rev-parse", "--git-dir"]).strip().decode("utf8")
            ).resolve()

            for path in Path(self.code_dir).resolve().rglob("*"):
                if (
                    path.is_file()
                    # ignore files in .git
                    and not str(path).startswith(str(git_dir_path))  # noqa: W503
                    # ignore files ignored by git
                    and (  # noqa: W503
                        subprocess.run(["git", "check-ignore", "-q", str(path)]).returncode == 1
                    )
                ):
                    code.add_file(str(path), name=str(path.relative_to(self.code_dir)))

        else:
            for path in Path(self.code_dir).resolve().rglob("*.py"):
                code.add_file(str(path), name=str(path.relative_to(self.code_dir)))

        experiment.log_artifact(code)

        
        
        
class UploadCheckpointsAsArtifact(pl.Callback):
    """Upload checkpoints to wandb as an artifact, at the end of run."""

    def __init__(self, ckpt_dir: str = "checkpoints/", upload_best_only: bool = False):
        self.ckpt_dir = ckpt_dir
        self.upload_best_only = upload_best_only

    @rank_zero_only
    def on_keyboard_interrupt(self, trainer, pl_module):
        self.on_train_end(trainer, pl_module)

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        ckpts = wandb.Artifact("experiment-ckpts", type="checkpoints")

        if self.upload_best_only:
            ckpts.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for path in Path(self.ckpt_dir).rglob("*.ckpt"):
                ckpts.add_file(str(path))

        experiment.log_artifact(ckpts)        
        
        

class LogConfusionMatrix(pl.Callback):
    """Generate confusion matrix every epoch and send it to wandb.
    Expects validation step to return predictions and targets.
    """

    def __init__(self):
        self.preds = []
        self.targets = []
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module) -> None:
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Gather data from single batch."""
        if self.ready:
            self.preds.append(outputs["y_pred"].detach().cpu())
            self.targets.append(outputs["y_true"].detach().cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate confusion matrix."""
        if self.ready:
            logger = get_wandb_logger(trainer)
            experiment = logger.experiment

            preds = torch.cat(self.preds).numpy()
            targets = torch.cat(self.targets).numpy()

            confusion_matrix = metrics.confusion_matrix(y_true=targets, y_pred=preds)

            plt.figure(figsize=(14, 8))
            sns.set(font_scale=1.4)
            sns.heatmap(confusion_matrix, annot=True, annot_kws={"size": 8}, fmt="g")
            experiment.log({f"confusion_matrix/{experiment.name}": wandb.Image(plt)}, commit=False)
            plt.clf()
            self.preds.clear()
            self.targets.clear()

            
            

class LogF1PrecRecHeatmap(pl.Callback):
    """Generate f1, precision, recall heatmap every epoch and send it to wandb.
    Expects validation step to return predictions and targets.
    """

    def __init__(self, class_names: List[str] = None):
        self.preds = []
        self.targets = []
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Gather data from single batch."""
        if self.ready:
            self.preds.append(outputs["y_pred"].detach().cpu())
            self.targets.append(outputs["y_true"].detach().cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate f1, precision and recall heatmap."""
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment

            preds = torch.cat(self.preds).numpy()
            targets = torch.cat(self.targets).numpy()
            f1 = f1_score(preds, targets, average=None)
            r = recall_score(preds, targets, average=None)
            p = precision_score(preds, targets, average=None)
            data = [f1, p, r]

            # set figure size
            plt.figure(figsize=(14, 3))

            # set labels size
            sns.set(font_scale=1.2)

            # set font size
            sns.heatmap(
                data,
                annot=True,
                annot_kws={"size": 10},
                fmt=".3f",
                yticklabels=["F1", "Precision", "Recall"],
            )

            # names should be uniqe or else charts from different experiments in wandb will overlap
            experiment.log({f"f1_p_r_heatmap/{experiment.name}": wandb.Image(plt)}, commit=False)

            # reset plot
            plt.clf()

            self.preds.clear()
            self.targets.clear()


# def get_wandb_logger(trainer: pl.Trainer):
#     logger = trainer.logger
#     if isinstance(logger, list):
#         for l in logger:
#             if isinstance(l, pl.loggers.wandb.WandbLogger):
#                 return l
#     else:
#         return logger






class ImagePredictionLogger(pl.Callback):
    
    """
    To do: Switch to a running stack of top k or bottom k samples seen since the start of each epoch
    """
    def __init__(self, 
                 top_k_per_batch: int=5,
                 bottom_k_per_batch: int=5,
                 frequency: int=50):
        super().__init__()
        self.top_k_per_batch = top_k_per_batch
        self.bottom_k_per_batch = bottom_k_per_batch
        self.frequency = frequency
#         self.num_samples = num_samples
#         self.val_imgs, self.val_labels = val_samples['image'], val_samples['target']

    def on_sanity_check_start(self, trainer, pl_module):
        self._sanity_check = True

    def on_sanity_check_end(self, trainer, pl_module):
        self._sanity_check = False
    
        
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if ("loss" not in outputs) or self._sanity_check:
            return
        if batch_idx % self.frequency != 0:
            return
        
        y_true = batch[1]
        loss = nn.CrossEntropyLoss(reduction="none")(outputs["y_logit"], y_true).cpu().numpy()        
        
    
        y_pred = [pl_module.label_encoder.idx2class[i] for i in outputs["y_pred"].cpu().numpy()]
        imgs = np.transpose(batch[0].cpu().numpy(), (0,2,3,1))
        labels = [pl_module.label_encoder.idx2class[i] for i in batch[1].cpu().numpy()]
        probs = outputs["y_logit"].softmax(dim=1).cpu().numpy().tolist()

        sorted_idx = np.argsort(loss)#.cpu().numpy())
        top_k_idx = sorted_idx[:self.top_k_per_batch]
        bottom_k_idx = sorted_idx[::-1][:self.bottom_k_per_batch]
        top_k = len(top_k_idx)
        
        logger = get_wandb_logger(trainer)

        logger.experiment.log({"epoch":trainer.current_epoch,
                                       **{f"bottom_k_per_batch":
                                    wandb.Image(imgs[k,:,:,:], caption=f"Pred:{y_pred[k]}, Label:{labels[k]}, prob: {np.max(probs[k]):.4f}, loss:{loss[k]:.4f}")
                                    for k in bottom_k_idx}
                                      }, commit=False)

        logger.experiment.log({"epoch":trainer.current_epoch,
                                       **{f"top_k_per_batch":
                                    wandb.Image(imgs[k,:,:,:], caption=f"Pred:{y_pred[k]}, Label:{labels[k]}, prob: {np.max(probs[k]):.4f}, loss:{loss[k]:.4f}") 
                                    for k in top_k_idx}
                                      }, commit=False)






