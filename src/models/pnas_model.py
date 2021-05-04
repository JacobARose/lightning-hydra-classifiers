from typing import Any, List, Optional, Dict

import torch
from torch import nn
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics.classification import Accuracy
import timm
# from src.models.modules.simple_dense_net import SimpleDenseNet


class Classifier(nn.Module):
    """A Classifier class for Co-Tuning.
    Args:
        backbone (torch.nn.Module): Any backbone to extract 2-d features from data.
        num_classes (int): Number of classes.
        head_source (torch.nn.Module): Classifier head of source model.
        head_target (torch.nn.Module, optional): Any classifier head. Use :class:`torch.nn.Linear` by default
        finetune (bool): Whether finetune the classifier or train from scratch. Default: True
    Inputs:
        - x (tensor): input data fed to backbone
    Outputs:
        - y_s: predictions of source classifier head
        - y_t: predictions of target classifier head
    Shape:
        - Inputs: (b, *) where b is the batch size and * means any number of additional dimensions
        - y_s: (b, N), where b is the batch size and N is the number of classes
        - y_t: (b, N), where b is the batch size and N is the number of classes
    """
    tasks = ["source", "target"]
    
    def __init__(self, 
                 backbone: nn.Module, 
                 head_source: Optional[nn.Module] = None,
                 head_target: Optional[nn.Module] = None,
                 num_classes: int=0,
                 finetune: bool=True):
        super().__init__()
        self.automatic_optimization=False
        
        backbone_name = 'resnet50' #'xception41'
        backbone = timm.create_model(backbone_name, pretrained=True)

        
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.num_classes = num_classes
#         self.bottleneck = nn.Sequential(
#             nn.AdaptiveAvgPool2d(output_size=(1, 1)),
#             nn.Flatten()
#         )
        
        num_filters = self.backbone.fc.in_features
        layers = list(self.backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)#, self.bottleneck)

        
        self._features_dim = self.backbone.num_features
        self.head_source = head_source or nn.Linear(self._features_dim, num_classes)
        self.head_target = head_target or nn.Linear(self._features_dim, num_classes)
        self.finetune = finetune
        self.set_current_task(task='source')
        
        self._loss = torch.nn.CrossEntropyLoss()

        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

        self.metric_hist = {
            "train/acc": [],
            "val/acc": [],
            "train/loss": [],
            "val/loss": [],
        }
        
        
    def set_current_task(self, task: float='source'):
        assert task in self.tasks, f"task {task} is not in registered tasks: {self.tasks}"
        
        if task == 'source':
            self._set_source_task()
        if task == 'target':
            self._set_target_task()

    def _set_source_task(self):
        self.head = self.head_source

    def _set_target_task(self):
        self.head = self.head_target
            
            
    @property
    def features_dim(self) -> int:
        """The dimension of features before the final `head` layer"""
        return self._features_dim

    def loss(self, y_pred, y_true):
        return self._loss(y_pred, y_true)
    
    def forward(self, x: torch.Tensor):
        """"""
#         f = self.backbone(x)
#         f = self.bottleneck(f)
        f = self.feature_extractor(x)
        y = self.head(f)
        return y

    def get_parameters(self, base_lr: float=0.1) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
            {"params": self.head_source.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
            {"params": self.bottleneck.parameters(), "lr": 1.0 * base_lr},
            {"params": self.head_target.parameters(), "lr": 1.0 * base_lr},
        ]
        return params
    
    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        acc = self.train_accuracy(preds, targets)
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: Dict[Any, Any]):
        # log best so far train acc and train loss
        self.metric_hist["train/acc"].append(self.trainer.callback_metrics["train/acc"])
        self.metric_hist["train/loss"].append(self.trainer.callback_metrics["train/loss"])
        self.log("train/acc_best", max(self.metric_hist["train/acc"]), prog_bar=False)
        self.log("train/loss_best", min(self.metric_hist["train/loss"]), prog_bar=False)

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_accuracy(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        # log best so far val acc and val loss
        self.metric_hist["val/acc"].append(self.trainer.callback_metrics["val/acc"])
        self.metric_hist["val/loss"].append(self.trainer.callback_metrics["val/loss"])
        self.log("val/acc_best", max(self.metric_hist["val/acc"]), prog_bar=False)
        self.log("val/loss_best", min(self.metric_hist["val/loss"]), prog_bar=False)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_accuracy(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(
            params=self.parameters(), 
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )

    
    
    
    
#         def training_step(self, batch: Any, batch_idx: int):
#         loss, preds, targets = self.step(batch)

#         # log train metrics
#         acc = self.train_accuracy(preds, targets)
# #         self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
# #         self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

#         # we can return here dict with any tensors
#         # and then read it in some callback or in training_epoch_end() below
#         # remember to always return loss from training_step, or else backpropagation will fail!
#         return {"loss": loss, "preds": preds, "targets": targets}

#     def training_epoch_end(self, outputs: List[Any]):
#         # log best so far train acc and train loss
#         self.metric_hist["train/acc"].append(self.trainer.callback_metrics["train/acc"])
#         self.metric_hist["train/loss"].append(self.trainer.callback_metrics["train/loss"])
#         self.log("train/acc_best", max(self.metric_hist["train/acc"]), prog_bar=False)
#         self.log("train/loss_best", min(self.metric_hist["train/loss"]), prog_bar=False)

#     def validation_step(self, batch: Any, batch_idx: int):
#         loss, preds, targets = self.step(batch)

#         # log val metrics
#         acc = self.val_accuracy(preds, targets)
#         self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
#         self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

#         return {"loss": loss, "preds": preds, "targets": targets}

#     def validation_epoch_end(self, outputs: List[Any]):
#         # log best so far val acc and val loss
#         self.metric_hist["val/acc"].append(self.trainer.callback_metrics["val/acc"])
#         self.metric_hist["val/loss"].append(self.trainer.callback_metrics["val/loss"])
#         self.log("val/acc_best", max(self.metric_hist["val/acc"]), prog_bar=False)
#         self.log("val/loss_best", min(self.metric_hist["val/loss"]), prog_bar=False)

#     def test_step(self, batch: Any, batch_idx: int):
#         loss, preds, targets = self.step(batch)

#         # log test metrics
#         acc = self.test_accuracy(preds, targets)
#         self.log("test/loss", loss, on_step=False, on_epoch=True)
#         self.log("test/acc", acc, on_step=False, on_epoch=True)

#         return {"loss": loss, "preds": preds, "targets": targets}

#     def test_epoch_end(self, outputs: List[Any]):
#         pass

#     def configure_optimizers(self):
#         """Choose what optimizers and learning-rate schedulers to use in your optimization.
#         Normally you'd need one. But in the case of GANs or similar you might have multiple.

#         See examples here:
#             https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
#         """
#         return torch.optim.Adam(
#             params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
#         )
