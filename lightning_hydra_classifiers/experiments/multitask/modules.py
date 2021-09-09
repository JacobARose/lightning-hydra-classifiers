"""

experiments/multitask/modules.py


Pytorch Lightning Module extensions for multitask experiments.

Created by: Friday September 3rd, 2021
Author: Jacob A Rose


"""




import pytorch_lightning as pl
from munch import Munch
from torch import nn
import torch
from typing import Union, Optional, List
from lightning_hydra_classifiers.models.backbones import backbone
from lightning_hydra_classifiers.utils.metric_utils import get_per_class_metrics, get_scalar_metrics


__all__ = ["LitMultiTaskModule"]


class LitMultiTaskModule(pl.LightningModule):
    
    def __init__(self, config): #, ckpt_path: Optional[str]=None):
        super().__init__()
#         config = Munch(config)
#         self.config = config
        self.lr = config.lr
        self.num_classes = config.num_classes
        self.save_hyperparameters({"config":dict(config)})
        
        self.init_model(config)
        self.metrics = self.init_metrics(stage='all')
        self.criterion = nn.CrossEntropyLoss()

    def update_config(self, config):
        self.config.update(Munch(config))
        self.hparams.config.update(Munch(config))
        
    def forward(self, x, *args, **kwargs):
        return self.model(x)

    @property
    def config(self):
        return self._config
    
    @config.setter
    def config(self, config):
        config = Munch(config)
        self._config = config
        self.lr = config.lr
    
    def configure_optimizers(self):
        print(f"self.hparams={self.hparams}")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)#hparams.lr)
#         self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.t_max, eta_min=self.config.min_lr)

        return {'optimizer': self.optimizer}#, 'lr_scheduler': self.scheduler}

    def step(self, batch, batch_idx):
        image, y_true = batch[0], batch[1]
        y_logit = self.model(image)
        y_pred = torch.argmax(y_logit, dim=-1)
        return y_logit, y_true, y_pred

    def log_metric_step(self,
                        stage: str='train',
                        omit_metric_types: Optional[List[str]]=None,
                        omit_metric_keys: Optional[List[str]]=None):
        omit_metric_types = omit_metric_types or []
        omit_metric_keys = omit_metric_keys or []
        
        for metric_type, metric_collection in self.all_metrics[stage].items():
            if metric_type in omit_metric_types:
                continue
            if metric_type == "scalar":
                self.log_dict({k:v for k,v in metric_collection.items() if k not in omit_metric_keys},
                               on_step=False, on_epoch=True, prog_bar=True, logger=True)

            elif metric_type == "per_class":
                for k,v in metric_collection.items():
                    if k in omit_metric_keys:
                        continue
                    results = v.compute()
                    for class_idx, result in enumerate(results): #range(len(results)):
                        self.log(f"{k}_class_{class_idx}", result,
                                 on_step=False, on_epoch=True, prog_bar=False, logger=True)
            else:
                logging.warning(f"[Warning] {metric_type} requires specialized handling in lightningmodule.log_metric_step().")
    
    
    def update_metric_step(self,
                           y_logit,
                           y_true,
                           stage: str='train'):

            for metric_type, metric_collection in self.all_metrics[stage].items():
                metric_collection(y_logit, y_true)    
    
    
    def training_step(self, batch, batch_idx):
        y_logit, y_true, y_pred = self.step(batch, batch_idx)
        loss = self.criterion(y_logit, y_true)
        self.update_metric_step(y_logit,
                                y_true,
                                stage="train")
        self.log_dict({"train_loss": loss,
                       "train_acc": self.metrics_train["train/acc_top1"],
                       'lr': self.optimizer.param_groups[0]['lr']},
                       on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log_metric_step(stage='train',
                             omit_metric_types=None,
                             omit_metric_keys=None)
        
#         self.log_dict(self.metrics_train,
#                  on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_logit, y_true, y_pred = self.step(batch, batch_idx)
        loss = self.criterion(y_logit, y_true)
#         y_pred = torch.argmax(y_logit, dim=-1)
#         scores = self.metrics_val(y_logit, y_true)

        self.update_metric_step(y_logit,
                                y_true,
                                stage="val")        
        self.log("val_loss", loss,
                  on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", self.metrics_val["val/acc_top1"],
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log_metric_step(stage='val',
                             omit_metric_types=None,
                             omit_metric_keys=["val/acc_top1"])
#         self.log_dict({k:v for k,v in self.metrics_val.items() if k != "val/acc_top1"},
#                  on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return {"loss":loss,
                "y_logit":y_logit,
                "y_pred":y_pred}
    
    
    def test_step(self, batch, batch_idx):
        y_logit, y_true, y_pred = self.step(batch, batch_idx)
        loss = self.criterion(y_logit, y_true)
#         scores = self.metrics_test(y_logit, y_true)

        self.update_metric_step(y_logit,
                                y_true,
                                stage="test")
        self.log_dict({"test_loss": loss},
                       on_step=True, on_epoch=True, logger=True)
        self.log_metric_step(stage='test',
                             omit_metric_types=None,
                             omit_metric_keys=None)
#         self.log_dict(self.metrics_test,
#                  on_step=False, on_epoch=True, logger=True)
        return loss

        
        
#         y_logit, y_true, y_pred = self.step(batch, batch_idx)
#         return {'test_loss': F.cross_entropy(y_hat, y)}

#     def test_end(self, outputs):
#         # OPTIONAL
#         avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
#         tensorboard_logs = {'test_loss': avg_loss}
#         return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}
    
#     def test_epoch_end(self, outputs):
#         avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
#         logs = {'test_loss': avg_loss}
#         return {'test_loss': avg_loss, 'log': logs, 'progress_bar': logs}
    
    
    
    
    def init_model(self, config):
        self.model =  backbone.build_model(model_name=config.model_name,
                                           pretrained=config.pretrained,
                                           num_classes=config.num_classes)
        
        self.freeze_up_to(layer=config.init_freeze_up_to)
        
        
        
    def freeze_up_to(self, layer: Union[int, str]=None):
        
        if isinstance(layer, int):
            if layer < 0:
                layer = len(list(self.model.parameters())) + layer
            
#         self.model.enable_grad = True
        self.model.requires_grad = True
        if not layer:
            return
        for i, (name, param) in enumerate(self.model.named_parameters()):
            
            if isinstance(layer, int):
                if layer == i:
                    break
            elif isinstance(layer, str):
                if layer in name:
                    break
            param.requires_grad = False
    
    
    def init_metrics(self, stage: str='train', tag: Optional[str]=None):
        tag = tag or ""
        if not hasattr(self, "all_metrics"):
            self.all_metrics = {}
        
        if stage in ['train', 'all']:
            prefix=f'{tag}_train'.strip("_")
            self.metrics_train = get_scalar_metrics(num_classes=self.num_classes, average='macro', prefix=prefix)
            self.metrics_train_per_class = get_per_class_metrics(num_classes=self.num_classes, prefix='train')
            self.all_metrics['train'] = {"scalar":self.metrics_train,
                                         "per_class":self.metrics_train_per_class}
            
        if stage in ['val', 'all']:
            prefix=f'{tag}_val'.strip("_")
            self.metrics_val = get_scalar_metrics(num_classes=self.num_classes, average='macro', prefix=prefix)
            self.metrics_val_per_class = get_per_class_metrics(num_classes=self.num_classes, prefix='val')
            self.all_metrics['val'] = {"scalar":self.metrics_val,
                                       "per_class":self.metrics_val_per_class}
            
        if stage in ['test', 'all']:
            prefix=f'{tag}_test'.strip("_")
            self.metrics_test = get_scalar_metrics(num_classes=self.num_classes, average='macro', prefix=prefix)
            self.metrics_test_per_class = get_per_class_metrics(num_classes=self.num_classes, prefix='test')
            self.all_metrics['test'] = {"scalar":self.metrics_test,
                                        "per_class":self.metrics_test_per_class}

#     def reset_metrics(self, stage: str='all'):
        
#         self.all_metrics = self.all_metrics or {}
#         for k, v in all_metrics:
#             try:
#                 del v
#             except:
#                 pass
        

    @classmethod
    def available_backbones(self):
        return backbone.AVAILABLE_MODELS