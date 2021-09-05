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
from typing import Union
from lightning_hydra_classifiers.models.backbones import backbone
from lightning_hydra_classifiers.utils.metric_utils import get_per_class_metrics, get_scalar_metrics


__all__ = ["LitMultiTaskModule"]


class LitMultiTaskModule(pl.LightningModule):
    
    def __init__(self, config): #, ckpt_path: Optional[str]=None):
        super().__init__()
        config = Munch(config)
        self.config = config
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

    def configure_optimizers(self):
        print(f"self.hparams={self.hparams}")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)#hparams.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.t_max, eta_min=self.config.min_lr)

        return {'optimizer': self.optimizer, 'lr_scheduler': self.scheduler}

    def step(self, batch, batch_idx):
        image, y_true = batch[0], batch[1]
        y_logit = self.model(image)
        y_pred = torch.argmax(y_logit, dim=-1)
        return y_logit, y_true, y_pred

    
    def training_step(self, batch, batch_idx):
        y_logit, y_true, y_pred = self.step(batch, batch_idx)
        loss = self.criterion(y_logit, y_true)
        scores = self.metrics_train(y_logit, y_true)
        self.log_dict({"train_loss": loss, 'lr': self.optimizer.param_groups[0]['lr']},
                      on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(self.metrics_train,
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_logit, y_true, y_pred = self.step(batch, batch_idx)
        loss = self.criterion(y_logit, y_true)
#         y_pred = torch.argmax(y_logit, dim=-1)
        scores = self.metrics_val(y_logit, y_true)
        
        self.log("val_loss", loss,
                  on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", self.metrics_val["val/acc_top1"],
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict({k:v for k,v in self.metrics_val.items() if k != "val/acc_top1"},
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return {"loss":loss,
                "y_logit":y_logit,
                "y_pred":y_pred}
    
    
    def test_step(self, batch, batch_idx):
        y_logit, y_true, y_pred = self.step(batch, batch_idx)
        loss = self.criterion(y_logit, y_true)
        scores = self.metrics_test(y_logit, y_true)
        self.log_dict("test_loss", loss,
                      on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(self.metrics_test,
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
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
        for i, (name, param) in enumerate(self.model.named_parameters()):
            
            if isinstance(layer, int):
                if layer == i:
                    break
            elif isinstance(layer, str):
                if layer in name:
                    break
            param.requires_grad = False
    
    
    def init_metrics(self, stage: str='train'):
        
        if stage in ['train', 'all']:
            self.metrics_train = get_scalar_metrics(num_classes=self.num_classes, average='macro', prefix='train')
#             self.metrics_train_per_class = get_per_class_metrics(num_classes=self.num_classes, prefix='train')
            
        if stage in ['val', 'all']:
            self.metrics_val = get_scalar_metrics(num_classes=self.num_classes, average='macro', prefix='val')
#             self.metrics_val_per_class = get_per_class_metrics(num_classes=self.num_classes, prefix='val')
            
        if stage in ['test', 'all']:
            self.metrics_test = get_scalar_metrics(num_classes=self.num_classes, average='macro', prefix='test')
#             self.metrics_test_per_class = get_per_class_metrics(num_classes=self.num_classes, prefix='test')


