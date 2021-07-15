"""
lightning_hydra_classifiers/models/base.py

Author: Jacob A Rose
Created: Saturday May 29th, 2021

"""

import torch
from torch import nn
import torchmetrics as metrics
# from torchsummary import summary
from pathlib import Path
import numpy as np
# from typing import Any, List, Optional, Dict, Tuple
import pytorch_lightning as pl
import os

from lightning_hydra_classifiers.utils.metric_utils import get_per_class_metrics, get_scalar_metrics
from lightning_hydra_classifiers.utils.logging_utils import get_wandb_logger
import wandb
__all__ = ["BaseModule", "BaseLightningModule"]


class BaseModule(nn.Module):
    """
    Models should subclass this in place of nn.Module. This is a custom base class to implement standard interfaces & implementations across all custom pytorch modules in this library.
    
    Instance methods:
    
    * def forward(self, x):
    * def get_trainable_parameters(self):
    * def get_frozen_parameters(self):

    Class methods:

    * def freeze(cls, model: nn.Module):
    * def unfreeze(cls, model: nn.Module):
    * def initialize_weights(cls, modules):
    * def pack_checkpoint(
                        cls,
                        model=None,
                        criterion=None,
                        optimizer=None,
                        scheduler=None,
                        **kwargs
                        ):
    * def unpack_checkpoint(
                          cls,
                          checkpoint,
                          model=None,
                          criterion=None,
                          optimizer=None,
                          scheduler=None,
                          **kwargs,
                          ):
    * def save_checkpoint(self, checkpoint, path):
    * def load_checkpoint(self, path):
    
    
    
    """
    
    def forward(self, x):
        """
        Identity function by default. Subclasses should redefine this method.
        """
        return x
    
    def get_trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def get_frozen_parameters(self):
        return (p for p in self.parameters() if not p.requires_grad)


    @classmethod
    def freeze(cls, model: nn.Module):
        for param in model.parameters():
            param.requires_grad = False
            
    @classmethod
    def unfreeze(cls, model: nn.Module):
        for param in model.parameters():
            param.requires_grad = True

    @classmethod
    def initialize_weights(cls, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    @classmethod
    def pack_checkpoint(
                        cls,
                        model=None,
                        criterion=None,
                        optimizer=None,
                        scheduler=None,
                        **kwargs
                        ):
        content = {}
        if model is not None:
            content["model_state_dict"] = model.state_dict()
        if criterion is not None:
            content["criterion_state_dict"] = criterion.state_dict()
        if optimizer is not None:
            content["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            content["scheduler_state_dict"] = scheduler.state_dict()
        return content

    @classmethod
    def unpack_checkpoint(
                          cls,
                          checkpoint,
                          model=None,
                          criterion=None,
                          optimizer=None,
                          scheduler=None,
                          **kwargs,
                          ):
        state_dicts = {"model":model,
                       "criterion":criterion,
                       "optimizer":optimizer,
                       "scheduler":scheduler}
        
        for state_dict, part in state_dicts.items():
            if f"{state_dict}_state_dict" in checkpoint and part is not None:
                part.load_state_dict(checkpoint[f"{state_dict}_state_dict"])

    @classmethod
    def save_checkpoint(self, checkpoint, path):
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        return checkpoint
    
    
#     def save_model(self, path:str):
#         path = str(path)
#         if not Path(path).suffix=='.ckpt':
#             path = path + ".ckpt"
#         torch.save(self.state_dict(), path)
        
        
#     def load_model(self, path:str):
#         path = str(path)
#         if not Path(path).suffix=='.ckpt':
#             path = path + ".ckpt"
#         self.load_state_dict(torch.load(path))
        

    
##################################
##################################

##################################
##################################



class BaseLightningModule(pl.LightningModule):
    
    """
    Implements some more custom boiler plate for custom lightning modules
    
    """

    def _init_metrics(self, stage: str='train'):
        
        if stage in ['train', 'all']:
            self.metrics_train = get_scalar_metrics(num_classes=self.num_classes, average='macro', prefix='train')
            self.metrics_train_per_class = get_per_class_metrics(num_classes=self.num_classes, prefix='train')
            
        if stage in ['val', 'all']:
            self.metrics_val = get_scalar_metrics(num_classes=self.num_classes, average='macro', prefix='val')
            self.metrics_val_per_class = get_per_class_metrics(num_classes=self.num_classes, prefix='val')
            
        if stage in ['test', 'all']:
            self.metrics_test = get_scalar_metrics(num_classes=self.num_classes, average='macro', prefix='test')
            self.metrics_test_per_class = get_per_class_metrics(num_classes=self.num_classes, prefix='test')
    
    
    
    

    def training_step(self, batch, batch_idx):
        # 1. Forward pass:
        x, y = batch[:2]
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        
#         y_prob = self.probs(y_hat)
#         y_pred = torch.max(y_prob, dim=1)[1]
        
        return {'loss':loss,
                'log':{
                       'train_loss':loss,
                        'y_hat':y_hat,
#                        'y_prob':y_prob,
#                        'y_pred':y_pred,
                       'y_true':y,
                       'batch_idx':batch_idx
                       }
               }
        
    def training_step_end(self, outputs):
        logs = outputs['log']
        loss = outputs['loss']
        idx = logs['batch_idx']
#         y_prob, y_pred, y = logs['y_prob'], logs['y_pred'], logs['y_true']
        y_hat, y = logs['y_hat'], logs['y_true']
        
        y_prob = self.probs(y_hat.float())
        y_pred = torch.max(y_prob, dim=1)[1]
        
        batch_metrics = self.metrics_train(y_prob, y)
        
        for k in self.metrics_train.keys():
            if 'acc_top1' in k: continue
            self.log(k,
                     self.metrics_train[k],
                     on_step=True,
                     on_epoch=True,
                     logger=True,
                     prog_bar=False)
        
        self.log('train/acc',
                 self.metrics_train['train/acc_top1'], 
                 on_step=True,
                 on_epoch=True,
                 logger=True, 
                 prog_bar=True)
        self.log('train_loss', loss,
                 on_step=True,# on_epoch=True)#,
                 logger=True, 
                 prog_bar=True
                )
        
        return outputs
        
    def validation_step(self, batch, batch_idx):
        x, y = batch[:2]
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        y_prob = self.probs(y_hat)
        y_pred = torch.max(y_prob, dim=1)[1]
        return {'loss':loss,
                'log':{
                       'val_loss':loss,
                       'y_hat':y_hat,
#                        'y_prob':y_prob,
                       'y_pred':y_pred,
                       'y_true':y,
                       'batch_idx':batch_idx
                       }
               }

    def validation_step_end(self, outputs):
        
        logs = outputs['log']
        loss = logs['val_loss']
#         y_prob, y_pred, y = logs['y_prob'], logs['y_pred'], logs['y_true']
#         y_hat = logs['y_hat']        
        y_hat, y = logs['y_hat'], logs['y_true']
#         print("y_hat.dtype=", y_hat.dtype)
        y_prob = self.probs(y_hat.float())
        y_pred = torch.max(y_prob, dim=1)[1]


#         print("y_prob.dtype, y_prob_end.dtype)=", y_prob.dtype, y_prob_end.dtype)
#         print("y_prob.half().dtype, y_prob_end.half().dtype=", y_prob.half().dtype, y_prob_end.half().dtype)
#         print("y_prob.float().dtype, y_prob_end.float().dtype=", y_prob.float().dtype, y_prob_end.float().dtype)
#         print(f'y_prob.is_floating_point() = {y_prob.is_floating_point()}')
#         print("torch.isclose(y_prob.sum(dim=1), torch.ones_like(y_prob.sum(dim=1))).all()=")
#         print(torch.isclose(y_prob.sum(dim=1), torch.ones_like(y_prob.sum(dim=1))).all())
                
        self.metrics_val(y_prob, y)
        self.metrics_val_per_class(y_prob, y)
#         batch_metrics = self.metrics_val(y_pred, y)
#         breakpoint()
        for k in self.metrics_val.keys():
            prog_bar = bool('acc' in k)
            self.log(k,
                     self.metrics_val[k],
                     on_step=False,
                     on_epoch=True,
                     logger=True,
                     prog_bar=prog_bar)

        self.log('val_loss',loss,
                 on_step=False, on_epoch=True,
                 logger=True, prog_bar=True)#,
#                  sync_dist=True)

        outputs['y_prob'] = y_prob #.cpu().numpy()
        outputs['y_true'] = y #.cpu().numpy()
        
        return outputs



    def validation_epoch_end(self, validation_step_outputs):
        
        local_rank = os.environ.get("LOCAL_RANK", 0)
        print(f'local_rank={local_rank}')
        if str(local_rank)!="0":
            print(f'Skipping val/confusion matrix logging on local_rank={local_rank}')
            return None
        
        y_prob, y = [], []
        for batch in validation_step_outputs:
            y_prob.extend(batch['y_prob'].cpu().numpy())
            y.extend(batch['y_true'].cpu().numpy())
            
        
        y_prob=np.stack(y_prob)
        y=np.stack(y)
        
        logger = get_wandb_logger(self.trainer)
        experiment = logger.experiment

        if experiment:
            print(y_prob[0].shape)
            experiment.log({"val/confusion_matrix" : wandb.plot.confusion_matrix(probs=y_prob, #np.concatenate(y_prob,axis=0),
                                                                                 y_true=y, #np.stack(y),
                                                                                 class_names=self.classes,
                                                                                 title="val/confusion_matrix")})#,
#             else:

#                             "global_step": self.trainer.global_step},
#                            commit=False)
        
#         f1 = self.metrics_val_per_class['val/F1'].compute()
#         cm = self.metrics_val_per_class['val/ConfusionMatrix'].compute()
        
#         class_names = list(range(len(f1)))
#         if hasattr(self, "classes"):
#             class_names = self.classes
#         assert isinstance(class_names, list)
        
        
#         logger = get_wandb_logger(trainer)
#         experiment = logger.experiment
        
        
#         table = wandb.Table(columns=class_names)
        
#         for k,v in validation_step_outputs.items():
#             v["log"][]
        
#         self.metrics_val_per_class.reset()
        
    
    def test_step(self, batch, batch_idx):
        x, y = batch[:2]
        y_hat = self(x)
        loss = self.loss(y_hat, y)
#         y_prob = self.probs(y_hat)
#         y_pred = torch.max(y_prob, dim=1)[1]
        return {'loss':loss,
                'log':{
                       'test_loss':loss,
                       'y_hat':y_hat,
#                        'y_prob':y_prob,
#                        'y_pred':y_pred,
                       'y_true':y,
                       'batch_idx':batch_idx
                       }
               }

    def test_step_end(self, outputs):
        
        logs = outputs['log']
        loss = logs['test_loss']
#         y_prob, y_pred, y = logs['y_prob'], logs['y_pred'], logs['y_true']
#         y_hat = logs['y_hat']        
        y_hat, y = logs['y_hat'], logs['y_true']
#         print("y_hat.dtype=", y_hat.dtype)
        y_prob = self.probs(y_hat.float())
        y_pred = torch.max(y_prob, dim=1)[1]
        
        
        self.metrics_test(y_prob, y)
        self.metrics_test_per_class(y_prob, y)
#         batch_metrics = self.metrics_val(y_pred, y)
#         breakpoint()
        for k in self.metrics_test.keys():
            prog_bar = bool('acc' in k)
            self.log(k,
                     self.metrics_test[k],
                     on_step=False,
                     on_epoch=True,
                     logger=True,
                     prog_bar=prog_bar)

        self.log('test_loss',loss,
                 on_step=False, on_epoch=True,
                 logger=True, prog_bar=True)#,

        
        
        
#         outputs['y_prob'] = y_prob.cpu().numpy()
#         outputs['y_true'] = y.cpu().numpy()
        
        return outputs

###############################
#         plt.figure(figsize=(14, 8))
#         sns.set(font_scale=1.4)
#         sns.heatmap(cm.numpy(), annot=True, annot_kws={"size": 8}, fmt="g")        
        
#         experiment.log({f"confusion_matrix/{experiment.name}": wandb.Image(plt)}, commit=False)

        
#         print(f"type(f1) = {type(f1)}")
#         print(f"type(cm) = {type(cm)}")
        
#         if hasattr(f1, "shape"):
#             print(f1.shape)

#         if hasattr(cm, "shape"):
#             print(cm.shape)
#         if hasattr(f1, "__len__"):
#             print(len(f1))

#         if hasattr(cm, "__len__"):
#             print(len(cm))
#     wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
#                             preds=top_pred_ids, y_true=ground_truth_class_ids,
#                             class_names=self.flat_class_names)})
            
            
#         self.metrics_val_per_class.reset()
#         for k, v in validation_step_outputs[0].items():
#         print(type(validation_step_outputs), len(validation_step_outputs), type(validation_step_outputs[0]))
#         for k, v in validation_step_outputs[0].items():
#             print(k, type(v))
#             if hasattr(v, "shape"):
#                 print(v.shape)
#             elif isinstance(v, dict):
#                 for kk, vv in v.items():
#                     print(k, kk, type(vv))
#                     if hasattr(vv, "shape"):
#                         print(vv.shape)
                    
#         for pred in validation_step_outputs:








#     def _init_metrics(self, stage: str='train'):
        
#         if stage in ['train', 'all']:
#             self.metrics_train_avg = get_scalar_metrics(num_classes=self.num_classes, average='macro', prefix='train')
#             self.metrics_train_per_class = get_per_class_metrics(num_classes=self.num_classes, prefix='train')
            
#         if stage in ['val', 'all']:
#             self.metrics_val_avg = get_scalar_metrics(num_classes=self.num_classes, average='macro', prefix='val')
#             self.metrics_val_per_class = get_per_class_metrics(num_classes=self.num_classes, prefix='val')
            
#         if stage in ['test', 'all']:
#             self.metrics_test_avg = get_scalar_metrics(num_classes=self.num_classes, average='macro', prefix='test')
#             self.metrics_test_per_class = get_per_class_metrics(num_classes=self.num_classes, prefix='test')






    
# class LitCassava(pl.LightningModule):
#     def __init__(self, model):
#         super(LitCassava, self).__init__()
#         self.model = model
#         self.metric = pl.metrics.F1(num_classes=CFG.num_classes, average='macro')
#         self.criterion = nn.CrossEntropyLoss()
#         self.lr = CFG.lr

#     def forward(self, x, *args, **kwargs):
#         return self.model(x)

#     def configure_optimizers(self):
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
#         self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=CFG.t_max, eta_min=CFG.min_lr)

#         return {'optimizer': self.optimizer, 'lr_scheduler': self.scheduler}

#     def training_step(self, batch, batch_idx):
#         image = batch['image']
#         target = batch['target']
#         output = self.model(image)
#         loss = self.criterion(output, target)
#         score = self.metric(output.argmax(1), target)
#         logs = {'train_loss': loss, 'train_f1': score, 'lr': self.optimizer.param_groups[0]['lr']}
#         self.log_dict(
#             logs,
#             on_step=False, on_epoch=True, prog_bar=True, logger=True
#         )
#         return loss

#     def validation_step(self, batch, batch_idx):
#         image = batch['image']
#         target = batch['target']
#         output = self.model(image)
#         loss = self.criterion(output, target)
#         score = self.metric(output.argmax(1), target)
#         logs = {'valid_loss': loss, 'valid_f1': score}
#         self.log_dict(
#             logs,
#             on_step=False, on_epoch=True, prog_bar=True, logger=True
#         )
#         return loss

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
class Registry(dict):
    '''
    A helper class for managing registering modules, it extends a dictionary
    and provides register functions.
    Access of module is just like using a dictionary, eg:
        f = some_registry["foo_module"]
        
        
    [code source] https://julienbeaulieu.github.io/2020/03/16/building-a-flexible-configuration-system-for-deep-learning-models/
    '''
    
    # Instanciated objects will be empyty dictionaries
    def __init__(self, *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)

    # Decorator factory. Here self is a Registry dict
    def register(self, module_name, module=None):
        
        # Inner function used as function call
        if module is not None:
            _register_generic(self, module_name, module)
            return

        # Inner function used as decorator -> takes a function as argument
        def register_fn(fn):
            _register_generic(self, module_name, fn)
            return fn

        return register_fn # decorator factory returns a decorator function
    

def _register_generic(module_dict, module_name, module):
    assert module_name not in module_dict
    module_dict[module_name] = module