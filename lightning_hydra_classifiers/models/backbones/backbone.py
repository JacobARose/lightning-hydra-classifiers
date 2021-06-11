"""
lightning_hydra_classifiers/models/backbone.py

Author: Jacob A Rose
Created: Saturday May 29th, 2021

"""

from typing import Any, List, Optional, Dict, Tuple

import torch
from torch import nn, functional as F
import torchvision
from pytorch_lightning import LightningModule

import torchmetrics as metrics
# from pytorch_lightning import metrics
#.classification import Accuracy
import timm
from torchsummary import summary
import pandas as pd
from pathlib import Path
from stuf import stuf
import wandb
import numpy as np








# 3.a Optional: Register a custom backbone
# This is useful to create new backbone and make them accessible from `ImageClassifier`
# @ImageClassifier.backbones(name="resnet18")
def fn_resnet(pretrained: bool = True):
    model = torchvision.models.resnet18(pretrained)
    # remove the last two layers & turn it into a Sequential model
    backbone = nn.Sequential(*list(model.children())[:-2])
    num_features = model.fc.in_features
    # backbones need to return the num_features to build the head
    return backbone, num_features
    
    


def create_classifier(num_features: int, num_classes: int, pool_type='avg', bias: bool=True):
    global_pool = nn.AdaptiveAvgPool2d(1)
    flatten_layer = nn.Flatten()
    linear_layer = nn.Linear(num_features, num_classes, bias=bias)
    return [global_pool, flatten_layer, linear_layer]



























# class ResNet(nn.Module):
class ResNet(LightningModule):
    example_input_size = (2, 3, 224, 224)
    def __init__(self,
                 model_name: Optional[str]='resnet50',
                 num_classes: Optional[int]=1000,
                 input_shape: Tuple[int]=(3,224,224),
                 batch_size: Optional[int]=None,
                 optimizer=stuf({'name':"Adam", 'lr':0.01}),
                 seed: int=None): #, 'weight_decay':0.0})):
        super().__init__()
#         pl.trainer.seed_everything(seed=seed)
        self.save_hyperparameters()
        self.hparams.lr = optimizer.lr
        self.optimizer_hparams = stuf(optimizer)
        self.example_input_array = torch.rand(self.example_input_size)
        assert 'resnet' in model_name
        model = timm.create_model(model_name, pretrained=True)
        self.freeze(model)
        del model.fc
#         for key, val in model.__dict__.items():
#             self.__dict__[key] = val
#         self.base_model = model
        self.relu = nn.ReLU()
        
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        
        self.stem = nn.Sequential(
                        self.conv1,
                        self.bn1,
                        self.relu,
                        self.maxpool
                        )
        
        self.features = nn.Sequential(
                        self.layer1,
                        self.layer2,
                        self.layer3,
                        self.layer4
                        )
        self.num_features = model.num_features
#         self.num_features = self._get_conv_output(input_shape)
        self.input_shape = input_shape
        print(self.num_features, self._get_conv_output(input_shape))
        self.reset_classifier(num_classes, global_pool='avg')
                
        self.criterion = nn.CrossEntropyLoss()
        scalar_metrics = get_scalar_metrics(num_classes=num_classes)
        
        self.train_metrics = scalar_metrics.clone(prefix='train/')
        self.val_metrics = scalar_metrics.clone(prefix='val/')
        self.test_metrics = scalar_metrics.clone(prefix='test/')
        
        self.automatic_optimization = False

    def forward(self, x):
        out = self.forward_features(x)
        out = self.classifier(out)
        return out
    
    
    def forward_features(self, x):
        out = self.stem(x)
        out = self.features(out)
        return out







def get_scalar_metrics(num_classes: int, average: str='macro', prefix: str=''):
    default = {'acc_top1': metrics.Accuracy(top_k=1),
               'acc_top3': metrics.Accuracy(top_k=3),
               'precision_top1': metrics.Precision(num_classes=num_classes, top_k=1, average=average),
#                'precision_top3': metrics.Precision(num_classes=num_classes, top_k=3, average=average),
               'recall_top1': metrics.Recall(num_classes=num_classes, top_k=1, average=average)}#,
#                'recall_top3': metrics.Recall(num_classes=num_classes, top_k=3, average=average)}
    if len(prefix)>0:
        for k in list(default.keys()):
            default[prefix + r'/' + k] = default[k]
            del default[k]
    
    return metrics.MetricCollection(default)


def get_expensive_metrics(num_classes: int, normalize: Optional[str]=None, threshold: float=0.5, num_bins: int=100):
    default = {'ConfusionMatrix': metrics.ConfusionMatrix(num_classes=num_classes, normalize=normalize, threshold=threshold),
               'BinnedPRCurve': metrics.BinnedPrecisionRecallCurve(num_classes=num_classes, num_thresholds=num_bins),
               'BinnedAvgPrecision': metrics.BinnedAveragePrecision(num_classes=num_classes, num_thresholds=num_bins)}
    return metrics.MetricCollection(default)


# def _load_pretrained_model(model_name: str):
#     if model_name.startswith('timm_'):
#         from timm import create_model
#     elif model_name.startswith('torchvision_'):
#         from torchvision.models import 
import pdb

# class ResNet(nn.Module):
class ResNet(LightningModule):
    example_input_size = (2, 3, 224, 224)
    def __init__(self,
                 model_name: Optional[str]='resnet50',
                 num_classes: Optional[int]=1000,
                 input_shape: Tuple[int]=(3,224,224),
                 batch_size: Optional[int]=None,
                 optimizer=stuf({'name':"Adam", 'lr':0.01}),
                 seed: int=None): #, 'weight_decay':0.0})):
        super().__init__()
#         pl.trainer.seed_everything(seed=seed)
        self.save_hyperparameters()
        self.hparams.lr = optimizer.lr
        self.optimizer_hparams = stuf(optimizer)
        self.example_input_array = torch.rand(self.example_input_size)
        assert 'resnet' in model_name
        model = timm.create_model(model_name, pretrained=True)
        self.freeze(model)
        del model.fc
#         for key, val in model.__dict__.items():
#             self.__dict__[key] = val
#         self.base_model = model
        self.relu = nn.ReLU()
        
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        
        self.stem = nn.Sequential(
                        self.conv1,
                        self.bn1,
                        self.relu,
                        self.maxpool
                        )
        
        self.features = nn.Sequential(
                        self.layer1,
                        self.layer2,
                        self.layer3,
                        self.layer4
                        )
        self.num_features = model.num_features
#         self.num_features = self._get_conv_output(input_shape)
        self.input_shape = input_shape
        print(self.num_features, self._get_conv_output(input_shape))
        self.reset_classifier(num_classes, global_pool='avg')
                
        self.criterion = nn.CrossEntropyLoss()
        scalar_metrics = get_scalar_metrics(num_classes=num_classes)
        
        self.train_metrics = scalar_metrics.clone(prefix='train/')
        self.val_metrics = scalar_metrics.clone(prefix='val/')
        self.test_metrics = scalar_metrics.clone(prefix='test/')
        
        self.automatic_optimization = False

    def forward(self, x):
        out = self.forward_features(x)
        out = self.classifier(out)
        return out
    
    
    def forward_features(self, x):
        out = self.stem(x)
        out = self.features(out)
        return out
        
    def _get_conv_output(self, shape): # returns the size of the output tensor going into Linear layer from the conv block.
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self.forward_features(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size
    
    def step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        y_prob = self.probs(y_hat)
        y_pred = torch.max(y_prob, dim=1)[1]
        return {'loss':loss,
                'log':{
                       'y_prob':y_prob,
                       'y_pred':y_pred,
                       'y_true':y,
                       'batch_idx':batch_idx
                       }
               }
    
                
    def training_step(self, batch, batch_idx):      
        opt = self.optimizers()
        x, y = batch
        y_hat = self(x)
        
        def closure():#opt):
            self._current_loss = self.loss(y_hat, y)
            opt.zero_grad()
            self.manual_backward(self._current_loss)

        opt.step(closure=closure)
        
        y_prob = self.probs(y_hat)
        y_hat_int, y_pred = torch.max(y_prob, dim=1)
        
        loss = self._current_loss.detach()
        self.trainer.train_loop.running_loss.append(loss)
        return {'loss':loss,
                'log':{
                       'train_loss':loss,
                       'y_prob':y_prob,
                       'y_pred':y_pred,
                       'y_true':y,
                       'batch_idx':batch_idx
                       }
               }
        
        
    def training_step_end(self, outputs):
        run = self.logger.experiment[0]
        logs = outputs['log']
        loss = outputs['loss']
        idx = logs['batch_idx']
        y_prob, y_pred, y = logs['y_prob'], logs['y_pred'], logs['y_true']

        self.log_dict(self.train_metrics(y_prob, y))

        self.log('train/loss', loss,
                 on_step=True,# on_epoch=True)#, 
                 logger=True, prog_bar=True)
        run.log({
                 "train/y_pred":
                 wandb.Histogram(np_histogram=np.histogram(y_pred.cpu(), bins=self.num_classes)),
                 "train/y_true":
                 wandb.Histogram(np_histogram=np.histogram(y.cpu(), bins=self.num_classes)),
                "train/batch_idx":
                 idx,
                "global_step": self.trainer.global_step
                })
        
        
    def training_epoch_end(self, outputs):
        self.log_dict(self.train_metrics)
        
        run = self.logger.experiment[0]
        run.log({"epoch":self.trainer.current_epoch,
                 "global_step": self.trainer.global_step})
        
#         self.train_metrics.reset()
        
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        y_prob = self.probs(y_hat)
        y_pred = torch.max(y_prob, dim=1)[1]
        return {'loss':loss,
                'log':{
                       'val_loss':loss,
                       'y_prob':y_prob,
                       'y_pred':y_pred,
                       'y_true':y,
                       'batch_idx':batch_idx
                       }
               }

    def validation_step_end(self, outputs):
        logs = outputs['log']
        loss = logs['val_loss']
        idx = logs['batch_idx']
        
        y_prob, y_pred, y = logs['y_prob'], logs['y_pred'], logs['y_true']        

        self.val_metrics(y_prob, y)
#         for k in val_metrics_step.keys():
#             self.log(k, val_metrics_step[k], on_step=True, on_epoch=False)        

        self.log('val/loss',loss,
                 on_step=True, on_epoch=True,
                 logger=True, prog_bar=True)
        
        run = self.logger.experiment[0]
        run.log({"val/y_pred":
                 wandb.Histogram(np_histogram=np.histogram(y_pred.cpu())),
                 "val/y_true":
                 wandb.Histogram(np_histogram=np.histogram(y.cpu())),
                 "val/batch_idx":
                 idx,
                 "global_step": self.trainer.global_step,
                 **self.val_metrics(y_prob, y)
                })
        
    def validation_epoch_end(self, outputs):
        self.log_dict(self.val_metrics.compute())
        
        run = self.logger.experiment[0]
        run.log({"epoch":self.trainer.current_epoch,
                 "global_step": self.trainer.global_step,})
        
        self.val_metrics.reset()

        
    def on_after_backward(self):
        # example to inspect gradient information in tensorboard
        if self.trainer.global_step % 25 == 0:  # don't make the tf file huge
            for k, v in self.named_parameters():
                self.logger.experiment[0].add_histogram(
                    tag=f'grads/{k}', values=v.grad, global_step=self.trainer.global_step
                )
            
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        sample_imgs = x[:10]
        grid = torchvision.utils.make_grid(sample_imgs).permute(1,2,0).cpu().numpy()
        self.logger.experiment[0].log({"test/examples": [wandb.Image(grid, caption="test images")],
                                    "global_step": self.trainer.global_step})
        
        y_pred = torch.argmax(y_hat, dim=1)
        y_prob = self.probs(y_hat)
#         self.test_metrics(y_hat, y)
        self.log('test/loss', loss, logger=True)
        
        test_metrics_step = self.test_metrics(y_prob, y)
        for k in test_metrics_step.keys():
            self.log(k, test_metrics_step[k], logger=True) #, on_step=True, on_epoch=False)
    
        return test_metrics_step

    
    def test_epoch_end(self, outputs):
        
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()
    
    
    @property
    def loss(self):
        return self.criterion

    
    def probs(self, x):
        return x.softmax(dim=-1)
    
    
    def predict(self, batch, batch_idx, dataloader_idx):
        return (batch[0].detach().cpu(), self(batch[0]), *batch[1:])


    def reset_classifier(self, num_classes, global_pool: str='avg'):
        self.num_classes = num_classes
        self.global_pool, self.flatten, self.fc = create_classifier(self.num_features, 
                                                                    self.num_classes, 
                                                                    pool_type=global_pool)
        self.classifier = nn.Sequential(
                                        self.global_pool,
                                        self.flatten,
                                        self.fc
                                        )
        self.initialize_weights(self.classifier)

        
    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(
                                params=self.parameters(), 
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.optimizer.weight_decay
                                )

