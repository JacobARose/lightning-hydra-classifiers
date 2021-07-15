"""

lightning_hydra_classifiers/models/transfer.py

Author: Jacob A Rose
Created: Wednesday June 23rd, 2021




Based on the pytorch lightning script located at the following:
source: https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/domain_templates/computer_vision_fine_tuning.py



Computer vision example on Transfer Learning.

This computer vision example illustrates how one could fine-tune a pre-trained
network (by default, a ResNet50 is used) using pytorch-lightning. For the sake
of this example, the 'cats and dogs dataset' (~60MB, see `DATA_URL` below) and
the proposed network (denoted by `TransferLearningModel`, see below) is
trained for 15 epochs.
The training consists of three stages.

1. From epoch 0 to 4, the feature extractor (the pre-trained network) is frozen except
maybe for the BatchNorm layers (depending on whether `train_bn = True`). The BatchNorm
layers (if `train_bn = True`) and the parameters of the classifier are trained as a
single parameters group with lr = 1e-2.

2. From epoch 5 to 9, the last two layer groups of the pre-trained network are unfrozen
and added to the optimizer as a new parameter group with lr = 1e-4 (while lr = 1e-3
for the first parameter group in the optimizer).

3. Eventually, from epoch 10, all the remaining layer groups of the pre-trained network
are unfrozen and added to the optimizer as a third parameter group. From epoch 10,
the parameters of the pre-trained network are trained with lr = 1e-5 while those of
the classifier is trained with lr = 1e-4.
Note:
    See: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
"""

import logging
import os
from pathlib import Path
from typing import Union, Callable, Tuple, Optional, Dict, Any

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
# from torchvision.datasets.utils import download_and_extract_archive

import pytorch_lightning as pl
from pl_examples import cli_lightning_logo
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities.cli import LightningCLI


from lightning_hydra_classifiers.models import heads, base
from lightning_hydra_classifiers.models.base import BaseLightningModule

from lightning_hydra_classifiers.models.heads.classifier import Classifier


log = logging.getLogger(__name__)
# DATA_URL = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"

#  --- Finetuning Callback ---


class MilestonesFinetuning(BaseFinetuning):
    
    def __init__(self,
                 milestones: tuple = (3, 5, 10),
                 train_bn: bool = False):
        super().__init__()
        self.milestones = milestones
        self.train_bn = train_bn

    def freeze_before_training(self, pl_module: pl.LightningModule):
        """
        This method is called before ``configure_optimizers``
        and should be used to freeze any modules parameters.
        """
        unfreeze_layers = ['layer4']
        
        modules = dict(pl_module.feature_extractor.named_children())
        num_params = len(modules)
        
        for k in list(modules.keys()):
            for l in unfreeze_layers:
                if k.startswith(l):
                    modules.pop(k)

        modules = list(modules.values())        

        log.info(f'Freezing {len(modules)} layers before training, out of {num_params}. Unfrozen layer names: {unfreeze_layers}')

        self.freeze(modules=nn.Sequential(*modules), train_bn=self.train_bn)

    def finetune_function(self, pl_module: pl.LightningModule, epoch: int, optimizer: Optimizer, opt_idx: int):
        """
        This method is called on every train epoch start and should be used to
        ``unfreeze`` any parameters.
        
        Those parameters needs to be added in a new ``param_group``
        within the optimizer.
    .. note:: Make sure to filter the parameters based on ``requires_grad``.
        
        """
        
        if epoch == self.milestones[0]:
            unfreeze_layers = ['layer4']
            modules = [p for name, p in pl_module.feature_extractor.named_parameters() if name in unfreeze_layers]
            self.unfreeze_and_add_param_group(
                modules=nn.Sequential(*modules), optimizer=optimizer, train_bn=self.train_bn
            )

        elif epoch == self.milestones[1]:
            unfreeze_layers = ['layer3']
            modules = [p for name, p in pl_module.feature_extractor.named_parameters() if name in unfreeze_layers]
            self.unfreeze_and_add_param_group(
                modules=nn.Sequential(*modules), optimizer=optimizer, train_bn=self.train_bn
            )

            
        if epoch == self.milestones[2]:
            unfreeze_layers = ['layer2']
            modules = [p for name, p in pl_module.feature_extractor.named_parameters() if name in unfreeze_layers]
            self.unfreeze_and_add_param_group(
                modules=nn.Sequential(*modules), optimizer=optimizer, train_bn=self.train_bn
            )
            
            
            
#  --- Pytorch-lightning module ---


class TransferLearningModel(BaseLightningModule):

    classifier_factory: Callable = heads.Classifier

    def __init__(
                 self,
                 classifier: heads.Classifier=None,
                 train_bn: bool = False,
                 milestones: tuple = (2, 4, 8),
                 batch_size: int = 32,
                 optimizer: str = "Adam",
                 lr: float = 1e-3,
                 lr_scheduler_gamma: float = 1e-1,
                 classifier_kwargs: Optional[Dict[str, Any]]=None,
#                  num_workers: int = 6,
                 **kwargs
                 ) -> None:
        """TransferLearningModel
        Args:
            classifier: heads.Classifier instance (subclass of nn.Module) containing 3 named children:
                1. backbone
                2. bottleneck
                3. head
            train_bn: Whether the BatchNorm layers should be trainable. Defaults to False.
            milestones: List of two epochs milestones at which to update the trainable layers
            optimizer: str
                Name of optimizer to use.
            lr: Initial learning rate
            lr_scheduler_gamma: Factor by which the learning rate is reduced at each milestone
        """
        super().__init__()
        self.classifier = classifier
        self.classifier_kwargs = classifier_kwargs or {}
        if classifier is None:
            self.num_classes = self.classifier_kwargs['num_classes']
        else:
            self.num_classes = classifier.num_classes
        
        self.train_bn = train_bn
        self.milestones = milestones
        self.batch_size = batch_size
        self.lr = lr
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.optimizer_func = getattr(optim, optimizer)

#         self.num_workers = num_workers

        self._build_classifier()
        self._init_metrics('all')

#         self.save_hyperparameters()

    def _build_classifier(self):
        """Define model layers & loss."""
        if self.classifier is None:
            self.classifier = self.classifier_factory(**self.classifier_kwargs)
        self.feature_extractor = self.classifier.backbone
    
        self.fc = self.classifier.head        
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        """Forward pass. Returns logits."""
        x = self.classifier(x)
        return x

    def loss(self, logits, labels):
        return self.criterion(logits, labels)

    def probs(self, x):
        return x.softmax(dim=-1)
    
    def configure_optimizers(self):
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        rank_zero_info(
            f"The model will start training with only {len(trainable_parameters)} "
            f"trainable parameters out of {len(parameters)}."
        )
        log.info(f"self.lr={self.lr}")
        optimizer = self.optimizer_func(trainable_parameters, lr=self.lr)
        scheduler = MultiStepLR(optimizer, milestones=self.milestones, gamma=self.lr_scheduler_gamma)
        return [optimizer], [scheduler]
    
    
#     def training_step(self, batch, batch_idx):
#         # 1. Forward pass:
#         x, y = batch
#         y_hat = self(x)
#         loss = self.loss(y_hat, y)
        
#         y_prob = self.probs(y_hat)
#         y_pred = torch.max(y_prob, dim=1)[1]
        
#         return {'loss':loss,
#                 'log':{
#                        'train_loss':loss,
#                        'y_prob':y_prob,
#                        'y_pred':y_pred,
#                        'y_true':y,
#                        'batch_idx':batch_idx
#                        }
#                }
        
#     def training_step_end(self, outputs):
#         logs = outputs['log']
#         loss = outputs['loss']
#         idx = logs['batch_idx']
#         y_prob, y_pred, y = logs['y_prob'], logs['y_pred'], logs['y_true']
        
#         batch_metrics = self.metrics_train_avg(y_prob, y)
#         self.log_dict(batch_metrics)
        
#         self.log('train/acc',
#                  self.metrics_train_avg['train/acc_top1'], 
#                  on_step=True,
#                  on_epoch=True,
#                  prog_bar=True)
#         self.log('train/loss', loss,
#                  on_step=True,# on_epoch=True)#,
#                  logger=True, prog_bar=True)
        
        
#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self(x)
#         loss = self.loss(y_hat, y)
#         y_prob = self.probs(y_hat)
#         y_pred = torch.max(y_prob, dim=1)[1]
#         return {'loss':loss,
#                 'log':{
#                        'val_loss':loss,
#                        'y_prob':y_prob,
#                        'y_pred':y_pred,
#                        'y_true':y,
#                        'batch_idx':batch_idx
#                        }
#                }

#     def validation_step_end(self, outputs):
#         logs = outputs['log']
#         loss = logs['val_loss']
#         y_prob, y_pred, y = logs['y_prob'], logs['y_pred'], logs['y_true']
#         batch_metrics = self.metrics_val(y_prob, y)
        
#         for k in self.metrics_val.keys():
#             self.log(k,
#                      self.metrics_val[k],
#                      on_step=True, 
#                      on_epoch=True)        

#         self.log('val/loss',loss,
#                  on_step=True, on_epoch=True,
#                  logger=True, prog_bar=True)







# class FeedForwardBackbone(pl.LightningModule):
#     def __init__(self, config: DictConfig, **kwargs):
#         self.embedding_cat_dim = sum([y for x, y in config.embedding_dims])
#         super().__init__()
#         self.save_hyperparameters(config)
































# class MyLightningCLI(LightningCLI):

#     def add_arguments_to_parser(self, parser):
#         parser.add_class_arguments(MilestonesFinetuning, 'finetuning')
#         parser.link_arguments('data.batch_size', 'model.batch_size')
# #         parser.link_arguments('data.num_classes', 'model.num_classes', apply_on="instantiate")
#         parser.link_arguments('finetuning.milestones', 'model.milestones')
#         parser.link_arguments('finetuning.train_bn', 'model.train_bn')
#         parser.set_defaults({
#             'trainer.max_epochs': 15,
#             'trainer.weights_summary': None,
#             'trainer.progress_bar_refresh_rate': 1,
#             'trainer.num_sanity_val_steps': 0,
#         })
        
        

#     def instantiate_trainer(self):
#         finetuning_callback = MilestonesFinetuning(**self.config_init['finetuning'])
#         self.trainer_defaults['callbacks'] = [finetuning_callback]
#         super().instantiate_trainer()

        
        
        





from omegaconf import OmegaConf
from contrastive_learning.data.pytorch.datamodules import get_datamodule

    
def get_configs(DATASET_NAME: str="PNAS_family_100_512",
                MODEL_NAME: str="resnet18",
                batch_size: int=12,
                image_size: Tuple[int]=(512,512),
                channels: int=3):

    data_config = OmegaConf.create(
                    dict(
                         name=DATASET_NAME,
                         batch_size=batch_size,
                         val_split=None, #0.2,
                         num_workers=8,
                         seed=None,
                         debug=False,
                         normalize=True,
                         image_size=image_size,
                         channels=channels,
                         dataset_dir=None
                        )
    )

    model_config = OmegaConf.create(
                    dict(
                         name=MODEL_NAME,
                         pretrained=True,
                         input_size=(2, data_config.channels, *data_config.image_size),
                         lr=1e-3,
                         lr_scheduler_gamma=1e-1
                        )
    )
    
    
    trainer_config = OmegaConf.create(
                    dict(
                         gpus = 1,
                         min_epochs = 1,
                         max_epochs = 40,
                         weights_summary = "top",
                         progress_bar_refresh_rate = 10,
                         profiler = "simple",
                         log_every_n_steps = 50,
                         fast_dev_run = False,
                         limit_train_batches = 1.0,
                         limit_val_batches = 1.0,
                         auto_lr_find = False,
                         auto_scale_batch_size = False
                        )
    )
    
    callback_config = OmegaConf.create(
                    dict(finetuning=
                             dict(
                                 milestones=[5,10],
                                 train_bn=False
                                 )
                        )
    )

    return data_config, model_config, trainer_config, callback_config
    

def setup_train(data_config,
                model_config,
                trainer_config,
                callback_config,
                working_dir: str,
                verbose: bool=1):

    datamodule = get_datamodule(data_config = data_config)
    model_config.num_classes = len(datamodule.classes)

#     backbone = backbones.build_model(model_config.name,
#                                      pretrained=model_config.pretrained)

    classifier = Classifier(backbone_name=model_config.name,
                       num_classes=model_config.num_classes,
                       finetune=True)
    
    model = TransferLearningModel(
                                  classifier=classifier,
                                  train_bn = callback_config['finetuning']['train_bn'],
                                  milestones = callback_config['finetuning']['milestones'],
                                  batch_size = data_config.batch_size,
                                  optimizer = "Adam",
                                  lr = model_config.lr,
                                  lr_scheduler_gamma = model_config.lr_scheduler_gamma
#                                   num_workers = 6,
                                 )
    
    loggers = [pl.loggers.wandb.WandbLogger(
                                            entity = "jrose",
                                            project = "image_classification",
                                            job_type = "train",
                                            group="stage_0",
                                            config={'data':OmegaConf.to_container(data_config, resolve=True),
                                                    'model':OmegaConf.to_container(model_config, resolve=True)
                                                   }),
               pl.loggers.csv_logs.CSVLogger(
                                             save_dir = f'{working_dir}/logs',
                                             name = "csv/"
                                            )
              ]    
    finetuning_callback = MilestonesFinetuning(**callback_config['finetuning'])
    callbacks = [finetuning_callback]
    
    train_config = OmegaConf.to_container(trainer_config, resolve=True)
    trainer = pl.Trainer(**train_config,
                         logger=loggers,
                         callbacks=callbacks)
    

#     log_model_summary(model=model,
#                       working_dir=working_dir,
#                       input_size=list(model_config.input_size),
#                       full_summary=True,
#                       verbose=verbose)

    return datamodule, model, trainer

    
    

def cli_main(config_overrides = dict(DATASET_NAME="PNAS_family_100_512",
                                     MODEL_NAME="resnet50",
                                     batch_size=12,
                                     image_size=(512,512),
                                     channels=3),
             seed=1265
            ):
    
    seed_everything(seed)
    MODEL_NAME = config_overrides['MODEL_NAME']
    DATASET_NAME = config_overrides['DATASET_NAME']
    
    working_dir = f"/media/data/jacob/GitHub/lightning-hydra-classifiers/notebooks/playground_results/{MODEL_NAME}-{DATASET_NAME}"
    os.makedirs(working_dir, exist_ok=True)
    
    data_config, model_config, trainer_config, callback_config = get_configs(**config_overrides)
    
    
    datamodule, model, trainer = setup_train(data_config,
                                             model_config,
                                             trainer_config,
                                             callback_config,
                                             working_dir=working_dir,
                                             verbose=1)

    
#     print(f'Model:\n{dir(model)}')
#     print(f'DataModule:\n{dir(datamodule)}')
    
    trainer.fit(model, datamodule=datamodule)
    
    trainer.test()

    
    
if __name__ == "__main__":
#     cli_lightning_logo()
    cli_main()
    
    
    
    
    
    
    
    
    
    
    
    #     def training_step(self, batch, batch_idx):
#         # 1. Forward pass:
#         x, y = batch
#         y_hat = self(x)
#         loss = self.loss(y_hat, y)
        
#         y_prob = self.probs(y_hat)
#         y_pred = torch.max(y_prob, dim=1)[1]
        
#         return {'loss':loss,
#                 'log':{
#                        'train_loss':loss,
#                        'y_prob':y_prob,
#                        'y_pred':y_pred,
#                        'y_true':y,
#                        'batch_idx':batch_idx
#                        }
#                }
        
#     def training_step_end(self, outputs):
#         logs = outputs['log']
#         loss = outputs['loss']
#         idx = logs['batch_idx']
#         y_prob, y_pred, y = logs['y_prob'], logs['y_pred'], logs['y_true']
        
#         batch_metrics = self.train_metrics(y_prob, y)
#         self.log_dict(batch_metrics)
        
#         self.log('train/acc',
#                  self.train_metrics['train/acc_top1'], 
#                  on_step=True,
#                  on_epoch=True,
#                  prog_bar=True)
#         self.log('train/loss', loss,
#                  on_step=True,# on_epoch=True)#,
#                  logger=True, prog_bar=True)
        
        
#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self(x)
#         loss = self.loss(y_hat, y)
#         y_prob = self.probs(y_hat)
#         y_pred = torch.max(y_prob, dim=1)[1]
#         return {'loss':loss,
#                 'log':{
#                        'val_loss':loss,
#                        'y_prob':y_prob,
#                        'y_pred':y_pred,
#                        'y_true':y,
#                        'batch_idx':batch_idx
#                        }
#                }

#     def validation_step_end(self, outputs):
        
#         logs = outputs['log']
#         loss = logs['val_loss']
#         y_prob, y_pred, y = logs['y_prob'], logs['y_pred'], logs['y_true']
#         batch_metrics = self.val_metrics(y_prob, y)
        
#         for k in self.val_metrics.keys():
#             self.log(k,
#                      self.val_metrics[k],
#                      on_step=True, 
#                      on_epoch=True)        

#         self.log('val/loss',loss,
#                  on_step=True, on_epoch=True,
#                  logger=True, prog_bar=True)

# #         self.log('val/acc',
# #                  self.val_metrics['val/acc_top1'],
# #                  on_step=True, 
# #                  on_epoch=True,
# #                  prog_bar=True)