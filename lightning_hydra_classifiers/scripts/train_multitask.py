#!/usr/bin/env python
# coding: utf-8

# # train_multi-task.py
# 
# Based on the notebook: `multi-task_model-train.ipynb`
# 
# End of August attempts to create good model training workflows for multi-task experiments
# 
# Author: Jacob A Rose  
# Created on: Monday August 29th, 2021


"""


python "/media/data/jacob/GitHub/lightning-hydra-classifiers/lightning_hydra_classifiers/scripts/train_multitask.py"
"""



import numpy as np
import collections
import os
if 'TOY_DATA_DIR' not in os.environ: 
    os.environ['TOY_DATA_DIR'] = "/media/data_cifs/projects/prj_fossils/data/toy_data"
default_root_dir = os.environ['TOY_DATA_DIR']
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 200)

import torch
from torch import nn
import torchvision
from torchvision import transforms
import pytorch_lightning as pl
import timm

torch.backends.cudnn.benchmark = True


from rich import print as pp
import matplotlib.pyplot as plt
from munch import Munch
import argparse
import json

from lightning_hydra_classifiers.data.utils.make_catalogs import *
from lightning_hydra_classifiers.utils.metric_utils import get_per_class_metrics, get_scalar_metrics
from lightning_hydra_classifiers.utils.logging_utils import get_wandb_logger
import wandb
# torch.manual_seed(17)
from lightning_hydra_classifiers.experiments.transfer_experiment import TransferExperiment
# from lightning_hydra_classifiers.models.backbones.backbone import build_model
from lightning_hydra_classifiers.models.backbones import backbone
# from torchinfo import summary
# model_stats = summary(your_model, (1, 3, 28, 28), verbose=0)
# from lightning_hydra_classifiers.utils.common_utils import LabelEncoder






class PlantDataModule(pl.LightningDataModule):
#     valid_tasks = (0, 1)
    
    def __init__(self, 
                 batch_size,
                 task_id: int=0,
                 image_size: int=224,
                 image_buffer_size: int=32,
                 num_workers: int=4,
                 pin_memory: bool=True):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        
        self.experiment = TransferExperiment()
        self.set_task(task_id)        
        
        self.image_size = image_size
        self.image_buffer_size = image_buffer_size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        # Train augmentation policy
        
        self.__init_transforms()
                
        self.tasks = self.experiment.get_multitask_datasets(train_transform=self.train_transform,
                                                            val_transform=self.val_transform)


    def __init_transforms(self):
        
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=self.image_size,
                                         scale=(0.25, 1.2),
                                         ratio=(0.7, 1.3),
                                         interpolation=2),
            torchvision.transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(self.mean, self.std),
            transforms.Grayscale(num_output_channels=3)
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize(self.image_size+self.image_buffer_size),
            torchvision.transforms.ToTensor(),
            transforms.CenterCrop(self.image_size),
            transforms.Normalize(self.mean, self.std),
            transforms.Grayscale(num_output_channels=3)            
        ])

    def set_task(self, task_id: int):
        assert task_id in self.experiment.valid_tasks
        self.task_id = task_id
        

    @property
    def current_task(self):
        return self.tasks[self.task_id]

    def setup(self, stage=None):
        task = self.current_task
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_dataset = task['train']
            self.val_dataset = task['val']
            
            self.classes = self.train_dataset.classes
            self.num_classes = len(self.train_dataset.label_encoder)
            self.label_encoder = self.train_dataset.label_encoder
            
        elif stage == 'test':
            self.test_dataset = task['test']
                        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          pin_memory=self.pin_memory,
                          shuffle=True,
                          num_workers=self.num_workers,
                          drop_last=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          pin_memory=self.pin_memory,
                          num_workers=self.num_workers)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          pin_memory=self.pin_memory,
                          num_workers=self.num_workers)


# ## Model & LightningModules



class LitMultiTaskModule(pl.LightningModule):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lr = config.lr
        self.num_classes = config.num_classes
        self.save_hyperparameters()
        
        self.init_model(config)
        self.metrics = self.init_metrics(stage='all')
        self.criterion = nn.CrossEntropyLoss()        

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def configure_optimizers(self):
        print(f"self.hparams={self.hparams}")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)#hparams.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.t_max, eta_min=self.config.min_lr)

        return {'optimizer': self.optimizer, 'lr_scheduler': self.scheduler}

    def training_step(self, batch, batch_idx):
        image = batch[0]
        target = batch[1]
        output = self.model(image)
        loss = self.criterion(output, target)
#         scores = self.metrics_train(output.argmax(1), target)
        scores = self.metrics_train(output, target)
        self.log_dict({"train_loss": loss, 'lr': self.optimizer.param_groups[0]['lr']},
                      on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(scores,
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image = batch[0]
        target = batch[1]
        output = self.model(image)
        loss = self.criterion(output, target)
#         scores = self.metrics_val(output.argmax(1), target)
        scores = self.metrics_val(output, target)
        y_pred = torch.argmax(output, dim=-1)
        
        self.log("val_loss", loss,
                  on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(scores,
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return {"loss":loss.cpu().numpy(),
                "y_logit":output,#.cpu().numpy(),
                "y_pred":y_pred}#.cpu().numpy()}
    
    
    def init_model(self, config):
        self.model =  backbone.build_model(model_name=config.model_name,
                                           pretrained=config.pretrained,
                                           num_classes=config.num_classes)
        
        self.freeze_up_to(layer=config.init_freeze_up_to)
        
        
        
    def freeze_up_to(self, layer: Union[int, str]=None):
        
        if isinstance(layer, int):
            if layer < 0:
                layer = len(list(self.model.parameters())) + layer
            
        self.model.enable_grad = True
        for i, (name, param) in enumerate(self.model.named_parameters()):
            param.enable_grad = False
            if isinstance(layer, int):
                if layer == i:
                    break
            elif isinstance(layer, str):
                if layer == name:
                    break

        
        
    
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

    


# Custom Callback
class ImagePredictionLogger(pl.Callback):
    def __init__(self, top_k_per_batch: int=5, bottom_k_per_batch: int=5):
        super().__init__()
        self.top_k_per_batch = top_k_per_batch
        self.bottom_k_per_batch = bottom_k_per_batch
#         self.num_samples = num_samples
#         self.val_imgs, self.val_labels = val_samples['image'], val_samples['target']
        
        
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if "loss" not in outputs:
            return
        
#         loss = outputs["loss"]
        y_true = batch[1]
        loss = nn.CrossEntropyLoss(reduction="none")(outputs["y_logit"], y_true).cpu().numpy()

    
        y_pred = [pl_module.label_encoder.idx2class[i] for i in outputs["y_pred"].cpu().numpy()]
#         print(type(batch[0]))
        imgs = np.transpose(batch[0].cpu().numpy(), (0,2,3,1))
#         print(type(imgs))
        labels = [pl_module.label_encoder.idx2class[i] for i in batch[1].cpu().numpy()]
        probs = outputs["y_logit"].softmax(dim=1).cpu().numpy().tolist()

        
#         print(type(imgs[0]))
#         print(f"loss: {loss}")
        sorted_idx = np.argsort(loss)#.cpu().numpy())
        top_k_idx = sorted_idx[:self.top_k_per_batch]
        bottom_k_idx = sorted_idx[::-1][:self.bottom_k_per_batch]
        top_k = len(top_k_idx)
#         print(len(sorted_idx), type(sorted_idx))
        print(f'Logging top & bottom top_k_idx={top_k_idx} samples.')
#         print(f"imgs.shape={imgs.shape}, y_pred[0]={y_pred[0]}, labels[0]:{labels[0]}")
        print(f"len(probs[0]): {len(probs[0])}")
        trainer.logger.experiment.log({
            f"bottom_{top_k}_per_batch":
                                    wandb.Image(imgs[k,:,:,:], caption=f"Pred:{y_pred[k]}, Label:{labels[k]}, prob: {np.max(probs[k]):.4f}, loss:{loss[k]:.4f}")
                                    for k in bottom_k_idx
            }, commit=False)

        trainer.logger.experiment.log({
            f"top_{top_k}_per_batch":
                                    wandb.Image(imgs[k,:,:,:], caption=f"Pred:{y_pred[k]}, Label:{labels[k]}, prob: {np.max(probs[k]):.4f}, loss:{loss[k]:.4f}") 
                                    for k in top_k_idx
            
#                                     wandb.Image(x, caption=f"Pred:{pred}, Label:{y}") 
#                                     for x, pred, y in zip(imgs[top_k_idx], 
#                                                           y_pred[top_k_idx], 
#                                                           labels[top_k_idx])
            }, commit=False)


        
        
#         logits = pl_module(val_imgs)
#         preds = torch.argmax(logits, -1)
#         # Log the images as wandb Image
#         trainer.logger.experiment.log({
#             "examples":[wandb.Image(x, caption=f"Pred:{pred}, Label:{y}") 
#                            for x, pred, y in zip(val_imgs[:self.num_samples], 
#                                                  preds[:self.num_samples], 
#                                                  val_labels[:self.num_samples])]
#             }, commit=False)
        
            
            
        
#     def on_validation_epoch_end(self, trainer, pl_module):
#         # Bring the tensors to CPU
#         val_imgs = self.val_imgs.to(device=pl_module.device)
#         val_labels = self.val_labels.to(device=pl_module.device)
#         # Get model prediction
#         logits = pl_module(val_imgs)
#         preds = torch.argmax(logits, -1)
#         # Log the images as wandb Image
#         trainer.logger.experiment.log({
#             "examples":[wandb.Image(x, caption=f"Pred:{pred}, Label:{y}") 
#                            for x, pred, y in zip(val_imgs[:self.num_samples], 
#                                                  preds[:self.num_samples], 
#                                                  val_labels[:self.num_samples])]
#             }, commit=False)


#     config = Munch({
#         "seed":42,
#         "num_epochs": 10,
#         "precision": 16,
#         "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     })

#     config.data = {
#         "image_size": 224,
#         "image_buffer_size": 32, 
#         "batch_size": 32,
#         "num_workers": 4,
#         "pin_memory": True
#     }
#     config.model = {"model_name": "resnet50",
#                     "pretrained": "imagenet",
#                     "lr": 3e-4,
#                     "num_classes": None,
#                     "t_max": 20,
#                     "min_lr": 1e-6}

#     config.callbacks = {"monitor": {"metric":"val_loss",
#                                     "mode": "min"}
#                        }




def train_source_task(config: argparse.Namespace):
    
    
    pl.seed_everything(config['seed'])

    # ## DataModule
    datamodule = PlantDataModule(batch_size=config.data.batch_size,
                                 task_id=0,
                                 image_size=config.data.image_size,
                                 image_buffer_size=config.data.image_buffer_size,
                                 num_workers=config.data.num_workers,
                                 pin_memory=config.data.pin_memory)
    datamodule.setup("fit")
    config.model.num_classes = datamodule.num_classes
    pp(config)

    model = LitMultiTaskModule(config.model)
    model.label_encoder = datamodule.label_encoder

    # ## Callbacks
    # Checkpoint
    k=15
    img_prediction_callback = ImagePredictionLogger(top_k_per_batch=k,
                                                    bottom_k_per_batch=k)
    
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor=config.callbacks.monitor.metric,
                                                       save_top_k=1,
                                                       save_last=True,
                                                       save_weights_only=True,
                                                       filename='checkpoint/{epoch:02d}-{val_loss:.4f}-{val_f1:.4f}',
                                                       verbose=True,
                                                       mode=config.callbacks.monitor.mode)
    earlystopping = pl.callbacks.EarlyStopping(monitor=config.callbacks.monitor.metric,
                                               patience=3,
                                               mode=config.callbacks.monitor.mode)

    # ## Logger
    wandb_logger = pl.loggers.WandbLogger(entity = "jrose",
                                          project = "image_classification_train",
                                          job_type = "train_supervised",
                                          config=dict(config),
                                          group=f'{config.model.model_name}',
                                          dir=config.output_dir)


    # ## Trainer
    trainer = pl.Trainer(
#                 limit_train_batches=0.1,
#                 limit_val_batches=0.1,
                max_epochs=config.num_epochs,
                gpus=1,
    #             accumulate_grad_batches=CONFIG['accum'],
                precision=config.precision,
                callbacks=[earlystopping,
                           checkpoint_callback,
                          img_prediction_callback],
    #                        ImagePredictionLogger(val_samples)],
    #             checkpoint_callback=checkpoint_callback,
                logger=wandb_logger,
                weights_summary='top')


    # # TRAIN
    trainer.fit(model, datamodule)
    wandb.finish() 

######################################


# from lightning_hydra_classifiers.experiments.transfer_experiment import TransferExperiment
# output_root_dir = "/media/data_cifs/projects/prj_fossils/users/jacob/experiments/July2021-Nov2021/csv_datasets/experimental_datasets"
# experiment = TransferExperiment()
# experiment.export_experiment_spec(output_root_dir=output_root_dir)




def cmdline_args():
    
    parser = argparse.ArgumentParser(description="")
    subparsers = parser.add_subparsers()

    p = subparsers.add_parser('', help='model args')
    # subparser_one = parser_one.add_subparsers()
    p.add_argument("-o", "--output_dir", dest="output_dir", type=str,
                   default="/media/data_cifs/projects/prj_fossils/users/jacob/experiments/July2021-Nov2021/experiment_logs",
                   help="Output root directory for experiment logs.")
    p.add_argument("-e", "--num_epochs", dest="num_epochs", type=int, default=100,
                   help="Number of training epochs")
    p.add_argument("-res", "--image_size", dest="image_size", type=int, default=224,
                   help="image_size/model input resolution")
    p.add_argument("-buffer", "--image_buffer_size", dest="image_buffer_size", type=int, default=32,
                   help="Additional resolution to resize images before either random (train) or center (test) crop to image_size.")
    p.add_argument("-bsz", "--batch_size", dest="batch_size", type=int, default=32,
                   help="batch_size")
    p.add_argument("-nproc", "--num_workers", dest="num_workers", type=int, default=4,
                   help="num_workers per dataloader")
    p.add_argument("-model", "--model_name", dest="model_name", type=str, default="resnet50",
                   help="model backbone architecture")
    p.add_argument("-init_freeze", "--init_freeze_up_to", dest="init_freeze_up_to", default="layer4",
                   help="freeze up to and including layer name or index.")    
    p.add_argument("-pre", "--pretrained", dest="pretrained", default="imagenet", choices=["imagenet", True, False],
                   help="Use pretrained imagenet weights or randomly initialize from scratch.")
    p.add_argument("-lr", "--learning_rate", dest="learning_rate", type=float, default=3e-4,
                   help="Initial learning rate.")

    args = parser.parse_args([""])
    print("Args:")
    pp(args)

    config = Munch({
        "seed":42,
        "num_epochs": args.num_epochs,
        "precision": 16,
        "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        "output_dir":args.output_dir
    })

    config.data = Munch({
        "image_size": args.image_size,
        "image_buffer_size": args.image_buffer_size,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True
    })
    
    config.model = Munch({"model_name": args.model_name,
                          "init_freeze_up_to":args.init_freeze_up_to,
                    "pretrained": args.pretrained,
                    "lr": args.learning_rate,
                    "num_classes": None,
                    "t_max": 20,
                    "min_lr": 1e-6})

    config.callbacks = Munch({"monitor":
                              Munch({"metric":"val_loss",
                                     "mode": "min"})
                             })

    return args, config


if __name__ == '__main__':
    
    args, config = cmdline_args()
    os.makedirs(config.output_dir, exist_ok=True)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    
    train_source_task(config)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




#     out = json.dumps(config, default=str)
    
#     config = Munch({
#         "seed":42,
#         "num_epochs": 10,
#         "precision": 16,
#         "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     })

#     config.data = {
#         "image_size": 224,
#         "image_buffer_size": 32, 
#         "batch_size": 32,
#         "num_workers": 4,
#         "pin_memory": True
#     }
#     config.model = {"model_name": "resnet50",
#                     "pretrained": "imagenet",
#                     "lr": 3e-4,
#                     "num_classes": None,
#                     "t_max": 20,
#                     "min_lr": 1e-6}

#     config.callbacks = {"monitor": {"metric":"val_loss",
#                                     "mode": "min"}
#                        }
    
    
    
    
    
    
