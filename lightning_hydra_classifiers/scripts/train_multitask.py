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
from lightning_hydra_classifiers.utils.dataset_management_utils import Extract
from lightning_hydra_classifiers.utils.metric_utils import get_per_class_metrics, get_scalar_metrics
from lightning_hydra_classifiers.utils.logging_utils import get_wandb_logger
import wandb
from lightning_hydra_classifiers.experiments.transfer_experiment import TransferExperiment

from lightning_hydra_classifiers.experiments.reference_transfer_experiment import CIFAR10DataModule
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
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(self.mean, self.std),
            transforms.Grayscale(num_output_channels=3)
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize(self.image_size+self.image_buffer_size),
            transforms.ToTensor(),
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
    
    def __init__(self, config, ckpt_path: Optional[str]=None):
        super().__init__()
        self.config = Munch(config)
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

    def training_step(self, batch, batch_idx):
        image = batch[0]
        target = batch[1]
        output = self.model(image)
        loss = self.criterion(output, target)
#         scores = self.metrics_train(output.argmax(1), target)
        scores = self.metrics_train(output, target)
        self.log_dict({"train_loss": loss, 'lr': self.optimizer.param_groups[0]['lr']},
                      on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(self.metrics_train, #scores,
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image = batch[0]
        target = batch[1]
        output = self.model(image)
        loss = self.criterion(output, target)
#         scores = self.metrics_val(output.argmax(1), target)
#         try:
        scores = self.metrics_val(output, target)
#         except ValueError as e:
        
        y_pred = torch.argmax(output, dim=-1)
        
        self.log("val_loss", loss,
                  on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        self.log("val_acc", self.metrics_val["val/acc_top1"],
                 on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict({k:v for k,v in self.metrics_val.items() if k != "val/acc_top1"},
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
#         self.log_dict(scores,
#                       on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return {"loss":loss, #.cpu().numpy(),
                "y_logit":output, #.cpu().numpy(),
                "y_pred":y_pred} #.cpu().numpy()}
    
    
    def init_model(self, config):
        self.model =  backbone.build_model(model_name=config.model_name,
                                           pretrained=config.pretrained,
                                           num_classes=config.num_classes)
#         if os.path.isfile(self.ckpt_path):
            
        
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
                    
#         print(f"Validating model's layer freezing function, model.freeze_up_to(layer={layer})")
#         for i, (name, param) in enumerate(self.model.named_parameters()):
#             print(f"{name}.requires_grad: {param.requires_grad}")

        
        
    
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

    def on_sanity_check_start(self, trainer, pl_module):
        self._sanity_check = True

    def on_sanity_check_end(self, trainer, pl_module):
        self._sanity_check = False
    
        
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if ("loss" not in outputs) or self._sanity_check:
            return
        
#         loss = outputs["loss"]
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

        trainer.logger.experiment.log({"epoch":trainer.current_epoch,
                                       **{f"bottom_k_per_batch":
                                    wandb.Image(imgs[k,:,:,:], caption=f"Pred:{y_pred[k]}, Label:{labels[k]}, prob: {np.max(probs[k]):.4f}, loss:{loss[k]:.4f}")
                                    for k in bottom_k_idx}
                                      }, commit=False)

        trainer.logger.experiment.log({"epoch":trainer.current_epoch,
                                       **{f"top_k_per_batch":
                                    wandb.Image(imgs[k,:,:,:], caption=f"Pred:{y_pred[k]}, Label:{labels[k]}, prob: {np.max(probs[k]):.4f}, loss:{loss[k]:.4f}") 
                                    for k in top_k_idx}
                                      }, commit=False)




########################################################
########################################################
########################################################


def run_lr_tuner(trainer,
                 model,
                 datamodule,
                 config: argparse.Namespace,
                 run=None):
    """
    Learning rate tuner
    
    Adapted and refactored from "lightning-hydra-classifiers/lightning_hydra_classifiers/scripts/train_basic.py"
    """
#     pipeline_stage = str(pipeline_stage)
#     stage_config = config.wandb[pipeline_stage]

#     if not config.tuner.trainer_kwargs.auto_lr_find:
#         print(f'config.trainer.auto_lr_find is set to False, Skipping `run_lr_tuner(config, pipeline_stage={pipeline_stage})`')
#         print(f'Proceeding with:\n')
#         print(f'Learning rate = {config.model.lr:.3e}')
#         print(f'Batch size = {config.model.batch_size}')
        
#         return config.model.lr, None, config

    if os.path.isfile(config.lr_finder_results_path): #Path(config.experiment_dir,"lr_finder","hparams.yaml")):# and not config.tuner.options.lr.force_rerun:
        
        best_hparams = Extract.config_from_yaml(config.lr_finder_results_path)
            
        best_lr = best_hparams['lr']
        config.model.lr = best_lr
        assert config.model.lr == best_lr

        print(f'[FOUND] Previously completed trial. Results located in file:\n`{config.lr_finder_results_path}`')
        print(f'[LOADING] Previous results + avoiding repetition of tuning procedure.')
        print(f'Proceeding with learning rate, lr = {config.model.lr:.3e}')
        print('Model hparams =')
        pp(best_hparams)
        return config.model.lr, None, config
    
    
#     os.environ["WANDB_PROJECT"] = stage_config.init.project
#     run = wandb.init(entity=stage_config.init.entity,
#                      project=stage_config.init.project,
#                      job_type=stage_config.init.job_type,
#                      group=stage_config.init.group,
#                      dir=stage_config.init.run_dir,
#                      config=OmegaConf.to_container(stage_config.init.config, resolve=True))

#     datamodule, data_artifact = fetch_datamodule_from_dataset_artifact(dataset_config=config.datamodule,
#                                                                        artifact_config=config.artifacts.input_dataset_artifact,
#                                                                        run_or_api=run)
#     config.model.num_classes = len(datamodule.classes)
    # assert (num_classes == 19) | (num_classes == 179)
    ########################################
    ########################################
#     model, model_artifact = build_and_log_model_to_artifact(model_config=config.model,
#                                         artifact_config=config.artifacts.input_model_artifact,
#                                         run_or_api=run)

#     model, model_artifact = use_model_artifact(artifact_config=config.artifacts.input_model_artifact,
#                                                model_config=config.model,
#                                                run_or_api=run)


#     trainer = configure_trainer(config)
    
    try:
        if ("batch_size" not in model.hparams) or (model.hparams.batch_size is None):
            model.hparams.batch_size = config.data.batch_size
#         model.hparams = OmegaConf.create(model.hparams) #, resolve=True)
        print('Continuing with model.hparams:', model.hparams)
    except Exception as e:
        print(e)
        print('conversion from Omegaconf failed', model.hparams)
        print('continuing')    
    
    lr_tuner = trainer.tuner.lr_find(model, datamodule)#, **config.tuner.tuner_kwargs.lr)

    # TODO: pickle lr_tuner object
    lr_tuner_results = lr_tuner.results
    best_lr = lr_tuner.suggestion()
    
    suggestion = {"lr": best_lr,
                  "loss":lr_tuner_results['loss'][lr_tuner._optimal_idx]}

    model.hparams.lr = suggestion["lr"]
#     config.model.optimizer.lr = model.hparams.lr
    config.model.lr = model.hparams.lr
#     run.config.update(config, allow_val_change=True)
    
        
    best_hparams = DictConfig({"optimized_hparam_key": "lr",
                               "lr":best_lr,
                               "batch_size":config.data.batch_size})
#                                   "input_shape": model.hparams.input_shape,
#                                   "image_size":config.datamodule.image_size})
    
#     results_dir = Path(config.experiment_dir, "lr_finder")
    results_dir = Path(config.lr_finder_results_path).parent
    os.makedirs(results_dir, exist_ok=True)
    Extract.config2yaml(best_hparams, config.lr_finder_results_path)
    print(f'Saved best lr value (along w/ batch_size, image_size) to file located at: {str(config.lr_finder_results_path)}') # {str(results_dir / "hparams.yaml")}')
    print(f'File contents expected to contain: \n{dict(best_hparams)}')
    
        
    fig = lr_tuner.plot(suggest=True)
    plot_fname = 'lr_tuner_results_loss-vs-lr.png'
    plot_path = results_dir / plot_fname
    plt.suptitle(f"Suggested lr={best_lr:.4e} |\n| Searched {lr_tuner.num_training} lr values $\in$ [{lr_tuner.lr_min},{lr_tuner.lr_max}] |\n| bsz = {config.data.batch_size}")
    plt.tight_layout()
    plt.savefig(plot_path)
    if run is not None:
#         run.summary['lr_finder/plot'] = wandb.Image(fig, caption=plot_fname)
        run.summary['lr_finder/plot'] = wandb.Image(str(plot_path), caption=plot_fname)
        run.summary['lr_finder/best/loss'] = suggestion["loss"]
        run.summary['lr_finder/best/lr'] = suggestion["lr"]
        run.summary['lr_finder/batch_size'] = config.data.batch_size
        run.summary['image_size'] = config.data.image_size
        run.summary['lr_finder/results'] = dict(best_hparams)

#     run.finish()
    
    
    print(f'FINISHED: `run_lr_tuner(config)`')
    print(f'Proceeding with:\n')
    print(f'Learning rate = {config.model.lr:.3e}')
    print(f'Batch size = {config.data.batch_size}')
    
#     del datamodule
#     del model
#     del trainer
    
    return suggestion, lr_tuner_results, config






def load_data_and_model(config: argparse.Namespace):


    if config.debug == True:
        print(f"Debug mode activated, loading CIFAR10 datamodule")
        datamodule = CIFAR10DataModule(batch_size=config.data.batch_size,
                                       task_id=0,
                                       image_size=config.data.image_size,
                                       image_buffer_size=config.data.image_buffer_size,
                                       num_workers=config.data.num_workers,
                                       pin_memory=config.data.pin_memory)
    else:
        datamodule = PlantDataModule(batch_size=config.data.batch_size,
                                     task_id=0,
                                     image_size=config.data.image_size,
                                     image_buffer_size=config.data.image_buffer_size,
                                     num_workers=config.data.num_workers,
                                     pin_memory=config.data.pin_memory)
    datamodule.setup("fit")
    config.model.num_classes = datamodule.num_classes
    pp(config)
    
    if os.path.isfile(str(config.model.ckpt_path)):
        print(f"Loading from model checkpoint: {str(config.model.ckpt_path)}")
        model = LitMultiTaskModule.load_from_checkpoint(config.model.ckpt_path, config=config.model)
    else:
        if isinstance(config.model.ckpt_path, (str, Path)):
            print(f"User specified checkpoint path doesn't exist. Best checkpoint produced during training will be copied to that location: {config.model.ckpt_path}")
        print(f"Instantiating model from scratch with hparams:")
        print(config.model)
        model = LitMultiTaskModule(config.model)
    model.label_encoder = datamodule.label_encoder

    return datamodule, model




                    
        


# def train_source_task(config: argparse.Namespace):
#     pl.seed_everything(config.seed)
    # ## DataModule
#     if config.debug == True:
#         print(f"Debug mode activated, loading CIFAR10 datamodule")
#         datamodule = CIFAR10DataModule(batch_size=config.data.batch_size,
#                                        task_id=0,
#                                        image_size=config.data.image_size,
#                                        image_buffer_size=config.data.image_buffer_size,
#                                        num_workers=config.data.num_workers,
#                                        pin_memory=config.data.pin_memory)
#     else:
#         datamodule = PlantDataModule(batch_size=config.data.batch_size,
#                                      task_id=0,
#                                      image_size=config.data.image_size,
#                                      image_buffer_size=config.data.image_buffer_size,
#                                      num_workers=config.data.num_workers,
#                                      pin_memory=config.data.pin_memory)
#     datamodule.setup("fit")
#     config.model.num_classes = datamodule.num_classes
#     pp(config)
    
#     if os.path.isfile(str(config.model.ckpt_path)):
#         print(f"Loading from model checkpoint: {str(config.model.ckpt_path)}")
#         model = LitMultiTaskModule.load_from_checkpoint(config.model.ckpt_path, config=config.model)

#     else:
#         if isinstance(config.model.ckpt_path, (str, Path)):
#             print(f"User specified checkpoint path doesn't exist. Best checkpoint produced during training will be copied to that location: {config.model.ckpt_path}")
#         print(f"Instantiating model from scratch with hparams:")
#         print(config.model)
#         model = LitMultiTaskModule(config.model)
#     model.label_encoder = datamodule.label_encoder


def train_source_task(config: argparse.Namespace):
    pl.seed_everything(config.seed)

    datamodule, model = load_data_and_model(config=config)
    

    k=15
    img_prediction_callback = ImagePredictionLogger(top_k_per_batch=k,
                                                    bottom_k_per_batch=k)    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor=config.callbacks.monitor.metric,
                                                       save_top_k=1,
                                                       save_last=True,
                                                       dirpath=str(Path(config.model_ckpt_dir, "task_0")),
                                                       filename='{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}',
                                                       verbose=True,
                                                       mode=config.callbacks.monitor.mode)
    earlystopping = pl.callbacks.EarlyStopping(monitor=config.callbacks.monitor.metric,
                                               patience=3,
                                               mode=config.callbacks.monitor.mode)
    wandb_logger = pl.loggers.WandbLogger(entity = "jrose",
                                          project = "image_classification_train",
                                          job_type = "train_supervised",
                                          config=dict(config),
                                          group=f'{config.model.model_name}_task_0',
                                          reinit=True,
                                          dir=config.experiment_dir)

    # ## Trainer
    trainer = pl.Trainer(
#                 limit_train_batches=0.1,
#                 limit_val_batches=0.1,
                resume_from_checkpoint=config.trainer.resume_from_checkpoint,
                max_epochs=config.trainer.num_epochs,
                gpus=config.trainer.gpus,
                auto_lr_find=config.stages.lr_tuner,
                precision=config.trainer.precision,
                callbacks=[earlystopping,
                           checkpoint_callback,
                           img_prediction_callback],
#                 overfit_batches=5,
                logger=wandb_logger,
#                 track_grad_norm=2,
                weights_summary='top')

    if config.debug == True:
        import pdb; pdb.set_trace()

        
    if config.stages.lr_tuner == True:
        
        with wandb.init(entity = "jrose",
                        project = "image_classification_train",
                        job_type = "lr_tune",
                        config=dict(config),
                        group=f'{config.model.model_name}_task_0',
                        reinit=True,
                        dir=config.experiment_dir) as run:

            print(f"[Initiating Stage] lr_tuner")
            suggestion, lr_tuner_results, config = run_lr_tuner(trainer=trainer,
                                                                model=model,
                                                                datamodule=datamodule,
                                                                config=config,
                                                                run=run)
        
        
    try:
        trainer.fit(model, datamodule)
    except KeyboardInterrupt as e:
        print("Interruption:", e)
    finally:
        print(f"checkpoint_callback.best_model_path: {checkpoint_callback.best_model_path}")
        print(f"checkpoint_callback.best_model_score: {checkpoint_callback.best_model_score}")
    
    
#     wandb.finish()

    print(f"[Initiating TESTING on task_0]")

    test_results = trainer.test(datamodule=datamodule)
######################################


# from lightning_hydra_classifiers.experiments.transfer_experiment import TransferExperiment
# output_root_dir = "/media/data_cifs/projects/prj_fossils/users/jacob/experiments/July2021-Nov2021/csv_datasets/experimental_datasets"
# experiment = TransferExperiment()
# experiment.export_experiment_spec(output_root_dir=output_root_dir)




def cmdline_args(arg_overrides=None):
    
#     parser = argparse.ArgumentParser(description="")
#     subparsers = parser.add_subparsers()
#     p = subparsers.add_parser('', help='model args')
    p = argparse.ArgumentParser(description="")
    # subparser_one = parser_one.add_subparsers()
    p.add_argument("-o", "--output_dir", dest="output_dir", type=str,
                   default="/media/data_cifs/projects/prj_fossils/users/jacob/experiments/July2021-Nov2021/experiment_logs/Transfer_Experiments",
                   help="Output root directory for experiment logs.")
    p.add_argument("-ckpt", "--load_from_checkpoint", dest="load_from_checkpoint", type=str,
                   default=None,
                   help="Attempt to load model weights from checkpoint prior to training. If user desires to also load previous epoch info, use resume_from_checkpoint instead. If pointing to nonexistent file path, will attempt to copy best checkpoint produced by ModelCheckpoint callback to the specified location.")
    p.add_argument("-resume", "--resume_from_checkpoint", dest="resume_from_checkpoint", type=str,
                   default=None,
                   help="Provide a path to checkpoint in order to resume previous training from beginning of the last epoch. This will override --load_from_checkpoint flag.")
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
    p.add_argument("--gpus", dest="gpus", type=int, default=1, nargs="*",
                   help="Specify number of gpus or specific gpu ids.")
    p.add_argument("-d", "--debug", dest="debug", action="store_true", default=False,
                   help="Flag for activating debug-related settings. Currently limited to switching out datamodule to use CIFAR10")
    args = p.parse_args(arg_overrides)#[""])
    print("Args:")
    pp(args)

    config = Munch({
        "seed":42,
#         "num_epochs": args.num_epochs,
#         "precision": 16,
        "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        "output_dir":args.output_dir,
        "debug":args.debug
    })

    config.trainer = Munch({"precision": 16,
                            "num_epochs": args.num_epochs,
                            "gpus": args.gpus,
                            "resume_from_checkpoint": args.resume_from_checkpoint
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
                          "image_size": args.image_size,
                          "ckpt_path": args.load_from_checkpoint,
                          "t_max": 20,
                          "min_lr": 1e-6})

    config.callbacks = Munch({"monitor":
                              Munch({"metric":"val_acc",
                                     "mode": "max"})
#                               Munch({"metric":"val_loss",
#                                      "mode": "min"})
                             })

    config.stages = Munch({"lr_tuner":True,
                           "task_0":None, #Munch(),
                           "task_1":None})
    
    
    if config.debug:
        config.stages.task_0 = "CIFAR10"
    else:
        config.stages.task_0 = "Extant-PNAS"
        config.stages.task_1 = "PNAS"

    if config.stages.task_1 is not None:
        task_tags = config.stages.task_0 + "-to-" + config.stages.task_1
    else:
        task_tags = config.stages.task_0
        
    if config.model.pretrained in ("imagenet", True):
        weights_name = "imagenet_weights"
    else:
        weights_name = "random_weights"
        
    config.experiment_name = "_".join([task_tags, config.model.model_name, weights_name])
    config.experiment_dir = os.path.join(config.output_dir, config.experiment_name)
    
    config.model_ckpt_dir = str(Path(config.experiment_dir,"checkpoints"))
    config.lr_finder_results_path = str(Path(config.experiment_dir,"lr_finder","hparams.yaml"))
                      
                      
                      
                      
    return args, config


if __name__ == '__main__':
    
    args, config = cmdline_args()
    os.makedirs(config.experiment_dir, exist_ok=True)
#     os.makedirs(config.output_dir, exist_ok=True)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    
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
    
    
    
    
    
    
