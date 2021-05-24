#!/usr/bin/env python
# coding: utf-8

# # Model Training with WandB Versioned Data `[version 2]`
# 
# `model_training_w_wandb_versioned_data-[version 2]-refactor_input-and-output_artifact_config.ipynb`
# 
# Author: Jacob A Rose  
# Created on: Wednesday May 12th, 2021
# 
# -----
# Based on [this](https://colab.research.google.com/drive/1PRnwxttjPst6OmGiw9LDoVYRDApbcIjE) notebook (or see the original [report](https://wandb.ai/stacey/mendeleev/reports/DSViz-for-Image-Classification--VmlldzozNjE3NjA)), in which the [iNaturalist 2017](https://github.com/visipedia/inat_comp/tree/master/2017) dataset is re-sampled to several different versions, each with different #s of samples per class.
# * These are then used to train and validate a keras model, then log many image predictions to W&B so that the versioned input images are directly associated with each new log. 
# * This allows a flexible and scalable opportunity for researchers and even untrained members of the public to easily interact with the data and interrogate model decisions.
# 
# * Users can query subsets of the prediction results using basic logical queries w/ an effective autocomplete providing real-time suggestions for valid queries.

# ----------------------------
# 



from pathlib import Path
import os
import wandb
# os.environ['WANDB_CACHE_DIR'] = "/media/data/jacob/wandb_cache"
# os.makedirs("/media/data/jacob/wandb_cache", exist_ok=True)
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import numpy as np
import pytorch_lightning as pl
import torchvision
import torch
from torch import nn
import matplotlib.pyplot as plt
from contrastive_learning.data.pytorch.tensor import tensor_nbytes
from contrastive_learning.data.pytorch.pnas import PNASLightningDataModule
from contrastive_learning.data.pytorch.extant import ExtantLightningDataModule
from contrastive_learning.data.pytorch.common import DataStageError, colorbar, LeavesLightningDataModule
from contrastive_learning.data.pytorch.datamodules import get_datamodule, fetch_datamodule_from_dataset_artifact

from lightning_hydra_classifiers.callbacks.wandb_callbacks import LogPerClassMetricsToWandb, WandbClassificationCallback
from lightning_hydra_classifiers.models.resnet import ResNet
import lightning_hydra_classifiers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pl_bolts.callbacks import ModuleDataMonitor, BatchGradientVerificationCallback

import inspect
pl.trainer.seed_everything(seed=389)

from stuf import stuf
from box import Box
from rich import print as pp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')


from tqdm.auto import trange, tqdm
from dataclasses import dataclass, field, asdict, astuple, InitVar
from typing import Tuple, List, Dict, Any, Type, Union
from enum import Enum
from pprint import pprint as pp


########################################


def build_or_restore_model_from_artifact(model_config: Box, 
                                         artifact_config: Box,
                                         run_or_api=None) -> pl.LightningModule:
    """
    
    
    """
    model = ResNet(model_name=model_config.name,
                   num_classes=model_config.num_classes,
                   input_shape=model_config.input_shape,
                   optimizer=model_config.optimizer)
    model.reset_classifier(model_config.num_classes,'avg')
    model.unfreeze(getattr(model, model_config.unfreeze[0]))

    artifact_config.init_model_path = Path(artifact_config.init_model_dir,
                                           artifact_config.version,
                                           "imagenet")
    os.makedirs(os.path.dirname(artifact_config.init_model_path), exist_ok=True)
    model.save_model(artifact_config.init_model_path)

    artifact = wandb.Artifact(artifact_config.init_name,
                              type=artifact_config.input_type,
                              description=artifact_config.description,
                              metadata=dict({
                                            {"artifact":**artifact_config},
                                             "model":{**model_config}
                                           })
                             )
    artifact.add_dir(os.path.dirname(artifact_config.init_model_path))
    run.log_artifact(model_config)
    

    return model, artifact


def get_labels_from_filepath(path: str, fix_catalog_number: bool = False) -> Dict[str,str]:
    """
    Splits a precisely-formatted filename with the expectation that it is constructed with the following fields separated by '_':
    1. family
    2. genus
    3. species
    4. collection
    5. catalog_number
    
    If fix_catalog_number is True, assume that the collection is not included and must separately be extracted from the first part of the catalog number.
    
    """
    family, genus, species, collection, catalog_number = Path(path).stem.split("_", maxsplit=4)
    if fix_catalog_number:
        catalog_number = '_'.join([collection, catalog_number])
    return {"family":family,
            "genus":genus,
            "species":species,
            "collection":collection,
            "catalog_number":catalog_number}


def log_predictions(trainer, model, datamodule, fix_catalog_number=True):
    subset = datamodule.predict_on_split
    datamodule.setup(stage="predict")
    classes = datamodule.classes
    
    columns = ['catalog_number',
               'image',
               'guess',
               'truth',
               'softmax_guess',
               'softmax_truth']
    for j, class_name in enumerate(classes):
        columns.append(f'score_{class_name}')

    
    loss_func = torch.nn.CrossEntropyLoss(reduction='none')
    batches = trainer.predict(model, datamodule=datamodule)
#         x, y_logit, y_true, img_paths = list(np.concatenate(i) for i in zip(*batches))    
    prediction_rows = []
    for x, y_logit, y_true, img_paths in tqdm(batches, position=0, desc="epoch->", unit='batches'):

        y_true = torch.from_numpy(y_true).to(dtype=torch.long)
        y_logit = torch.from_numpy(y_logit)

        loss = loss_func(y_logit, y_true)

        y_pred = torch.argmax(y_logit, -1)
        y_score = y_logit.softmax(1)

        x = torch.from_numpy(x).permute(0,2,3,1)
        x = (255 * (x - x.min()) / (x.max() - x.min())).numpy().astype(np.uint8)
#         for i in tqdm(range(len(y_pred)), position=1, desc="batch->", unit='sample'):
        for i in trange(len(y_pred), position=1, leave=False, desc="batch->", unit='sample'):
            labels = get_labels_from_filepath(path=img_paths[i],
                                              fix_catalog_number=fix_catalog_number)
            row = [
                    labels['catalog_number'],
                    wandb.Image(x[i,...]),
                    classes[y_pred[i]],
                    classes[y_true[i]],
                    y_score[i,y_pred[i]],
                    y_score[i,y_true[i]]
            ]

            for j, score_j in enumerate(y_score[i,:].tolist()):
                row.append(np.round(score_j, 4))
            prediction_rows.append(row)


    prediction_table = wandb.Table(data=prediction_rows, columns=columns)
    artifact_name = f"{wandb.run.name}_{wandb.run.id}"
    prediction_artifact = wandb.Artifact(artifact_name,
                                         type=f"{subset}_predictions")
    
    prediction_artifact.add(prediction_table, f"{subset}_predictions")
    wandb.run.log_artifact(prediction_artifact)
    return prediction_artifact



########################################
########################################

def test_model(trainer,
               model,
               model_artifact,
               datamodule,
               config,
               fix_catalog_number=False, predictions_only=False):

    run = wandb.init(entity=config.wandb.init.entity,
                     project=config.wandb.init.project,
                     job_type="test",
                     group=config.wandb.init.group,
                     dir=config.wandb.init.run_dir,
                     config=config)
    
#     model_artifact.wait()
#     model_input_artifact = run.use_artifact(model_artifact)
    
    model_input_artifact = run.use_artifact(config.wandb.model_artifact.output_name + ":latest")
    model_input_dir = model_input_artifact.download(config.wandb.model_artifact.output_model_dir)
    
    assert Path(config.wandb.model_artifact.output_model_path).name in os.listdir(model_input_dir)
#     model.load_from_checkpoint(config.wandb.model_artifact.output_model_path)
    model.load_model(config.wandb.model_artifact.output_model_path)
    
    
    if not predictions_only:
        test_results = trainer.test(model, datamodule=datamodule)
        if isinstance(test_results, list):
            assert len(test_results) == 1
            test_results = test_results[0]
            
    prediction_artifact = log_predictions(trainer, model, datamodule, fix_catalog_number=fix_catalog_number)
    
    run.finish()    
    return prediction_artifact, test_results

#### Callbacks + Trainer


def configure_trainer(config, **kwargs):
    """
    Example:
        trainer = configure_trainer(config)
    """
    monitor = ModuleDataMonitor()#log_every_n_steps=25)
    per_class_metric_plots_cb = LogPerClassMetricsToWandb()
    early_stop_callback = EarlyStopping(monitor='val_loss',
                                        patience=3,
                                        verbose=False,
                                        mode='min')

    logger=pl.loggers.wandb.WandbLogger(name=f"{config.dataset.name}-timm-{config.model.name}",
                                        config=config)

    trainer = pl.Trainer(gpus=config.trainer.gpus,
                         logger=logger,
                         max_epochs=config.trainer.max_epochs,
                         weights_summary=config.trainer.weights_summary,
                         profiler=config.trainer.profiler,
                         log_every_n_steps=config.trainer.log_every_n_steps,
                         auto_scale_batch_size=config.trainer.auto_scale_batch_size,
                         callbacks=[per_class_metric_plots_cb,
                                    monitor,
                                    early_stop_callback],
                        **kwargs)
    return trainer

def log_model_checkpoint_2_artifact(model, 
                                    ckpt_path: str,
                                    artifact_config,
                                    run=None):
    
    run = run or wandb.run
    os.makedirs(os.path.dirname(artifact_config.output_model_path), exist_ok=True)
    
    model = model.load_from_checkpoint(ckpt_path)
    model.save_model(artifact_config.output_model_path)

    output_model_artifact = wandb.Artifact(
                                    artifact_config.output_name,
                                    type=artifact_config.output_type,
                                    description=artifact_config.description,
                                    metadata=dict(**artifact_config,
                                                  **config.model)
                                    )
    output_model_artifact.add_dir(artifact_config.output_model_dir)
    run.log_artifact(output_model_artifact)

    print("Output Model Artifact Checkpoint path:\n", 
          artifact_config.output_model_path)

########################################
####
########################################

########################################
#### Instantiate dataset ####
########################################


########################################
#### Instantiate model ####
########################################

# datamodule, artifact = fetch_datamodule_from_dataset_artifact(config=config, run_or_api=run)

# assert (num_classes == 19) | (num_classes == 179)

########################################



os.environ["WANDB_PROJECT"] = config.wandb.init.project
run = wandb.init(entity='jrose',
                 project=config.wandb.init.project,
                 job_type=config.wandb.init.job_type,
                 group=config.wandb.init.group,
                 dir=config.wandb.init.run_dir,
                 config=config)

########################################
#### Instantiate model ####
########################################

datamodule, data_artifact = fetch_datamodule_from_dataset_artifact(config=config, run_or_api=run)

# assert (num_classes == 19) | (num_classes == 179)

model, model_artifact = build_or_restore_model_from_artifact(model_config=model_config,
                                                             artifact_config=artifact_config,
                                                             run_or_api=run)


# if config.trainer.auto_scale_batch_size:
#     trainer = configure_trainer(config) #, auto_scale_batch_size=config.trainer.auto_scale_batch_size)
#     bsz_tuner = trainer.tune(model, datamodule=datamodule)
    
if config.trainer.auto_lr_find:
    trainer = configure_trainer(config)
    lr_finder = trainer.tuner.lr_find(model, datamodule, num_training=config.trainer.auto_lr_num_training)
    
    trainer.tuner.lr_find


# ## Friday, May 21st 2021
# 
# * TODO: Perform bootstrapping and/or k-val split for this ~50-100 sample learning rate search
# 
# * Idea: What other hyperparameters can we constrain/further optimize with these wide->narrow sweeps

# In[19]:


# Pick point based on plot, or get suggestion
new_lr = lr_finder.suggestion()

print(f'Suggested starting learning rate: {new_lr:.2e}')

model.hparams.lr = new_lr
config.model.optimizer.init_lr = new_lr

fig = lr_finder.plot(suggest=True)
fig.show()


# In[7]:


get_ipython().run_cell_magic('prun', '', 'trainer = configure_trainer(config, log_gpu_memory=True) # limit_train_batches=2, limit_val_batches=2)\n\ntrainer.fit(model, datamodule=datamodule)\nbest_ckpt = trainer.callbacks[-1].best_model_path\n\n\nlog_model_checkpoint_2_artifact(model, \n                                ckpt_path: str,\n                                artifact_config=config.wandb.output_artifacts[0],\n                                run=run)\n\n\nfix_catalog_number = "PNAS" in config.dataset.name\ntrainer = configure_trainer(config) #, limit_test_batches=2)\n\ntest_results = test_model(trainer,\n                          model,\n                          output_model_artifact,\n                          datamodule,\n                          config,\n                          fix_catalog_number=fix_catalog_number)\n\n\n\nwandb.finish()\nprint(trainer.current_epoch, test_results)')





##########################################
##########################################

fix_catalog_number = "PNAS" in config.dataset.name
trainer = configure_trainer(config) #, limit_test_batches=2)

test_results = test_model(trainer,
                          model,
                          output_model_artifact,
                          datamodule,
                          config,
                          fix_catalog_number=fix_catalog_number)



wandb.finish()
print(trainer.current_epoch, test_results)



# import torch
# from GPUtil import showUtilization as gpu_usage

# def test_pytorch_gpu_empty_cache(num_GB_2_gpu=10):
#     print("Initial GPU Usage")
#     gpu_usage()
#     tensorList = []
#     for x in range(10):
#         tensorList.append(torch.randn(int(10e6), num_GB_2_gpu).cuda())   # reduce the size of tensor if you are getting OOM
        
#         gpu_usage()

#     print("GPU Usage after allocating a bunch of Tensors")
#     gpu_usage()
#     del tensorList
#     print("GPU Usage after deleting the Tensors")
#     gpu_usage()

#     print("GPU Usage after emptying the cache")
#     torch.cuda.empty_cache()
#     gpu_usage()
    
    
    
# test_pytorch_gpu_empty_cache(12)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')


########################################

# # Notes
# 

# ### Left to do 5-16-2021
# 
# 1. trainer.fit ***
# 2. trainer.test ***
# 
# 3. Refactor ValLog Callback to log PyTorch Lightning predictions + per-class scores *****
# 
# 4. Add data tests for WandB downloaded data \*\*\*
# 
#     4a. Implement Data Drift framework!!!
#     
# 5. Add Salient Map interpretability callback \*\*\*
#     * https://github.com/MisaOgura/flashtorch
#     
# priority: \*, \*\*,\*\*\*, ****, *****