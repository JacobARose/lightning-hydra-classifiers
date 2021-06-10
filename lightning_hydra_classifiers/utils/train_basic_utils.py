"""
lightning_hydra_classifiers/utils/train_basic_utils.py


Initial offloaded set of functions for use in training script `lightning_hydra_classifiers/train_basic.py`.
Originally located in the jupyter notebook `lightning_hydra_classifiers/notebooks/model_training_w_wandb_versioned_data-[version 2]-refactor_input-and-output_artifact_config.ipynb`.


Created on: Monday, May 24th, 2021
Author: Jacob A Rose


"""

from box import Box
import wandb
import os

import pytorch_lightning as pl
from typing import Tuple, Dict
import torch
from pathlib import Path
import numpy as np

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pl_bolts.callbacks import ModuleDataMonitor

from tqdm.auto import tqdm, trange

from contrastive_learning.data.pytorch.datamodules import fetch_datamodule_from_dataset_artifact
from lightning_hydra_classifiers.models.resnet import ResNet
from omegaconf import DictConfig, ListConfig, OmegaConf
import hydra


from typing import List, Optional
# from pytorch_lightning import LightningModule, LightningDataModule, Callback, Trainer
from pytorch_lightning.loggers import LightningLoggerBase

from lightning_hydra_classifiers.utils import template_utils
log = template_utils.get_logger(__name__)


def build_model(config: Box) -> pl.LightningModule:
    
    if isinstance(config.input_shape, (ListConfig, DictConfig)):
        config.input_shape = OmegaConf.to_container(config.input_shape, resolve=True)
        
    model = ResNet(model_name=config.basename,
                   num_classes=config.num_classes,
                   input_shape=config.input_shape,
                   optimizer=config.optimizer)
    model.reset_classifier(config.num_classes,'avg')
    model.unfreeze(getattr(model, config.unfreeze[0]))
    return model



def log_model_artifact(model=None,
                       artifact_config: Box=None,
                       model_config: Box=None,
                       run_or_api=None,
                       finalize: bool=True):
    """
    
    
    if model=None, expects model checkpoint to already exist at the path in: artifact_config.model_path
    """
    run = run_or_api or wandb.Api()
    model_config = model_config or {}
    model_metadata = {}
    
    if model is not None:
        os.makedirs(os.path.dirname(artifact_config.model_path), exist_ok=True)
        model.save_model(artifact_config.model_path)

        model_metadata = {"input_shape": model.input_shape,
                          "num_features": model.num_features,
                          "num_classes": model.num_classes}
    
    artifact = wandb.Artifact(artifact_config.name,
                              type=artifact_config.type,
                              description=artifact_config.description,
                              metadata=dict({
                                            "artifact":artifact_config,
                                             "model":model_config,
                                             **model_metadata
                                           })
                             )
    artifact.add_dir(os.path.dirname(artifact_config.model_path))
    if finalize:
        run.log_artifact(artifact)
    
    return artifact


def build_and_log_model_to_artifact(model_config: Box,
                                    artifact_config: Box,
                                    run_or_api=None) -> Tuple[pl.LightningModule, wandb.Artifact]:
    """
    
    
    """
    from omegaconf import DictConfig, ListConfig, OmegaConf

    if isinstance(model_config, DictConfig):
        model_config = DictConfig(OmegaConf.to_container(model_config, resolve=True))
        
    if isinstance(model_config.input_shape, ListConfig):
        model_config.input_shape = OmegaConf.to_container(model_config.input_shape, resolve=True)
        
    if isinstance(artifact_config, DictConfig):
        artifact_config = DictConfig(OmegaConf.to_container(artifact_config, resolve=True))
        
    model = build_model(model_config)

    model_artifact = log_model_artifact(model, artifact_config, model_config, run_or_api=run_or_api)

    return model, model_artifact

############################
############################


# associative learning
# metaphysical learning


############################
############################


def use_model_artifact(model=None,
                       artifact_config: Box=None,
                       model_config: Box=None,
                       run_or_api=None):
    run = run_or_api or wandb.Api()
    model_config = model_config or {}
    
    
    artifact = run.use_artifact(artifact_config.uri)
    artifact_dir = artifact.download(root=artifact_config.model_dir)
    
    if model is None:
        model = build_model(model_config)
        
    try:
        model.load_model(artifact_config.model_path)
    except Exception as e: # RunTimeError
        print('model.load_model failed. Attempting model.load_from_checkpoint')
        model = model.load_from_checkpoint(artifact_config.model_path) #config.artifacts.output_model_artifact.model_path)
    return model, artifact


############################
############################

############################
############################

def configure_trainer(config, **kwargs):
    """
    Example:
        trainer = configure_trainer(config)
    """
    from lightning_hydra_classifiers.callbacks.wandb_callbacks import LogPerClassMetricsToWandb
    
 

    monitor = ModuleDataMonitor()#log_every_n_steps=25)
    per_class_metric_plots_cb = LogPerClassMetricsToWandb()
    early_stop_callback = EarlyStopping(monitor=config.callbacks.early_stopping.monitor,
                                        patience=3,
                                        verbose=False,
                                        mode=config.callbacks.early_stopping.mode)

#     logger=pl.loggers.wandb.WandbLogger(name=f"{config.datamodule.name}-timm-{config.model.name}",
#                                         config=config)
    
    # Init Lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    
    
    train_config = OmegaConf.to_container(config.trainer, resolve=True)
    train_config.pop("_target_")
    trainer = pl.Trainer(**train_config, # gpus=config.trainer.gpus,
                         logger=logger,
#                          max_epochs=config.trainer.max_epochs,
#                          weights_summary=config.trainer.weights_summary,
#                          profiler=config.trainer.profiler,
#                          log_every_n_steps=config.trainer.log_every_n_steps,
                         auto_scale_batch_size=config.tuner.trainer_kwargs.auto_scale_batch_size,
#                          fast_dev_run=config.trainer.fast_dev_run,
                         callbacks=[per_class_metric_plots_cb,
                                    monitor,
                                    early_stop_callback],
                        **kwargs)
    return trainer






############################
############################

############################
############################


def load_model_from_checkpoint_artifact(model,
                                        artifact_config: Box,
                                        model_config: Box=None,
                                        run_or_api=None):
    run = run_or_api or wandb.Api()
    model_config = model_config or {}
    
    os.makedirs(artifact_config.model_dir, exist_ok=True)
    
    model = model.load_from_checkpoint(artifact_config.model_path)
    model.save_model(artifact_config.model_path)

    output_model_artifact = wandb.Artifact(
                                    artifact_config.name,
                                    type=artifact_config.type,
                                    description=artifact_config.description,
                                    metadata=dict(**artifact_config,
                                                  **model_config)
                                    )
    output_model_artifact.add_dir(artifact_config.model_dir)
    run.log_artifact(output_model_artifact)

    print("Output Model Artifact Checkpoint path:\n", 
          artifact_config.model_path)




def log_predictions(trainer,
                    model,
                    datamodule,
                    fix_catalog_number=True,
                    finalize=True):
    subset = datamodule.predict_on_split
    datamodule.return_paths = True
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

#         x = torch.from_numpy(x).permute(0,2,3,1)
#         x = (255 * (x - x.min()) / (x.max() - x.min())).numpy().astype(np.uint8)

        x = np.transpose(x, (0,2,3,1))
        x = (255 * (x - x.min()) / (x.max() - x.min())).astype(np.uint8)

        y_pred = y_pred.numpy()
        y_true = y_true.numpy()
        y_score = y_score.numpy()
        
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
    
    if finalize:
        wandb.run.log_artifact(prediction_artifact)
    return prediction_artifact



########################################
########################################

def run_test_model(config,
                   fix_catalog_number=False,
                   predictions_only=False,
                   pipeline_stage="3"):

    
    pipeline_stage = str(pipeline_stage)
    stage_config = config.wandb[pipeline_stage]
    
    os.environ["WANDB_PROJECT"] = stage_config.init.project
    run = wandb.init(entity=stage_config.init.entity,
                     project=stage_config.init.project,
                     job_type=stage_config.init.job_type,
                     group=stage_config.init.group,
                     dir=stage_config.init.run_dir,
                     config=OmegaConf.to_container(config, resolve=True))

#     dataset_artifact = run.use_artifact(config.artifacts.input_dataset_artifact.uri,
#                                         type=config.artifacts.input_dataset_artifact.type) 
    config.datamodule.return_paths = True
    datamodule, data_artifact = fetch_datamodule_from_dataset_artifact(dataset_config=config.datamodule,
                                                                       artifact_config=config.artifacts.input_dataset_artifact,
                                                                       run_or_api=run)
    config.model.num_classes = len(datamodule.classes)
    # assert (num_classes == 19) | (num_classes == 179)

    model, model_artifact = use_model_artifact(model=None,
                                               artifact_config=config.artifacts.output_model_artifact,
                                               model_config=config.model,
                                               run_or_api=run)

#     model_input_artifact = run.use_artifact(config.wandb.model_artifact.output_name + ":latest")
#     model_input_dir = model_input_artifact.download(config.wandb.model_artifact.output_model_dir)
#     assert Path(config.wandb.model_artifact.output_model_path).name in os.listdir(model_input_dir)    
    model = model.load_from_checkpoint(config.artifacts.output_model_artifact.model_path)
    
    trainer = configure_trainer(config)
    
    test_results = None
    if not predictions_only:
        test_results = trainer.test(model, datamodule=datamodule)
        if isinstance(test_results, list):
            assert len(test_results) == 1
            test_results = test_results[0]
            
            
    datamodule.predict_on_split = "test"
    datamodule.return_paths = True
    prediction_artifact = log_predictions(trainer,
                                          model,
                                          datamodule,
                                          fix_catalog_number=fix_catalog_number,
                                          finalize=True)#False)
    
    run.finish()    
    return prediction_artifact, test_results




#### Callbacks + Trainer







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







# def log_model_artifact(model,
#                        artifact_config: Box,
#                        model_config: Box=None,
#                        run_or_api=None):
#     run = run_or_api or wandb.Api()
#     model_config = model_config or {}
    
#     artifact_config.init_model_path = Path(artifact_config.init_model_dir,
#                                            artifact_config.version,
#                                            "imagenet")
#     os.makedirs(os.path.dirname(artifact_config.init_model_path), exist_ok=True)
#     model.save_model(artifact_config.init_model_path)

#     artifact = wandb.Artifact(artifact_config.init_name,
#                               type=artifact_config.input_type,
#                               description=artifact_config.description,
#                               metadata=dict({
#                                             "artifact":artifact_config,
#                                              "model":model_config
#                                            })
#                              )
#     artifact.add_dir(os.path.dirname(artifact_config.init_model_path))
#     run.log_artifact(model_config)
    
#     return artifact



# def log_model_checkpoint_2_artifact(model,
#                                     artifact_config: Box,
#                                     model_config: Box=None,
#                                     run_or_api=None):
    
# def load_model_from_checkpoint_artifact(model,
#                                     artifact_config: Box,
#                                     model_config: Box=None,
#                                     run_or_api=None):
    
#     run = run or wandb.run
#     os.makedirs(os.path.dirname(artifact_config.output_model_path), exist_ok=True)
    
#     model = model.load_from_checkpoint(ckpt_path)
#     model.save_model(artifact_config.output_model_path)

#     output_model_artifact = wandb.Artifact(
#                                     artifact_config.output_name,
#                                     type=artifact_config.output_type,
#                                     description=artifact_config.description,
#                                     metadata=dict(**artifact_config,
#                                                   **config.model)
#                                     )
#     output_model_artifact.add_dir(artifact_config.output_model_dir)
#     run.log_artifact(output_model_artifact)

#     print("Output Model Artifact Checkpoint path:\n", 
#           artifact_config.output_model_path)



    
# def build_and_log_model_to_artifact(model_config: Box,
#                                     artifact_config: Box,
#                                     run_or_api=None) -> Tuple[pl.LightningModule, wandb.Artifact]:
#     """
    
    
#     """
#     model = build_model(model_config)

#     model_artifact = log_model_artifact(model, artifact_config, model_config, run_or_api=run_or_api)

#     return model, model_artifact