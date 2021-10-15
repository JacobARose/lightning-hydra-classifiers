
"""

lightning_hydra_classifiers/experiments/configs/file_manager.py


The configs for an experiment file & dir orchestration

Author: Jacob A Rose
Created: Wednesday Sept 15th, 2021


"""
from pathlib import Path
import os
from typing import *
from dataclasses import dataclass, field, asdict
from rich import print as pp
# from lightning_hydra_classifiers.experiments.configs.model import *
# from lightning_hydra_classifiers.experiments.configs.trainer import *
# from lightning_hydra_classifiers.experiments.configs.data import *
# from lightning_hydra_classifiers.experiments.configs.pretrain.lr_tuner import *

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from lightning_hydra_classifiers.experiments.configs.utils import hash_utils

from lightning_hydra_classifiers.experiments.configs.base import BaseConfig

__all__ =  ["SystemConfig", "TaskFileSystemConfig", "MultiTaskFileSystemConfig"]

# @dataclass
# class MetaDataConfig
# experiment_dir / task_name / hparam_md5_hash / <all loggable experiment contents>

@dataclass
class SystemConfig(BaseConfig):
    """
    All directories created or logged to during experiment
    """
    replicate: int
    experiment_name: str
    root_dir: str

    overwrite_previous_runs: Optional[bool] = field(default=False)
#     hashname: Optional[str] = field(default="")
    experiment_dir: str = field(init=False)
    
    def __post_init__(self):
        self.experiment_dir = str(Path(self.root_dir, self.experiment_name))
        
        if self.overwrite_previous_runs:
            self.replicate = self.get_next_unused_replicate(self.replicate)

    
    
    def run_dir(self, hashname: str, replicate: int, stage: Optional[str] = 'main'):
        """The location where any run is stored."""

        if not isinstance(replicate, int) or replicate <= 0:
            raise ValueError('Bad replicate: {}'.format(replicate))
            
        parts = [self.experiment_dir,
                 hashname,
                 self.name_prefix,
                 f'replicate_{replicate}']
#                  "__".join([self.name_prefix, self.hashname]),
#                  f'replicate_{replicate}']
        if stage is not None:
            parts.append(stage)
            
        return os.path.join(*parts)
    
    @property
    def name_prefix(self) -> str:
        return "main" # self.experiment_name
    
    def get_next_unused_replicate(self, hashname: str, lowest_replicate: int=1) -> int:
        replicate = lowest_replicate
        if not isinstance(replicate, int) or replicate <= 0:
            raise ValueError('Bad replicate: {}'.format(replicate))

        while not os.path.exists(self.run_dir(hashname=hashname, replicate=replicate)):
            replicate += 1
            
        return replicate

        
        
#         self.unique_config_id = hash_utils.get_hash(hparams)
#     def __hash__(self):
#         """Make the instances hashable."""
#         return hash(self)
        

@dataclass
class TaskFileSystemConfig(SystemConfig):
    task_id: str = MISSING #field(init=False)
    hashname: Optional[str] = field(default="")

    model_ckpt_dir: str = "" #field(init=False)
    lr_tuner_dir: str = "" #field(init=False)
    lr_tuner_results_path: str = "" # field(init=False)
    lr_tuner_hparams_path: str = "" # = field(init=False)
        
    def __post_init__(self):
        super().__post_init__()
        self.model_ckpt_dir = str(Path(self.run_dir(stage=self.task_id),
                                       "checkpoints"))
        self.model_ckpt_path = str(Path(self.model_ckpt_dir,
                                       "model.ckpt"))
        
        self.lr_tuner_dir = str(Path(self.run_dir(stage=self.task_id),
                                     "lr_tuner"))
        self.lr_tuner_results_path = str(Path(self.lr_tuner_dir, "results.csv"))
        self.lr_tuner_hparams_path = str(Path(self.lr_tuner_dir, "hparams.yaml"))

    @property
    def name_prefix(self) -> str:
        return "tasks"

        
    def run_dir(self, stage: str = "main"):
        return super().run_dir(hashname=self.hashname, replicate=self.replicate, stage=stage)
        
        
        
        
@dataclass
class MultiTaskFileSystemConfig(SystemConfig): #(TaskFileSystemConfig):
    task_ids: Tuple[str] = field(default_factory=lambda: ("task_0", "task_1"))
    hashname: Optional[str] = field(default="")
    tasks: Dict[str, TaskFileSystemConfig] = field(init=False, default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        
        for task_id in self.task_ids:
            self.tasks[task_id] = TaskFileSystemConfig(task_id=task_id,
                                                       hashname=self.hashname,
                                                       replicate=self.replicate,
                                                       experiment_name=self.experiment_name,
                                                       root_dir=self.root_dir)


    @property
    def name_prefix(self) -> str:
        return "tasks"
    
    
    
if __name__ == "__main__":
    
    
    print(f"MultiTaskFileSystemConfig -- default configuration:")
    pp(asdict(MultiTaskFileSystemConfig(replicate=1, experiment_name="default", root_dir='.')))

#     @property
#     def name_prefix(self) -> str:
#         return self.task_id


        

#     if config.debug:
#         config.stages.task_0 = OmegaConf.create({"name":"CIFAR10",
#                                       "task_id":0,
#                                       "skip_lr_tuner":False})
#         del config.stages.task_1
#         config.trainer.update(OmegaConf.create(max_epochs=2,
#                                     limit_train_batches=4,
#                                     limit_val_batches=4,
#                                     limit_test_batches=4,
#                                     auto_lr_find=bool(config.model.lr is not None)))
        
#     else:
#         config.stages.task_0 = OmegaConf.create({"name":"Extant-PNAS",
#                                       "task_id":0,
#                                       "skip_lr_tuner":False})
#         config.stages.task_1 = OmegaConf.create({"name":"PNAS",
#                                       "task_id":1,
#                                       "skip_lr_tuner":False})

#     if "task_1" in config.stages:
#         task_tags = config.stages.task_0.name + "-to-" + config.stages.task_1.name
#     else:
#         task_tags = config.stages.task_0.name
        
# #     import pdb; pdb.set_trace()
#     if config.model.pretrained in ("imagenet", True):
#         weights_name = "imagenet_weights"
#     else:
#         weights_name = "random_weights"
        
#     config.experiment_name = "_".join([task_tags, config.model.model_name, weights_name])
#     config.experiment_dir = os.path.join(config.output_dir, config.experiment_name)
    
#     for task in config.stages.keys():
#         if config.stages[task] is None: continue
#         config.stages[task].model_ckpt_dir = str(Path(config.experiment_dir, task, "checkpoints"))
#         config.stages[task].lr_tuner_dir = str(Path(config.experiment_dir, task, "lr_tuner"))
#         config.stages[task].lr_tuner_results_path = str(Path(config.stages[task].lr_tuner_dir, "results.csv"))
#         config.stages[task].lr_tuner_hparams_path = str(Path(config.stages[task].lr_tuner_dir, "hparams.yaml"))

        
#     config.lr_tuner = OmegaConf.structured(lr_tuner.LRTunerConfig(min_lr = 1e-07,
#                                                                   max_lr = 1.2,
#                                                                   num_training = 150,
#                                                                   mode = 'exponential',
#                                                                   early_stop_threshold = 8.0))

