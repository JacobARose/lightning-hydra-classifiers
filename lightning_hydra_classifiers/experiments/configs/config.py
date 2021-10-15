
"""

lightning_hydra_classifiers/experiments/configs/config.py


The main config for an experiment orchestration

Author: Jacob A Rose
Created: Monday Sept 13th, 2021


"""

import os
from typing import *
from dataclasses import dataclass, field

from lightning_hydra_classifiers.experiments.configs.model import *
from lightning_hydra_classifiers.experiments.configs.trainer import *
from lightning_hydra_classifiers.experiments.configs.data import *
from lightning_hydra_classifiers.experiments.configs.pretrain.lr_tuner import *
from lightning_hydra_classifiers.experiments.configs.file_manager import TaskFileSystemConfig, MultiTaskFileSystemConfig, SystemConfig
from .base import BaseConfig


import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


__all__ =  ["MultiTaskExperimentConfig", "BaseExperimentConfig", "register_configs"]

# @dataclass
# class MetaDataConfig
# experiment_dir / task_name / hparam_md5_hash / <all loggable experiment contents>

# defaults = ["replicate=1",
#             "seed=97865"]#,
# #             "root_dir=."
# ]
defaults = []#"replicate=1"]
#      {"seed": "97865"}]
#      {"root_dir": "."}
# ]

@dataclass
class BaseExperimentConfig(BaseConfig):

#     defaults: List[Any] = field(default_factory=list)
    
    root_dir: str = ""
    seed: int = 3899#MISSING
    replicate: int = 1 #MISSING
    overwrite_previous_runs: Optional[bool] = False
        
    model: BaseLitModuleConfig = LitMultiTaskModuleConfig()
    data: MultiTaskDataModuleConfig = MultiTaskDataModuleConfig()        
    trainer: TrainerConfig = TrainerConfig()
    system: Optional[SystemConfig] = None        
    task_ids: Tuple[str] = field(default_factory=tuple)
        
        
#     def __post_init__(self):
        
#         if self.overwrite_previous_runs:
#             self.replicate = self.get_next_unused_replicate(self.replicate)
        

        
    def get_experiment_name(self):
        return "__".join([self.data.dataset_name,
                   self.model.backbone.backbone_name,
                   f"{self.model.backbone.weights_tag}-weights"])
        
    
    def run_dir(self, hashname: str=None, replicate: int=None):
        """The location where any run is stored."""
        hashname = hashname or self.hashname
        replicate = replicate or self.replicate

#         if not isinstance(replicate, int) or replicate <= 0:
#             raise ValueError('Bad replicate: {}'.format(replicate))
            
        parts = [self.system.experiment_dir,
                 hashname,
                 self.system.name_prefix,
                 f'replicate_{replicate}']
#                  "__".join([self.name_prefix, self.hashname]),
#                  f'replicate_{replicate}']
#         if stage is not None:
#             parts.append(stage)
            
        return os.path.join(*parts)
    
        
#     def __post_init__(self):
        
#         os.makedirs(self.root_dir)
        

        
        
        
        
@dataclass
class SingleTaskExperimentConfig(BaseExperimentConfig):


#     root_dir: str = "."
#     seed: int = 1234
#     replicate: int = 1



    model: LitSingleTaskModuleConfig = LitSingleTaskModuleConfig()
    data: MultiTaskDataModuleConfig = MultiTaskDataModuleConfig()        
    trainer: TrainerConfig = TrainerConfig()
    pretrain: LRTunerConfig = LRTunerConfig()
    system: Optional[TaskFileSystemConfig] = None        
#     task_ids: Tuple[str] = field(default_factory=lambda: ("task_0"))




# _defaults = [{"model@config.model":"LitMultiTaskModuleConfig"}]
#             "_self_"]
#      {"seed": "97865"}]
#      {"root_dir": "."}

@dataclass
class MonitorMetricConfig:
    metric: str = "val_acc"
    mode: str = "max"

@dataclass
class CallbacksConfig:
    
    monitor: MonitorMetricConfig = field(default_factory=MonitorMetricConfig)




@dataclass
class MultiTaskExperimentConfig(BaseExperimentConfig):


#     root_dir: str = "."
#     seed: int = 1234
#     replicate: int = 1
#     overwrite_previous_runs: bool = False
#     defaults: List[Any] = field(default_factory=lambda: _defaults)

    model: LitMultiTaskModuleConfig = field(default_factory=LitMultiTaskModuleConfig)
    data: MultiTaskDataModuleConfig = field(default_factory=MultiTaskDataModuleConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    callbacks: CallbacksConfig = field(default_factory=CallbacksConfig)
        
    pretrain: LRTunerConfig = field(default_factory=LRTunerConfig)
    system: Optional[MultiTaskFileSystemConfig] = field(init=False) # None
        
        
    debug: bool = False
#     task_ids: Tuple[str] = field(default_factory=lambda: ("task_0", "task_1"))
    # TODO: Add structured config for orchestrating image statistics logging pre-running anything else
    
    def __post_init__(self):
#         super().__post_init__()
        self.system = MultiTaskFileSystemConfig(replicate=self.replicate,
                                                experiment_name=self.get_experiment_name(),
                                                root_dir=self.root_dir,
                                                hashname=self.hashname)
#         self.model.ckpt_path =


def register_configs():
    cs = ConfigStore.instance()
    cs.store(name="config",
             node=MultiTaskExperimentConfig)



