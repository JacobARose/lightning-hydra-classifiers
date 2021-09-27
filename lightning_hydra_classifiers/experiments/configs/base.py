
"""

lightning_hydra_classifiers/experiments/configs/base.py


The base config for all experiment-related configs to inherit from. Inspired heavily by Google's open_lth

Author: Jacob A Rose
Created: Wednesday Sept 15th, 2021


"""

import os
from typing import *
from dataclasses import dataclass, field
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

import abc
import argparse
import copy
__all__ =  ["BaseConfig"]

# @dataclass
# class MetaDataConfig
# experiment_dir / task_name / hparam_md5_hash / <all loggable experiment contents>

from lightning_hydra_classifiers.experiments.configs.utils import hash_utils

# @dataclass
# class Desc(abc.ABC):
#     """The bundle of hyperparameters necessary for a particular kind of job. Contains many hparams objects.
#     Each hparams object should be a field of this dataclass.
#     """


@dataclass
class BaseConfig(abc.ABC):
    
    @staticmethod
    def name_prefix() -> str:
        """The name to prefix saved runs with."""
        pass

    @property
    def hashname(self) -> str:
        """The name under which experiments with these hyperparameters will be stored."""
        hash_str = hash_utils.get_hash(self)
        return hash_str
    
#         if len(self.name_prefix()):
#             return "_".join([self.name_prefix(), hash_str])
#         else:
#             return hash_str

#         fields_dict = {f.name: getattr(self, f.name) for f in fields(self)}
#         hparams_strs = [str(fields_dict[k]) for k in sorted(fields_dict) if isinstance(fields_dict[k], Hparams)]
#         hash_str = hashlib.md5(";".join(hparams_strs).encode('utf-8')).hexdigest()
#########################################
    
#     @staticmethod
#     @abc.abstractmethod
#     def add_args(parser: argparse.ArgumentParser, defaults: 'Desc' = None) -> None:
#         """Add the necessary command-line arguments."""
#         pass

#     @staticmethod
#     @abc.abstractmethod
#     def create_from_args(args: argparse.Namespace) -> 'Desc':
#         """Create from command line arguments."""

#         pass

#     def save(self, output_location):
#         if not get_platform().is_primary_process: return
#         if not get_platform().exists(output_location): get_platform().makedirs(output_location)

#         fields_dict = {f.name: getattr(self, f.name) for f in fields(self)}
#         hparams_strs = [fields_dict[k].display for k in sorted(fields_dict) if isinstance(fields_dict[k], Hparams)]
#         with get_platform().open(paths.hparams(output_location), 'w') as fp:
#             fp.write('\n'.join(hparams_strs))



# @dataclass
# class BaseExperimentConfig:

#     resolution: int
#     channels: int
#     root_dir: str
#     seed: int

#     model: BaseLitModuleConfig = LitMultiTaskModuleConfig()
#     data: MultiTaskDataModuleConfig = MultiTaskDataModuleConfig()        
#     trainer: TrainerConfig = TrainerConfig()
        
#     task_ids: Tuple[str] = field(default_factory=tuple)
        
        
# @dataclass
# class SingleTaskExperimentConfig(BaseExperimentConfig):

#     resolution: int = 512
#     channels: int = 3
#     root_dir: str = "."
#     seed: int = 1234

#     model: LitSingleTaskModuleConfig = LitSingleTaskModuleConfig()
#     data: MultiTaskDataModuleConfig = MultiTaskDataModuleConfig()        
#     trainer: TrainerConfig = TrainerConfig()
#     pretrain: LRTunerConfig = LRTunerConfig()
        
#     task_ids: Tuple[str] = field(default_factory=lambda: ("task_0"))



# @dataclass
# class MultiTaskExperimentConfig(BaseExperimentConfig):

#     resolution: int = 512
#     channels: int = 3
#     root_dir: str = "."
#     seed: int = 1234

#     model: LitMultiTaskModuleConfig = LitMultiTaskModuleConfig()
#     data: MultiTaskDataModuleConfig = MultiTaskDataModuleConfig()        
#     trainer: TrainerConfig = TrainerConfig()
#     pretrain: LRTunerConfig = LRTunerConfig()
    
#     task_ids: Tuple[str] = field(default_factory=lambda: ("task_0", "task_1"))
#     # TODO: Add structured config for orchestrating image statistics logging pre-running anything else
    


# cs = ConfigStore.instance()
# cs.store(name="multitask_experiment_config", node=MultiTaskExperimentConfig)



