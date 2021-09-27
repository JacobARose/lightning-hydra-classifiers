
"""

lightning_hydra_classifiers/experiments/configs/misc.py


The config for managing system-wide parameters for experiment orchestration that ultimately work best when classified under `config.misc`.

Author: Jacob A Rose
Created: Tuesday Sept 14th, 2021


"""


from typing import *
from dataclasses import dataclass, field

from lightning_hydra_classifiers.experiments.configs.model import *
from lightning_hydra_classifiers.experiments.configs.trainer import *
from lightning_hydra_classifiers.experiments.configs.data import *
from lightning_hydra_classifiers.experiments.configs.pretrain.lr_tuner import *

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


# __all__ =  ["MultiTaskExperimentConfig", "BaseExperimentConfig"]

@dataclass
class MiscConfig:

    resolution: int
    channels: int
    root_dir: str
    seed: int

#     model: BaseLitModuleConfig = LitMultiTaskModuleConfig()
#     data: MultiTaskDataModuleConfig = MultiTaskDataModuleConfig()        
#     trainer: TrainerConfig = TrainerConfig()

        
#     def __post_init__(self):
        
