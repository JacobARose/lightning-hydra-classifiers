
"""

lightning_hydra_classifiers/experiments/configs/optimizer.py

Collection of dataclass configs for lightning experiments.

Author: Jacob A Rose
Created: Monday Sept 14th, 2021


"""


from typing import *
from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore



__all__ = ["AdamOptimizerConfig", "AdamWOptimizerConfig", "OptimizerConfig", "register_configs"]

@dataclass(unsafe_hash=True)
class OptimizerConfig:
    lr: float = 0.001
    betas: Tuple[float] = (0.9, 0.999)
    eps: float = 1e-08
    weight_decay: float = 0.01
    amsgrad: bool = False

@dataclass
class AdamWOptimizerConfig(OptimizerConfig):
    _target_: str = "torch.optim.AdamW"
    weight_decay: float = 0.01

@dataclass
class AdamOptimizerConfig(OptimizerConfig):
    _target_: str = "torch.optim.Adam"
    weight_decay: float = 0.0
        
        
        
def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group="optimizer",
#         package="config",
        name="AdamW",
        node=AdamWOptimizerConfig,
    )
    cs.store(
        group="optimizer",
#         package="config",
        name="Adam",
        node=AdamOptimizerConfig,
    )

    
register_configs()