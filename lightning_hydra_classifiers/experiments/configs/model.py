"""

lightning_hydra_classifiers/experiments/configs/model.py

Collection of dataclass configs for lightning experiments.

Author: Jacob A Rose
Created: Monday Sept 14th, 2021


"""


from typing import *
from dataclasses import dataclass, field

from lightning_hydra_classifiers.experiments.configs.base import BaseConfig
from lightning_hydra_classifiers.experiments.configs.optimizer import *

__all__ = ["ClassifierConfig", "MultiTaskClassifierConfig", "BackboneConfig", "LitMultiTaskModuleConfig", "BaseLitModuleConfig",
           "LitSingleTaskModuleConfig"]

@dataclass(unsafe_hash=True)
class ClassifierConfig(BaseConfig):
    in_features: Optional[int] = None
    num_classes: Optional[int] = None

@dataclass(unsafe_hash=True)
class MultiTaskClassifierConfig(BaseConfig):
    task_0: ClassifierConfig = field(default_factory=ClassifierConfig) #(None, num_classes=91)
    task_1: ClassifierConfig = field(default_factory=ClassifierConfig) #ClassifierConfig(None, num_classes=19)


@dataclass(unsafe_hash=True)
class BackboneConfig(BaseConfig):
    backbone_name: str
    pretrained: str="imagenet"
    global_pool_type: str="avg"
    drop_rate: float=0.0
    init_freeze_up_to: Optional[str]=None
        
        
    @property
    def weights_tag(self):
        if isinstance(self.pretrained, str):
            return self.pretrained
        if self.pretrained == False:
            return "random"

        
###########################################
###########################################
        
        
        
        
        
        
        
@dataclass(unsafe_hash=True)
class BaseLitModuleConfig(BaseConfig):

    backbone: BackboneConfig = BackboneConfig(backbone_name="resnet50")
    optimizer: OptimizerConfig = AdamWOptimizerConfig()
    

        

        
@dataclass(unsafe_hash=True)
class LitSingleTaskModuleConfig(BaseLitModuleConfig):
    # TBD: Use this somewhere. Currently just exists to organize coherent task structure for config subclasses.
#     backbone: BackboneConfig
#     optimizer: OptimizerConfig = AdamWOptimizerConfig()
    
    task: ClassifierConfig = ClassifierConfig(None, num_classes=19)
        
        
@dataclass(unsafe_hash=True)
class LitMultiTaskModuleConfig(BaseLitModuleConfig):
    
#     backbone: BackboneConfig = BackboneConfig(backbone_name="resnet50")
#     optimizer: OptimizerConfig = AdamWOptimizerConfig()
        
    multitask: MultiTaskClassifierConfig = MultiTaskClassifierConfig()
        
        
import hydra
from hydra.core.config_store import ConfigStore

        
        
def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group="model",
#         package="config",
        name="LitMultiTaskModuleConfig",
        node=LitMultiTaskModuleConfig
    )
#     cs.store(
#         group="optimizer",
#         package="config",
#         name="Adam",
#         node=AdamOptimizerConfig,
#     )


register_configs() 