
"""

lightning_hydra_classifiers/experiments/configs/data.py

Collection of dataclass configs for lightning experiments.

Author: Jacob A Rose
Created: Monday Sept 13th, 2021


"""


from typing import *
from dataclasses import dataclass, field

from .base import BaseConfig

__all__ = ["MultiTaskDataModuleConfig"]



@dataclass
class DataModuleConfig(BaseConfig):
    _target_: str # ="lightning_hydra_classifiers.experiments.multitask.datamodules.MultiTaskDataModule"
    image_size: int = 512
    image_buffer_size: int = 32
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
        
        
    dataset_name: str = ""


@dataclass
class SingleTaskDataModuleConfig(DataModuleConfig):
    _target_: str = "lightning_hydra_classifiers.experiments.multitask.datamodules.SingleTaskDataModule"
    dataset_name: str = "Extant_Leaves"
#     image_size: int = 512
#     image_buffer_size: int = 32
#     batch_size: int = 32
#     num_workers: int = 4
#     pin_memory: bool = True
        
        
        
        
@dataclass
class MultiTaskDataModuleConfig(DataModuleConfig):
    _target_: str = "lightning_hydra_classifiers.experiments.multitask.datamodules.MultiTaskDataModule"
        
    dataset_name: str = "Extant-PNAS_to_PNAS"
    transfer_experiment: Optional["TransferExperimentConfig"] = None #field(init=False)
        
        
    def __post_init__(self):
        self.transfer_experiment = TransferExperimentConfig(experiment_root_dir = EXPERIMENTAL_DATASETS_DIR,
                                                            experiment_name = "Extant-to-PNAS-512-transfer_benchmark")
        
from lightning_hydra_classifiers.data.utils.make_catalogs import CSV_CATALOG_DIR_V1_0, EXPERIMENTAL_DATASETS_DIR
        
@dataclass
class TransferExperimentConfig:
    source_root_dir: str = CSV_CATALOG_DIR_V1_0
    # TBD: Auto format the date for `experiment_root_dir`
    experiment_root_dir: str = "/media/data/jacob/GitHub/lightning-hydra-classifiers/notebooks/experiments_September_2021"
    experiment_name: str = "Extant-to-PNAS-512-transfer_benchmark"
                                                       
                                                       
                                                       
                                                       

#     image_size: int = 512
#     image_buffer_size: int = 32
#     batch_size: int = 32
#     num_workers: int = 4
#     pin_memory: bool = True