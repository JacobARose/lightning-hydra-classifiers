
"""

lightning_hydra_classifiers/experiments/configs/data.py

Collection of dataclass configs for lightning experiments.

Author: Jacob A Rose
Created: Monday Sept 13th, 2021


"""


from typing import *
from dataclasses import dataclass, field

from .base import BaseConfig

from hydra.core.config_store import ConfigStore

__all__ = ["MultiTaskDataModuleConfig", "TransferExperimentConfig", "Extant_to_PNAS_ExperimentConfig", "Extant_to_Fossil_ExperimentConfig", "TaskConfig", "register_configs"]



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
    experiment: Optional["TransferExperimentConfig"] = None #field(init=False)
        
        
    def __post_init__(self):
        
        if self.dataset_name == "Extant-PNAS_to_PNAS":
            self.experiment = Extant_to_PNAS_ExperimentConfig()
        elif self.dataset_name == "Extant_to_Fossil":
            self.experiment = Extant_to_Fossil_ExperimentConfig()

        
#         self.experiment = TransferExperimentConfig(experiment_root_dir = EXPERIMENTAL_DATASETS_DIR,
#                                                             experiment_name = "Extant-to-PNAS-512-transfer_benchmark")
        
# from lightning_hydra_classifiers.data.utils.make_catalogs import CSV_CATALOG_DIR_V1_0, EXPERIMENTAL_DATASETS_DIR
        
# @dataclass
# class TransferExperimentConfig:
#     source_root_dir: str = CSV_CATALOG_DIR_V1_0
#     # TBD: Auto format the date for `experiment_root_dir`
#     experiment_root_dir: str = "/media/data/jacob/GitHub/lightning-hydra-classifiers/notebooks/experiments_September_2021"
#     experiment_name: str = "Extant-to-PNAS-512-transfer_benchmark"
################
#     image_size: int = 512
#     image_buffer_size: int = 32
#     batch_size: int = 32
#     num_workers: int = 4
#     pin_memory: bool = True

from lightning_hydra_classifiers.data.utils.make_catalogs import CSV_CATALOG_DIR_V1_0, EXPERIMENTAL_DATASETS_DIR

@dataclass
class TaskConfig:
    name: str
    val_split: Optional[float] = 0.2
    test_split: Optional[float] = None
        

@dataclass
class TransferExperimentConfig:
    source_root_dir: str = CSV_CATALOG_DIR_V1_0
    experiment_root_dir: str = EXPERIMENTAL_DATASETS_DIR # '/media/data_cifs/projects/prj_fossils/users/jacob/experiments/July2021-Nov2021/csv_datasets/experimental_datasets',
    experiment_name: Optional[str] = None
    task_0: Optional[TaskConfig] = None
    task_1: Optional[TaskConfig] = None
        
#     task_0_name: Optional[str] = None
#     task_1_name: Optional[str] = None
#     seed: int = 99
        

@dataclass
class Extant_to_PNAS_ExperimentConfig(TransferExperimentConfig):
    experiment_name: str = "Extant-to-PNAS-512-transfer_benchmark"
    task_0: Optional[TaskConfig] = TaskConfig(name = 'Extant_Leaves_family_10_512_minus_PNAS_family_100_512',
                                              val_split = 0.2,
                                              test_split = None)
    task_1: Optional[TaskConfig] = TaskConfig(name = 'PNAS_family_100_512_minus_Extant_Leaves_family_10_512',
                                              val_split = 0.2,
                                              test_split = None)        
        
#     task_0_name: str = 'Extant_Leaves_family_10_512_minus_PNAS_family_100_512'
#     task_1_name: str = 'PNAS_family_100_512_minus_Extant_Leaves_family_10_512'

        
@dataclass
class Extant_to_Fossil_ExperimentConfig(TransferExperimentConfig):
    experiment_name: str = "Extant-to-Fossil-512-transfer_benchmark"
        
    task_0: Optional[TaskConfig] = TaskConfig(name = "Extant_Leaves_family_3_512",
                                              val_split = 0.2,
                                              test_split = 0.3)
    task_1: Optional[TaskConfig] = TaskConfig(name = "Fossil_family_3_512",
                                              val_split = 0.2,
                                              test_split = 0.3)

        
#     task_0_name: str = "Extant_Leaves_family_3_512"
#     task_1_name: str = "Fossil_family_3_512"


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group="data",
        package="config",
        name="Extant_to_PNAS",
        node=Extant_to_PNAS_ExperimentConfig)
    
    cs.store(
        group="data",
        package="config",
        name="Extant_to_Fossil",
        node=Extant_to_Fossil_ExperimentConfig)
    
    
# register_configs()