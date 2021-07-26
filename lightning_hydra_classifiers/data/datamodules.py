"""
Convenience functions for flexibly interfacing with any PyTorch or PyTorch Lightning datamodules defined in contrastive_learning.

Created by: Sunday May 16th, 2021
Author: Jacob A Rose


"""

from box import Box
import wandb
from typing import Tuple, Any
from omegaconf import DictConfig

from contrastive_learning.data.pytorch.pnas import PNASLightningDataModule
from contrastive_learning.data.pytorch.extant import ExtantLightningDataModule
from contrastive_learning.data.pytorch.fossil import FossilLightningDataModule
from contrastive_learning.data.pytorch.common import LeavesLightningDataModule
from contrastive_learning.data.pytorch.utils.file_utils import ensure_dir_exists
__all__ = ["get_datamodule", "fetch_datamodule_from_dataset_artifact"]




def build_datamodule(**kwargs):
    if 'Extant' in kwargs.get('name',''):
        datamodule = ExtantLightningDataModule(**kwargs)
    elif 'PNAS' in kwargs.get('name',''):
        datamodule = PNASLightningDataModule(**kwargs)
    elif 'Fossil' in kwargs.get('name',''):
        datamodule = FossilLightningDataModule(**kwargs)
    return datamodule

########################################
def get_datamodule(data_config: DictConfig):
    """
    Return an instance of a LeavesLightningDataModule by passing a config object containing all required arguments.
    
    datamodule = get_datamodule(config.dataset)

    Config Args:
    
        name: str=default_name,
        batch_size: int=32
        val_split: float=0.2
        num_workers=0
        seed: int=None
        debug: bool=False
        normalize: bool=True
        image_size: Union[int,str] = None
        channels: int=None
        dataset_dir: Optional[str]=None
    
    """
#     ensure_dir_exists(data_config.dataset_dir)
    
#     if 'Extant' in data_config.name:
#         datamodule = ExtantLightningDataModule(**vars(data_config))
#     elif 'PNAS' in data_config.name:
#         datamodule = PNASLightningDataModule(**vars(data_config))
    
    datamodule = build_datamodule(**vars(data_config))

    try:
        datamodule.setup(stage="fit")
        data_config.classes = datamodule.classes
        data_config.num_classes = len(data_config.classes)
    except:
        pass
        
    return datamodule, data_config




def fetch_datamodule_from_dataset_artifact(dataset_config: Box, 
                                           artifact_config: Box,
                                           run_or_api=None) -> Tuple[LeavesLightningDataModule, wandb.Artifact]:
    """
    Download a versioned dataset artifact from wandb and output a configured datamodule.
    
    # TODO (Jacob): Write up unit tests
    """
    run = run_or_api or wandb.Api()
    artifact = run.use_artifact(artifact_config.uri,
                                type=artifact_config.type) #'jrose/image_classification_datasets/PNAS_family_100_1024:v0', type='raw_data')
    ensure_dir_exists(dataset_config.dataset_dir)
    dataset_artifact_dir = artifact.download(root=dataset_config.dataset_dir)


    datamodule = get_datamodule(dataset_config)
    datamodule.setup('fit')
    datamodule.setup('test')
    ########################
#     config.model.num_classes = dataset_config.num_classes
    dataset_config.classes = datamodule.classes
    dataset_config.num_classes = len(dataset_config.classes)
    artifact_config.num_classes = dataset_config.num_classes

    return datamodule, artifact