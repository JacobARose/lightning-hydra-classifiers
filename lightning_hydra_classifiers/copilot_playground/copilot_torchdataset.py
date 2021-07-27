def __init__(self,
                 config: DictConfig,
                 files: Union[Dict[str, List[Path]], List[Path]]=None):
        """
        Parameters
        ----------
        config : DictConfig
            OmegaConf DictConfig object containing the following keys:
                'data_dir' : str
                    Path to the directory containing the dataset
                'train_dir' : str
                    Path to the directory containing the training data
                'val_dir' : str
                    Path to the directory containing the validation data
                'test_dir' : str
                    Path to the directory containing the test data
                'train_transforms' : List[Callable]
                    List of transforms to apply to each training sample
                'val_transforms' : List[Callable]
                    List of transforms to apply to each validation sample
                'test_transforms' : List[Callable]
                    List of transforms to apply to each test sample
                'target_transforms' : List[Callable]
                    List of transforms to apply to each target sample
                'batch_size' : int
                    Batch size for the training module
                'num_workers' : int
                    Number of workers for the training module
                'pin_memory' : bool
                    Flag indicating whether to pin memory for the training module
        files : Union[Dict[str, List[Path]], List[Path]]
            Dictionary of file paths to the training, validation, and test data, or list of paths to all data
        """
        self.config = config
        self.data_dir = Path(config.data_dir)
        self.train_dir = self.data_dir / config.train_dir
        self.val_dir = self.data_dir / config.val_dir
        self.test_dir = self.data_dir / config.test_dir
        self.train_transforms = config.train_transforms
        self.val_transforms = config.val_transforms
        self.test_transforms = config.test_transforms
        self.target_transforms = config.target_transforms
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.pin_memory = config.pin_memory
        self.train_dataset = self.get




==============


        """
        Args:
            config (DictConfig): OmegaConf DictConfig object containing the following keys:
                - dataset_dir (Union[str, Path]): Path to the directory containing the dataset.
                - dataset_type (str): Name of the dataset type (e.g. "kaggle").
                - dataset_name (str): Name of the dataset (e.g. "cifar10").
                - dataset_config (DictConfig): OmegaConf DictConfig object containing the configuration
                    for the dataset type.
                - dataset_transforms (DictConfig): OmegaConf DictConfig object containing the
                    transformations to apply to the dataset.
                - dataset_split_method (str): Name of the dataset split method (e.g. "split_none").
                - dataset_split_config (DictConfig): OmegaConf DictConfig object containing the
                    configuration for the dataset split method.
                - dataset_split_options (DictConfig): OmegaConf DictConfig object containing the
                    options to pass to the dataset split method.
                - dataset_loader_options (DictConfig): OmegaConf DictConfig object containing the
                    options to pass to the dataset loader.
                - dataset_train_subdir (str): Name of the subdirectory containing the training data.
                - dataset_valid_subdir (str): Name of the subdirectory containing the validation data.
                - dataset_test_subdir (str): Name of the subdirectory containing the test data.
                - dataset_train_files (List[str]): List of file names to include in the training set.
                - dataset_valid_files (List[str]): List of file names to include in the validation set.
                - dataset_test_files (List[str]): List of file names to include in the test set.
            files (Union[Dict[str, List[Path]], List[Path]]): Files to load.
        """
        self.config = config
        self.dataset_dir = Path(self.config.dataset_dir)
        self.dataset_type = self.config.dataset_type
        self.dataset_name =






from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import pytorch_lightning as pl
from torchvision import transforms
import torchdata
from typing import *

# class CommonDataset(torchdata.datasets.Files): #(CommonDataArithmetic, CommonDataSelect, CommonDataSplitter):
#     """
#     Custom subclass for extracting, loading, and transforming labeled image data for training modules with PyTorch Lightning
    
#     """
#     transform = None
#     target_transform = None

#     def __init__(self,
#                     config: DictConfig,
#                     files: Union[Dict[str, List[Path]], List[Path]]=None):

#             """
#             Initialize a CommonDataset instance

#             Parameters
#             ----------
#             config : DictConfig
#                 OmegaConf object containing configuration for the dataset
#             files : Union[Dict[str, List[Path]], List[Path]]
#                 A dict of files or a list of files to pull data from.
#             """
#             super().__init__(files)
#             self.config = config
#             self.transform = self.config.transform
#             self.target_transform = self.config.target_transform
#             self.root = self.config.root
#             self.train = self.config.train
#             self.download = self.config.download
#             self.transform = transforms.Compose(self.config.transform)
#             self.target_transform = transforms.Compose(self.config.target_transform)
#             self.files = self.config.files