# from . import common#, fossil, pnas, extant
from .utils import catalog_registry
from .datasets import common

# from lightning_hydra_classifiers.data.common import CommonDataset
# CommonDataset.available_datasets.update(fossil.available_datasets)
# CommonDataset.available_datasets.update(extant.available_datasets)
# CommonDataset.available_datasets.update(pnas.available_datasets)

from . import datasets
from . import datamodules
from . import utils
from . import wandb