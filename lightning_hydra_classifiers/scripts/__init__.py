# from . import callbacks, datamodules, models, utils, experiments


# from . import data
from . import (train,
			   train_basic,
			   finetune,
			   train_multi_gpu,
			   train_BYOL)

from . import multitask
from .multitask import train as train_multitask