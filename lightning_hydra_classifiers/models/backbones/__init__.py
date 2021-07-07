"""
lightning_hydra_classifiers/models/backbones/__init__.py

Author: Jacob A Rose
Created: Tuesday June 22nd, 2021

"""

# backbones = Registry()


from . import resnet
from . import senet
from . import efficientnet

from .backbone import build_model