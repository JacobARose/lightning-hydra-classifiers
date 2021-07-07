from ..base import Registry


Heads = Registry()


# from . import linear_mlp
from . import classifier
from . import domain_discriminator

from .classifier import Classifier
