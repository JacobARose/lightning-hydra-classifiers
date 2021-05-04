"""
Created on: Tuesday, April 20th, 2021
Author: Jacob A Rose


"""

from torch import nn
import collections



# source: https://github.com/BloodAxe/pytorch-toolbelt/blob/master/pytorch_toolbelt/utils/torch_utils.py
def transfer_weights(model: nn.Module, model_state_dict: collections.OrderedDict):
    """
    Copy weights from state dict to model, skipping layers that are incompatible.
    This method is helpful if you are doing some model surgery and want to load
    part of the model weights into different model.
    :param model: Model to load weights into
    :param model_state_dict: Model state dict to load weights from
    :return: None
    """
    for name, value in model_state_dict.items():
        try:
            model.load_state_dict(collections.OrderedDict([(name, value)]), strict=False)
        except Exception as e:
            print(e)
