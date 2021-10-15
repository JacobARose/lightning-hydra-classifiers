"""
Created on: Tuesday, April 20th, 2021
Author: Jacob A Rose


"""

import torchmetrics as metrics
from torch import nn
import collections



from torchinfo import summary
import os
from typing import Tuple, Optional
from torch import nn

from torchinfo import summary


__all__ = ["log_model_summary"]


def log_model_summary(model: nn.Module,
                      input_size: Tuple[int],
                      full_summary: bool=True,
                      working_dir: str=".",
                      model_name: Optional[str]=None,
                      verbose: bool=1):
    """
    produce a text file with the model summary
    
    TODO: Add this to Eval Plugins
    
    log_model_summary(model=model,
                  working_dir=working_dir,
                  input_size=(1, data_config.channels, *data_config.image_size),
                  full_summary=True)

    """

    if full_summary:
        col_names=("kernel_size", "input_size","output_size", "num_params", "mult_adds")
    else:
        col_names=("input_size","output_size", "num_params")

    model_summary = summary(model.cuda(),
                            input_size=input_size,
                            row_settings=('depth', 'var_names'),
                            col_names=col_names,
                            verbose=verbose)

    if (model_name is None) and (hasattr(model, "name")):
        model_name = model.name
    if (model_name is None):
        summary_path = os.path.join(working_dir, f'model_summary.txt')
    else:
        summary_path = os.path.join(working_dir, f'{model_name}_model_summary.txt')
    
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    with open(summary_path, "w") as f:
        f.write(str(model_summary))
        
    return model_summary







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
