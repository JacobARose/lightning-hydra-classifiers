"""
lightning_hydra_classifiers/utils/ckpt_utils.py



Created on: Monday, September 13th, 2021
Author: Jacob A Rose


"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import numbers
from typing import Union, List, Any, Tuple, Dict, Optional, Sequence
import collections
from sklearn.model_selection import train_test_split
import json
import torchdata

log = template_utils.get_logger(__name__)


__all__ = []

from collections import namedtuple



_STATE_DICT_KEY = "state_dict"


def _is_lightning_checkpoint(checkpoint: Dict[str, Any]) -> bool:
    """Returns true if we believe this checkpoint to be a Lightning checkpoint."""
    return _STATE_DICT_KEY in checkpoint


# def _is_d2go_checkpoint(checkpoint: Dict[str, Any]) -> bool:
#     """Returns true if we believe this to be a D2Go checkpoint."""
#     d2_go_keys = [_OLD_STATE_DICT_KEY, "iteration"]
#     for key in d2_go_keys:
#         if key not in checkpoint:
#             return False
#     return True



def set_requires_grad(model, reg_exps, value):
    """
    source: https://github.com/facebookresearch/d2go/blob/main/d2go/modeling/model_freezing_utils.py
    """
    total_num_parameters = 0
    unmatched_parameters = []
    unmatched_parameter_names = []
    matched_parameters = []
    matched_parameter_names = []
    for name, parameter in model.named_parameters():
        total_num_parameters += 1
        matched = False
        for frozen_layers_regex in reg_exps:
            if re.match(frozen_layers_regex, name):
                matched = True
                parameter.requires_grad = value
                matched_parameter_names.append(name)
                matched_parameters.append(parameter)
                break
        if not matched:
            unmatched_parameter_names.append(name)
            unmatched_parameters.append(parameter)
    logger.info(
        "Matched layers (require_grad={}): {}".format(value, matched_parameter_names)
    )
    logger.info("Unmatched layers: {}".format(unmatched_parameter_names))
    return matched_parameter_names, unmatched_parameter_names