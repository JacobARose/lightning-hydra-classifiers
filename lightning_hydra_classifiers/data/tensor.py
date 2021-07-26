"""
contrastive_learning/data/pytorch/tensor.py


Common classes & functions for making use of torch.Tensors.

Created by: Thursday May 21st, 2021
Author: Jacob A Rose

"""

import torch





def tensor_nbytes(tensor: torch.Tensor, units="MB"):
    """
    Return the size of the tensor's memory footprint in Kilobytes, Megabytes, or Gigabytes
    """
    unit_scales = {"KB":1e3, "MB":1e6, "GB":1e9}
    return int(tensor.nelement() * tensor.element_size / unit_scales[units])