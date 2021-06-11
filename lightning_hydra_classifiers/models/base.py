"""
lightning_hydra_classifiers/models/base.py

Author: Jacob A Rose
Created: Saturday May 29th, 2021

"""

import torch
from torch import nn
import torchmetrics as metrics
# from torchsummary import summary
from pathlib import Path
# from typing import Any, List, Optional, Dict, Tuple


__all__ = ["BaseModel"]


class BaseModule(nn.Module):
    """
    Models should subclass this in place of nn.Module. This is a custom base class to implement standard interfaces & implementations across all custom pytorch modules in this library.
    
    """
    
    def forward(self, x):
        """
        Identity function by default. Subclasses should redefine this method.
        """
        return x
    
    
#     def save_model(self, path:str):
#         path = str(path)
#         if not Path(path).suffix=='.ckpt':
#             path = path + ".ckpt"
#         torch.save(self.state_dict(), path)
        
        
#     def load_model(self, path:str):
#         path = str(path)
#         if not Path(path).suffix=='.ckpt':
#             path = path + ".ckpt"
#         self.load_state_dict(torch.load(path))
        


    def freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False
            
    def unfreeze(self, model):
        for param in model.parameters():
            param.requires_grad = True

    def get_trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def get_frozen_parameters(self):
        return (p for p in self.parameters() if not p.requires_grad)

    
    def initialize_weights(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)




                
                
                
    def pack_checkpoint(
                        self,
                        model=None,
                        criterion=None,
                        optimizer=None,
                        scheduler=None,
                        **kwargs
                        ):
        content = {}
        if model is not None:
            content["model_state_dict"] = model.state_dict()
        if criterion is not None:
            content["criterion_state_dict"] = criterion.state_dict()
        if optimizer is not None:
            content["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            content["scheduler_state_dict"] = scheduler.state_dict()
        return content

    def unpack_checkpoint(
                          self,
                          checkpoint,
                          model=None,
                          criterion=None,
                          optimizer=None,
                          scheduler=None,
                          **kwargs,
                          ):
        state_dicts = ("model", "criterion", "optimizer", "scheduler")
        parts = (model, criterion, optimizer, scheduler)
        for state_dict, part in zip(state_dicts, parts):
            if f"{state_dict}_state_dict" in checkpoint and part is not None:
                part.load_state_dict(checkpoint[f"{state_dict}_state_dict"])

    
    def save_checkpoint(self, checkpoint, path):
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        return checkpoint
    
    
    
    
    
    
    
# Inherits from dict
class Registry(dict):
    '''
    A helper class for managing registering modules, it extends a dictionary
    and provides register functions.
    Access of module is just like using a dictionary, eg:
        f = some_registry["foo_module"]
        
        
    [code source] https://julienbeaulieu.github.io/2020/03/16/building-a-flexible-configuration-system-for-deep-learning-models/
    '''
    
    # Instanciated objects will be empyty dictionaries
    def __init__(self, *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)

    # Decorator factory. Here self is a Registry dict
    def register(self, module_name, module=None):
        
        # Inner function used as function call
        if module is not None:
            _register_generic(self, module_name, module)
            return

        # Inner function used as decorator -> takes a function as argument
        def register_fn(fn):
            _register_generic(self, module_name, fn)
            return fn

        return register_fn # decorator factory returns a decorator function
    

def _register_generic(module_dict, module_name, module):
    assert module_name not in module_dict
    module_dict[module_name] = module