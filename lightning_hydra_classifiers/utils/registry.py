"""
lightning_hydra_classifiers/utils/registry.py



Created on: Wednesday, July 15th, 2021
Author: Jacob A Rose


"""

import os
from pathlib import Path
import numpy as np
import numbers
from lightning_hydra_classifiers.utils import template_utils
from typing import Union, List, Any, Tuple, Dict

log = template_utils.get_logger(__name__)


__all__ = ["Registry", "register"]



#############################################################
#############################################################


import importlib
from collections import defaultdict



class RegistryError(ValueError):
    pass




class Registry:
    _available = defaultdict(dict)
    _defaults = dict()
    _option_functions = []

    @staticmethod
    def add(namespace, name, cls):
        Registry._available[namespace][name] = cls

    @staticmethod
    def keys(namespace):
        return list(Registry._available[namespace].keys())

    @staticmethod
    def get(namespace, name):
        return Registry._available[namespace].get(name)

    @staticmethod
    def set_default(namespace, name):
        if namespace in Registry._defaults:
            raise RegistryError(f'namespace {namespace} already has a default: {Registry._defaults[namespace]}')
        Registry._defaults[namespace] = name

    @staticmethod
    def default(namespace):
        return Registry._defaults.get(namespace)

    @staticmethod
    def add_option_function(f):
        Registry._option_functions.append(f)

    @staticmethod
    def option_functions():
        return list(Registry._option_functions)

def register(namespace, name, default=False):
    def inner(cls):
        Registry.add(namespace, name, cls)
        if default:
            Registry.set_default(namespace, name)
        return cls

    return inner

def with_option_parser(f):
    Registry.add_option_function(f)
























############################################################
############################################################