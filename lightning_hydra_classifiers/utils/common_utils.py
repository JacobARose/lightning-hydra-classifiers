"""
lightning_hydra_classifiers/utils/common_utils.py



Created on: Wednesday, July 14th, 2021
Author: Jacob A Rose


"""

import os
from pathlib import Path
import numpy as np
import numbers
from lightning_hydra_classifiers.utils import template_utils
from typing import Union, List, Any, Tuple, Dict, Optional
import collections
from sklearn.model_selection import train_test_split
import json
log = template_utils.get_logger(__name__)


__all__ = ["LabelEncoder", "trainval_split", "trainvaltest_split"]



class LabelEncoder(object):
    
    """Label encoder for tag labels."""
    def __init__(self,
                 class2idx: Dict[str,int]=None,
                 replace: Optional[Dict[str,str]]=None):
        self.class2idx = class2idx or {}
        self.replace = replace or {}
        self.index2class = {v: k for k, v in self.class2idx.items()}
        self.classes = [k for k in self.class2idx.keys() if k not in self.replace.keys()]
        
        self.replace_class2idx_items()
        self.num_samples = 0

        
    def replace_class2idx_items(self):
        if (len(self.replace) == 0) \
        or (len([k for k in self.replace.keys() if k in self.class2idx.keys()]) == 0):
            return
        
        log.info(f'LabelEncoder replacing {len(self.replace.keys())} class encodings with that other an another class')
        log.info('Replacing: ' + str({k:v for k,v in self.replace.items() if k in self.class2idx}))
        for old, new in self.replace.items():
            if old in list(self.class2idx.keys()):
                self.class2idx[old] = self.class2idx[new]
        self.index2class = {v: k for k, v in self.class2idx.items()}
        self.classes = [k for k in self.class2idx.keys() if k not in self.replace.keys()]                
        
    def __len__(self):
        return len(self.classes)

    def __str__(self):
        msg = f"<LabelEncoder(num_classes={len(self)})>"
        if len(self.replace) > 0:
            msg += "\n" + f"<num_replaced_classes={len(self.replace)}"
        return msg

    def fit(self, y):
        
        counts = collections.Counter(y)
        self.num_samples += sum(counts.values())
        
        classes = list(counts.keys())

        old_num_classes = len(self)
        new_classes = [label for label in classes if label not in self.classes]
        
        for i, label in enumerate(new_classes):
            self.class2idx[label] = old_num_classes + i
        self.index2class = {v: k for k, v in self.class2idx.items()}
        
        self.classes = [k for k in self.class2idx.keys() if k not in self.replace.keys()]        
        self.replace_class2idx_items()

        new_classes = [c for c in new_classes if c not in self.replace.keys()]
        if len(new_classes):
            log.debug(f"[FITTING] {len(y)} samples with {len(classes)} classes, adding {len(new_classes)} new class labels. Latest num_classes = {len(self)}")
        assert len(self) == (old_num_classes + len(new_classes))
        return self

    def encode(self, y):
        if not hasattr(y,"__len__"):
            y = [y]
        print(self.class2idx)
        return np.array([self.class2idx[label] for label in y])

    def decode(self, y):
        if not hasattr(y,"__len__"):
            y = [y]
        return np.array([self.index2class[label] for label in y])

    def save(self, fp):
        with open(fp, "w") as fp:
            contents = self.getstate() # {"class2idx": self.class2idx}
            json.dump(contents, fp, indent=4, sort_keys=False)

    @classmethod
    def load(cls, fp):
        with open(fp, "r") as fp:
            kwargs = json.load(fp=fp)
        return cls(**kwargs)
    
    def getstate(self):
        return {"class2idx": self.class2idx,
                "replace": self.replace}
    
    def __repr__(self):
        disp = f"""<{str(type(self)).strip("'>").split('.')[-1]}>:\n"""
        disp += f"    num_classes: {len(self)}\n"
        disp += f"    fit on num_samples: {self.num_samples}"
        return disp
        
#     def encode(self, y):
#         y_one_hot = np.zeros((len(y), len(self.class2idx)), dtype=int)
#         for i, item in enumerate(y):
#             for class_ in item:
#                 y_one_hot[i][self.class2idx[class_]] = 1
#         return y_one_hot

#     def decode(self, y):
#         classes = []
#         for i, item in enumerate(y):
#             indices = np.where(item == 1)[0]
#             classes.append([self.index2class[index] for index in indices])
#         return classes

####################################################


def trainval_split(x: Union[List[Any],np.ndarray]=None,
                   y: Union[List[Any],np.ndarray]=None,
                   val_train_split: float=0.2,
                   random_state: int=None,
                   stratify: bool=True
                   ) -> Dict[str,Tuple[np.ndarray]]:
    """
    Wrapper function to split data into 3 stratified subsets specified by `splits`.
    
    User specifies absolute fraction of total requested for each subset (e.g. splits=[0.5, 0.2, 0.3])
    
    Function calculates adjusted fractions necessary in order to use sklearn's builtin train_test_split function over a sequence of 2 steps.
    
    Step 1: Separate test set from the rest of the data (constituting the union of train + val)
    
    Step 2: Separate the train and val sets from the remainder produced by step 1.

    Output:
        Dict: {'train':(x_train, y_train),
                'val':(x_val_y_val),
                'test':(x_test, y_test)}
                
    Example:
        >> data = torch.data.Dataset(...)
        >> y = data.targets
        >> data_splits = trainvaltest_split(x=None,
                                            y=y,
                                            splits=(0.5, 0.2, 0.3),
                                            random_state=0,
                                            stratify=True)
    
    """
    
    train_split = 1.0 - val_train_split
    
    if stratify and (y is None):
        raise ValueError("If y is not provided, stratify must be set to False.")
    
    y = np.array(y)
    if x is None:
        x = np.arange(len(y))
    else:
        x = np.array(x)
    
    stratify_y = y if stratify else None    
    x_train, x_val, y_train, y_val = train_test_split(x, y,
                                                      test_size=val_train_split, 
                                                      random_state=random_state,
                                                      stratify=stratify_y)

    x = np.concatenate((x_train, x_val)).tolist()
    assert len(set(x)) == len(x), f"[Warning] Check for possible data leakage. len(set(x))={len(set(x))} != len(x)={len(x)}"
    
    log.debug(f"x_train.shape={x_train.shape}, y_train.shape={y_train.shape}")
    log.debug(f"x_val.shape={x_val.shape}, y_val.shape={y_val.shape}")
    log.debug(f'Absolute splits: {[train_split, val_train_split]}')
    
    return {"train":(x_train, y_train),
            "val":(x_val, y_val)}








####################################################

def trainvaltest_split(x: Union[List[Any],np.ndarray]=None,
                       y: Union[List[Any],np.ndarray]=None,
                       splits: List[float]=(0.5, 0.2, 0.3),
                       random_state: int=None,
                       stratify: bool=True
                       ) -> Dict[str,Tuple[np.ndarray]]:
    """
    Wrapper function to split data into 3 stratified subsets specified by `splits`.
    
    User specifies absolute fraction of total requested for each subset (e.g. splits=[0.5, 0.2, 0.3])
    
    Function calculates adjusted fractions necessary in order to use sklearn's builtin train_test_split function over a sequence of 2 steps.
    
    Step 1: Separate test set from the rest of the data (constituting the union of train + val)
    
    Step 2: Separate the train and val sets from the remainder produced by step 1.

    Output:
        Dict: {'train':(x_train, y_train),
                'val':(x_val_y_val),
                'test':(x_test, y_test)}
                
    Example:
        >> data = torch.data.Dataset(...)
        >> y = data.targets
        >> data_splits = trainvaltest_split(x=None,
                                            y=y,
                                            splits=(0.5, 0.2, 0.3),
                                            random_state=0,
                                            stratify=True)
    
    """
    
    
    assert len(splits) == 3, "Must provide eactly 3 float values for `splits`"
    assert np.isclose(np.sum(splits), 1.0), f"Sum of all splits values {splits} = {np.sum(splits)} must be 1.0"
    
    train_split, val_split, test_split = splits
    val_relative_split = val_split/(train_split + val_split)
    train_relative_split = train_split/(train_split + val_split)
    
    
    if stratify and (y is None):
        raise ValueError("If y is not provided, stratify must be set to False.")
    
    y = np.array(y)
    if x is None:
        x = np.arange(len(y))
    else:
        x = np.array(x)
    
    stratify_y = y if stratify else None    
    x_train_val, x_test, y_train_val, y_test = train_test_split(x, y,
                                                        test_size=test_split, 
                                                        random_state=random_state,
                                                        stratify=y)
    
    stratify_y_train = y_train_val if stratify else None
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val,
                                                      test_size=val_relative_split,
                                                      random_state=random_state, 
                                                      stratify=y_train_val)
    
    x = np.concatenate((x_train, x_val, x_test)).tolist()
    assert len(set(x)) == len(x), f"[Warning] Check for possible data leakage. len(set(x))={len(set(x))} != len(x)={len(x)}"
    
    log.debug(f"x_train.shape={x_train.shape}, y_train.shape={y_train.shape}")
    log.debug(f"x_val.shape={x_val.shape}, y_val.shape={y_val.shape}")
    log.debug(f"x_test.shape={x_test.shape}, y_test.shape={y_test.shape}")
    log.debug(f'Absolute splits: {[train_split, val_split, test_split]}')
    log.debug(f'Relative splits: [{train_relative_split:.2f}, {val_relative_split:.2f}, {test_split}]')
    
    return {"train":(x_train, y_train),
            "val":(x_val, y_val),
            "test":(x_test, y_test)}




#############################################################
#############################################################


# import os
# import importlib
# from collections import defaultdict


# from 

# class RegistryError(ValueError):
#     pass




# class Registry:
#     _available = defaultdict(dict)
#     _defaults = dict()
#     _option_functions = []

#     @staticmethod
#     def add(namespace, name, cls):
#         Registry._available[namespace][name] = cls

#     @staticmethod
#     def keys(namespace):
#         return list(Registry._available[namespace].keys())

#     @staticmethod
#     def get(namespace, name):
#         return Registry._available[namespace].get(name)

#     @staticmethod
#     def set_default(namespace, name):
#         if namespace in Registry._defaults:
#             raise RegistryError(f'namespace {namespace} already has a default: {Registry._defaults[namespace]}')
#         Registry._defaults[namespace] = name

#     @staticmethod
#     def default(namespace):
#         return Registry._defaults.get(namespace)

#     @staticmethod
#     def add_option_function(f):
#         Registry._option_functions.append(f)

#     @staticmethod
#     def option_functions():
#         return list(Registry._option_functions)

# def register(namespace, name, default=False):
#     def inner(cls):
#         Registry.add(namespace, name, cls)
#         if default:
#             Registry.set_default(namespace, name)
#         return cls

#     return inner

# def with_option_parser(f):
#     Registry.add_option_function(f)


############################################################
############################################################

def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    
    Source: scikit-learn
    
    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


