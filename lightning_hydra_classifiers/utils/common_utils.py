"""
lightning_hydra_classifiers/utils/common_utils.py



Created on: Wednesday, July 14th, 2021
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

from lightning_hydra_classifiers.utils import template_utils
from lightning_hydra_classifiers.utils.plot_utils import colorbar


log = template_utils.get_logger(__name__)


__all__ = ["LabelEncoder", "trainval_split", "trainvaltest_split", "DataSplitter",
           "plot_split_distributions", "plot_class_distributions", "filter_df_by_threshold",
           "compute_class_counts"]


class LabelEncoder(object):
    """
    Label encoder for tag labels.
    
    len(idx2class) <= len(class2idx)
    num_classes == len(idx2class) <= len(class2idx)
    """
    def __init__(self,
                 class2idx: Dict[str,int]=None,
                 replacements: Optional[Dict[str,str]]=None):
        self.class2idx = class2idx or {}
        self.replacements = replacements or {}
        assert len(self.classes) == len(self.idx2class) <= len(self.class2idx)
        self.num_samples = 0
        self.verbose=False
#         self.replace_class2idx_items()
        

    @property
    def idx2class(self):
        return {v: k for k, v in self.class2idx.items() if k not in self.replacements.keys()}
    
    @property
    def classes(self):
        return [k for k in self.class2idx.keys() if k not in self.replacements.keys()]

    def __len__(self):
        return len(self.idx2class)

    @property
    def num_classes(self):
        return len(self)

    def __str__(self):
        msg = f"<LabelEncoder(num_classes={len(self)})>"
        if len(self.replacements) > 0:
            msg += "\n" + f"<num_replaced_classes={len(self.replacements)}>"
        return msg

    def fit(self, y):
        
        counts = collections.Counter(y)
        self.num_samples += sum(counts.values())
        
        classes = sorted(list(counts.keys()))
        new_classes = sorted([label for label in classes if (label not in self.classes) and (label not in self.replacements.keys())])
        replace_classes = sorted([label for label in classes if label in self.replacements.keys()])
        
        old_num_classes = len(self)        
        old_highest_class = None
        idx = 0
        if len(self.idx2class) > 0:
            old_highest_class = max(self.idx2class.keys())
            idx = old_highest_class + 1
            
        for label in new_classes:
            self.class2idx[label] = idx
            idx += 1
        for label in replace_classes:
            if self.replacements[label] in self.class2idx:
                self.class2idx[label] = self.class2idx[self.replacements[label]]
            else:
                print(f"[Warning]: label {label} marked for replacement, but its replacement label has yet to be assigned an int encoding in class2idx.")
            
        new_classes = [c for c in new_classes if c not in self.replacements.keys()]
        if len(new_classes):
            log.debug(f"[FITTING] {len(y)} samples with {len(classes)} classes, adding {len(new_classes)} new class labels. Latest num_classes = {len(self)}")
        assert len(self) == (old_num_classes + len(new_classes)), f"len(self)={len(self)}, (old_num_classes={old_num_classes}, len(new_classes)={len(new_classes)})"
        assert np.all([label in self.idx2class.values() for label in new_classes])
        return self

    def encode(self, y):
        if not hasattr(y,"__len__"):
            y = [y]
        return np.array([self.class2idx[label] for label in y])

    def decode(self, y):
        if not hasattr(y,"__len__"):
            y = [y]
        return np.array([self.idx2class[label] for label in y])

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
                "replacements": self.replacements}
    
    def __repr__(self):
        disp = f"""<{str(type(self)).strip("'>").split('.')[-1]}>:\n"""
        disp += f"    num_classes: {len(self)}\n"
        disp += f"    fit on num_samples: {self.num_samples}"
        return disp


#################################
#################################
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


class DataSplitter:

    @classmethod
    def create_trainvaltest_splits(cls,
                                   data: torchdata.Dataset,
                                   val_split: float=0.2,
                                   test_split: Optional[Union[str, float]]=None, #0.3,
                                   shuffle: bool=True,
                                   seed: int=3654,
                                   stratify: bool=True,
                                   plot_distributions: bool=False) -> Tuple["FossilDataset"]:
        
        if (test_split == "test") or (test_split is None):
            train_split = 1 - val_split
            if hasattr(data, f"test_dataset"):
                data = getattr(data, f"train_dataset")            
        elif isinstance(test_split, float):
            train_split = 1 - (test_split + val_split)
        else:
            raise ValueError(f"Invalid split arguments: val_train_split={val_train_split}, test_split={test_split}")


        splits=(train_split, val_split, test_split)
        splits = list(filter(lambda x: isinstance(x, float), splits))
        y = data.targets

        if len(splits)==2:
            data_splits = trainval_split(x=None,
                                         y=y,
                                         val_train_split=splits[-1],
                                         random_state=seed,
                                         stratify=stratify)

        else:
            data_splits = trainvaltest_split(x=None,
                                             y=y,
                                             splits=splits,
                                             random_state=seed,
                                             stratify=stratify)

        dataset_splits={}
        for split, (split_idx, split_y) in data_splits.items():
            print(split, len(split_idx))
            dataset_splits[split] = data.filter(indices=split_idx, subset_key=split)
        
        
        label_encoder = LabelEncoder()
        label_encoder.fit(dataset_splits["train"].targets)
        
        for d in [*list(dataset_splits.values()), data]:
            d.label_encoder = label_encoder
        return dataset_splits


#############################################################
#############################################################


def plot_class_distributions(targets: List[Any], 
                             sort_by: Optional[Union[str, bool, Sequence]]="count",
                             ax=None,
                             xticklabels: bool=True):
    """
    Example:
        counts = plot_class_distributions(targets=data.targets, sort=True)
    """
    
    counts = compute_class_counts(targets,
                                  sort_by=sort_by)
                        
    keys = list(counts.keys())
    values = list(counts.values())

    if ax is None:
        plt.figure(figsize=(20,12))
    ax = sns.histplot(x=keys, weights=values, discrete=True, ax=ax)
    plt.sca(ax)
    if xticklabels:
        xtick_fontsize = "medium"
        if len(keys) > 100:
            xtick_fontsize = "x-small"
        elif len(keys) > 75:
            xtick_fontsize = "small"
        plt.xticks(
            rotation=90, #45, 
            horizontalalignment='right',
            fontweight='light',
            fontsize=xtick_fontsize
        )
        if len(keys) > 100:
            for label in ax.xaxis.get_ticklabels()[::2]:
                label.set_visible(False)
        
    else:
        ax.set_xticklabels([])
    
    return counts


#############################################################
#############################################################


def plot_split_distributions(data_splits: Dict[str, "CommonDataset"]):
    """
    Create 3 vertically-stacked count plots of train, val, and test dataset class label distributions
    """
    assert isinstance(data_splits, dict)
    num_splits = len(data_splits)
    
    if num_splits < 4:
        rows = num_splits
        cols = 1
    else:
        rows = int(num_splits // 2)
        cols = int(num_splits % 2)
    fig, ax = plt.subplots(rows, cols, figsize=(20*cols,10*rows))
    ax = ax.flatten()
    
    
    train_key = [k for k,v in data_splits.items() if "train" in k]
    sort_by = True
    if len(train_key)==1:
        sort_by = compute_class_counts(data_splits[train_key[0]].targets,
                                       sort_by="count")
        log.info(f'Sorting by count for {train_key} subset, and applying order to all other subsets')
#         log.info(f"len(sort_by)={len(sort_by)}")

    num_classes = len(set(list(data_splits.values())[0].targets))    
    xticklabels=False
    num_samples = 0
    counts = {}
    for i, (k, v) in enumerate(data_splits.items()):
        if i == len(data_splits)-1:
            xticklabels=True
        counts[k] = plot_class_distributions(targets=v.targets, 
                                             sort_by=sort_by,
                                             ax = ax[i],
                                             xticklabels=xticklabels)
        num_nonzero_classes = len([name for name, count_i in counts[k].items() if count_i > 0])
        
        title = f"{k} [n={len(v)}"
        if num_nonzero_classes < num_classes:
            title += f", num_classes@(count > 0) = {num_nonzero_classes}-out-of-{num_classes} classes in dataset"
        title += "]"
        plt.gca().set_title(title, fontsize='large')
        
        num_samples += len(v)
    
    suptitle = '-'.join(list(data_splits.keys())) + f"_splits (total samples={num_samples}, total classes = {num_classes})"
    
    plt.suptitle(suptitle, fontsize='x-large')
    plt.subplots_adjust(bottom=0.1, top=0.94, wspace=None, hspace=0.08)
    
    return fig, ax


#############################################################
#############################################################



#############################################################
#############################################################


def filter_df_by_threshold(df: pd.DataFrame,
                           threshold: int,
                           y_col: str='family'):
    """
    Filter rare classes from dataset in a pd.DataFrame
    
    Input:
        df (pd.DataFrame):
            Must contain at least 1 column with name given by `y_col`
        threshold (int):
            Exclude any rows from df that contain a `y_col` value with fewer than `threshold` members in all of df.
        y_col (str): default="family"
            The column in df to look for rare classes to exclude.
    Output:
        (pd.DataFrame):
            Returns a dataframe with the same number of columns as df, and an equal or lower number of rows.
    """
    return df.groupby(y_col).filter(lambda x: len(x) >= threshold)


#############################################################
#############################################################



def compute_class_counts(targets: Sequence,
                         sort_by: Optional[Union[str, bool, Sequence]]="count"
                        ) -> Dict[str, int]:
    
    counts = collections.Counter(targets)
#     if hasattr(sort_by, "__len__"):
    if isinstance(sort_by, dict):
        counts = {k: counts[k] for k in sort_by.keys()}
    if isinstance(sort_by, list):
        counts = {k: counts[k] for k in sort_by}
    elif (sort_by == "count"):
        counts = dict(sorted(counts.items(), key = lambda x:x[1], reverse=True))
    elif (sort_by is True):
        counts = dict(sorted(counts.items(), key = lambda x:x[0], reverse=True))
        
    return counts

#############################################################
#############################################################














#############################################################
#############################################################
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


