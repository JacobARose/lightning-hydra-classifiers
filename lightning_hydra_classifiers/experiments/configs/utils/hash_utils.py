"""

lightning_hydra_classifiers/experiments/configs/utils/hash_utils.py


Hash utils for reproducible configs.

Author: Jacob A Rose
Created: Wednesday Sept 16th, 2021


"""

from typing import *
import json
import hashlib
import dataclasses

__all__ = ["get_hash"]

def dict_drop_empty(pairs):
    return dict(
        (k, v)
        for k, v in pairs
        if not (
            v is None
            or not v and isinstance(v, Collection)
        )
    )

def json_default(thing):
    return dataclasses.asdict(thing, dict_factory=dict_drop_empty)


def json_dumps(thing):
    return json.dumps(
        thing,
        default=json_default,
        ensure_ascii=False,
        sort_keys=True,
        indent=None,
        separators=(',', ':'),
    )


def get_hash(thing):
    """
    Produce deterministic md5 hash codes from dataclass configs.
    
    Based on blogpost: https://death.andgravity.com/stable-hashing
    
    
    """
    return hashlib.md5(json_dumps(thing).encode('utf-8')).hexdigest()

# print(json_dumps(asdict(cfg)))
# print(get_hash(cfg))