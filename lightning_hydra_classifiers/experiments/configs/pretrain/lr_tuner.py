"""
lightning_hydra_classifiers/experiments/configs/pretrain/lr_tuner.py


Created on: Monday Sept 13th, 2021
Author: Jacob A Rose

"""


from dataclasses import dataclass

__all__ = ["LRTunerConfig"]


@dataclass
class LRTunerConfig:
    
    min_lr: float = 1e-08
    max_lr: float = 1.0
    num_training: int = 100
    mode: str = 'exponential'
    early_stop_threshold: float = 4.0