"""
lr_schedulers.py

"""






from torch.optim import Optimizer
from typing import Optional

import torch
import math


def linear_warmup_decay(warmup_steps, total_steps, cosine=True, linear=False):
    """Linear warmup for warmup_steps, optionally with cosine annealing or linear decay to 0 at total_steps."""
    assert not (linear and cosine)

    def fn(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))

        if not (cosine or linear):
            # no decay
            return 1.0

        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        if cosine:
            # cosine decay
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        # linear decay
        return 1.0 - progress

    return fn



class DelayedCosineAnnealingWarmRestarts(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
    def __init__(self,
                 optimizer, 
                 T_0,
                 T_mult=2,
                 eta_min=0,
                 start_epoch=0):
        self.start_epoch = start_epoch
        super().__init__(optimizer=optimizer, 
                         T_0=T_0,
                         T_mult=T_mult,
                         eta_min=eta_min)
    def step(self, epoch=0):
        if epoch >= self.start_epoch:
            super().step(epoch=epoch-self.start_epoch)
            
            
            







def configure_schedulers(optimizer,
                         config):
    """
    Options are:
    
    - linear_warmup_cosine_decay
    - linear_warmup_cosine_decay_w_warm_restarts
    
    """
    
    if config.scheduler_type == "linear_warmup_cosine_decay":
        warmup_steps = config.get("warmup_steps", 5)
        total_steps = config.get("total_steps", 20)
        lr_lambda = linear_warmup_decay(warmup_steps, total_steps, cosine=True, linear=False)
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [{"scheduler":warmup_scheduler, "interval": "epoch"}]
    elif config.scheduler_type == "linear_warmup_cosine_decay_warm_restarts":
        warmup_steps = config.get("warmup_steps", 5)
        total_steps = config.get("total_steps", 20)
        lr_lambda = linear_warmup_decay(warmup_steps, total_steps, cosine=True, linear=False)
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        min_lr = config.get("min_lr", )
        decay_scheduler = DelayedCosineAnnealingWarmRestarts(optimizer,
                                                             T_0=config.get("T_0",1),
                                                             T_mult=config.get("T_mult", 2),
                                                             eta_min=config.get("eta_min",0),
                                                             start_epoch=config.get("start_epoch", warmup_steps))
        return [{"scheduler":warmup_scheduler, "interval": "epoch"},
                {"scheduler":decay_scheduler, "interval": "epoch"}]
    elif config.scheduler_type is False:
        return []
    else:
        raise Exception(f"Misconfigured Scheduler config:{config}")


















################################################


# from lr_scheduler.lr_scheduler import LearningRateScheduler
# from lr_scheduler.reduce_lr_on_plateau_lr_scheduler import ReduceLROnPlateauScheduler
# from lr_scheduler.warmup_lr_scheduler import WarmupLRScheduler


# class WarmupLRScheduler(LearningRateScheduler):
#     """
#     Warmup learning rate until `total_steps`
#     Args:
#         optimizer (Optimizer): wrapped optimizer.
#     """
#     def __init__(
#             self,
#             optimizer: Optimizer,
#             init_lr: float,
#             peak_lr: float,
#             warmup_steps: int,
#     ) -> None:
#         super(WarmupLRScheduler, self).__init__(optimizer, init_lr)
#         self.init_lr = init_lr
#         if warmup_steps != 0:
#             warmup_rate = peak_lr - init_lr
#             self.warmup_rate = warmup_rate / warmup_steps
#         else:
#             self.warmup_rate = 0
#         self.update_steps = 1
#         self.lr = init_lr
#         self.warmup_steps = warmup_steps

#     def step(self, val_loss: Optional[torch.FloatTensor] = None):
#         if self.update_steps < self.warmup_steps:
#             lr = self.init_lr + self.warmup_rate * self.update_steps
#             self.set_lr(self.optimizer, lr)
#             self.lr = lr
#         self.update_steps += 1
#         return self.lr


# class WarmupReduceLROnPlateauScheduler(LearningRateScheduler):
#     r"""
#     source: https://github.com/sooftware/pytorch-lr-scheduler/blob/main/lr_scheduler/warmup_reduce_lr_on_plateau_scheduler.py
    
#     Warmup learning rate until `warmup_steps` and reduce learning rate on plateau after.
#     Args:
#         optimizer (Optimizer): wrapped optimizer.
#         init_lr (float): Initial learning rate.
#         peak_lr (float): Maximum learning rate.
#         warmup_steps (int): Warmup the learning rate linearly for the first N updates.
#         patience (int): Number of epochs with no improvement after which learning rate will be reduced.
#         factor (float): Factor by which the learning rate will be reduced. new_lr = lr * factor.
#     """
#     def __init__(
#             self,
#             optimizer: Optimizer,
#             init_lr: float,
#             peak_lr: float,
#             warmup_steps: int,
#             patience: int = 1,
#             factor: float = 0.3,
#     ) -> None:
#         super(WarmupReduceLROnPlateauScheduler, self).__init__(optimizer, init_lr)
#         self.warmup_steps = warmup_steps
#         self.update_steps = 0
#         self.warmup_rate = (peak_lr - init_lr) / self.warmup_steps \
#             if self.warmup_steps != 0 else 0
#         self.schedulers = [
#             WarmupLRScheduler(
#                 optimizer=optimizer,
#                 init_lr=init_lr,
#                 peak_lr=peak_lr,
#                 warmup_steps=warmup_steps,
#             ),
#             ReduceLROnPlateauScheduler(
#                 optimizer=optimizer,
#                 lr=peak_lr,
#                 patience=patience,
#                 factor=factor,
#             ),
#         ]

#     def _decide_stage(self):
#         if self.update_steps < self.warmup_steps:
#             return 0, self.update_steps
#         else:
#             return 1, None

#     def step(self, val_loss: Optional[float] = None):
#         stage, steps_in_stage = self._decide_stage()

#         if stage == 0:
#             self.schedulers[0].step()
#         elif stage == 1:
#             self.schedulers[1].step(val_loss)

#         self.update_steps += 1

#         return self.get_lr()