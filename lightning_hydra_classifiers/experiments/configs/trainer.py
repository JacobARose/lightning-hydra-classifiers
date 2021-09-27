
"""

lightning_hydra_classifiers/experiments/configs/trainer.py

Collection of dataclass configs for lightning experiments.

Author: Jacob A Rose
Created: Monday Sept 13th, 2021


"""


from typing import *
from dataclasses import dataclass, field

__all__ = ["TrainerConfig"]


from omegaconf import MISSING


@dataclass
class TrainerConfig:
    _target_: str = "pytorch_lightning.trainer.Trainer"
    logger: Any = True
    checkpoint_callback: bool = True
    callbacks: Any = None
    default_root_dir: Optional[str] = None
    gradient_clip_val: float = 0
    process_position: int = 0
    num_nodes: int = 1
    num_processes: int = 1
    gpus: Any = None
    auto_select_gpus: bool = False
    tpu_cores: Any = None
    log_gpu_memory: Optional[str] = None
    progress_bar_refresh_rate: int = 1
    overfit_batches: Any = 0.0  # Union[int, float]
    track_grad_norm: Any = -1  # Union[int, float, str]
    check_val_every_n_epoch: int = 1
    fast_dev_run: Any = False  # Union[int, bool]
    accumulate_grad_batches: Any = 1  # Union[int, Dict[int, int], List[list]]
    max_epochs: int = 1000
    min_epochs: int = 1
    max_steps: Optional[int] = None
    min_steps: Optional[int] = None
    limit_train_batches: Any = 1.0  # Union[int, float]
    limit_val_batches: Any = 1.0  # Union[int, float]
    limit_test_batches: Any = 1.0  # Union[int, float]
    val_check_interval: Any = 1.0  # Union[int, float]
    flush_logs_every_n_steps: int = 100
    log_every_n_steps: int = 50
    accelerator: Any = None  # Union[str, Accelerator, NoneType]
    sync_batchnorm: bool = False
    precision: int = 32
    weights_summary: Optional[str] = "full"
    weights_save_path: Optional[str] = None
    num_sanity_val_steps: int = 2
    truncated_bptt_steps: Optional[int] = None
    resume_from_checkpoint: Any = None  # Union[str, Path, NoneType]
    profiler: Any = None  # Union[BaseProfiler, bool, str, NoneType]
    benchmark: bool = False
    deterministic: bool = False
    reload_dataloaders_every_epoch: bool = False
    auto_lr_find: Any = False  # Union[bool, str]
    replace_sampler_ddp: bool = True
    terminate_on_nan: bool = False
    auto_scale_batch_size: Any = False  # Union[str, bool]
    prepare_data_per_node: bool = True
    plugins: Any = None   # Union[str, list, NoneType]
    amp_backend: str = "native"
    amp_level: str = "O2"
    distributed_backend: Optional[str] = None
    automatic_optimization: Optional[bool] = None
    move_metrics_to_cpu: bool = False
    enable_pl_optimizer: bool = False