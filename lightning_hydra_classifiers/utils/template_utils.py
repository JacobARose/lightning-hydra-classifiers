import logging
import warnings
from typing import List, Sequence
import os
import pytorch_lightning as pl
import rich
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from rich.syntax import Syntax
from rich.tree import Tree
import hydra

def get_logger(name=__name__, level=logging.INFO):
    """Initializes python logger."""

    logger = logging.getLogger(name)
    
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


log = get_logger()


def extras(config: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file.
        - disabling warnings
        - easier access to debug mode
        - forcing debug friendly configuration
        - forcing multi-gpu friendly configuration
    Args:
        config (DictConfig): [description]
    """

    # enable adding new keys to config
    OmegaConf.set_struct(config, False)

    # disable python warnings if <config.disable_warnings=True>
    if config.get("disable_warnings"):
        log.info(f"Disabling python warnings! <config.disable_warnings=True>")
        warnings.filterwarnings("ignore")

    # set <config.trainer.fast_dev_run=True> if <config.debug=True>
    if config.get("debug"):
        log.info("Running in debug mode! <config.debug=True>")
        config.trainer.fast_dev_run = True

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    if config.trainer.get("fast_dev_run"):
        log.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        # Debuggers don't like GPUs or multiprocessing
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0

    # force multi-gpu friendly configuration if <config.trainer.accelerator=ddp>
    if config.trainer.get("accelerator") in ["ddp", "ddp_spawn", "dp", "ddp2"]:
        log.info("Forcing ddp friendly configuration! <config.trainer.accelerator=ddp>")
        # ddp doesn't like num_workers>0 or pin_memory=True
        if config.datamodule.get("num_workers"):
#             config.datamodule.num_workers = 0
            gpus = config.trainer.get("gpus")
            if isinstance(gpus, list):
                config.datamodule.num_workers = 4 * len(gpus)
            elif isinstance(gpus, int):
                config.datamodule.num_workers = 4
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False

    # disable adding new keys to config
    OmegaConf.set_struct(config, True)


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "trainer",
        "model",
        "datamodule",
        "callbacks",
        "logger",
        "hparams",
        "seed",
    ),
    resolve: bool = True,
    file: str = None
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Config.
        fields (Sequence[str], optional): Determines which main fields from config will be printed
        and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = Tree(f":gear: CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(Syntax(branch_content, "yaml"))

    rich.print(tree, file=file)


def empty(*args, **kwargs):
    pass


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.

    Additionaly saves:
        - sizes of train, val, test dataset
        - number of trainable model parameters
    """

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["trainer"] = config["trainer"]
    hparams["model"] = config["model"]
    hparams["datamodule"] = config["datamodule"]
    if "optimizer" in config:
        hparams["optimizer"] = config["optimizer"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]
    if "hparams" in config:
        hparams["hparams"] = config["hparams"]

    # save sizes of each dataset
    # (requires calling `datamodule.setup()` first to initialize datasets)
    # datamodule.setup()
    # if hasattr(datamodule, "data_train") and datamodule.data_train:
    #     hparams["datamodule/train_size"] = len(datamodule.data_train)
    # if hasattr(datamodule, "data_val") and datamodule.data_val:
    #     hparams["datamodule/val_size"] = len(datamodule.data_val)
    # if hasattr(datamodule, "data_test") and datamodule.data_test:
    #     hparams["datamodule/test_size"] = len(datamodule.data_test)

    # save number of model parameters
    hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )
    
    for logger in trainer.logger:
        if hasattr(logger, "log_hyperparams"):
            trainer.logger.log_hyperparams(hparams)
        if hasattr(logger, 'save_dir'):
            os.makedirs(logger.save_dir, exist_ok=True)
    
    if 'wandb' in config.logger.keys():
        wandb.watch(model.classifier, criterion=model.criterion, log='all')
        log.info(f'Logging classifier gradients to wandb using wandb.watch()')

    # disable logging any more hyperparameters for all loggers
    # (this is just a trick to prevent trainer from logging hparams of model, since we already did that above)
    trainer.logger.log_hyperparams = empty


from torch import distributed as dist
    
# @rank_zero_only
def init(config: DictConfig):
#     wandb.init(..., reinit=dist.is_available() and dist.is_initialized() and dist.get_rank() == 0)
    if config.trainer.accelerator == "ddp":
#     if wandb.run is None:
        config.wandb.init.reinit = True# = dist.is_available() and dist.is_initialized() and (dist.get_rank() == 0)
#         print(f"dist.is_available()={dist.is_available()}")
#         print(f"dist.is_initialized()={dist.is_initialized()}")
#         print(f"dist.get_rank() == 0)={(dist.get_rank() == 0)}")
    
        logging.info(f"Since trainer.accelerator={config.trainer.accelerator}, setting config.wandb.init.reinit to: {config.wandb.init.reinit}")
        
#             logging.info(f"torch.distributed.get_rank() = {dist.get_rank()}")
        
        local_rank = os.environ.get("LOCAL_RANK", 0)
        print(f'local_rank={local_rank}')
        if str(local_rank)=="0":
            hydra.utils.instantiate(config.wandb.init)
            print(f'Just may have successfully initiated wandb')
        else:
            print(f'Skipping wandb.init b/c local_rank={local_rank}')
        
    
# def init_ddp_connection(self, *args, **kwargs):
#     super().init_ddp_connection(*args, **kwargs)

#     if torch.distributed.get_rank() == 0:
#         import wandb
#         wandb.run = None
    
    
def finish(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, WandbLogger):
            wandb.finish()
