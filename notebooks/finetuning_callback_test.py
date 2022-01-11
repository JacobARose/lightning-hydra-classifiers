

import os
import pytest
from typing import *
import pytorch_lightning as pl
import torch
from lightning_hydra_classifiers.models.transfer import *
from torch.utils.data import DataLoader

from torch import nn
# from pytorch_lightning import LightningModule, seed_everything, Trainer
import logging
import json
logging.basicConfig(level=logging.DEBUG)
logger = logging.Logger(__name__)
logger.setLevel('INFO')
pylog = logging.getLogger()


BN_TYPE = (torch.nn.modules.batchnorm._BatchNorm,)

def is_bn(layer: nn.Module) -> bool:
    """ Return True if layer's type is one of the batch norms."""
    return isinstance(layer, BN_TYPE)

def grad_check(tensor: torch.Tensor) -> bool:
    """ Returns True if tensor.requires_grad==True, else False."""
    return tensor.requires_grad == True


# os.chdir("/media/data/jacob/GitHub/lightning-hydra-classifiers")#/tests")

class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=2000, shape=(3,64,64)):
        self.num_samples = num_samples
        self.shape = shape
        self.data = torch.randn(num_samples, *shape)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples

class RandomTupleSupervisedDataset(RandomDataset):
    
    def __init__(self, num_classes=1000, num_samples=2000, shape=(3,64,64)):
        super().__init__(num_samples, shape)
        self.num_classes = num_classes
        
        self.targets = torch.randperm(num_classes)[:num_samples]
        
    def __getitem__(self, index):
        return self.data[index], self.targets[index]


    
    
class FinetuningLightningCallback(pl.callbacks.Callback):
    
# class FinetuningLightningPlugin:
    mode_dict = {"min": torch.lt, "max": torch.gt}
    order_dict = {"min": "<", "max": ">"}
    
    
    def __init__(self,
                 monitor: str="val_loss",
                 mode: str="min",
                 patience: int=4):
        
#         if pl_module.hparams.finetuning_strategy == "finetuning_unfreeze_layers_on_plateau":
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
#         self.best_metric = 0
        self.milestone_index = 0
        
        self.min_delta *= 1 if self.monitor_op == torch.gt else -1
        torch_inf = torch.tensor(np.Inf)
        self.best_score = torch_inf if self.monitor_op == torch.lt else -torch_inf
    
        self.milestone_logs = []
        
    def on_fit_start(self,
                     trainer,
                     pl_module):
        self.milestones = pl_module.finetuning_milestones
        print(f"Setting milestones: {pl_module.finetuning_milestones}")
    
    
    def finetuning_pretrained_strategy(self,
                                       trainer: "pl.Trainer",
                                       pl_module):
        """
        
        
        """
        epoch = trainer.current_epoch
        logs = trainer.callback_metrics
        current = logs.get(self.monitor)
        
        if self.mode == "min":
            new_best = current < self.best_score
        elif self.mode == "max":
            new_best = current > self.best_score
            
        if new_best or self.wait_epochs >= self.patience:
            self.best_score = current
            self.wait_epochs = 0
            next_to_unfreeze = self.milestones[-self.milestone_index]
            print(f"epoch: {epoch} unfreezing down to: {next_to_unfreeze}")
            pl_module.unfreeze_backbone_top_layers(unfreeze_down_to=next_to_unfreeze)
            self.milestone_index += 1
            self.milestone_logs.append({"epoch":epoch,
                                        "unfreeze_at_layer":next_to_unfreeze,
                                        "trainable_params":model.get_trainable_parameters(count_params=True),
                                        "nontrainable_params":model.get_nontrainable_parameters(count_params=True)})
        else:
            self.wait_epochs += 1
    
    @property
    def monitor_op(self) -> Callable:
        return self.mode_dict[self.mode]
    
    def on_epoch_end(self, trainer, pl_module):
        """Called when the epoch ends."""

        self.finetuning_pretrained_strategy(trainer=trainer, pl_module=pl_module)
        try:
            trainer.logger.log_dict(self.milestone_logs[-1], on_epoch=True)
        except Exception as e:
            print(e)
            print(f"logging to wandb didnt work bro")

            
            
            
            

class TestLightningClassifier(LightningClassifier):

    def __init__(self,
                 backbone_name='resnet50',
                 pretrained: Union[bool, str]=True,
                 num_classes: int=1000,
                 finetuning_strategy: str="feature_extractor",
                 seed: int=None,
                 **kwargs):

        super().__init__(backbone_name=backbone_name,
                         pretrained=pretrained,
                         num_classes=num_classes,
                         pool_type="avgdrop",
                         head_type="linear",
                         hidden_size=None, lr=0.01, backbone_lr_mult=0.1,
                         weight_decay=0.01,
                         finetuning_strategy=finetuning_strategy,
                         seed=42,
                        **kwargs)
        self._verbose=True
        
        
    
        
    def training_step(self, batch, batch_idx):
        self.log("train_loss",1)
        return {"loss": 1}
    
    def validation_step(self, batch, batch_idx):
        self.log("val_loss",1)
        return {"loss": 1}
    
    
#         output = super().training_step(batch, batch_idx)
#         self._verbose=False
#         return output

    def training_step_end(self, outputs):
        super().training_step_end(outputs)

    def print(self, *args):
        if self._verbose:
            print(*args)

    def train_dataloader(self):
        return DataLoader(RandomTupleSupervisedDataset(num_classes=1000, num_samples=50, shape=(3,64,64)), batch_size=2)

    def val_dataloader(self):
        return DataLoader(RandomTupleSupervisedDataset(num_classes=1000, num_samples=50, shape=(3,64,64)), batch_size=2)


def save_log(log, fp):
    with open(fp, "w") as fp:
        json.dump(log, fp, indent=4, sort_keys=False)

# @pytest.mark.parametrize("finetuning_strategy",
#                         [("feature_extractor",)
#                          "feature_extractor_+_bn.eval()",
#                          "feature_extractor_+_except_bn"])

# @pytest.mark.parametrize("finetuning_strategy, expected_layer_counts",
#     [
#         ("feature_extractor",
#             {"is_training":{'True': 53, 'False': 0, 'Total': 53}, 
#              "requires_grad":{'True': 0, 'False': 53, 'Total': 53}}
#         ),
#         ("feature_extractor_+_bn.eval()",
#             {"is_training":{'True': 0, 'False': 53, 'Total': 53}, 
#              "requires_grad":{'True': 0, 'False': 53, 'Total': 53}}
#         ),
#         ("feature_extractor_+_except_bn",
#             {"is_training":{'True': 53, 'False': 0, 'Total': 53}, 
#              "requires_grad":{'True': 53, 'False': 0, 'Total': 53}}
#         )
#     ]
#                         )
# @pytest.mark.parametrize()
def test_finetuning_callback(tmpdir):#, finetuning_strategy: str, expected_layer_counts: Dict[str,Dict[str,int]]):#, expectations: Dict[str,Any]):
    """Test finetuning strategy works as expected."""

    pl.seed_everything(42)
    
    callbacks = [FinetuningLightningCallback(monitor="val_loss",
                                             mode="min",
                                             patience=4)]

    model = TestLightningClassifier(finetuning_strategy=finetuning_strategy)
#     callback = TestBackboneFinetuningCallback(unfreeze_backbone_at_epoch=3, verbose=False)

    trainer = pl.Trainer(limit_train_batches=2,
                         limit_val_batches=2,
                         default_root_dir=tmpdir,
                         callbacks=callbacks,
                         max_epochs=20)
    trainer.fit(model)
    
    
#     pylog.info(f"strategy: {finetuning_strategy}")
    model._verbose = True
    layer_counts = model.count_trainable_batchnorm_layers()
    
#     save_log(log=model.method_calls,
#              fp=os.path.abspath(f"{finetuning_strategy}-method_calls.json"))
    print(f"method_calls logs saved to: {os.path.abspath(f'{finetuning_strategy}-method_calls.json')}")
    print(f"method_calls: {model.method_calls}")

#     print(f"strategy: {finetuning_strategy}")
#     print(f"Expected layer counts: {expected_layer_counts}")
#     print(f"count trainable batchnorm layers: {model.count_trainable_batchnorm_layers()}")
#     print(f"count trainable layers: {model.get_trainable_parameters(count_layers=True)}")
#     print(f"count nontrainable layers: {model.get_nontrainable_parameters(count_layers=True)}")
    

    pylog.info(f"strategy: {finetuning_strategy}")
    pylog.info(f"Expected layer counts: {expected_layer_counts}")
    pylog.info(f"count trainable batchnorm layers: {model.count_trainable_batchnorm_layers()}")
    pylog.info(f"count trainable layers: {model.get_trainable_parameters(count_layers=True)}")
    pylog.info(f"count nontrainable layers: {model.get_nontrainable_parameters(count_layers=True)}")
    
    
#     assert expected_layer_counts["is_training"]["True"] == layer_counts[0]["True"]
#     assert expected_layer_counts["is_training"]["False"] == layer_counts[0]["False"]

#     assert expected_layer_counts["requires_grad"]["True"] == layer_counts[1]["True"]
#     assert expected_layer_counts["requires_grad"]["False"] == layer_counts[1]["False"]
