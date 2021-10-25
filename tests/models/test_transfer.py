"""

tests/models/test_transfer.py

tests for code defined in lightning_hydra_classifiers/models/transfer.py


Created by: Jacob A Rose
Creayed on: Friday Oct 22nd, 2021


"""

from lightning_hydra_classifiers.models.transfer import BaseLightningModule, LightningClassifier

from tests.helpers import RandomDataset, RandomTupleSupervisedDataset


import pytest
from typing import *
import pytorch_lightning as pl
import torch
from lightning_hydra_classifiers.models.transfer import *
from torch.utils.data import DataLoader

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.Logger(__name__)
logger.setLevel('INFO')
pylog = logging.getLogger(__name__)



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
        

# @pytest.fixture(params=["feature_extractor",
#                         "feature_extractor_+_bn.eval()",
#                         "feature_extractor_+_except_bn",
#                         "broken_strategy"])
# def finetuning_strategy(strategy):
#     with pytest.raises(ValueError, match="Invalid strategy"):
#         if strategy == "broken_strategy":
#             raise ValueError("Invalid strategy")
#     return strategy


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
        output = super().training_step(batch, batch_idx)
        self._verbose=False
        return output

    def training_step_end(self, outputs):
        output = super().training_step_end(outputs)
        assert output==None
        assert outputs != None

    def print(self, *args):
        if self._verbose:
            print(*args)

    def train_dataloader(self):
        return DataLoader(RandomTupleSupervisedDataset(num_classes=1000, num_samples=50, shape=(3,64,64)), batch_size=2)

    def val_dataloader(self):
        return DataLoader(RandomTupleSupervisedDataset(num_classes=1000, num_samples=50, shape=(3,64,64)), batch_size=2)






# @pytest.mark.parametrize("finetuning_strategy", # pytest.lazy_fixture(
#                         ["feature_extractor",
#                         "feature_extractor_+_bn.eval()",
#                         "feature_extractor_+_except_bn"])
# # @pytest.mark.parametrize()
# def test_finetuning_strategy(tmpdir, finetuning_strategy: str):#, expectations: Dict[str,Any]):
#     """Test finetuning strategy works as expected."""

#     pl.seed_everything(42)


#     class TestLightningClassifier(LightningClassifier):

#         def __init__(self,
#                      backbone_name='resnet50',
#                      pretrained: Union[bool, str]=True,
#                      num_classes: int=1000,
#                      finetuning_strategy: str="feature_extractor",
#                      seed: int=None,
#                      **kwargs):

#             super().__init__(backbone_name=backbone_name,
#                              pretrained=pretrained,
#                              num_classes=num_classes,
#                              pool_type="avgdrop",
#                              head_type="linear",
#                              hidden_size=None, lr=0.01, backbone_lr_mult=0.1,
#                              weight_decay=0.01,
#                              finetuning_strategy=finetuning_strategy,
#                              seed=42,
#                             **kwargs)
#             self._verbose=True

#         def training_step(self, batch, batch_idx):
#             output = super().training_step(batch, batch_idx)
#             self._verbose=False
#             return output

#         def training_step_end(self, outputs):
#             output = super().training_step_end(outputs)
            
#         def print(self, *args):
#             if self._verbose:
#                 print(*args)
    
# #         def configure_optimizers(self):
# #             optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
# #             lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
# #             return [optimizer], [lr_scheduler]

#         def train_dataloader(self):
#             return DataLoader(RandomTupleSupervisedDataset(num_classes=1000, num_samples=50, shape=(3,64,64)), batch_size=2)

#         def val_dataloader(self):
#             return DataLoader(RandomTupleSupervisedDataset(num_classes=1000, num_samples=50, shape=(3,64,64)), batch_size=2)

        
#     model = TestLightningClassifier(finetuning_strategy=finetuning_strategy)

#     trainer = pl.Trainer(limit_train_batches=4, default_root_dir=tmpdir, callbacks=[], max_epochs=2)
#     trainer.fit(model)

#     print(f"strategy: {finetuning_strategy}")
#     print("count trainable batchnorm layers: ", model.count_trainable_batchnorm_layers())
#     print("count trainable layers: ", model.get_trainable_parameters(count_layers=True))
#     print("count nontrainable layers: ", model.get_nontrainable_parameters(count_layers=True))
    
    
    
    
    
    
@pytest.mark.parametrize("finetuning_strategy, expected_layer_counts",
    [
        ("feature_extractor",
            {"is_training":{'True': 53, 'False': 0, 'Total': 53}, 
             "requires_grad":{'True': 0, 'False': 53, 'Total': 53}}
        ),
        ("feature_extractor_+_bn.eval()",
            {"is_training":{'True': 0, 'False': 53, 'Total': 53}, 
             "requires_grad":{'True': 0, 'False': 53, 'Total': 53}}
        ),
        ("feature_extractor_+_except_bn",
            {"is_training":{'True': 53, 'False': 0, 'Total': 53}, 
             "requires_grad":{'True': 53, 'False': 0, 'Total': 53}}
        )
    ]
                        )
# @pytest.mark.parametrize()
def test_finetuning_strategy(tmpdir, finetuning_strategy: str, expected_layer_counts: Dict[str,Dict[str,int]]):#, expectations: Dict[str,Any]):
    """Test finetuning strategy works as expected."""

    pl.seed_everything(42)

    model = TestLightningClassifier(finetuning_strategy=finetuning_strategy)
#     callback = TestBackboneFinetuningCallback(unfreeze_backbone_at_epoch=3, verbose=False)

    trainer = pl.Trainer(limit_train_batches=2, limit_val_batches=2, default_root_dir=tmpdir, callbacks=[], max_epochs=2)
    trainer.fit(model)
    
#     pylog.info(f"strategy: {finetuning_strategy}")
    model._verbose = True
    layer_counts = model.count_trainable_batchnorm_layers()
    
    pylog.info(f"strategy: {finetuning_strategy}")
    pylog.info(f"Expected layer counts: {expected_layer_counts}")
    pylog.info(f"count trainable batchnorm layers: {model.count_trainable_batchnorm_layers()}")
    pylog.info(f"count trainable layers: {model.get_trainable_parameters(count_layers=True)}")
    pylog.info(f"count nontrainable layers: {model.get_nontrainable_parameters(count_layers=True)}")
    
    
    assert expected_layer_counts["is_training"]["True"] == layer_counts[0]["True"]
    assert expected_layer_counts["is_training"]["False"] == layer_counts[0]["False"]

    assert expected_layer_counts["requires_grad"]["True"] == layer_counts[1]["True"]
    assert expected_layer_counts["requires_grad"]["False"] == layer_counts[1]["False"]