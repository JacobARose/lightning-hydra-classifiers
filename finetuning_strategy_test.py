
import pytest
from typing import *
import pytorch_lightning as pl
import torch
from lightning_hydra_classifiers.models.transfer import *
from torch.utils.data import DataLoader

# from pytorch_lightning import LightningModule, seed_everything, Trainer
import logging
import json
logging.basicConfig(level=logging.INFO) #logging.DEBUG)
pylog = logging.getLogger()

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

    

class TestLightningFinetuneClassifier(TestLightningClassifier):

    
    method_calls = {"feature_extractor_strategy":{},
                         "freeze":{},
                         "unfreeze":{},
                         "freeze_bn":{},
                         "unfreeze_bn":{},
                         "set_bn_eval":{}}
    for k in method_calls:
        method_calls[k]["num_calls"] = 0
    
    def __init__(self,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)
        
    def feature_extractor_strategy(self,
                                   freeze_bn: bool=True,
                                   eval_bn: bool=False):
        """
        Defaults to PyTorch default, which is to freeze the gradients for batchnorm layers, but not necessarily apply eval() to them upon freezing, thus allowing the running mean & std to continue training on each incoming batch.
        
        Allows the option of both freezing and setting to eval mode all batch norm layers, thus fully removing the possibility of data leakage or accidental injection of noise.

        Arguments:
           freeze_bn: bool, default=True
               If True, set bn layers' attribute requires_grad to False.
           eval_bn: bool, default=False
               If True, apply layer.eval() to bn layers. If False, apply layer.train() to bn layers.
        
        """
        self.method_calls["feature_extractor_strategy"]["num_calls"] += 1
        self.method_calls["feature_extractor_strategy"]["kwargs"] = {"freeze_bn":freeze_bn,
                                                                     "eval_bn": eval_bn}
        super().feature_extractor_strategy(freeze_bn=freeze_bn,
                                           eval_bn=eval_bn)
    
    
    @classmethod
    def freeze(cls,
               module,
               freeze_bn: bool=True,
               filter_pattern: Optional[str]=None):
        
        cls.method_calls["freeze"]["num_calls"] += 1
        cls.method_calls["freeze"]["kwargs"] = {"freeze_bn":freeze_bn,
                                                "filter_pattern":filter_pattern}
        cls.method_calls["freeze"]["set_requires_grad=False"] = []
        
#         super().freeze(module, freeze_bn, filter_pattern)
        modules = list(module.named_modules())
        
        for n, m in modules:
            if isinstance(filter_pattern, str) and (filter_pattern not in n):
                continue
            for p_name, p in m.named_parameters():
                if isinstance(filter_pattern, str) and (filter_pattern not in n):
                    continue
                if freeze_bn or not is_bn(m):
                    cls.method_calls["freeze"]["set_requires_grad=False"].append(f"module:{n}-param:{p_name}")
                    p.requires_grad=False
            cls.freeze_bn(m, freeze_bn)      

            
    @classmethod
    def unfreeze(cls,
                 module,
                 unfreeze_bn: bool=True,
                 filter_pattern: Optional[str]=None):
        
        cls.method_calls["unfreeze"]["num_calls"] += 1
        cls.method_calls["unfreeze"]["kwargs"] = {"unfreeze_bn":unfreeze_bn,
                                                 "filter_pattern":filter_pattern}
        cls.method_calls["unfreeze"]["set_requires_grad=True"] = []

#         out = (p for _, p in cls.get_named_parameters(model=module))
        
        if isinstance(module, (Generator, Sequence)):
            modules = module
        else:
            modules = list(module.named_modules())
        for n, m in modules:
            if isinstance(filter_pattern, str) and (filter_pattern not in n):
                continue
            if is_bn(m) and not unfreeze_bn:
                continue
            cls.unfreeze_bn(m, unfreeze_bn)
            for p_name, p in m.named_parameters():
                cls.method_calls["freeze"]["set_requires_grad=True"].append(f"module:{n}-param:{p_name}")
                p.requires_grad=True
            m.train()

    @classmethod
    def freeze_bn(cls, module: nn.Module, freeze_bn: bool=True):
        bn_mods = list(cls.get_batchnorm_modules(model=module))
        cls.method_calls["freeze_bn"]["num_calls"] += 1
        cls.method_calls["freeze_bn"]["kwargs"] = {"freeze_bn":freeze_bn}
        cls.method_calls["freeze_bn"]["set_requires_grad=False"] = []
        cls.method_calls["freeze_bn"]["num_modules"] = len(bn_mods)

        
        for n, m in bn_mods:
            if freeze_bn:
                for p_name, p in m.named_parameters():
                    p.requires_grad = False
                    cls.method_calls["freeze"]["set_requires_grad=False"].append(f"module:{n}-param:{p_name}")
                    if cls._verbose: logger.debug(f"[freeze_bn][Layer={n}] Set requires_grad=False")

    @classmethod
    def unfreeze_bn(cls, module: nn.Module, unfreeze_bn: bool=True):
        bn_mods = list(cls.get_batchnorm_modules(model=module))
        cls.method_calls["unfreeze_bn"]["num_calls"] += 1
        cls.method_calls["unfreeze_bn"]["kwargs"] = {"unfreeze_bn":unfreeze_bn}
        cls.method_calls["unfreeze_bn"]["set_requires_grad=True"] = []
        cls.method_calls["unfreeze_bn"]["num_modules"] = len(bn_mods)
        
        for n, m in bn_mods: #cls.get_batchnorm_modules(model=module):
            if unfreeze_bn:
                for p_name, p in m.named_parameters():
                    p.requires_grad = True
                    cls.method_calls["freeze"]["set_requires_grad=True"].append(f"module:{n}-param:{p_name}")
                    if cls._verbose: logger.debug(f"[unfreeze_bn][Layer={n}] Set requires_grad=True")

                    
    @classmethod
    def set_bn_eval(cls, module: nn.Module)->None:
        "Set bn layers in eval mode for all recursive children of `m`."
        
        bn_mods = []
        cls.method_calls["set_bn_eval"]["num_calls"] += 1
        cls.method_calls["set_bn_eval"]["kwargs"] = {"unfreeze_bn":unfreeze_bn}
        cls.method_calls["set_bn_eval"]["set_requires_grad=True"] = []
        

        
        for n, l in module.named_children():
#             if isinstance(l, nn.BatchNorm2d) and not next(l.parameters()).requires_grad:
            if is_bn(l) and not next(l.parameters()).requires_grad:
                l.eval()
                bn_mods.append(n)
                if cls._verbose: logger.debug(f"[set_bn_eval][Layer={n}] Called layer.eval()")

#                 continue
            cls.set_bn_eval(l)
    
        cls.method_calls["set_bn_eval"]["called_modules"] = bn_mods
#         super().__init__(backbone_name=backbone_name,
#                          pretrained=pretrained,
#                          num_classes=num_classes,
#                          pool_type="avgdrop",
#                          head_type="linear",
#                          hidden_size=None, lr=0.01, backbone_lr_mult=0.1,
#                          weight_decay=0.01,
#                          finetuning_strategy=finetuning_strategy,
#                          seed=42,
#                         **kwargs)
#         self._verbose=True

#     def training_step(self, batch, batch_idx):
#         output = super().training_step(batch, batch_idx)
#         self._verbose=False
#         return output

#     def training_step_end(self, outputs):
#         output = super().training_step_end(outputs)
#         assert output==None
#         assert outputs != None

#     def print(self, *args):
#         if self._verbose:
#             print(*args)

#     def train_dataloader(self):
#         return DataLoader(RandomTupleSupervisedDataset(num_classes=1000, num_samples=50, shape=(3,64,64)), batch_size=2)

#     def val_dataloader(self):
#         return DataLoader(RandomTupleSupervisedDataset(num_classes=1000, num_samples=50, shape=(3,64,64)), batch_size=2)






def save_log(log, fp):
    with open(fp, "w") as fp:
        json.dump(log, fp, indent=4, sort_keys=False)

# @pytest.mark.parametrize("finetuning_strategy",
#                         [("feature_extractor",)
#                          "feature_extractor_+_bn.eval()",
#                          "feature_extractor_+_except_bn"])

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

    model = TestLightningFinetuneClassifier(finetuning_strategy=finetuning_strategy)
#     callback = TestBackboneFinetuningCallback(unfreeze_backbone_at_epoch=3, verbose=False)

    trainer = pl.Trainer(limit_train_batches=4, default_root_dir=tmpdir, callbacks=[], max_epochs=2)
    trainer.fit(model)
    
    
#     pylog.info(f"strategy: {finetuning_strategy}")
    model._verbose = True
    layer_counts = model.count_trainable_batchnorm_layers()
    
    save_log(log=model.method_calls,
             fp=os.abspath(f"{finetuning_strategy}-method_calls.json"))
    print(f"method_calls logs saved to: {os.abspath(f"{finetuning_strategy}-method_calls.json")}")
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
    
    
    assert expected_layer_counts["is_training"]["True"] == layer_counts[0]["True"]
    assert expected_layer_counts["is_training"]["False"] == layer_counts[0]["False"]

    assert expected_layer_counts["requires_grad"]["True"] == layer_counts[1]["True"]
    assert expected_layer_counts["requires_grad"]["False"] == layer_counts[1]["False"]

    
#     pylog.debug(f"strategy: {finetuning_strategy}")
#     pylog.debug(f"Expected layer counts: {expected_layer_counts}")
#     pylog.debug(f"count trainable batchnorm layers: {model.count_trainable_batchnorm_layers()}")
#     pylog.debug(f"count trainable layers: {model.get_trainable_parameters(count_layers=True)}")
#     pylog.debug(f"count nontrainable layers: {model.get_nontrainable_parameters(count_layers=True)}")
