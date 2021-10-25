"""

experiments/multitask/modules.py


Pytorch Lightning Module extensions for multitask experiments.

Created by: Friday September 3rd, 2021
Author: Jacob A Rose


"""




import pytorch_lightning as pl
from munch import Munch
from collections import OrderedDict
from omegaconf import DictConfig, OmegaConf
from torch import nn
import torch
import re
from typing import Union, Optional, List, Tuple
from lightning_hydra_classifiers.models.heads import ClassifierHead
from lightning_hydra_classifiers.models.backbones import backbone
from lightning_hydra_classifiers.utils.metric_utils import get_per_class_metrics, get_scalar_metrics
from lightning_hydra_classifiers.models.base import BaseModule
from lightning_hydra_classifiers.experiments.configs.model import *
# from lightning_hydra_classifiers.experiments.configs.trainer import *


__all__ = ["LitMultiTaskModule"]#, "AdamWOptimizerConfig", "AdamOptimizerConfig"]




from dataclasses import dataclass, fields, asdict


# @dataclass
# class OptimizerConfig:
#     lr: float = 0.001
#     betas: Tuple[float] = (0.9, 0.999)
#     eps: float = 1e-08
#     weight_decay: float = 0.01
#     amsgrad: bool = False

# @dataclass
# class AdamWOptimizerConfig(OptimizerConfig):
#     _target_
#     weight_decay: float = 0.01

# @dataclass
# class AdamOptimizerConfig(OptimizerConfig):
#     weight_decay: float = 0.0


def _is_pool_type(l): return re.search(r'Pool[123]d$', l.__class__.__name__)

def has_pool_type(m: nn.Module) -> bool:
    "Return `True` if `m` is a pooling layer or has one in its children"
    if _is_pool_type(m): return True
    for l in m.children():
        if has_pool_type(l): return True
    return False


AVAILABLE_GLOBAL_POOL_LAYERS = {"avg":nn.AdaptiveAvgPool2d,
                                "max":nn.AdaptiveMaxPool2d}


# @dataclass(unsafe_hash=True)
# class ClassifierConfig:
#     in_features: Optional[int] = None
#     num_classes: Optional[int] = None

# @dataclass(unsafe_hash=True)
# class MultiTaskClassifierConfig:
#     task_0: ClassifierConfig = ClassifierConfig(None, num_classes=91)
#     task_1: ClassifierConfig = ClassifierConfig(None, num_classes=19)


# @dataclass(unsafe_hash=True)
# class BackboneConfig:
#     backbone_name: str
#     pretrained: str="imagenet"
#     global_pool_type: str="avg"
#     drop_rate: float=0.0
#     init_freeze_up_to: Optional[str]=None
        
# @dataclass(unsafe_hash=True)
# class LitMultiTaskModuleConfig:
    
#     backbone_config: BackboneConfig = BackboneConfig(backbone_name="resnet50")
#     multitask: MultiTaskClassifierConfig = MultiTaskClassifierConfig()
    
    
#     model_name: str
#     pretrained: Union[str, bool]=False,
#     progress: bool=True,
#     num_classes: int=1000,
#     global_pool_type: str='avg',
#     drop_rate: float=0.0,
#     init_freeze_up_to: str=None,

    
    
#     model_name: str,
#     pretrained: Union[str, bool]=False,
#     progress: bool=True,
#     num_classes: int=1000,
#     global_pool_type: str='avg',
#     drop_rate: float=0.0,
#     init_freeze_up_to: str=None,
        


        
        




class LitMultiTaskModule(pl.LightningModule, BaseModule):
    
    layers = ["backbone", "global_pool", "classifier"]
    
    def __init__(self, config: LitMultiTaskModuleConfig):
        super().__init__()
        self.current_task = "task_0"
        self.config = config
        try:
            config = asdict(config)
        except:
            config = dict(config)
#         import pdb; pdb.set_trace()
        self.save_hyperparameters({
                                   "config":config
        })
        self.heads = {}
        self.init_backbone(config=self.config.backbone)
        self.config.multitask.task_0.in_features = self.out_features
#         self.config.multitask.task_0.num_classes = config.num_classes
        self.config.multitask.task_1.in_features = self.out_features
#         self.config.multitask.task_1.num_classes = config.num_classes
        
        self.init_classifier(key="task_0",
                             config=self.config.multitask.task_0)
        self.init_classifier(key="task_1",
                             config=self.config.multitask.task_1)
        self.set_current_classifier(key="task_0")        
        
        self.metrics = self.init_metrics(stage='all')
        self.criterion = nn.CrossEntropyLoss()
        
    def features(self, x):
#         import pdb;pdb.set_trace()
        x = self.backbone(x)
        x = self.global_pool(x)
        return x
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def init_classifier(self,
                        key: str,
                        config: ClassifierConfig):
        self.heads[key] = ClassifierHead(in_features=config.in_features,
                                         num_classes=config.num_classes)        
        
    def set_current_classifier(self,
                               key: str):
        assert key in self.heads
        self.classifier = self.heads[key]
        self.num_classes = self.classifier.num_classes
        self.current_task = key
    
    def init_backbone(self,
                      config: BackboneConfig,
                      **kwargs):
        

        GlobalPoolFactory = AVAILABLE_GLOBAL_POOL_LAYERS[config.global_pool_type]
#         self.backbone = backbone.build_model(model_name=config.backbone_name,
#                                              pretrained=config.pretrained,
# #                                              num_classes=num_classes,
# #                                              global_pool_type=config.global_pool_type,
#                                              drop_rate=config.drop_rate)


        backbone_module = backbone.build_model(model_name=config.backbone_name,
                                                 pretrained=config.pretrained,
                                                 drop_rate=config.drop_rate)
                         
        self.backbone = nn.Sequential(OrderedDict(backbone_module.backbone.named_children()))

        self.out_features = backbone_module.out_features

        self.global_pool = GlobalPoolFactory(1) #self.out_features//4) #self.backbone.global_pool
        
        
#         self.classifier = self.backbone.classifier
        
#         self.global_pool = GlobalPoolFactory(output_size=1) #self.out_features)
#         self.classifier = ClassifierHead(in_features=self.out_features,
#                                          num_classes=self.num_classes)

        self.freeze_up_to(self.backbone, layer=config.init_freeze_up_to)

    

    @property
    def config(self):
        return self._config
    
    @config.setter
    def config(self, config):
        config = OmegaConf.structured(config)
#         config = OmegaConf.create(dict(config))
        self._config = config
        print(type(config.multitask))
        self.tasks = list(dict(config.multitask).keys())
#         self.tasks = list(config.multitask.__dataclass_fields__.keys())
#         self.hparams.config.update(config)
        self.lr = config.optimizer.lr
        self.weight_decay = config.optimizer.weight_decay
        self.num_classes = config.multitask[self.current_task].num_classes
    
    def configure_optimizers(self):
        print(f"self.hparams={self.hparams}")
        self.optimizer = torch.optim.AdamW([{"params":self.backbone.parameters()},
                                            {"params":self.classifier.parameters()}],
                                           lr=self.lr, weight_decay=self.weight_decay)#hparams.lr)
#         self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.t_max, eta_min=self.config.min_lr)

        return {'optimizer': self.optimizer}

    def step(self, batch, batch_idx):
        image, y_true = batch[0], batch[1]
        y_logit = self(image)
        y_pred = torch.argmax(y_logit, dim=-1)
        return y_logit, y_true, y_pred

    def log_metric_step(self,
                        stage: str='train',
                        omit_metric_types: Optional[List[str]]=None,
                        omit_metric_keys: Optional[List[str]]=None):
        omit_metric_types = omit_metric_types or []
        omit_metric_keys = omit_metric_keys or []
        
        for metric_type, metric_collection in self.all_metrics[stage].items():
            if metric_type in omit_metric_types:
                continue
            if metric_type == "scalar":
                self.log_dict({k:v for k,v in metric_collection.items() if k not in omit_metric_keys},
                               on_step=False, on_epoch=True, prog_bar=True, logger=True)

            elif metric_type == "per_class":
                for k,v in metric_collection.items():
                    if k in omit_metric_keys:
                        continue
                    results = v.compute()
                    for class_idx, result in enumerate(results): #range(len(results)):
                        self.log(f"{k}_class_{class_idx}", result,
                                 on_step=False, on_epoch=True, prog_bar=False, logger=True)
            else:
                logging.warning(f"[Warning] {metric_type} requires specialized handling in lightningmodule.log_metric_step().")
    
    
    def update_metric_step(self,
                           y_logit,
                           y_true,
                           stage: str='train'):

            for metric_type, metric_collection in self.all_metrics[stage].items():
                metric_collection(y_logit, y_true)    
    
    
    def training_step(self, batch, batch_idx):
        y_logit, y_true, y_pred = self.step(batch, batch_idx)
        loss = self.criterion(y_logit, y_true)
        self.update_metric_step(y_logit,
                                y_true,
                                stage="train")
        self.log_dict({"train_loss": loss,
                       "train_acc": self.metrics_train["train/acc_top1"]},
                       on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict({'lr': self.optimizer.param_groups[0]['lr']},
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log_metric_step(stage='train',
                             omit_metric_types=None,
                             omit_metric_keys=["train/acc_top1"])
        
#         self.log_dict(self.metrics_train,
#                  on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_logit, y_true, y_pred = self.step(batch, batch_idx)
        
        loss = self.criterion(y_logit, y_true)
#         y_pred = torch.argmax(y_logit, dim=-1)
#         scores = self.metrics_val(y_logit, y_true)

        self.update_metric_step(y_logit,
                                y_true,
                                stage="val")        
        self.log("val_loss", loss,
                  on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", self.metrics_val["val/acc_top1"],
                 on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log_metric_step(stage='val',
                             omit_metric_types=None,
                             omit_metric_keys=["val/acc_top1"])
#         self.log_dict({k:v for k,v in self.metrics_val.items() if k != "val/acc_top1"},
#                  on_step=False, on_epoch=True, prog_bar=True, logger=True)
        path, catalog_number = getattr(batch, "path"), getattr(batch, "catalog_number")
        return {"loss":loss,
                "y_logit":y_logit,
                "y_pred":y_pred,
                "y_true":y_true,
                "path":path,
                "catalog_number":catalog_number}
    
    def test_step(self, batch, batch_idx):
        y_logit, y_true, y_pred = self.step(batch, batch_idx)
        loss = self.criterion(y_logit, y_true)
#         scores = self.metrics_test(y_logit, y_true)

        self.update_metric_step(y_logit,
                                y_true,
                                stage="test")
        self.log_dict({"test_loss": loss},
                       on_step=True, on_epoch=True, logger=True)
        self.log_metric_step(stage='test',
                             omit_metric_types=None,
                             omit_metric_keys=None)
#         self.log_dict(self.metrics_test,
#                  on_step=False, on_epoch=True, logger=True)
        return loss

#     layers = OrderedDict({k:v for k,v in model.model.model.named_children() if k not in ["pool", "classifier"]})
    
    
    @classmethod
    def get_default_classifier_key(cls, model: nn.Module):
        children = dict(model.named_children()).keys()
        if "classifier" in children:
            return "classifier"
        if "fc" in children:
            return "fc"
        return None
        
        
    @classmethod
    def freeze_up_to(cls,
                     model: nn.Module,
                     layer: Union[int, str]=None):
        
        if isinstance(layer, int):
            if layer < 0:
                layer = len(list(model.parameters())) + layer
            
#         self.model.enable_grad = True
        model.requires_grad = True
        if layer in [None, 0]:
            return
        for i, (name, param) in enumerate(model.named_parameters()):
            if isinstance(layer, int):
                if layer == i:
                    break
            elif isinstance(layer, str):
                if layer in name:
                    break
            param.requires_grad = False
    
    
    def init_metrics(self, stage: str='train', tag: Optional[str]=None):
        tag = tag or ""
        if not hasattr(self, "all_metrics"):
            self.all_metrics = {}
        
        print(f"self.num_classes={self.num_classes}")
        if stage in ['train', 'all']:
            prefix=f'{tag}_train'.strip("_")
            self.metrics_train = get_scalar_metrics(num_classes=self.num_classes, average='macro', prefix=prefix)
            self.metrics_train_per_class = get_per_class_metrics(num_classes=self.num_classes, prefix='train')
            self.all_metrics['train'] = {"scalar":self.metrics_train,
                                         "per_class":self.metrics_train_per_class}
            
        if stage in ['val', 'all']:
            prefix=f'{tag}_val'.strip("_")
            self.metrics_val = get_scalar_metrics(num_classes=self.num_classes, average='macro', prefix=prefix)
            self.metrics_val_per_class = get_per_class_metrics(num_classes=self.num_classes, prefix='val')
            self.all_metrics['val'] = {"scalar":self.metrics_val,
                                       "per_class":self.metrics_val_per_class}
            
        if stage in ['test', 'all']:
            prefix=f'{tag}_test'.strip("_")
            self.metrics_test = get_scalar_metrics(num_classes=self.num_classes, average='macro', prefix=prefix)
            self.metrics_test_per_class = get_per_class_metrics(num_classes=self.num_classes, prefix='test')
            self.all_metrics['test'] = {"scalar":self.metrics_test,
                                        "per_class":self.metrics_test_per_class}

            
    def load_from_checkpoint(self, ckpt_path):
        
        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        state_dict = self.state_dict()
        if "state_dict" in ckpt:
            for k,v in ckpt["state_dict"].items():
                if v.shape == state_dict[k].shape:
                    state_dict[k] = v
            self.load_state_dict(state_dict)
        return self
        
            
            
#     def reset_metrics(self, stage: str='all'):
        
#         self.all_metrics = self.all_metrics or {}
#         for k, v in all_metrics:
#             try:
#                 del v
#             except:
#                 pass
        

    @classmethod
    def available_backbones(self):
        return backbone.AVAILABLE_MODELS


#     def show_batch(self, win_size=(10, 10)):

#         def _to_vis(data):
#             return tensor_to_image(torchvision.utils.make_grid(data, nrow=8))

#         # get a batch from the training set: try with `val_datlaoader` :)
#         imgs, labels = next(iter(self.train_dataloader()))
#         imgs_aug = self.transform(imgs)  # apply transforms
#         # use matplotlib to visualize
#         plt.figure(figsize=win_size)
#         plt.imshow(_to_vis(imgs))
#         plt.figure(figsize=win_size)
#         plt.imshow(_to_vis(imgs_aug))




# convenience funtion to log predictions for a batch of test images
def log_test_predictions(images, labels, outputs, predicted, test_table, log_counter):
    # obtain confidence scores for all classes
    scores = F.softmax(outputs.data, dim=1)
    log_scores = scores.cpu().numpy()
    log_images = images.cpu().numpy()
    log_labels = labels.cpu().numpy()
    log_preds = predicted.cpu().numpy()
    # adding ids based on the order of the images
    _id = 0
    for img, label, p, s in zip(log_images, log_labels, log_preds, log_scores):
        
        # id, image pixels, model's guess, true label, scores for all classes
        img_id = str(_id) + "_" + str(log_counter)
        test_table.add_data(img_id, wandb.Image(img), p, l, *s)
        _id += 1
        if _id == NUM_IMAGES_PER_BATCH:
            break




























    
########################################
# source: https://github.com/PyTorchLightning/pytorch-lightning/pull/5468
    
# class MyDataset(RandomDataset):

#     def __getitem__(self, index):
#         return {"index": index, "batch": self.data[index]}   

# class MyModel(LightningModule):

#     def test_step(self, batch, batch_idx, dataloader_idx=None):
#         x = batch["batch"]
#         output = self.layer(x)
#         loss = self.loss(batch, output)

#         # need to provide a key `id` being a number.
#         self.add_predictions([
#             {"id": idx.item(), "predictions": o} 
#             for idx, o in zip(batch["index"], output)])

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



#         y_logit, y_true, y_pred = self.step(batch, batch_idx)
#         return {'test_loss': F.cross_entropy(y_hat, y)}

#     def test_end(self, outputs):
#         # OPTIONAL
#         avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
#         tensorboard_logs = {'test_loss': avg_loss}
#         return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}
    
#     def test_epoch_end(self, outputs):
#         avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
#         logs = {'test_loss': avg_loss}
#         return {'test_loss': avg_loss, 'log': logs, 'progress_bar': logs}



# class ClassifierPool:
"""

Mon Sept 13th, 2021: Eh, maybe wont go ahead with this particular abstraction.
"""
#     __heads: OrderedDict = OrderedDict({})
    
#     @property
#     def heads(self):
#         return self.__heads
    
#     @property
#     def current_head(self):
#         return self.heads[self.current_name]    
    
#     def add_head(self, head: ClassifierHead, name: str):
#         assert isinstance(head, ClassifierHead)
#         if name in self.heads:
#             logger.info(f"Replacing classifier head: {name}")
#             del self.__heads[name]
#         self.__heads[name] = head
#         self.set_current_head(name=name)
    
#     def set_current_head(self, name: Union[int, str]):
#         assert name in self.heads
#         self.current_name = name
#         logger.info(f"Setting current classifier head to be: {name}")
    
# self.classifier_pool = ClassifierPool()

# self.classifier = ClassifierHead(in_features=self.out_features,
#                                  num_classes=num_classes)