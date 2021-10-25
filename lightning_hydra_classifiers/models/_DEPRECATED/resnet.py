"""

TBD: DEPRECATE (Added notice Monday August 30th, 2021)

"""

from typing import Any, List, Optional, Dict, Tuple

import torch
from torch import nn, functional as F
import torchvision
from pytorch_lightning import LightningModule

import torchmetrics as metrics
# from pytorch_lightning import metrics
#.classification import Accuracy
import timm
from torchsummary import summary
import pandas as pd
from pathlib import Path
from stuf import stuf
import wandb
import numpy as np










def display_model_params_requires_grad(model, prefix=''):
    for name, p in model.named_parameters():
        print(prefix + name, type(p), p.shape, f'requires_grad=={p.requires_grad}')
    
    

    
# 3.a Optional: Register a custom backbone
# This is useful to create new backbone and make them accessible from `ImageClassifier`
# @ImageClassifier.backbones(name="resnet18")
def fn_resnet(pretrained: bool = True):
    model = torchvision.models.resnet18(pretrained)
    # remove the last two layers & turn it into a Sequential model
    backbone = nn.Sequential(*list(model.children())[:-2])
    num_features = model.fc.in_features
    # backbones need to return the num_features to build the head
    return backbone, num_features
    
    


def create_classifier(num_features: int, num_classes: int, pool_type='avg', bias: bool=True):
    global_pool = nn.AdaptiveAvgPool2d(1)
    flatten_layer = nn.Flatten()
    linear_layer = nn.Linear(num_features, num_classes, bias=bias)
    return [global_pool, flatten_layer, linear_layer]


def get_scalar_metrics(num_classes: int, average: str='macro', prefix: str=''):
    default = {'acc_top1': metrics.Accuracy(top_k=1),
               'acc_top3': metrics.Accuracy(top_k=3),
               'precision_top1': metrics.Precision(num_classes=num_classes, top_k=1, average=average),
#                'precision_top3': metrics.Precision(num_classes=num_classes, top_k=3, average=average),
               'recall_top1': metrics.Recall(num_classes=num_classes, top_k=1, average=average)}#,
#                'recall_top3': metrics.Recall(num_classes=num_classes, top_k=3, average=average)}
    if len(prefix)>0:
        for k in list(default.keys()):
            default[prefix + r'/' + k] = default[k]
            del default[k]
    
    return metrics.MetricCollection(default)


def get_expensive_metrics(num_classes: int, normalize: Optional[str]=None, threshold: float=0.5, num_bins: int=100):
    default = {'ConfusionMatrix': metrics.ConfusionMatrix(num_classes=num_classes, normalize=normalize, threshold=threshold),
               'BinnedPRCurve': metrics.BinnedPrecisionRecallCurve(num_classes=num_classes, num_thresholds=num_bins),
               'BinnedAvgPrecision': metrics.BinnedAveragePrecision(num_classes=num_classes, num_thresholds=num_bins)}
    return metrics.MetricCollection(default)


# def _load_pretrained_model(model_name: str):
#     if model_name.startswith('timm_'):
#         from timm import create_model
#     elif model_name.startswith('torchvision_'):
#         from torchvision.models import 
import pdb

# class ResNet(nn.Module):
class ResNet(LightningModule):
    example_input_size = (2, 3, 224, 224)
    def __init__(self,
                 model_name: Optional[str]='resnet50',
                 num_classes: Optional[int]=1000,
                 input_shape: Tuple[int]=(3,224,224),
                 batch_size: Optional[int]=None,
                 optimizer=stuf({'name':"Adam", 'lr':0.01}),
                 seed: int=None,
                 **kwargs): #, 'weight_decay':0.0})):
        super().__init__()
#         pl.trainer.seed_everything(seed=seed)
        self.save_hyperparameters()
        self.hparams.lr = optimizer.lr
        self.optimizer_hparams = stuf(optimizer)
        self.example_input_array = torch.rand(self.example_input_size)
        assert 'resnet' in model_name
        model = timm.create_model(model_name, pretrained=True)
        self.freeze(model)
        del model.fc
#         for key, val in model.__dict__.items():
#             self.__dict__[key] = val
#         self.base_model = model
        self.relu = nn.ReLU()
        
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        
        self.stem = nn.Sequential(
                        self.conv1,
                        self.bn1,
                        self.relu,
                        self.maxpool
                        )
        
        self.features = nn.Sequential(
                        self.layer1,
                        self.layer2,
                        self.layer3,
                        self.layer4
                        )
        self.num_features = model.num_features
#         self.num_features = self._get_conv_output(input_shape)
        self.input_shape = input_shape
        print(self.num_features, self._get_conv_output(input_shape))
        self.reset_classifier(num_classes, global_pool='avg')
                
        self.criterion = nn.CrossEntropyLoss()
        scalar_metrics = get_scalar_metrics(num_classes=num_classes)
        
        self.train_metrics = scalar_metrics.clone(prefix='train/')
        self.val_metrics = scalar_metrics.clone(prefix='val/')
        self.test_metrics = scalar_metrics.clone(prefix='test/')
        
        self.automatic_optimization = False

    def forward(self, x):
        out = self.forward_features(x)
        out = self.classifier(out)
        return out
    
    
    def forward_features(self, x):
        out = self.stem(x)
        out = self.features(out)
        return out
        
    def _get_conv_output(self, shape): # returns the size of the output tensor going into Linear layer from the conv block.
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self.forward_features(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size
    
    def step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        y_prob = self.probs(y_hat)
        y_pred = torch.max(y_prob, dim=1)[1]
        return {'loss':loss,
                'log':{
                       'y_prob':y_prob,
                       'y_pred':y_pred,
                       'y_true':y,
                       'batch_idx':batch_idx
                       }
               }
    
                
    def training_step(self, batch, batch_idx):      
        opt = self.optimizers()
        x, y = batch
        y_hat = self(x)
        
        def closure():#opt):
            self._current_loss = self.loss(y_hat, y)
            opt.zero_grad()
            self.manual_backward(self._current_loss)

        opt.step(closure=closure)
        
        y_prob = self.probs(y_hat)
        y_hat_int, y_pred = torch.max(y_prob, dim=1)
        
        loss = self._current_loss.detach()
        self.trainer.train_loop.running_loss.append(loss)
        return {'loss':loss,
                'log':{
                       'train_loss':loss,
                       'y_prob':y_prob,
                       'y_pred':y_pred,
                       'y_true':y,
                       'batch_idx':batch_idx
                       }
               }
        
        
    def training_step_end(self, outputs):
        run = self.logger.experiment[0]
        logs = outputs['log']
        loss = outputs['loss']
        idx = logs['batch_idx']
        y_prob, y_pred, y = logs['y_prob'], logs['y_pred'], logs['y_true']

        batch_metrics = self.train_metrics(y_prob, y)
        self.log_dict(batch_metrics)
        
        self.log('train/acc', batch_metrics['train/acc_top1'], on_step=True, prog_bar=True)
        self.log('train/loss', loss,
                 on_step=True,# on_epoch=True)#,
                 logger=True, prog_bar=True)
        run.log({
                 "train/y_pred":
                 wandb.Histogram(np_histogram=np.histogram(y_pred.cpu(), bins=self.num_classes)),
                 "train/y_true":
                 wandb.Histogram(np_histogram=np.histogram(y.cpu(), bins=self.num_classes)),
                "train/batch_idx":
                 idx,
                "global_step": self.trainer.global_step
                })
        
        
    def training_epoch_end(self, outputs):
        self.log_dict(self.train_metrics)
        
        run = self.logger.experiment[0]
        run.log({"epoch":self.trainer.current_epoch,
                 "global_step": self.trainer.global_step})
        
#         self.train_metrics.reset()
        
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        y_prob = self.probs(y_hat)
        y_pred = torch.max(y_prob, dim=1)[1]
        return {'loss':loss,
                'log':{
                       'val_loss':loss,
                       'y_prob':y_prob,
                       'y_pred':y_pred,
                       'y_true':y,
                       'batch_idx':batch_idx
                       }
               }

    def validation_step_end(self, outputs):
        logs = outputs['log']
        loss = logs['val_loss']
        idx = logs['batch_idx']
        
        y_prob, y_pred, y = logs['y_prob'], logs['y_pred'], logs['y_true']        

        batch_metrics = self.val_metrics(y_prob, y)
#         self.log_dict(batch_metrics)
        
        self.log('val/acc', batch_metrics['val/acc_top1'], on_step=True, prog_bar=True)

#         for k in val_metrics_step.keys():
#             self.log(k, val_metrics_step[k], on_step=True, on_epoch=False)        

        self.log('val/loss',loss,
                 on_step=True, on_epoch=True,
                 logger=True, prog_bar=True)
        
        run = self.logger.experiment[0]
        run.log({"val/y_pred":
                 wandb.Histogram(np_histogram=np.histogram(y_pred.cpu())),
                 "val/y_true":
                 wandb.Histogram(np_histogram=np.histogram(y.cpu())),
                 "val/batch_idx":
                 idx,
                 "global_step": self.trainer.global_step,
                 **self.val_metrics(y_prob, y)
                })
        
    def validation_epoch_end(self, outputs):
        self.log_dict(self.val_metrics.compute())
        
        run = self.logger.experiment[0]
        run.log({"epoch":self.trainer.current_epoch,
                 "global_step": self.trainer.global_step,})
        
        self.val_metrics.reset()

        
    def on_after_backward(self):
        # example to inspect gradient information in tensorboard
        if self.trainer.global_step % 25 == 0:  # don't make the tf file huge
            for k, v in self.named_parameters():
                self.logger.experiment[0].add_histogram(
                    tag=f'grads/{k}', values=v.grad, global_step=self.trainer.global_step
                )
            
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        sample_imgs = x[:10]
        grid = torchvision.utils.make_grid(sample_imgs).permute(1,2,0).cpu().numpy()
        self.logger.experiment[0].log({"test/examples": [wandb.Image(grid, caption="test images")],
                                    "global_step": self.trainer.global_step})
        
        y_pred = torch.argmax(y_hat, dim=1)
        y_prob = self.probs(y_hat)
#         self.test_metrics(y_hat, y)
        self.log('test/loss', loss, logger=True)
        
        test_metrics_step = self.test_metrics(y_prob, y)
        for k in test_metrics_step.keys():
            self.log(k, test_metrics_step[k], logger=True) #, on_step=True, on_epoch=False)
    
        return test_metrics_step

    
    def test_epoch_end(self, outputs):
        
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()
    
    
    @property
    def loss(self):
        return self.criterion

    
    def probs(self, x):
        return x.softmax(dim=-1)
    
    
    def predict(self, batch, batch_idx, dataloader_idx):
        return (batch[0].detach().cpu(), self(batch[0]), *batch[1:])


    def reset_classifier(self, num_classes, global_pool: str='avg'):
        self.num_classes = num_classes
        self.global_pool, self.flatten, self.fc = create_classifier(self.num_features, 
                                                                    self.num_classes, 
                                                                    pool_type=global_pool)
        self.classifier = nn.Sequential(
                                        self.global_pool,
                                        self.flatten,
                                        self.fc
                                        )
        self.initialize_weights(self.classifier)

        
    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(
                                params=self.parameters(), 
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.optimizer.weight_decay
                                )

    def save_model(self, path:str):
        path = str(path)
        if not Path(path).suffix=='.ckpt':
            path = path + ".ckpt"
        torch.save(self.state_dict(), path)
        
        
    def load_model(self, path:str):
        path = str(path)
        if not Path(path).suffix=='.ckpt':
            path = path + ".ckpt"
        self.load_state_dict(torch.load(path))
        


    def freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False
            
    def unfreeze(self, model):
        for param in model.parameters():
            param.requires_grad = True

    def get_trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def get_frozen_parameters(self):
        return (p for p in self.parameters() if not p.requires_grad)

    
    def initialize_weights(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#     def on_after_backward(self, training_step_output, batch_idx, untouched_loss):
#         print('Entered LightningModule method: lm.on_after_backward()')
#         print('Grad Norms:\n', self.grad_norm(2))        
#         print(self.device, 'self.training=', self.training)
#         summary(self, input_size=(3, 224, 224))
#         display_model_params_requires_grad(self)
#         import pdb; pdb.set_trace()        
#         super().on_after_backward(training_step_output, batch_idx, untouched_loss)

            
    
    
    
    
#         return {'hidden':logs}
#         return logs #outputs[['y_pred', 'y_true']]
#         super().validation_step_end(outputs)
            
            
#     def validation_epoch_end(self, epoch_outputs):
#         run = self.logger.experiment
#         print(len(epoch_outputs))
#         import pdb; pdb.set_trace()
#         run.log({k:v.cpu().numpy() for k,v in self.val_metrics.compute().items()})
#         run.log({'val/loss_epoch':torch.sum(['val/loss']),
#                  **self.val_metrics.compute()})



#     def batch_loss(self, batch):
#         x, y = batch
#         y_hat = self(x)
#         loss = self.loss(y_hat, y)
#         return loss

    
    
    
    
    
    
#     def loss(self, logits, labels):
#         return self.criterion(logits, labels)
#         return F.nll_loss(logits, labels)
        
#     def stem(self, x):
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.maxpool(out)
#         return out
        
#     def forward_features(self, x):
#         out = self.stem(x)
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         return out
    
#     def logits(self, x):
#         out = self.global_pool(x)
#         out = out.view(out.size(0), -1)
#         out = self.fc(out)
#         return out
    
    
#         self.train_metrics = {'scalar':scalar_metrics.clone(prefix='train/'),
#                               'vector':vector_metrics.clone(prefix='train/')}
#         self.val_metrics = {'scalar':scalar_metrics.clone(prefix='val/'),
#                             'vector':vector_metrics.clone(prefix='val/')}
#         self.test_metrics = {'scalar':scalar_metrics.clone(prefix='test/'),
#                              'vector':vector_metrics.clone(prefix='test/')}        
    
    
    
    
    
    
###########################################
###########################################
    
    
    
#     def validation_epoch_end(self, epoch_outputs):
#         logger = self.logger.experiment
#         cm = self.val_metrics['vector']['ConfusionMatrix']
#         pr_curve = self.val_metrics['vector']['BinnedPRCurve']
#         avg_precision = self.val_metrics['vector']['BinnedAvgPrecision']
        
#         for idx, batch in enumerate(epoch_outputs):
            
#             y_pred, y_true = batch['y_pred'].cpu(), batch['y_true'].cpu()
#             cm(y_pred, y_true)
#             pr_curve(y_pred, y_true)
#             avg_precision(y_pred, y_true)
            
            
            
#         cm_table = wandb.Table(dataframe=pd.DataFrame(cm.compute()))#.cpu().numpy()))
#         logger.log({"val/confusion_matrix":cm_table, "epoch":self.current_epoch}, commit=False)
        
#         return epoch_outputs

    
    
#         precision, recall, thresholds = pr_curve.compute()
#         pr_table = wandb.Table(['family', 'precision', 'recall'], [(class_idx, pr, re) for class_idx, pr, re in enumerate(zip(precision, recall))])
#         logger.log({"val/pr_curve":pr_table, "epoch":self.current_epoch}, commit=False)
            
#         avg_precision_table = wandb.Table(dataframe=pd.DataFrame([float(bin_) for bin_ in avg_precision.compute()]))
#         logger.log({"val/avg_precision":avg_precision_table, "epoch":self.current_epoch})
            

    
    
    
    
    
    
    
    
    
#         for idx, batch in enumerate(epoch_outputs):
#             y_pred, y_true = batch['y_pred'], batch['y_true']
            
#             cm = self.val_metrics['vector']['ConfusionMatrix'](y_pred, y_true)
#             confusion_matrix = wandb.Table(dataframe=cm)
#             logger.log({"confusion_matrix":pr_curve, "epoch":self.current_epoch}, step=idx, commit=False)

#             precision, recall, thresholds = self.val_metrics['vector']['BinnedPRCurve'](y_pred, y_true)
#             pr_curve = wandb.Table([(pr, re) for pr, re in zip(precision, recall)])
#             logger.log({"pr_curve":pr_curve, "epoch":self.current_epoch}, step=idx, commit=False)
# #             pr_curve(pred, target)
            
#             avg_precision = self.val_metrics['vector']['BinnedAvgPrecision'](y_pred, y_true)
#             avg_precision = wandb.Table(dataframe=avg_precision)
#             logger.log({"avg_precision":avg_precision, "epoch":self.current_epoch}, step=idx)




# class FCView(nn.Module):
#     def __init__(self):
#         super(FCView, self).__init__()

#     # noinspection PyMethodMayBeStatic
#     def forward(self, x):
#         n_b = x.data.size(0)
#         x = x.view(n_b, -1)
#         return x

#     def __repr__(self):
#         return 'view(nB, -1)'

    
# class TeeHeads(nn.Module):
#     def __init__(self, *nets):
#         """Create multi-head network (multiple outputs)
#         :param nets: modules to form a Tee
#         :type nets: nn.Module
#         """
#         super(TeeHeads, self).__init__()
#         for idx, net in enumerate(nets):
#             self.add_module("{}".format(idx), net)

#     def forward(self, *inputs):
#         outputs = []
#         for module in self._modules.values():
#             outputs.append(module(*inputs))
#         return outputs




######################
    
    
    
            
# class ResNet(LightningModule):
#     # Modify attributs	    
#     def __init__(self,
#                  backbone_name: Optional[str]='resnet50',
#                  num_classes: Optional[int]=1000,
#                  optimizer=stuf({'name':"Adam", 'lr':0.001, 'weight_decay':0.0})):
            
            
            
# class ImagenetTransferLearning(LightningModule):
#     def __init__(self):
#         super().__init__()

#         # init a pretrained resnet
#         backbone = models.resnet50(pretrained=True)
#         num_filters = backbone.fc.in_features
#         layers = list(backbone.children())[:-1]
#         self.feature_extractor = nn.Sequential(*layers)

#         # use the pretrained model to classify cifar-10 (10 image classes)
#         num_target_classes = 10
#         self.classifier = nn.Linear(num_filters, num_target_classes)

#     def forward(self, x):
#         self.feature_extractor.eval()
#         with torch.no_grad():
#             representations = self.feature_extractor(x).flatten(1)
#         x = self.classifier(representations)
#         ...





##############################################
##############################################
##############################################
##############################################
##############################################


# class ResNet(nn.Module):
#     """A ResNet model for feature extraction, fine-tuning, and transfer learning
#     Args:
#         backbone (torch.nn.Module): Any backbone to extract 2-d features from data.
#         num_classes (int): Number of classes.
#         head_source (torch.nn.Module): Classifier head of source model.
#         head_target (torch.nn.Module, optional): Any classifier head. Use :class:`torch.nn.Linear` by default
#         finetune (bool): Whether finetune the classifier or train from scratch. Default: True
#     Inputs:
#         - x (tensor): input data fed to backbone
#     Outputs:
#         - y_s: predictions of source classifier head
#         - y_t: predictions of target classifier head
#     Shape:
#         - Inputs: (b, *) where b is the batch size and * means any number of additional dimensions
#         - y_s: (b, N), where b is the batch size and N is the number of classes
#         - y_t: (b, N), where b is the batch size and N is the number of classes
#     """
#     tasks = ["source", "target"]
    
#     def __init__(self, 
#                  num_classes: int=1000):
#         super().__init__()
        
    
#     def _build_model(self, backbone: str='resnet50'):
#         assert 'resnet' in backbone_name
#         backbone = timm.create_model(backbone_name, pretrained=True)

#         named_layers = OrderedDict(model.named_children())
#         layers = list(named_layers.values())
#         layer_names = list(named_layers.keys())
#         layer_names
        
        
#         self.backbone = backbone
#         for param in self.backbone.parameters():
#             param.requires_grad = False
            
#         self.num_classes = num_classes
# #         self.bottleneck = nn.Sequential(
# #             nn.AdaptiveAvgPool2d(output_size=(1, 1)),
# #             nn.Flatten()
# #         )
        
#         num_filters = self.backbone.fc.in_features
#         layers = list(self.backbone.children())[:-1]
#         self.feature_extractor = nn.Sequential(*layers)#, self.bottleneck)

        
#         self._features_dim = self.backbone.num_features
#         self.head_source = head_source or nn.Linear(self._features_dim, num_classes)
#         self.head_target = head_target or nn.Linear(self._features_dim, num_classes)
#         self.finetune = finetune
#         self.set_current_task(task='source')
        
#         self._loss = torch.nn.CrossEntropyLoss()

#         self.train_accuracy = Accuracy()
#         self.val_accuracy = Accuracy()
#         self.test_accuracy = Accuracy()

#         self.metric_hist = {
#             "train/acc": [],
#             "val/acc": [],
#             "train/loss": [],
#             "val/loss": [],
#         }
        
        
#     def set_current_task(self, task: float='source'):
#         assert task in self.tasks, f"task {task} is not in registered tasks: {self.tasks}"
        
#         if task == 'source':
#             self._set_source_task()
#         if task == 'target':
#             self._set_target_task()

#     def _set_source_task(self):
#         self.head = self.head_source

#     def _set_target_task(self):
#         self.head = self.head_target
            
            
#     @property
#     def features_dim(self) -> int:
#         """The dimension of features before the final `head` layer"""
#         return self._features_dim

#     def loss(self, y_pred, y_true):
#         return self._loss(y_pred, y_true)
    
#     def forward(self, x: torch.Tensor):
#         """"""
# #         f = self.backbone(x)
# #         f = self.bottleneck(f)
#         f = self.feature_extractor(x)
#         y = self.head(f)
#         return y

#     def get_parameters(self, base_lr: float=0.1) -> List[Dict]:
#         """A parameter list which decides optimization hyper-parameters,
#             such as the relative learning rate of each layer
#         """
#         params = [
#             {"params": self.backbone.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
#             {"params": self.head_source.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
#             {"params": self.bottleneck.parameters(), "lr": 1.0 * base_lr},
#             {"params": self.head_target.parameters(), "lr": 1.0 * base_lr},
#         ]
#         return params
    
#     def step(self, batch: Any):
#         x, y = batch
#         logits = self.forward(x)
#         loss = self.criterion(logits, y)
#         preds = torch.argmax(logits, dim=1)
#         return loss, preds, y

#     def training_step(self, batch: Any, batch_idx: int):
#         loss, preds, targets = self.step(batch)
#         acc = self.train_accuracy(preds, targets)
#         return {"loss": loss, "preds": preds, "targets": targets}

#     def training_epoch_end(self, outputs: Dict[Any, Any]):
#         # log best so far train acc and train loss
#         self.metric_hist["train/acc"].append(self.trainer.callback_metrics["train/acc"])
#         self.metric_hist["train/loss"].append(self.trainer.callback_metrics["train/loss"])
#         self.log("train/acc_best", max(self.metric_hist["train/acc"]), prog_bar=False)
#         self.log("train/loss_best", min(self.metric_hist["train/loss"]), prog_bar=False)

#     def validation_step(self, batch: Any, batch_idx: int):
#         loss, preds, targets = self.step(batch)

#         # log val metrics
#         acc = self.val_accuracy(preds, targets)
#         self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
#         self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

#         return {"loss": loss, "preds": preds, "targets": targets}

#     def validation_epoch_end(self, outputs: List[Any]):
#         # log best so far val acc and val loss
#         self.metric_hist["val/acc"].append(self.trainer.callback_metrics["val/acc"])
#         self.metric_hist["val/loss"].append(self.trainer.callback_metrics["val/loss"])
#         self.log("val/acc_best", max(self.metric_hist["val/acc"]), prog_bar=False)
#         self.log("val/loss_best", min(self.metric_hist["val/loss"]), prog_bar=False)

#     def test_step(self, batch: Any, batch_idx: int):
#         loss, preds, targets = self.step(batch)

#         # log test metrics
#         acc = self.test_accuracy(preds, targets)
#         self.log("test/loss", loss, on_step=False, on_epoch=True)
#         self.log("test/acc", acc, on_step=False, on_epoch=True)

#         return {"loss": loss, "preds": preds, "targets": targets}

#     def test_epoch_end(self, outputs: List[Any]):
#         pass

#     def configure_optimizers(self):
#         """Choose what optimizers and learning-rate schedulers to use in your optimization.
#         Normally you'd need one. But in the case of GANs or similar you might have multiple.

#         See examples here:
#             https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
#         """
#         return torch.optim.Adam(
#             params=self.parameters(), 
#             lr=self.hparams.lr,
#             weight_decay=self.hparams.weight_decay
#         )

    
    
    
    
#         def training_step(self, batch: Any, batch_idx: int):
#         loss, preds, targets = self.step(batch)

#         # log train metrics
#         acc = self.train_accuracy(preds, targets)
# #         self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
# #         self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

#         # we can return here dict with any tensors
#         # and then read it in some callback or in training_epoch_end() below
#         # remember to always return loss from training_step, or else backpropagation will fail!
#         return {"loss": loss, "preds": preds, "targets": targets}

#     def training_epoch_end(self, outputs: List[Any]):
#         # log best so far train acc and train loss
#         self.metric_hist["train/acc"].append(self.trainer.callback_metrics["train/acc"])
#         self.metric_hist["train/loss"].append(self.trainer.callback_metrics["train/loss"])
#         self.log("train/acc_best", max(self.metric_hist["train/acc"]), prog_bar=False)
#         self.log("train/loss_best", min(self.metric_hist["train/loss"]), prog_bar=False)

#     def validation_step(self, batch: Any, batch_idx: int):
#         loss, preds, targets = self.step(batch)

#         # log val metrics
#         acc = self.val_accuracy(preds, targets)
#         self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
#         self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

#         return {"loss": loss, "preds": preds, "targets": targets}

#     def validation_epoch_end(self, outputs: List[Any]):
#         # log best so far val acc and val loss
#         self.metric_hist["val/acc"].append(self.trainer.callback_metrics["val/acc"])
#         self.metric_hist["val/loss"].append(self.trainer.callback_metrics["val/loss"])
#         self.log("val/acc_best", max(self.metric_hist["val/acc"]), prog_bar=False)
#         self.log("val/loss_best", min(self.metric_hist["val/loss"]), prog_bar=False)

#     def test_step(self, batch: Any, batch_idx: int):
#         loss, preds, targets = self.step(batch)

#         # log test metrics
#         acc = self.test_accuracy(preds, targets)
#         self.log("test/loss", loss, on_step=False, on_epoch=True)
#         self.log("test/acc", acc, on_step=False, on_epoch=True)

#         return {"loss": loss, "preds": preds, "targets": targets}

#     def test_epoch_end(self, outputs: List[Any]):
#         pass

#     def configure_optimizers(self):
#         """Choose what optimizers and learning-rate schedulers to use in your optimization.
#         Normally you'd need one. But in the case of GANs or similar you might have multiple.

#         See examples here:
#             https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
#         """
#         return torch.optim.Adam(
#             params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
#         )
