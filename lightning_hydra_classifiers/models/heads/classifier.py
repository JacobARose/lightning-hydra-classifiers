"""
lightning_hydra_classifiers/models/heads/classifier.py

Author: Jacob A Rose
Created: Monday June 21st, 2021


Based on soure: https://github.com/Neronjust2017/TransferBed/blob/master/common/modules/classifier.py

"""

from typing import Tuple, Optional, List, Dict, Any
import torch.nn as nn
import torch

from ..base import BaseModule
from .. import backbones

__all__ = ['Classifier']


class Classifier(BaseModule):
    """A generic Classifier class for domain adaptation.
    Args:
        backbone (torch.nn.Module): Any backbone to extract 2-d features from data
        num_classes (int): Number of classes
        bottleneck (torch.nn.Module, optional): Any bottleneck layer. Use no bottleneck by default
        bottleneck_dim (int, optional): Feature dimension of the bottleneck layer. Default: -1
        head (torch.nn.Module, optional): Any classifier head. Use :class:`torch.nn.Linear` by default
        finetune (bool): Whether finetune the classifier or train from scratch. Default: True
    .. note::
        Different classifiers are used in different domain adaptation algorithms to achieve better accuracy
        respectively, and we provide a suggested `Classifier` for different algorithms.
        Remember they are not the core of algorithms. You can implement your own `Classifier` and combine it with
        the domain adaptation algorithm in this algorithm library.
    .. note::
        The learning rate of this classifier is set 10 times to that of the feature extractor for better accuracy
        by default. If you have other optimization strategies, please over-ride :meth:`~Classifier.get_parameters`.
    Inputs:
        - x (tensor): input data fed to `backbone`
    Outputs:
        - predictions: classifier's predictions
        - features: features after `bottleneck` layer and before `head` layer
    Shape:
        - Inputs: (minibatch, *) where * means, any number of additional dimensions
        - predictions: (minibatch, `num_classes`)
        - features: (minibatch, `features_dim`)
    """
    
#     backbone: nn.Module = None
    num_classes: int = 1000
    backbone_name: str = 'resnet50'
    bottleneck_dim: int = -1
#     bottleneck: Optional[nn.Module] = None
#     head: Optional[nn.Module] = None
    finetune: bool = True
    

    def __init__(self,
                 backbone: nn.Module = None,
                 num_classes: int = 1000,
                 backbone_name: Optional[str] = 'resnet50',
                 bottleneck_dim: Optional[int] = -1,
                 bottleneck: Optional[nn.Module] = None,
                 head: Optional[nn.Module] = None,
                 finetune: bool=True):

        self.num_classes = num_classes
        self.classes = []
        super(Classifier, self).__init__()
        
        if backbone is None:
            backbone = self.build_backbone(name=backbone_name,
                                                pretrained=True)
            
        self.backbone = backbone
        self.backbone_name = self.backbone.name
        self.backbone_pretrained = self.backbone.pretrained
        
        if bottleneck is None:
            self.bottleneck = self.build_bottleneck(pool_size=(1,1))
            self._features_dim = self.backbone.out_features
        else:
            self.bottleneck = bottleneck
            
            if bottleneck_dim == -1:
                bottleneck_dim = self.backbone.out_features
            
            assert bottleneck_dim > 0
            self._features_dim = bottleneck_dim

        if head is None:
            self.head = self.build_head(input_dim=self._features_dim,
                                        num_classes=self.num_classes)
        else:
            self.head = head
        self.finetune = finetune

    @property
    def features_dim(self) -> int:
        """The dimension of features before the final `head` layer"""
        return self._features_dim
    
        

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """"""
        return self.__forward(x)[0]


    def __forward(self, x: torch.Tensor) -> torch.Tensor:
    
        f = self.backbone(x)
        f = self.bottleneck(f)
        predictions = self.head(f)
        return predictions, f
                                                
    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
            {"params": self.bottleneck.parameters(), "lr": 1.0 * base_lr},
            {"params": self.head.parameters(), "lr": 1.0 * base_lr},
        ]

        return params
    
    def copy_head(self) -> nn.Module:
        """Copy the origin fully connected layer"""
        return copy.deepcopy(self.head)

    def get_parser_args(self) -> Dict[str,Any]:
        return {"backbone": self.backbone,
                "num_classes":self.num_classes,
                "backbone_name":self.backbone_name,
                "bottleneck_dim":self.bottleneck_dim,
                "bottleneck":self.bottleneck,
                "head":self.head,
                "finetune":self.finetune}
    
    
    @classmethod
    def build_backbone(cls,
                       name: str,
                       pretrained: bool=False):
        
        return backbones.build_model(name,
                                     pretrained=pretrained)
    
    @classmethod
    def build_bottleneck(cls,
                         pool_size: Tuple[int]=(1,1)):
        return nn.Sequential(
                             nn.AdaptiveAvgPool2d(output_size=pool_size),
                             nn.Flatten()
                             )
    
    @classmethod
    def build_head(cls,
                   input_dim: int=512,
                   num_classes: int=1000):

        head = nn.Linear(input_dim, num_classes)
        cls.initialize_weights([head])
        
        return head
