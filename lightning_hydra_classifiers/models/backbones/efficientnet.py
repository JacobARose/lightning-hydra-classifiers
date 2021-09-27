"""
lightning_hydra_classifiers/models/backbones/efficientnet.py

Author: Jacob A Rose
Created: Thursday Junee 11th, 2021

"""




import timm
from timm import models
from torch import nn
import torch
from typing import Union
from .. import BaseModule
from ..heads.classifier import ClassifierHead




# efficientnet_layers = ['conv1', 'bn1', 'blocks.0' 'blocks.1', 'blocks.2', 'blocks.3', 'blocks.4', 'blocks.5', 'blocks.6', 'conv_head', 'bn2', 'classifier']
efficientnet_layers = ['conv_stem', 'bn1', 'blocks.0' 'blocks.1', 'blocks.2', 'blocks.3', 'blocks.4', 'blocks.5', 'blocks.6', 'conv_head', 'bn2', 'act2', 'global_pool', 'classifier']

AVAILABLE_GLOBAL_POOL_LAYERS = {"avg":nn.AdaptiveAvgPool2d,
                                "max":nn.AdaptiveMaxPool2d}

# def get_timm_backbone(model_name: str,
#                       pretrained: Union[str, bool]=False,
#                       progress: bool=True,
#                       num_classes: int=1000,
#                       global_pool_type: str='avg',
#                       **kwargs) -> nn.Module:
    
#     backbone = timm.create_model(model_name,
#                                  pretrained=pretrained,
#                                  progress=progress)

    
BackboneModelFactory = timm.create_model
    
    
    

class EfficientNetBackbone(BaseModule): #models.efficientnet.EfficientNet, BaseModule):
    """
    ResNets without fully connected layer
    
    Fully Connected layer is loaded if available, but omitted from the model's forward method.
    """
    layers = ['conv_stem', 'bn1', 'act1',
              'blocks',
              'conv_head', 'bn2', 'act2']
#               'global_pool', 
#               'classifier']
    
    blocks_list = ['blocks.0', 'blocks.1', 'blocks.2', 'blocks.3', 'blocks.4', 'blocks.5', 'blocks.6']

    def __init__(self, 
                 model_name: str,
                 pretrained: Union[str, bool]=False,
#                  progress: bool=True,
                 drop_rate: float = 0.0,
                 **kwargs) -> nn.Module:
        super().__init__()
        
        backbone = timm.create_model(model_name,
                                     pretrained=pretrained)

        
#         self.act_layer = backbone.act_layer
#         self.norm_layer = backbone.norm_layer
#         self.se_layer = backbone.se_layer
#         if not fix_stem:
#             stem_size = round_chs_fn(stem_size)
        self.conv_stem = backbone.conv_stem
        self.bn1 = backbone.bn1
        self.act1 = backbone.act1
        self.blocks = backbone.blocks
        self.feature_info = backbone.feature_info
        
        self.conv_head = backbone.conv_head
        self.bn2 = backbone.bn2
        self.act2 = backbone.act2

        self.model_name = model_name
        self.pretrained = pretrained
        self._out_features = backbone.num_features
        self.drop_rate = drop_rate


    def stem(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x
    
    def bottleneck(self, x):
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.bottleneck(x)
        return x
    
    @property
    def out_features(self) -> int:
        """The dimension of output features"""
        return self._out_features

            


class CustomEfficientNet(BaseModule):
    
    layers = ['backbone']
    
    def __init__(self, 
                 model_name: str,
                 pretrained: Union[str, bool]=False,
                 progress: bool=True,
                 num_classes: int=1000,
                 global_pool_type: str='avg',
                 drop_rate: float=0.0,
                 **kwargs) -> nn.Module: # *args, **kwargs):
        
        super().__init__()
        BackboneModelFactory = EfficientNetBackbone
#         GlobalPoolFactory = AVAILABLE_GLOBAL_POOL_LAYERS[global_pool_type]
        
        assert model_name in AVAILABLE_MODELS, f'[ERROR] Please only choose from available models: {AVAILABLE_MODELS}'
#         assert global_pool_type in AVAILABLE_GLOBAL_POOL_LAYERS
        
        self.model_name = model_name
#         self.num_classes = num_classes
        self.pretrained = pretrained
        self.drop_rate = drop_rate
            
        self.backbone = BackboneModelFactory(model_name=self.model_name,
                                             pretrained=self.pretrained,
#                                              progress=progress,
                                             drop_rate=self.drop_rate)
        self.out_features = self.backbone.out_features
#         self.global_pool = GlobalPoolFactory(output_size=1)#self.out_features)
#         self.classifier = ClassifierHead(in_features=self.out_features,
#                                          num_classes=self.num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        return x





AVAILABLE_MODELS = ['efficientnet_b0',
                    'efficientnet_b1',
                    'efficientnet_b1_pruned',
                    'efficientnet_b2',
                    'efficientnet_b2_pruned',
                    'efficientnet_b2a',
                    'efficientnet_b3',
                    'efficientnet_b3_pruned',
                    'efficientnet_b3a',
                    'efficientnet_b4',
                    'efficientnet_b5',
                    'efficientnet_b6',
                    'efficientnet_b7',
                    'efficientnet_b8',
                    'efficientnet_cc_b0_4e',
                    'efficientnet_cc_b0_8e',
                    'efficientnet_cc_b1_8e',
                    'efficientnet_el',
                    'efficientnet_em',
                    'efficientnet_es',
                    'efficientnet_l2',
                    'efficientnet_lite0',
                    'efficientnet_lite1',
                    'efficientnet_lite2',
                    'efficientnet_lite3',
                    'efficientnet_lite4']



def build_model(model_name: str,
                pretrained: Union[str, bool]=False,
                progress: bool=True,
                num_classes: int=1000,
                global_pool_type: str='avg',
                **kwargs) -> nn.Module:
    assert model_name in AVAILABLE_MODELS, f'[ERROR] Please only choose from available models: {AVAILABLE_MODELS}'    
#     model_func = globals()[model_name]
    if pretrained == True:
        pretrained = "imagenet"

    return CustomEfficientNet(model_name=model_name,
                              pretrained=pretrained,
                              progress=progress,
                              num_classes=num_classes,
                              global_pool_type=global_pool_type,
                              **kwargs)
