"""
lightning_hydra_classifiers/models/backbones/efficientnet.py

Author: Jacob A Rose
Created: Thursday Junee 11th, 2021

"""




import timm
from torch import nn
from typing import Union
from .. import BaseModule





efficientnet_layers = ['conv1', 'bn1', 'blocks.0' 'blocks.1', 'blocks.2', 'blocks.3', 'blocks.4', 'blocks.5', 'blocks.6', 'conv_head', 'bn2', 'classifier']

class CustomEfficientNet(BaseModule):#, timm.models.efficientnet.EfficientNet):
    def __init__(self, model_name='tf_efficientNet_b0_ns', num_classes=1000, pretrained=True, progress: bool=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.model.get_classifier().in_features
        self.model.classifier = nn.Linear(in_features, num_classes)
#         self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x
    
    def unfreeze_at(self, layer: str):
        assert layer in efficientnet_layers
        self.model.requires_grad = True        
        for name, param in model.named_parameters():
            if layer in name:
                break
            param.requires_grad = False




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
                **kwargs) -> nn.Module:
    assert model_name in AVAILABLE_MODELS, f'[ERROR] Please only choose from available models: {AVAILABLE_MODELS}'    
#     model_func = globals()[model_name]
    if pretrained == True:
        pretrained = "imagenet"

    return CustomEfficientNet(model_name=model_name, pretrained=pretrained, progress=progress, num_classes=num_classes, **kwargs)
    