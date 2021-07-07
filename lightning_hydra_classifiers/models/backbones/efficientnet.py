"""
lightning_hydra_classifiers/models/backbones/efficientnet.py

Author: Jacob A Rose
Created: Thursday Junee 11th, 2021

"""




import timm
from torch import nn




efficientnet_layers = ['conv1', 'bn1', 'blocks.0' 'blocks.1', 'blocks.2', 'blocks.3', 'blocks.4', 'blocks.5', 'blocks.6', 'conv_head', 'bn2', 'classifier']

class CustomEfficientNet(nn.Module):
    def __init__(self, model_name='tf_efficientNet_b0_ns', pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.model.get_classifier().in_features
        self.model.classifier = nn.Linear(in_features, CFG.num_classes)

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