"""
Simple ResNet34 Model for Waste Classification
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet34(nn.Module):
    """ResNet34-based model for waste classification"""
    
    def __init__(self, num_classes=9, pretrained=True):
        super(ResNet34, self).__init__()
        
        # Load ResNet34
        self.backbone = models.resnet34(pretrained=pretrained)
        # Replace classifier head for our number of classes
        num_features = self.backbone.fc.in_features
        self.head = nn.Linear(num_features, num_classes)

        # we don't need the classifier head
        self.backbone.fc = nn.Identity()

        # Initialize classifier for training from scratch
        if not pretrained:
            nn.init.xavier_uniform_(self.head.weight)
            nn.init.constant_(self.head.bias, 0)
    
    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False


def create_model(model_config):
    model = ResNet34(
        num_classes=model_config['num_classes'],
        pretrained=model_config['pretrained']
    )
    if model_config['freeze_backbone']:
        model.freeze_backbone()
    return model
    
