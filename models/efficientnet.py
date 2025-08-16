"""
EfficientNet Models for Waste Classification
"""

import torch
import torch.nn as nn
import torchvision.models as models


class EfficientNetB0(nn.Module):
    """EfficientNetB0-based model for waste classification"""
    
    def __init__(self, num_classes=9, pretrained=True):
        super(EfficientNetB0, self).__init__()
        
        # Load EfficientNetB0
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        # Replace classifier head for our number of classes
        num_features = self.backbone.classifier[1].in_features
        self.head = nn.Linear(num_features, num_classes)

        # Remove the original classifier
        self.backbone.classifier = nn.Identity()

        # Initialize classifier for training from scratch
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.constant_(self.head.bias, 0)
    
    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
