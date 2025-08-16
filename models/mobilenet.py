"""
MobileNet Models for Waste Classification
"""

import torch
import torch.nn as nn
import torchvision.models as models


class MobileNetV3Large(nn.Module):
    """MobileNetV3Large-based model for waste classification"""
    
    def __init__(self, num_classes=9, pretrained=True):
        super(MobileNetV3Large, self).__init__()
        
        # Load MobileNetV3Large
        self.backbone = models.mobilenet_v3_large(pretrained=pretrained)
        # Replace classifier head for our number of classes
        num_features = self.backbone.classifier[3].in_features
        self.head = nn.Linear(num_features, num_classes)

        # Remove the original classifier
        self.backbone.classifier = nn.Identity()

        # Initialize classifier
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.constant_(self.head.bias, 0)
    
    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
