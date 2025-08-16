"""
Models package for waste classification
"""

from .resnet34 import ResNet34
from .efficientnet import EfficientNetB0
from .factory import create_model

__all__ = ['ResNet34', 'EfficientNetB0', 'create_model']
