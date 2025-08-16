"""
Model factory for creating different waste classification models
"""

from .resnet34 import ResNet34
from .efficientnet import EfficientNetB0


def create_model(model_config):
    """Create a model based on configuration
    
    Args:
        model_config (dict): Configuration dictionary containing:
            - model_type (str): 'resnet34' or 'efficientnet_b0'
            - num_classes (int): Number of output classes
            - pretrained (bool): Whether to use pretrained weights
            - freeze_backbone (bool): Whether to freeze backbone parameters
    """
    model_type = model_config.get('model_type', 'resnet34').lower()
    
    if model_type == 'resnet34':
        model = ResNet34(
            num_classes=model_config['num_classes'],
            pretrained=model_config['pretrained']
        )
    elif model_type == 'efficientnet_b0':
        model = EfficientNetB0(
            num_classes=model_config['num_classes'],
            pretrained=model_config['pretrained']
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types: 'resnet34', 'efficientnet_b0'")
    
    if model_config.get('freeze_backbone', False):
        model.freeze_backbone()
    
    return model
