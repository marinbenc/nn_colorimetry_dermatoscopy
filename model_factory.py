import torchvision.models as models
from coral_pytorch.layers import CoralLayer
from torchvision.models.vgg import VGG11_BN_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
import torch.nn as nn

def get_vgg11_bn_coral(num_classes=6, weights=VGG11_BN_Weights.IMAGENET1K_V1):
    model = models.vgg11_bn(weights=weights)
    model.classifier[6] = CoralLayer(4096, num_classes)
    return model

def get_vgg11_bn_classification(num_classes=6, weights=VGG11_BN_Weights.IMAGENET1K_V1):
    model = models.vgg11_bn(weights=weights)
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model

def get_efficientnet_b0_coral(num_classes=6, weights=EfficientNet_B0_Weights.IMAGENET1K_V1):
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = CoralLayer(in_features, num_classes)
    return model

def get_efficientnet_b0_classification(num_classes=6, weights=EfficientNet_B0_Weights.IMAGENET1K_V1):
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

def get_efficientnet_b4_coral(num_classes=6, weights=EfficientNet_B4_Weights.IMAGENET1K_V1):
    model = efficientnet_b4(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = CoralLayer(in_features, num_classes)
    return model

def get_efficientnet_b4_classification(num_classes=6, weights=EfficientNet_B4_Weights.IMAGENET1K_V1):
    model = efficientnet_b4(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

MODEL_FACTORY = {
    'vgg11_bn_coral': get_vgg11_bn_coral,
    'vgg11_bn_classification': get_vgg11_bn_classification,
    'efficientnet_b0_coral': get_efficientnet_b0_coral,
    'efficientnet_b0_classification': get_efficientnet_b0_classification,
    'efficientnet_b4_coral': get_efficientnet_b4_coral,
    'efficientnet_b4_classification': get_efficientnet_b4_classification,
}

def get_model_from_string(model_name, num_classes=6, weights=None):
    if model_name not in MODEL_FACTORY:
        raise ValueError(f"Unknown model name: {model_name}")
    # If weights is None, use default weights for each model
    if weights is None:
        return MODEL_FACTORY[model_name](num_classes=num_classes)
    return MODEL_FACTORY[model_name](num_classes=num_classes, weights=weights)

