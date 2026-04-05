import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from src.models.base_classifier import BaseClassifier
import torch

class ResNetClassifier(BaseClassifier):
    """
    Resnet18
    """
    def __init__(
        self, 
        num_classes: int = 7, 
        lr: float = 1e-3, 
        weight_decay: float = 1e-4,
        optimizer_name: str = 'adam',
        scheduler_name: str = 'cosine',
        pretrained: bool = True,
        class_names: list = None,
        class_weights: torch.Tensor = None
    ):
        super().__init__(
            num_classes=num_classes, 
            lr=lr, 
            weight_decay=weight_decay,
            optimizer_name=optimizer_name, 
            scheduler_name=scheduler_name,
            class_names=class_names,
            class_weights=class_weights
        )
        
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.model = resnet18(weights=weights)
        
        # Wymiana ostatniej warstwy na naszą klasę
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
