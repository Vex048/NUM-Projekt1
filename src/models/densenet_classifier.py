import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights
from src.models.base_classifier import BaseClassifier

class DenseNetClassifier(BaseClassifier):
    """
    DenseNet-121 Classifier
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
        
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        self.model = densenet121(weights=weights)
        
        # Wymiana ostatniej warstwy na klasyfikator dla naszych klas
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_ftrs, num_classes)
