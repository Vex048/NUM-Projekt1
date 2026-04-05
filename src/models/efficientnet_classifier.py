import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from src.models.base_classifier import BaseClassifier

class EfficientNetClassifier(BaseClassifier):
    """
    EfficientNet-B0 Classifier
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
        class_weights: torch.Tensor = None,
        freeze_backbone: bool = False
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
        
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.model = efficientnet_b0(weights=weights)
        
        # Zamrożenie głównej części modelu, by uniknąć przeuczenia
        if freeze_backbone and pretrained:
            for param in self.model.features.parameters():
                param.requires_grad = False
                
        # Wymiana ostatniej warstwy i podbicie dropoutu
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[0] = nn.Dropout(p=0.5, inplace=True)
        self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)
