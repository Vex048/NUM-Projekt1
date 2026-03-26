import torch.nn as nn
from src.models.base_classifier import BaseClassifier

class BaselineClassifier(BaseClassifier):
    """
    Prosty CNN jako baseLine, potem można dać tu np. XgBoost
    """
    def __init__(self, num_classes: int = 10, lr: float = 1e-3, class_names: list = None):
        super().__init__(num_classes=num_classes, lr=lr, class_names=class_names)
        

        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
