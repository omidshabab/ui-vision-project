import torch
import torch.nn as nn

class UIDetectionNet(nn.Module):
    def __init__(self, output_size=500):  # Match with max_features
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)  # Output size matches max_features
        )
    
    def forward(self, x):
        features = self.backbone(x)
        classifications = self.classifier(features)
        return classifications
        features = self.backbone(x)
        classifications = self.classifier(features)
        return classifications