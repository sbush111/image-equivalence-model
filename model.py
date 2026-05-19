from config import Config
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Self

class ImagePairMatcher(nn.Module):

    def __init__(self, config: Config):

        super().__init__()

        self.cnn_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1), # 3 x 32 x 32 -> 16 x 32 x 32
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 16 x 32 x 32 -> 16 x 16 x 16
        )

        self.cnn_block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1), # 16 x 16 x 16 -> 32 x 16 x 16
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 32 x 8 x 8
            nn.Dropout2d(0.2)
        )

        self.cnn_block3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), #32 x 8 x 8 -> 64 x 8 x 8
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # 64 x 8 x 8 -> 64 x 4 x 4
        )

        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1)) # 64 x 4 x 4 -> 64 x 1 x 1
        self.flat = nn.Flatten() # 64 x 1 x 1 -> 64

        self.fc1 = nn.Sequential(
            nn.Linear(4 * 64, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.out = nn.Linear(16, 1)

    def _embed(self, x: Tensor) -> Tensor:
        x = self.cnn_block1(x)
        x = self.cnn_block2(x)
        x = self.cnn_block3(x)
        x = self.gap(x)
        return self.flat(x)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        features1 = self._embed(x1)
        features2 = self._embed(x2)
        difference = features1 - features2
        difference = torch.abs(difference)
        hadamard = features1 * features2
        x = torch.concat([features1, features2, difference, hadamard], dim=-1)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.out(x)

    def predict(self, x1: Tensor, x2: Tensor, threshold: float = 0.5) -> bool:
        return F.sigmoid(self.forward(x1, x2)) >= threshold

    @staticmethod
    def from_config(config: Config) -> Self:
        model = ImagePairMatcher(config) # TODO...
        return model

'''
TODO:
Turn magic values into variables
Populate from_config with code to set variables based on config values
'''