from config import Config
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Optional, Self

class ImagePairMatcher(nn.Module):

    def __init__(self, 
                 cnn_hidden_layers: list[int],
                 fc_hidden_layers: list[int],
                 cnn_dropout: float = 0.0, 
                 fc_dropout: float = 0.0):

        super().__init__()

        cnn_blocks = []
        cnn_hidden_layers = [3] + cnn_hidden_layers
        for layer_in, layer_out in zip(cnn_hidden_layers[:-1], cnn_hidden_layers[1:]):
            cnn_block = nn.Sequential(
                nn.Conv2d(in_channels=layer_in, out_channels = layer_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(num_features=layer_out),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(cnn_dropout)
            )
            cnn_blocks.append(cnn_block)
        self.cnn = nn.Sequential(*cnn_blocks)
        
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flat = nn.Flatten()

        fc_blocks = []
        fc_hidden_layers = [4 * cnn_hidden_layers[-1]] + fc_hidden_layers
        for layer_in, layer_out in zip(fc_hidden_layers[:-1], fc_hidden_layers[1:]):
            fc_block = nn.Sequential(
                nn.Linear(layer_in, layer_out),
                nn.ReLU(),
                nn.Dropout(fc_dropout)
            )
            fc_blocks.append(fc_block)
        self.fc = nn.Sequential(*fc_blocks)

        self.out = nn.Linear(fc_hidden_layers[-1], 1)

    def _embed(self, x: Tensor) -> Tensor:
        x = self.cnn(x)
        x = self.gap(x)
        return self.flat(x)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        features1 = self._embed(x1)
        features2 = self._embed(x2)
        difference = torch.abs(features1 - features2)
        hadamard = features1 * features2
        x = torch.cat([features1, features2, difference, hadamard], dim=-1)
        x = self.fc(x)
        return self.out(x).squeeze(-1)

    @torch.no_grad()
    def predict(self, x1: Tensor, x2: Tensor, threshold: float = 0.5) -> Tensor:
        return torch.sigmoid(self.forward(x1, x2)) >= threshold

    @staticmethod
    def from_config(config: Config) -> Self:
        model = ImagePairMatcher(cnn_hidden_layers=[config.CNN_HIDDEN_CHANNELS_1, config.CNN_HIDDEN_CHANNELS_2, config.CNN_OUT_CHANNELS],
                                 fc_hidden_layers=[config.FC_HIDDEN_1, config.FC_HIDDEN_2],
                                 cnn_dropout=config.CNN_DROPOUT,
                                 fc_dropout=config.FC_DROPOUT)
        return model
