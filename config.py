from dataclasses import dataclass
from typing import Any, Optional, Self

@dataclass
class Config:

    # Model Architecture
    CNN_HIDDEN_CHANNELS_1: int = 16
    CNN_HIDDEN_CHANNELS_2: int = 32
    CNN_OUT_CHANNELS: int = 64
    CNN_DROPOUT: float = 0.0

    FC_HIDDEN_1: int = 32
    FC_HIDDEN_2: int = 16
    FC_DROPOUT: float = 0.0
    
    # Training
    LEARNING_RATE: float = 0.001

    # TODO ...
    
    @staticmethod
    def generate_config(param_grid: dict[str, Any]) -> Self:
        config = Config()
        # TODO ...
        return config

if __name__ == '__main__':
    config = Config()
    print(config)