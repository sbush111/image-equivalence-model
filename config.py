from dataclasses import dataclass
from random import Random
from typing import Optional, Self, Sequence

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
    NUM_EPOCHS: int = 20
    
    LEARNING_RATE: float = 0.001
    WEIGHT_DECAY: float = 0.0
    
    PATIENCE: Optional[int] = None
    DELTA: Optional[float] = None
    
    @classmethod
    def generate_randomized(cls, param_grid: dict[str, Sequence], random_state: Optional[int] = None) -> Self:
        config = cls()
        rng = Random(random_state)
        for hyperparameter, options in param_grid.items():
            if hyperparameter not in list(filter(lambda s: s[0] != '_', dir(config))):
                raise ValueError(f'{hyperparameter} is not a valid hyperparameter')
            setattr(config, hyperparameter, rng.choice(options))
        return config

if __name__ == '__main__':
    config = Config()
    print(config)