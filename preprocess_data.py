import warnings
from numpy.exceptions import VisibleDeprecationWarning
warnings.filterwarnings('ignore', category=VisibleDeprecationWarning)

import numpy as np
from numpy.typing import NDArray
import os
import random
from transform import get_transform
import torch
from torchvision.datasets import CIFAR100
from torchvision.transforms.v2 import Transform
from tqdm import tqdm
from typing import Optional

def _render(cifar_obj: CIFAR100, 
            cifar_start: int, 
            cifar_end: int, 
            random_state: Optional[int] = None,
            show_progress: bool = True) -> tuple[NDArray, NDArray, NDArray]:

    if cifar_end - cifar_start < 2:
        raise Exception('Size of dataset (cifar_end - cifar_start) must be at least two to allow for non-matching pairs')

    rng = random.Random() if random_state is None else random.Random(random_state)

    length = cifar_end - cifar_start
    half_length = length // 2
    mask = [True for _ in range(half_length)] + [False for _ in range(length - half_length)]
    rng.shuffle(mask)

    firsts = []
    seconds = []
    targets = []

    for i in tqdm(range(cifar_start, cifar_end), disable=not show_progress):
        j = i
        while not mask[i - cifar_start] and j == i:
            j = rng.randrange(cifar_start, cifar_end)
        first_image = cifar_obj[i][0]
        assert isinstance(first_image, torch.Tensor)
        second_image = cifar_obj[j][0]
        assert isinstance(second_image, torch.Tensor)
        firsts.append(first_image)
        seconds.append(second_image)
        targets.append(1.0 if mask[i - cifar_start] else 0.0)

    return np.stack(firsts), np.stack(seconds), np.stack(targets)

if __name__ == '__main__':

    print('Loading data...')
    transform = get_transform()
    train_and_validate = CIFAR100(root='data', train=True, download=True, transform=transform)
    test = CIFAR100(root='data', train=False, download=True, transform=transform)
    print('Complete.')
    
    data = {}

    print('\nRendering training data...')
    data['train_firsts'], data['train_seconds'], data['train_targets'] = _render(train_and_validate, 0, 40000, random_state=0)
    print('Complete')
    
    print('\nRendering validation data...')
    data['validate_firsts'], data['validate_seconds'], data['validate_targets'] = _render(train_and_validate, 40000, 50000, random_state=1)
    print('Complete.')

    print('\nRendering testing data...')
    data['test_firsts'], data['test_seconds'], data['test_targets'] = _render(test, 0, len(test), random_state=2)
    print('Complete.')

    print('\nSaving data...')
    os.makedirs('data/preprocessed', exist_ok=True)
    for name in data:
        array = data[name]
        np.save(f'data/preprocessed/{name}.npy', array, allow_pickle=False)
    print('Complete.')

    # Debug
    name = 'train_firsts'
    array = data[name]
    print()
    print(name)
    print(type(array))
    print(array.dtype)
    print(array.shape)
    print()
    print(array)