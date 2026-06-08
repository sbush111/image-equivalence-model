from config import Config
from contextlib import nullcontext
from dataclasses import dataclass
import torch
from torch.nn import Module
from torch import device as Device
from torch.nn.modules.loss import _Loss as Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Literal, Optional, Self

class EarlyStopper:
    
    def __init__(self, patience: int, delta: float):
        self.patience = patience
        self.countdown = patience
        self.delta = delta
        self.best = float('inf')

    def consider_stop(self, validation: float) -> bool:
        if self.best - validation > self.delta:
            self.countdown = self.patience
            self.best = validation
        else:
            self.countdown -= 1
        return self.countdown <= 0

    def __str__(self) -> str:
        out = []
        out.append('EarlyStopper')
        out.append('============')
        out.append(f'patience={self.patience}')
        out.append(f'delta={round(self.patience, 3)}')
        out.append('------------')
        out.append(f'current_countdown={self.countdown}')
        out.append(f'current_best_val={self.best}')
        return '\n'.join(out)

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def from_config(config: Config) -> Self | None:
        if (config.PATIENCE is None) != (config.DELTA is None):
            raise ValueError('Cannot determine early stopping criteria. PATIENCE and DELTA must both be None or neither be None.')
        if config.PATIENCE is None:
            return None
        return EarlyStopper(config.PATIENCE, config.DELTA)
        

@dataclass
class TrainResults:
    train_losses: list[float]
    train_accuracies: list[float]
    validate_losses: list[float]
    validate_accuracies: list[float]
    
def train(model: Module, 
          criterion: Loss, 
          optimizer: Optimizer,
          num_epochs: int,
          train_loader: DataLoader,  
          validate_loader: DataLoader,
          device: Device,
          early_stop: Optional[EarlyStopper] = None) -> TrainResults:

    model = model.to(device)

    train_losses = []
    train_accuracies = []
    validate_losses = []
    validate_accuracies = []
    best_validate_loss = float('inf')

    batches_per_epoch = len(train_loader) + len(validate_loader)
    with tqdm(desc='train batches', total=num_epochs*batches_per_epoch) as progress_bar:

        for _ in range(num_epochs):

            train_loss, train_accuracy = _run_epoch('train', model, criterion, train_loader, device, progress_bar, optimizer)
            validate_loss, validate_accuracy = _run_epoch('eval', model, criterion, validate_loader, device, progress_bar)
            
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            validate_losses.append(validate_loss)
            validate_accuracies.append(validate_accuracy)

            if early_stop and early_stop.consider_stop(validate_loss):
                break
                
    return TrainResults(train_losses, train_accuracies, validate_losses, validate_accuracies)


@dataclass
class TestResults:
    test_loss: float
    test_accuracy: float

def test(model: Module, 
         criterion: Loss, 
         test_loader: DataLoader, 
         device: Optional[Device] = None) -> TestResults:

    if device is None:
        device = Device('cpu')

    model = model.to(device)

    with tqdm(desc='test batches', total=len(test_loader)) as progress_bar:
        test_loss, test_accuracy = _run_epoch('eval', model, criterion, test_loader, device, progress_bar)
        
    return TestResults(test_loss, test_accuracy)


def _run_epoch(mode: Literal['train', 'eval'], 
               model: Module, 
               criterion: Loss, 
               loader: DataLoader, 
               device: Device,
               progress_bar: tqdm,
               optimizer: Optional[Optimizer] = None) -> tuple[float, float]:

    if mode not in ['train', 'eval']:
        raise ValueError('"mode" parameter must be either "train" or "eval"')
    
    if mode == 'train' and optimizer is None:
        raise ValueError('Optimizer must be provided in train mode')

    if mode == 'train':
        model.train()

    if mode == 'eval':
        model.eval()

    running_loss = 0.0
    correct = 0

    with torch.inference_mode() if mode == 'eval' else nullcontext():

        for firsts, lasts, labels in loader:
            if mode == 'train':
                optimizer.zero_grad()
            firsts, lasts, labels = firsts.to(device), lasts.to(device), labels.to(device)
            outputs = model(firsts, lasts)
            loss = criterion(outputs, labels)
            correct += ((outputs >= 0.5) == labels).sum().item()
            batch_size = firsts.size(0)
            running_loss += loss.item() * batch_size
            progress_bar.update()
            if mode == 'eval':
                continue
            loss.backward()
            optimizer.step()

    return running_loss / len(loader.dataset), float(correct) / len(loader.dataset)