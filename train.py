from contextlib import nullcontext
from dataclasses import dataclass
import torch
from torch.nn import Module
from torch import device as Device
from torch.nn.modules.loss import _Loss as Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Literal, Optional


@dataclass
class TrainResults:
    train_losses: list[float]
    validate_losses: Optional[list[float]]
    
def train(model: Module, 
          criterion: Loss, 
          optimizer: Optimizer,
          num_epochs: int,
          train_loader: DataLoader,  
          validate_loader: Optional[DataLoader] = None,
          device: Optional[Device] = None,
          progress: bool = False) -> TrainResults:

    if device is None:
        device = Device('cpu')

    model = model.to(device)

    train_losses = []
    validate_losses = [] if validate_loader is not None else None

    num_batches = len(train_loader)
    if validate_loader is not None:
        num_batches += len(validate_loader)

    with (tqdm(desc='train batches', total=num_epochs*num_batches) if progress else nullcontext()) as progress_bar:

        for _ in range(num_epochs):

            train_loss = _run_epoch('train', model, criterion, train_loader, device, optimizer, progress_bar=(progress_bar if progress else None))
            train_losses.append(train_loss)
    
            if validate_loader is None:
                continue
    
            validate_loss = _run_epoch('eval', model, criterion, validate_loader, device, progress_bar=(progress_bar if progress else None))
            validate_losses.append(validate_loss)

    return TrainResults(train_losses, validate_losses)


@dataclass
class TestResults:
    test_loss: float

def test(model: Module, 
         criterion: Loss, 
         test_loader: DataLoader, 
         device: Optional[Device] = None,
         progress: bool = False) -> TestResults:

    if device is None:
        device = Device('cpu')

    model = model.to(device)

    with (tqdm(desc='test batches', total=len(test_loader)) if progress else nullcontext()) as progress_bar:
        test_loss = _run_epoch('eval', model, criterion, test_loader, device, progress_bar=(progress_bar if progress else None))
        
    return TestResults(test_loss)


def _run_epoch(mode: Literal['train', 'eval'], 
               model: Module, 
               criterion: Loss, 
               loader: DataLoader, 
               device: Device, 
               optimizer: Optional[Optimizer] = None,
               progress_bar: Optional[tqdm] = None) -> float:

    if mode not in ['train', 'eval']:
        raise ValueError('"mode" parameter must be either "train" or "eval"')
    
    if mode == 'train' and optimizer is None:
        raise ValueError('Optimizer must be provided in train mode')

    if mode == 'train':
        model.train()

    if mode == 'eval':
        model.eval()

    running_loss = 0.0

    with torch.no_grad() if mode == 'eval' else nullcontext():

        for firsts, lasts, labels in loader:
            if mode == 'train':
                optimizer.zero_grad()
            firsts, lasts, labels = firsts.to(device), lasts.to(device), labels.to(device)
            outputs = model(firsts, lasts)
            loss = criterion(outputs, labels)
            batch_size = firsts.size(0)
            running_loss += loss.item() * batch_size
            if progress_bar is not None:
                progress_bar.update()
            if mode == 'eval':
                continue
            loss.backward()
            optimizer.step()

    return running_loss / len(loader.dataset)