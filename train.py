from dataclasses import dataclass
import torch
from torch.nn import Module
from torch import device as Device
from torch.nn.modules.loss import _Loss as Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional

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

    iterator = range(num_epochs)
    if progress:
        iterator = tqdm(iterator)
    
    for epoch in iterator:

        train_loss = 0.0
        model.train()

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            batch_size = inputs.size(0)
            train_loss += loss.item() * batch_size

        train_losses.append(train_loss / len(train_loader.dataset))

        if validate_loader is None:
            continue

        validate_loss = 0.0
        model.eval()

        with torch.no_grad():

            for inputs, labels in validate_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                batch_size = inputs.size(0)
                validate_loss += loss.item() * batch_size

        validate_losses.append(validate_loss / len(validate_loader.dataset))

    return TrainResults(train_losses, validate_losses)

@dataclass
class TestResults:
    test_loss: float

def test(model: Module, 
         criterion: Loss, 
         test_loader: DataLoader, 
         device: Optional[Device] = None) -> TestResults:

    if device is None:
        device = Device('cpu')

    model = model.to(device)

    test_loss = 0.0

    model.eval()
    
    with torch.no_grad():

        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            batch_size = inputs.size(0)
            test_loss += loss.item() * batch_size

    return TestResults(test_loss / len(test_loader.dataset))