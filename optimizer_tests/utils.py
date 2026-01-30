"""
Utility functions for CIFAR-10 training and evaluation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np


def get_device():
    """
    Get the best available device (MPS for Apple Silicon, CUDA, or CPU).
    
    Returns:
        torch.device: Device to use for computation
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device


def get_cifar10_loaders(batch_size=64, data_dir='./data', num_workers=2):
    """
    Create CIFAR-10 data loaders with standard normalization.
    
    Args:
        batch_size: Batch size for DataLoader
        data_dir: Directory to store/load CIFAR-10 data
        num_workers: Number of worker processes for data loading
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    # CIFAR-10 mean and std for normalization
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    
    # Training transforms (with data augmentation)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Test transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Download and load datasets
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, test_loader


def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """
    Train the model for one epoch.
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        optimizer: Optimizer instance
        criterion: Loss function
        device: Device to use for computation
        epoch: Current epoch number
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Track metrics
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def evaluate_model(model, test_loader, criterion, device):
    """
    Evaluate the model on test data.
    
    Args:
        model: Neural network model
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to use for computation
        
    Returns:
        tuple: (average_loss, accuracy, per_class_accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # For per-class accuracy
    class_correct = np.zeros(10)
    class_total = np.zeros(10)
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Evaluating"):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Track metrics
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Per-class accuracy
            for i in range(len(target)):
                label = target[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
    
    avg_loss = running_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    # Calculate per-class accuracy
    per_class_acc = {}
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    for i, name in enumerate(class_names):
        if class_total[i] > 0:
            per_class_acc[name] = 100. * class_correct[i] / class_total[i]
        else:
            per_class_acc[name] = 0.0
    
    return avg_loss, accuracy, per_class_acc


def save_checkpoint(model, optimizer, epoch, loss, accuracy, filename):
    """
    Save model checkpoint.
    
    Args:
        model: Neural network model
        optimizer: Optimizer instance
        epoch: Current epoch number
        loss: Current loss value
        accuracy: Current accuracy value
        filename: Filename to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(model, optimizer, filename, device):
    """
    Load model checkpoint.
    
    Args:
        model: Neural network model
        optimizer: Optimizer instance (can be None)
        filename: Filename to load checkpoint from
        device: Device to load the model on
        
    Returns:
        tuple: (epoch, loss, accuracy)
    """
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    accuracy = checkpoint['accuracy']
    
    print(f"Checkpoint loaded from {filename}")
    print(f"Epoch: {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    return epoch, loss, accuracy
