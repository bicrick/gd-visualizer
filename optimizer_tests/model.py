"""
Simple CNN architecture for CIFAR-10 classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Simple CNN for CIFAR-10 classification.
    
    Architecture:
    - Conv1: 3 -> 32 channels, 3x3 kernel
    - ReLU + MaxPool (2x2)
    - Conv2: 32 -> 64 channels, 3x3 kernel
    - ReLU + MaxPool (2x2)
    - Flatten
    - FC1: -> 128 units
    - ReLU
    - FC2: -> 10 classes (CIFAR-10)
    
    ~50K parameters for quick iteration.
    """
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        # After two 2x2 poolings: 32x32 -> 16x16 -> 8x8
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Conv block 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)  # 32x32 -> 16x16
        
        # Conv block 2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)  # 16x16 -> 8x8
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x
    
    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(num_classes=10):
    """
    Create and return a SimpleCNN model.
    
    Args:
        num_classes: Number of output classes (default: 10 for CIFAR-10)
        
    Returns:
        SimpleCNN model instance
    """
    model = SimpleCNN(num_classes=num_classes)
    print(f"Model created with {model.count_parameters():,} parameters")
    return model


if __name__ == "__main__":
    # Test the model
    model = create_model()
    
    # Create dummy input
    dummy_input = torch.randn(4, 3, 32, 32)  # batch_size=4
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model architecture:\n{model}")
