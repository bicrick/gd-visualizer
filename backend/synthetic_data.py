"""
Generate synthetic training data for gradient descent visualization.
"""

import numpy as np


def generate_synthetic_dataset(n_samples=100, seed=42):
    """
    Generate synthetic 2D data points for training.
    These are used to compute gradients in batch vs stochastic settings.
    
    Args:
        n_samples: Number of data points to generate
        seed: Random seed for reproducibility
    
    Returns:
        Array of shape (n_samples, 2) containing (x, y) coordinates
    """
    np.random.seed(seed)
    
    # Generate points around multiple centers to create interesting gradient patterns
    centers = [
        (-2, -2),
        (2, -2),
        (-2, 2),
        (2, 2),
        (0, 0),
    ]
    
    data = []
    samples_per_center = n_samples // len(centers)
    
    for cx, cy in centers:
        x = np.random.normal(cx, 0.8, samples_per_center)
        y = np.random.normal(cy, 0.8, samples_per_center)
        data.append(np.column_stack([x, y]))
    
    # Add remaining samples randomly
    remaining = n_samples - len(centers) * samples_per_center
    if remaining > 0:
        x = np.random.uniform(-4, 4, remaining)
        y = np.random.uniform(-4, 4, remaining)
        data.append(np.column_stack([x, y]))
    
    return np.vstack(data)


def compute_loss_on_data(loss_func, data_point):
    """
    Compute loss for a single data point.
    This simulates the loss for one training example.
    
    Args:
        loss_func: Loss function that takes (x, y) parameters
        data_point: Array of shape (2,) containing (x, y) coordinates
    
    Returns:
        Loss value
    """
    return loss_func(data_point[0], data_point[1])


def compute_gradient_on_data(loss_func, data_point, h=1e-5):
    """
    Compute gradient for a single data point.
    
    Args:
        loss_func: Loss function
        data_point: Array of shape (2,) containing (x, y) coordinates
        h: Step size for numerical differentiation
    
    Returns:
        Gradient array of shape (2,)
    """
    x, y = data_point[0], data_point[1]
    
    fx_plus = loss_func(x + h, y)
    fx_minus = loss_func(x - h, y)
    grad_x = (fx_plus - fx_minus) / (2 * h)
    
    fy_plus = loss_func(x, y + h)
    fy_minus = loss_func(x, y - h)
    grad_y = (fy_plus - fy_minus) / (2 * h)
    
    return np.array([grad_x, grad_y])

