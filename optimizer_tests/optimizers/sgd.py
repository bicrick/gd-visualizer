"""
Stochastic Gradient Descent optimizer for PyTorch.

This implementation simulates the stochastic nature of mini-batch gradient descent
by adding controlled noise to gradients, mimicking the variance that comes from
using random mini-batches instead of the full dataset.
"""

import torch
from torch.optim.optimizer import Optimizer


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer with gradient noise simulation.
    
    Unlike batch gradient descent which computes exact gradients over the full
    dataset, SGD approximates gradients using mini-batches. This implementation
    simulates that stochastic behavior by:
    1. Adding gradient noise (perpendicular and magnitude components)
    2. Using a step multiplier for faster convergence
    3. Gradually reducing noise over time (simulating learning rate annealing)
    
    The visualization shows SGD "racing ahead" while bouncing around, representing
    the real-world trade-off: faster convergence but less stable path.
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 0.01)
        step_multiplier: Multiplier for effective step size (default: 3.0)
                        Higher values = SGD moves faster
        noise_scale: Initial standard deviation of gradient noise (default: 0.8)
                    Higher values = more "bouncing" behavior
        noise_decay: Multiplicative decay factor per iteration (default: 0.995)
                    Gradually reduces noise, allowing SGD to settle
    """
    
    def __init__(self, params, lr=0.01, step_multiplier=3.0, noise_scale=0.8, noise_decay=0.995):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if step_multiplier < 0.0:
            raise ValueError(f"Invalid step_multiplier: {step_multiplier}")
        if noise_scale < 0.0:
            raise ValueError(f"Invalid noise_scale: {noise_scale}")
        if noise_decay < 0.0 or noise_decay > 1.0:
            raise ValueError(f"Invalid noise_decay: {noise_decay}")
        
        defaults = dict(
            lr=lr,
            step_multiplier=step_multiplier,
            noise_scale=noise_scale,
            noise_decay=noise_decay
        )
        super(SGD, self).__init__(params, defaults)
        
        # Initialize global noise scale
        self.current_noise_scale = noise_scale
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step with gradient noise.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
            
        Returns:
            Loss value if closure is provided, otherwise None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            step_multiplier = group['step_multiplier']
            noise_decay = group['noise_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Compute gradient magnitude
                grad_norm = grad.norm()
                
                if grad_norm > 1e-8:
                    # Normalize gradient
                    grad_normalized = grad / grad_norm
                    
                    # Create perpendicular direction for sideways bouncing
                    # For multi-dimensional tensors, we add noise to each element
                    perpendicular_noise = torch.randn_like(grad) * self.current_noise_scale * 0.7
                    
                    # Magnitude noise (along gradient direction)
                    magnitude_noise = torch.randn_like(grad) * self.current_noise_scale * 0.3
                    
                    # Combined noisy gradient
                    noisy_grad = grad + perpendicular_noise + magnitude_noise
                else:
                    # If gradient is too small, just use it as is
                    noisy_grad = grad
                
                # Effective learning rate with step multiplier
                effective_lr = lr * step_multiplier
                
                # Update parameters with noisy gradient and larger step
                p.add_(noisy_grad, alpha=-effective_lr)
        
        # Decay noise for next iteration (allows SGD to settle over time)
        self.current_noise_scale *= noise_decay
        
        return loss
    
    def __repr__(self):
        return (f"SGD(lr={self.defaults['lr']}, step_multiplier={self.defaults['step_multiplier']}, "
                f"noise_scale={self.defaults['noise_scale']}, noise_decay={self.defaults['noise_decay']})")
