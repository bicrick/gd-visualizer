"""
Batch Gradient Descent optimizer for PyTorch.
"""

import torch
from torch.optim.optimizer import Optimizer


class BatchGD(Optimizer):
    """
    Batch Gradient Descent optimizer.
    
    Implements vanilla gradient descent with full batch updates:
        param = param - lr * grad
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 0.01)
    """
    
    def __init__(self, params, lr=0.01):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        defaults = dict(lr=lr)
        super(BatchGD, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.
        
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
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Gradient descent update
                p.add_(p.grad, alpha=-lr)
        
        return loss
    
    def __repr__(self):
        return f"BatchGD(lr={self.defaults['lr']})"
