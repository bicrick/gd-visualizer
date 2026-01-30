"""
Momentum Gradient Descent optimizer for PyTorch.
"""

import torch
from torch.optim.optimizer import Optimizer


class MomentumGD(Optimizer):
    """
    Gradient Descent with Momentum optimizer.
    
    Implements momentum-based gradient descent:
        v = momentum * v + grad
        param = param - lr * v
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 0.01)
        momentum: Momentum coefficient (default: 0.9)
    """
    
    def __init__(self, params, lr=0.01, momentum=0.9):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum > 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        
        defaults = dict(lr=lr, momentum=momentum)
        super(MomentumGD, self).__init__(params, defaults)
    
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
            momentum = group['momentum']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                param_state = self.state[p]
                
                # Initialize velocity buffer if needed
                if 'velocity' not in param_state:
                    param_state['velocity'] = torch.zeros_like(p.data)
                
                velocity = param_state['velocity']
                
                # Update velocity: v = momentum * v + grad
                velocity.mul_(momentum).add_(p.grad)
                
                # Update parameters: param = param - lr * v
                p.add_(velocity, alpha=-lr)
        
        return loss
    
    def __repr__(self):
        return f"MomentumGD(lr={self.defaults['lr']}, momentum={self.defaults['momentum']})"
