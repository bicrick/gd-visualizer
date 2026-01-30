"""
ADAM (Adaptive Moment Estimation) optimizer for PyTorch.
"""

import torch
from torch.optim.optimizer import Optimizer


class AdamGD(Optimizer):
    """
    ADAM (Adaptive Moment Estimation) optimizer.
    
    Combines momentum and RMSprop for adaptive learning rates:
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad^2
        m_hat = m / (1 - beta1^t)
        v_hat = v / (1 - beta2^t)
        param = param - lr * m_hat / (sqrt(v_hat) + epsilon)
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 0.001)
        beta1: Exponential decay rate for first moment estimates (default: 0.9)
        beta2: Exponential decay rate for second moment estimates (default: 0.999)
        epsilon: Small constant for numerical stability (default: 1e-8)
    """
    
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {beta2}")
        if epsilon < 0.0:
            raise ValueError(f"Invalid epsilon value: {epsilon}")
        
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon)
        super(AdamGD, self).__init__(params, defaults)
    
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
            beta1 = group['beta1']
            beta2 = group['beta2']
            epsilon = group['epsilon']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                param_state = self.state[p]
                
                # Initialize state if needed
                if len(param_state) == 0:
                    param_state['step'] = 0
                    param_state['m'] = torch.zeros_like(p.data)  # First moment
                    param_state['v'] = torch.zeros_like(p.data)  # Second moment
                
                m = param_state['m']
                v = param_state['v']
                param_state['step'] += 1
                t = param_state['step']
                
                # Update biased first moment estimate: m = beta1 * m + (1 - beta1) * grad
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update biased second moment estimate: v = beta2 * v + (1 - beta2) * grad^2
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Compute bias-corrected first moment estimate
                m_hat = m / (1 - beta1 ** t)
                
                # Compute bias-corrected second moment estimate
                v_hat = v / (1 - beta2 ** t)
                
                # Update parameters: param = param - lr * m_hat / (sqrt(v_hat) + epsilon)
                p.addcdiv_(m_hat, v_hat.sqrt() + epsilon, value=-lr)
        
        return loss
    
    def __repr__(self):
        return (f"AdamGD(lr={self.defaults['lr']}, beta1={self.defaults['beta1']}, "
                f"beta2={self.defaults['beta2']}, epsilon={self.defaults['epsilon']})")
