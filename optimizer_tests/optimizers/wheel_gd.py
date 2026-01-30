"""
Wheel Optimizer for PyTorch - models optimization as a wheel rolling down the loss landscape.

Unlike standard momentum, a rolling wheel has gyroscopic stability - it resists
turning when spinning fast. The gradient affects velocity indirectly through
angular momentum rather than directly.
"""

import torch
from torch.optim.optimizer import Optimizer


class WheelGD(Optimizer):
    """
    Wheel Optimizer - models optimization as a wheel rolling down the loss landscape.
    
    State per parameter:
        v: velocity vector (direction and speed of the wheel)
        L: angular momentum (scalar, how fast the wheel is spinning)
    
    The gradient is decomposed into:
        - Parallel component (along v): affects spin speed (L)
        - Perpendicular component (across v): tries to turn the wheel,
          but is resisted by L/I (gyroscopic resistance)
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 0.01)
        beta: Momentum decay factor (0-1), like standard momentum (default: 0.98)
        inertia: Moment of inertia. Higher = harder to accelerate, more turning resistance (default: 5.0)
        eps: Small constant for numerical stability (default: 1e-8)
    """
    
    def __init__(self, params, lr=0.01, beta=0.98, inertia=5.0, eps=1e-8):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if beta < 0.0 or beta > 1.0:
            raise ValueError(f"Invalid beta value: {beta}")
        if inertia <= 0.0:
            raise ValueError(f"Invalid inertia value: {inertia}")
        
        defaults = dict(lr=lr, beta=beta, inertia=inertia, eps=eps)
        super(WheelGD, self).__init__(params, defaults)
    
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
            beta = group['beta']
            I = group['inertia']
            eps = group['eps']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                param_state = self.state[p]
                
                # Initialize state if needed
                if len(param_state) == 0:
                    param_state['velocity'] = torch.zeros_like(p.data)
                    param_state['angular_momentum'] = 0.0
                
                v = param_state['velocity']
                L = param_state['angular_momentum']
                
                # Flatten for vector operations
                v_flat = v.view(-1)
                grad_flat = grad.view(-1)
                
                speed = torch.norm(v_flat).item()
                
                if speed > eps:
                    # Wheel is moving - decompose gradient into parallel and perpendicular components
                    v_hat = v_flat / speed
                    
                    # Parallel component: projection of gradient onto velocity direction
                    g_parallel_mag = torch.dot(grad_flat, v_hat).item()
                    g_parallel = g_parallel_mag * v_hat
                    
                    # Perpendicular component: what remains after removing parallel
                    g_perp = grad_flat - g_parallel
                    
                    # Update angular momentum from parallel component
                    # If gradient aligns with velocity, spin faster
                    # If gradient opposes velocity, spin slower
                    L = beta * L + g_parallel_mag
                    L = max(L, 0.0)  # L is non-negative
                    
                    # Update velocity direction with gyroscopic turn resistance
                    # Perpendicular gradient tries to turn the wheel toward it
                    # Resistance: I * (1 + L) combines baseline inertia with gyroscopic boost
                    gyro_resistance = I * (1 + L) + eps
                    direction_change = g_perp / gyro_resistance
                    
                    v_hat_new = v_hat + direction_change
                    v_hat_new_norm = torch.norm(v_hat_new).item()
                    if v_hat_new_norm > eps:
                        v_hat_new = v_hat_new / v_hat_new_norm
                    
                    # New speed from rolling constraint: v = L / I
                    speed_new = L / I
                    
                    # Combine direction and speed, reshape back to parameter shape
                    v_new = speed_new * v_hat_new
                    v.copy_(v_new.view_as(v))
                else:
                    # Cold start: apply gradient as initial push
                    # The wheel starts rolling from rest with a gentle push
                    grad_mag = torch.norm(grad_flat).item()
                    
                    if grad_mag > eps:
                        # Initial velocity from gradient, scaled by learning_rate
                        v_new = lr * grad_flat
                        v.copy_(v_new.view_as(v))
                        
                        # L follows from rolling constraint: L = I * speed
                        L = I * torch.norm(v_new).item()
                
                # Store updated angular momentum
                param_state['angular_momentum'] = L
                
                # Update parameters: move opposite to velocity direction
                p.add_(v, alpha=-lr)
        
        return loss
    
    def __repr__(self):
        return (f"WheelGD(lr={self.defaults['lr']}, beta={self.defaults['beta']}, "
                f"inertia={self.defaults['inertia']})")
