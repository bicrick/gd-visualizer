# The Wheel Optimizer

## Intuition

Imagine a wheel rolling down a hill instead of a ball sliding, or a spinning coin in one of those coin funnel machines at museums.

When a ball slides, a sideways push immediately changes its direction. But a rolling wheel has **gyroscopic stability** — it resists turning when spinning fast, like a bicycle that's easy to balance at high speed but wobbly when slow.

This is because a rolling wheel has two kinds of momentum:
- **Translational momentum** — resistance to speeding up or slowing down
- **Angular momentum** — creates resistance to turning (gyroscopic stability)

Standard optimizers with momentum only have the first kind. The Wheel optimizer has both.

The key insight: **perpendicular forces are resisted by angular momentum**. Instead of turning sharply:

```
perpendicular gradient → tries to turn wheel
angular momentum L → resists the turn
result → smooth curves when spinning fast, tighter turns when slow
```

When the wheel is spinning fast (high L), it maintains its direction despite sideways gradients. As it loses spin (L decreases through momentum decay), it can turn more sharply, creating a spiraling trajectory that tightens as it settles, just like a coin spinning down a funnel.

## High Level Algorithm

**State:**
- θ — model parameters
- v — velocity vector (direction and speed of the wheel)
- L — angular momentum (scalar, how fast the wheel is spinning)

**Each step:**

1. Receive gradient g
2. Decompose g into parallel (along v) and perpendicular (across v) components
3. Parallel component adds to angular momentum L (speeds up or slows the spin)
4. Perpendicular component tries to turn the wheel **toward** the gradient direction
5. Turning is **resisted** by gyroscopic stability proportional to L × I
6. Higher L or I means stronger resistance, smoother curves
7. New speed is determined by L through the rolling constraint: `speed = L / I`
8. Update parameters

**The moment of inertia I controls:**
- How hard it is to get the wheel rolling (high I = slow to accelerate)
- How much the wheel resists turning (high I = more gyroscopic resistance = smoother curves)

Both effects come from the same mechanism — this is physically correct.

## Hyperparameters

| Name | Symbol | Default | Meaning |
|------|--------|---------|---------|
| Learning rate | `lr` | 0.001 | Step size |
| Momentum decay | `beta` | 0.95 | How quickly angular momentum decays |
| Moment of inertia | `I` | 1.0 | Resistance to acceleration and turning |

## Python Implementation (NumPy)

```python
import numpy as np

class WheelOptimizer:
    """
    Wheel Optimizer
    
    Models optimization as a wheel rolling down the loss landscape.
    The wheel has gyroscopic stability — it resists turning when spinning fast.
    
    Args:
        lr: Learning rate
        beta: Momentum decay factor (like standard momentum)
        I: Moment of inertia. Higher = harder to accelerate, more turning resistance
    """
    
    def __init__(self, lr=0.001, beta=0.95, I=1.0, eps=1e-8):
        self.lr = lr
        self.beta = beta
        self.I = I
        self.eps = eps
        
        # State
        self.v = None  # velocity vector
        self.L = 0.0   # angular momentum (scalar)
    
    def step(self, params, grads):
        """
        Perform one optimization step.
        
        Args:
            params: list of parameter arrays
            grads: list of gradient arrays (same structure as params)
        
        Returns:
            Updated params
        """
        # Flatten everything for easier math
        flat_grad = self._flatten(grads)
        
        # Initialize velocity on first step
        if self.v is None:
            self.v = np.zeros_like(flat_grad)
        
        speed = np.linalg.norm(self.v)
        
        if speed > self.eps:
            # Decompose gradient into parallel and perpendicular components
            v_hat = self.v / speed
            g_parallel_mag = np.dot(flat_grad, v_hat)
            g_parallel = g_parallel_mag * v_hat
            g_perp = flat_grad - g_parallel
            
            # Update angular momentum from parallel component
            # If gradient aligns with velocity, spin faster
            # If gradient opposes velocity, spin slower
            self.L = self.beta * self.L + g_parallel_mag
            self.L = max(self.L, 0.0)  # L is non-negative; if it hits 0, we've stopped
            
            # Update velocity direction with gyroscopic turn resistance
            # Perpendicular gradient tries to turn the wheel toward it
            # Gyroscopic stability resists proportional to L * I
            gyro_resistance = self.L * self.I + self.eps
            direction_change = g_perp / gyro_resistance
            v_hat_new = v_hat + direction_change
            v_hat_new = v_hat_new / (np.linalg.norm(v_hat_new) + self.eps)
            
            # New speed from rolling constraint
            speed_new = self.L / self.I
            
            # Combine
            self.v = speed_new * v_hat_new
        
        else:
            # Cold start: wheel is stationary
            # Gradient gets it rolling in the gradient direction
            g_mag = np.linalg.norm(flat_grad)
            
            if g_mag > self.eps:
                self.L = g_mag
                speed_new = self.L / self.I
                v_hat_new = flat_grad / g_mag
                self.v = speed_new * v_hat_new
        
        # Update parameters
        flat_params = self._flatten(params)
        flat_params_new = flat_params - self.lr * self.v
        
        return self._unflatten(flat_params_new, params)
    
    def _flatten(self, arrays):
        return np.concatenate([a.flatten() for a in arrays])
    
    def _unflatten(self, flat, reference):
        result = []
        idx = 0
        for ref in reference:
            size = ref.size
            result.append(flat[idx:idx+size].reshape(ref.shape))
            idx += size
        return result
```

## PyTorch Implementation

```python
import torch
from torch.optim import Optimizer

class WheelOptimizer(Optimizer):
    """
    Wheel Optimizer for PyTorch.
    
    Models optimization as a wheel rolling down the loss landscape.
    The wheel has gyroscopic stability — it resists turning when spinning fast.
    
    Args:
        params: Model parameters
        lr: Learning rate
        beta: Momentum decay factor
        I: Moment of inertia. Higher = harder to accelerate, more turning resistance
    """
    
    def __init__(self, params, lr=0.001, beta=0.95, I=1.0, eps=1e-8):
        defaults = dict(lr=lr, beta=beta, I=I, eps=eps)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            I = group['I']
            eps = group['eps']
            
            # Gather all params and grads in this group
            params_with_grad = []
            grads = []
            
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad)
            
            if len(grads) == 0:
                continue
            
            # Flatten
            flat_grad = torch.cat([g.flatten() for g in grads])
            
            # Get or initialize state
            # Use id of first param as key for shared state
            first_param = group['params'][0]
            state = self.state[first_param]
            if len(state) == 0:
                state['v'] = torch.zeros_like(flat_grad)
                state['L'] = torch.tensor(0.0, device=flat_grad.device)
            
            v = state['v']
            L = state['L']
            
            speed = torch.linalg.norm(v)
            
            if speed > eps:
                # Decompose gradient
                v_hat = v / speed
                g_parallel_mag = torch.dot(flat_grad, v_hat)
                g_parallel = g_parallel_mag * v_hat
                g_perp = flat_grad - g_parallel
                
                # Update angular momentum
                L = beta * L + g_parallel_mag
                L = torch.clamp(L, min=0.0)
                
                # Update direction with gyroscopic turn resistance
                # Perpendicular gradient tries to turn the wheel toward it
                # Gyroscopic stability resists proportional to L * I
                gyro_resistance = L * I + eps
                direction_change = g_perp / gyro_resistance
                v_hat_new = v_hat + direction_change
                v_hat_new = v_hat_new / (torch.linalg.norm(v_hat_new) + eps)
                
                # New speed from rolling constraint
                speed_new = L / I
                
                # Combine
                v = speed_new * v_hat_new
            
            else:
                # Cold start
                g_mag = torch.linalg.norm(flat_grad)
                
                if g_mag > eps:
                    L = g_mag
                    speed_new = L / I
                    v_hat_new = flat_grad / g_mag
                    v = speed_new * v_hat_new
            
            # Save state
            state['v'] = v
            state['L'] = L
            
            # Update parameters
            idx = 0
            for p in params_with_grad:
                size = p.numel()
                p.add_(v[idx:idx+size].reshape(p.shape), alpha=-lr)
                idx += size
        
        return loss
```

## Usage Example

```python
# PyTorch
model = MyModel()
optimizer = WheelOptimizer(model.parameters(), lr=0.001, beta=0.95, I=1.0)

for data, target in dataloader:
    optimizer.zero_grad()
    loss = criterion(model(data), target)
    loss.backward()
    optimizer.step()
```

## Comparison to Standard Momentum

| Property | Standard Momentum | Wheel Optimizer |
|----------|-------------------|-----------------|
| Resists speeding up/slowing down | ✓ | ✓ |
| Turning behavior | Direct steering | Gyroscopic turn resistance |
| Trajectory shape | Smooth curves | Spirals (tighter as L decreases) |
| Gradient effect | Direct | Resisted by L × I |
| State tracked | v | v, L |
| Hyperparameters | lr, β | lr, β, I |

## Tuning Guide

**Try higher I when:**
- Gradients are noisy
- Loss landscape has many small local features
- You want smoother, more stable trajectories
- You want stronger resistance to direction changes
- Training is overshooting or oscillating

**Try lower I when:**
- You want tighter spirals and more responsive turning
- Loss landscape has well-defined valleys
- Training is too slow to start
- You need to quickly adapt to changing gradient directions

## Mathematical Summary

Given gradient g at step t:

```
# Decompose gradient
v̂ = v / |v|
g_∥ = (g · v̂) v̂
g_⊥ = g - g_∥

# Update angular momentum (from parallel component)
L_t = β L_{t-1} + (g · v̂)

# Gyroscopic turn resistance
# Perpendicular gradient tries to turn wheel toward it
# Resistance proportional to L × I
gyro_resistance = L_t × I
direction_change = g_⊥ / gyro_resistance

# Update velocity direction
v̂_new = normalize(v̂ + direction_change)

# Update speed (rolling constraint)  
|v_new| = L_t / I

# Update parameters
θ_t = θ_{t-1} - lr · v_new
```

Key insight: As L decreases (momentum decay), gyroscopic resistance decreases, allowing sharper turns. This creates the characteristic spiral trajectory that tightens as the wheel settles, like a coin spinning down a funnel.