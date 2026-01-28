"""
Stochastic Gradient Descent optimizer implementation.
"""

import numpy as np
from loss_functions import compute_gradient


def stochastic_gradient_descent(loss_func, initial_params, learning_rate, n_iterations, 
                                dataset=None, seed=42, convergence_threshold=1e-6, 
                                max_iterations=10000, bounds=None,
                                step_multiplier=3.0, noise_scale=0.8, noise_decay=0.995):
    """
    Stochastic Gradient Descent - simulates mini-batch behavior with larger,
    noisier steps compared to batch gradient descent.
    
    Unlike batch gradient descent which computes the exact gradient over the full
    dataset, SGD approximates the gradient using mini-batches. This results in:
    1. Faster convergence (larger effective step size via step_multiplier)
    2. Noisier gradients that don't always point directly downhill (noise_scale)
    3. Gradual noise reduction over time (noise_decay) for settling
    
    The visualization shows SGD "racing ahead" while bouncing around, which
    represents the real-world trade-off: faster convergence but less stable path.
    The noise decay simulates learning rate annealing in practice.
    
    Args:
        loss_func: Loss function
        initial_params: Starting point [x, y]
        learning_rate: Step size
        n_iterations: Number of optimization steps (or None for convergence-based)
        dataset: Training data (not used, kept for API compatibility)
        seed: Random seed
        convergence_threshold: Stop if gradient magnitude is below this
        max_iterations: Maximum iterations for convergence mode
        bounds: Tuple of (min, max) for parameter bounds, or None for no bounds
        step_multiplier: Multiplier for effective step size (default 3.0)
                        Higher values = SGD moves faster, converges quicker
        noise_scale: Initial standard deviation of gradient noise (default 0.8)
                    Higher values = more radical "bouncing" behavior
        noise_decay: Multiplicative decay factor per iteration (default 0.995)
                    Values < 1.0 gradually reduce noise, allowing SGD to settle
                    0.995 = ~60% noise at 100 steps, ~8% at 500 steps
    
    Returns:
        List of (x, y, loss) tuples representing the trajectory
    """
    np.random.seed(seed)
    
    params = np.array(initial_params, dtype=float)
    trajectory = []
    
    # Use convergence mode if n_iterations is None or negative
    use_convergence = n_iterations is None or n_iterations < 0
    iterations = max_iterations if use_convergence else n_iterations
    
    # Track current noise level (starts at noise_scale, decays over time)
    current_noise = noise_scale
    
    for i in range(iterations):
        # Compute loss at current position
        loss = loss_func(params[0], params[1])
        trajectory.append((float(params[0]), float(params[1]), float(loss)))
        
        # Compute gradient at current position
        grad = compute_gradient(loss_func, params[0], params[1])
        
        # Check convergence (use true gradient magnitude)
        grad_magnitude = np.linalg.norm(grad)
        if use_convergence and grad_magnitude < convergence_threshold:
            break
        
        # Add noise to simulate mini-batch gradient variance
        # Noise has two components:
        # 1. Directional noise: perpendicular to gradient (causes sideways bouncing)
        # 2. Magnitude noise: along gradient direction (causes speed variation)
        
        # Create perpendicular direction for sideways bouncing
        if grad_magnitude > 1e-8:
            grad_normalized = grad / grad_magnitude
            perpendicular = np.array([-grad_normalized[1], grad_normalized[0]])
        else:
            perpendicular = np.array([1.0, 0.0])
        
        # Perpendicular noise (sideways bouncing) - more dramatic
        # Uses current_noise which decays over time
        sideways_noise = np.random.normal(0, current_noise * 0.7) * perpendicular
        
        # Magnitude noise (speed variation along gradient)
        magnitude_noise = np.random.normal(0, current_noise * 0.3) * grad_normalized if grad_magnitude > 1e-8 else np.zeros(2)
        
        # Combined noisy gradient
        noisy_grad = grad + sideways_noise + magnitude_noise
        
        # Effective learning rate is multiplied for faster convergence
        effective_lr = learning_rate * step_multiplier
        
        # Update parameters with noisy gradient and larger step
        params = params - effective_lr * noisy_grad
        
        # Decay noise for next iteration (allows SGD to settle over time)
        current_noise *= noise_decay
        
        # Check if parameters are within bounds
        if bounds is not None:
            bounds_min, bounds_max = bounds
            if (params[0] < bounds_min or params[0] > bounds_max or 
                params[1] < bounds_min or params[1] > bounds_max):
                # Stop optimization if we exit the bounds
                break
    
    return trajectory
