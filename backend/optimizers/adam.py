"""
ADAM (Adaptive Moment Estimation) optimizer implementation.
"""

import numpy as np
from loss_functions import compute_gradient


def adam_optimizer(loss_func, initial_params, learning_rate, n_iterations,
                   beta1=0.9, beta2=0.999, epsilon=1e-8, dataset=None, seed=42,
                   convergence_threshold=1e-6, max_iterations=10000, bounds=None):
    """
    ADAM (Adaptive Moment Estimation) optimizer - combines momentum and RMSprop.
    
    Args:
        loss_func: Loss function
        initial_params: Starting point [x, y]
        learning_rate: Step size (alpha)
        n_iterations: Number of optimization steps (or None for convergence-based)
        beta1: Exponential decay rate for first moment estimates (default 0.9)
        beta2: Exponential decay rate for second moment estimates (default 0.999)
        epsilon: Small constant for numerical stability (default 1e-8)
        dataset: Training data (not used, kept for API compatibility)
        seed: Random seed
        convergence_threshold: Stop if gradient magnitude is below this
        max_iterations: Maximum iterations for convergence mode
        bounds: Tuple of (min, max) for parameter bounds, or None for no bounds
    
    Returns:
        List of (x, y, loss) tuples representing the trajectory
    """
    np.random.seed(seed)
    
    params = np.array(initial_params, dtype=float)
    m = np.zeros_like(params)  # First moment estimate
    v = np.zeros_like(params)  # Second moment estimate
    trajectory = []
    
    # Use convergence mode if n_iterations is None or negative
    use_convergence = n_iterations is None or n_iterations < 0
    iterations = max_iterations if use_convergence else n_iterations
    
    for t in range(1, iterations + 1):
        # Compute loss at current position
        loss = loss_func(params[0], params[1])
        trajectory.append((float(params[0]), float(params[1]), float(loss)))
        
        # Compute gradient at current parameter position
        grad = compute_gradient(loss_func, params[0], params[1])
        
        # Check convergence
        if use_convergence and np.linalg.norm(grad) < convergence_threshold:
            break
        
        # Update biased first moment estimate
        m = beta1 * m + (1 - beta1) * grad
        
        # Update biased second raw moment estimate
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        
        # Compute bias-corrected first moment estimate
        m_hat = m / (1 - beta1 ** t)
        
        # Compute bias-corrected second raw moment estimate
        v_hat = v / (1 - beta2 ** t)
        
        # Update parameters
        params = params - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        
        # Check if parameters are within bounds
        if bounds is not None:
            bounds_min, bounds_max = bounds
            if (params[0] < bounds_min or params[0] > bounds_max or 
                params[1] < bounds_min or params[1] > bounds_max):
                # Stop optimization if we exit the bounds
                break
    
    return trajectory
