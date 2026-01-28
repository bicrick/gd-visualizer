"""
Batch Gradient Descent optimizer implementation.
"""

import numpy as np
from loss_functions import compute_gradient


def batch_gradient_descent(loss_func, initial_params, learning_rate, n_iterations,
                           dataset=None, seed=42, convergence_threshold=1e-6,
                           max_iterations=10000, bounds=None):
    """
    Batch Gradient Descent - computes gradient at current parameters.
    
    Args:
        loss_func: Loss function
        initial_params: Starting point [x, y]
        learning_rate: Step size
        n_iterations: Number of optimization steps (or None for convergence-based)
        dataset: Training data (not used for pure GD, kept for API compatibility)
        seed: Random seed
        convergence_threshold: Stop if gradient magnitude is below this
        max_iterations: Maximum iterations for convergence mode
        bounds: Tuple of (min, max) for parameter bounds, or None for no bounds
    
    Returns:
        List of (x, y, loss) tuples representing the trajectory
    """
    np.random.seed(seed)
    
    params = np.array(initial_params, dtype=float)
    trajectory = []
    
    # Use convergence mode if n_iterations is None or negative
    use_convergence = n_iterations is None or n_iterations < 0
    iterations = max_iterations if use_convergence else n_iterations
    
    for i in range(iterations):
        # Compute loss at current position
        loss = loss_func(params[0], params[1])
        trajectory.append((float(params[0]), float(params[1]), float(loss)))
        
        # Compute gradient at current parameter position
        grad = compute_gradient(loss_func, params[0], params[1])
        
        # Check convergence
        if use_convergence and np.linalg.norm(grad) < convergence_threshold:
            break
        
        # Update parameters
        params = params - learning_rate * grad
        
        # Check if parameters are within bounds
        if bounds is not None:
            bounds_min, bounds_max = bounds
            if (params[0] < bounds_min or params[0] > bounds_max or 
                params[1] < bounds_min or params[1] > bounds_max):
                # Stop optimization if we exit the bounds
                break
    
    return trajectory
