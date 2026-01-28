"""
Wheel Optimizer - models optimization as a wheel rolling down the loss landscape.
"""

import numpy as np
from loss_functions import compute_gradient


def wheel_optimizer(loss_func, initial_params, learning_rate, n_iterations,
                    beta=0.95, I=1.0, eps=1e-8, dataset=None, seed=42,
                    convergence_threshold=1e-6, max_iterations=10000, bounds=None):
    """
    Wheel Optimizer - models optimization as a wheel rolling down the loss landscape.
    
    Unlike standard momentum, a rolling wheel has gyroscopic stability — it resists
    turning when spinning fast. The gradient affects velocity indirectly through
    angular momentum rather than directly.
    
    State:
        v: velocity vector (direction and speed of the wheel)
        L: angular momentum (scalar, how fast the wheel is spinning)
    
    The gradient is decomposed into:
        - Parallel component (along v): affects spin speed (L)
        - Perpendicular component (across v): tries to turn the wheel, 
          but is resisted by L/I (gyroscopic resistance)
    
    Args:
        loss_func: Loss function
        initial_params: Starting point [x, y]
        learning_rate: Step size
        n_iterations: Number of optimization steps (or None for convergence-based)
        beta: Momentum decay factor (0-1), like standard momentum (default 0.95)
        I: Moment of inertia. Higher = harder to accelerate, more turning resistance (default 1.0)
        eps: Small constant for numerical stability (default 1e-8)
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
    v = np.zeros_like(params)  # Velocity vector
    L = 0.0  # Angular momentum (scalar)
    trajectory = []
    
    # Use convergence mode if n_iterations is None or negative
    use_convergence = n_iterations is None or n_iterations < 0
    iterations = max_iterations if use_convergence else n_iterations
    
    # Debug output
    print(f"\n[WHEEL] Starting optimization:")
    print(f"  Initial params: {params}")
    print(f"  Learning rate: {learning_rate}, Beta: {beta}, I: {I}")
    print(f"  Iterations: {iterations if not use_convergence else f'max {iterations} (convergence mode)'}")
    
    for i in range(iterations):
        # Compute loss at current position
        loss = loss_func(params[0], params[1])
        trajectory.append((float(params[0]), float(params[1]), float(loss)))
        
        # Compute gradient at current parameter position
        grad = compute_gradient(loss_func, params[0], params[1])
        
        # Check convergence (use gradient magnitude)
        grad_norm = np.linalg.norm(grad)
        if use_convergence and grad_norm < convergence_threshold:
            print(f"[WHEEL] Converged at iteration {i}: grad_norm={grad_norm:.6e}")
            break
        
        speed = np.linalg.norm(v)
        
        # Debug output for first few iterations
        if i < 3 or i % 20 == 0:
            print(f"[WHEEL] Iter {i}: pos=({params[0]:.4f}, {params[1]:.4f}), loss={loss:.4f}, L={L:.4f}, speed={speed:.4f}, grad_norm={grad_norm:.4f}")
        
        if speed > eps:
            # Wheel is moving - decompose gradient into parallel and perpendicular components
            v_hat = v / speed
            g_parallel_mag = np.dot(grad, v_hat)
            g_parallel = g_parallel_mag * v_hat
            g_perp = grad - g_parallel
            
            # Update angular momentum from parallel component
            # If gradient aligns with velocity, spin faster
            # If gradient opposes velocity, spin slower
            L_old = L
            L = beta * L + g_parallel_mag
            L = max(L, 0.0)  # L is non-negative; if it hits 0, we've stopped
            
            # Update velocity direction with gyroscopic turn resistance
            # Perpendicular gradient tries to turn the wheel toward it
            # Resistance has two components:
            #   1. Baseline from I (inertia resists rotation even when stationary)
            #   2. Gyroscopic boost from L (spinning adds stability)
            # Formula: I * (1 + L) = I + I*L
            gyro_resistance = I * (1 + L) + eps
            direction_change = g_perp / gyro_resistance
            direction_change_mag = np.linalg.norm(direction_change)
            v_hat_new = v_hat + direction_change
            v_hat_new = v_hat_new / (np.linalg.norm(v_hat_new) + eps)
            
            # New speed from rolling constraint
            speed_new = L / I
            
            # Combine direction and speed
            v = speed_new * v_hat_new
            
            # Debug gyroscopic resistance
            if i < 3 or i % 20 == 0:
                g_perp_mag_val = np.linalg.norm(g_perp)
                gyro_resistance_val = I * (1 + L) + eps
                print(f"    g_parallel={g_parallel_mag:.4f}, g_perp_mag={g_perp_mag_val:.4f}")
                print(f"    L: {L_old:.4f} -> {L:.4f}, gyro_resistance={gyro_resistance_val:.4f}, dir_change_mag={direction_change_mag:.4f}")
        else:
            # Cold start: apply gradient as initial push
            # The wheel starts rolling from rest with a gentle push
            g_mag = np.linalg.norm(grad)
            
            if g_mag > eps:
                # Initial velocity from gradient, scaled by learning_rate
                # This matches standard GD behavior and keeps initial speed reasonable
                v = learning_rate * grad
                
                # L follows from rolling constraint: L = I * speed
                # This keeps L and speed coupled via the physics model
                L = I * np.linalg.norm(v)
                
                if i < 3:
                    speed_new = np.linalg.norm(v)
                    print(f"    Cold start: g_mag={g_mag:.4f}, initial L={L:.4f}, speed={speed_new:.4f}")
        
        # Update parameters
        params = params - learning_rate * v
        
        # Check if parameters are within bounds
        if bounds is not None:
            bounds_min, bounds_max = bounds
            if (params[0] < bounds_min or params[0] > bounds_max or 
                params[1] < bounds_min or params[1] > bounds_max):
                # Stop optimization if we exit the bounds
                print(f"[WHEEL] Exited bounds at iteration {i}")
                break
    
    print(f"[WHEEL] Finished: {len(trajectory)} steps, final pos=({params[0]:.4f}, {params[1]:.4f}), L={L:.4f}")
    return trajectory
