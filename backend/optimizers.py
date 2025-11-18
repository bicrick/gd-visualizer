"""
Gradient descent optimizer implementations for visualization.
"""

import numpy as np
from loss_functions import DEFAULT_LOSS_FUNCTION, compute_gradient


def stochastic_gradient_descent(loss_func, initial_params, learning_rate, n_iterations, 
                                dataset=None, seed=42, convergence_threshold=1e-6, 
                                max_iterations=10000, bounds=None):
    """
    Stochastic Gradient Descent - adds noise to simulate stochastic behavior.
    
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
        
        # Compute gradient at current position
        grad = compute_gradient(loss_func, params[0], params[1])
        
        # Add noise to simulate stochastic behavior
        noise = np.random.normal(0, 0.1, size=grad.shape)
        noisy_grad = grad + noise
        
        # Check convergence
        if use_convergence and np.linalg.norm(grad) < convergence_threshold:
            break
        
        # Update parameters with noisy gradient
        params = params - learning_rate * noisy_grad
        
        # Check if parameters are within bounds
        if bounds is not None:
            bounds_min, bounds_max = bounds
            if (params[0] < bounds_min or params[0] > bounds_max or 
                params[1] < bounds_min or params[1] > bounds_max):
                # Stop optimization if we exit the bounds
                break
    
    return trajectory


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


def momentum_gradient_descent(loss_func, initial_params, learning_rate, momentum, 
                              n_iterations, dataset=None, seed=42, 
                              convergence_threshold=1e-6, max_iterations=10000,
                              lr_decay=0.995, bounds=None):
    """
    Gradient Descent with Momentum - accumulates velocity across iterations.
    
    Args:
        loss_func: Loss function
        initial_params: Starting point [x, y]
        learning_rate: Step size
        momentum: Momentum coefficient (0-1)
        n_iterations: Number of optimization steps (or None for convergence-based)
        dataset: Training data (not used, kept for API compatibility)
        seed: Random seed
        convergence_threshold: Stop if gradient magnitude is below this
        max_iterations: Maximum iterations for convergence mode
        lr_decay: Learning rate decay factor applied each iteration (default 0.995)
                  1.0 means no decay, values < 1.0 cause gradual decay
        bounds: Tuple of (min, max) for parameter bounds, or None for no bounds
    
    Returns:
        List of (x, y, loss) tuples representing the trajectory
    """
    np.random.seed(seed)
    
    params = np.array(initial_params, dtype=float)
    velocity = np.zeros_like(params)
    trajectory = []
    current_lr = learning_rate
    
    # Use convergence mode if n_iterations is None or negative
    use_convergence = n_iterations is None or n_iterations < 0
    iterations = max_iterations if use_convergence else n_iterations
    
    for i in range(iterations):
        # Compute loss at current position
        loss = loss_func(params[0], params[1])
        trajectory.append((float(params[0]), float(params[1]), float(loss)))
        
        # Compute gradient at current parameter position
        grad = compute_gradient(loss_func, params[0], params[1])
        
        # Check convergence (use gradient magnitude)
        if use_convergence and np.linalg.norm(grad) < convergence_threshold:
            break
        
        # Update velocity with momentum and current learning rate
        velocity = momentum * velocity - current_lr * grad
        
        # Update parameters
        params = params + velocity
        
        # Check if parameters are within bounds
        if bounds is not None:
            bounds_min, bounds_max = bounds
            if (params[0] < bounds_min or params[0] > bounds_max or 
                params[1] < bounds_min or params[1] > bounds_max):
                # Stop optimization if we exit the bounds
                break
        
        # Decay learning rate
        current_lr *= lr_decay
    
    return trajectory


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


def find_collision_newton(x0, y0, z0, vx0, vy0, vz0, gravity, dt, loss_func, 
                         gradient_hint=None, loss_hint=None, max_iters=4):
    """
    Find collision using Newton's method with gradient prediction.
    Much faster than bisection: 2-4 evaluations instead of 15.
    
    Args:
        x0, y0, z0: Initial position at start of time step
        vx0, vy0, vz0: Initial velocity at start of time step
        gravity: Gravitational acceleration
        dt: Full time step
        loss_func: Loss function defining the surface
        gradient_hint: Gradient at previous point (for prediction)
        loss_hint: Loss at previous point (for prediction)
        max_iters: Maximum Newton iterations (default 4)
    
    Returns:
        Tuple: (t_collision, x_col, y_col, z_col, gradient_col)
        where gradient_col can be used as hint for next collision
        Returns None if no collision found
    """
    def position_at_time(t):
        """Calculate position at time t within the step."""
        x = x0 + vx0 * t
        y = y0 + vy0 * t
        z = z0 + vz0 * t - 0.5 * gravity * t * t
        return x, y, z
    
    # Step 1: Predict collision time using gradient hint (if available)
    if gradient_hint is not None and loss_hint is not None:
        # Solve analytically: z(t) = loss_hint + gradient · (xy(t) - xy0)
        # z0 + vz0*t - 0.5*g*t^2 = loss_hint + grad_x*(vx0*t) + grad_y*(vy0*t)
        # Rearranging: -0.5*g*t^2 + (vz0 - grad·v_xy)*t + (z0 - loss_hint) = 0
        
        a = -0.5 * gravity
        b = vz0 - np.dot(gradient_hint, [vx0, vy0])
        c = z0 - loss_hint
        
        # Solve quadratic
        discriminant = b**2 - 4*a*c
        if discriminant >= 0 and abs(a) > 1e-10:
            # Two solutions - take the one in [0, dt]
            t1 = (-b + np.sqrt(discriminant)) / (2*a)
            t2 = (-b - np.sqrt(discriminant)) / (2*a)
            
            # Choose the first positive collision time
            candidates = [t for t in [t1, t2] if 0 <= t <= dt]
            if candidates:
                t_predicted = min(candidates)
            else:
                t_predicted = dt / 2  # Fallback to midpoint
        else:
            t_predicted = dt / 2  # Fallback to midpoint
    else:
        t_predicted = dt / 2  # No hint - start at midpoint
    
    # Quick check: is there actually a collision?
    x_start, y_start, z_start = position_at_time(0)
    loss_start = loss_func(x_start, y_start)
    if z_start - loss_start < 0:
        # Started below surface - shouldn't happen but handle it
        return 0, x0, y0, z0, compute_gradient(loss_func, x0, y0)
    
    x_end, y_end, z_end = position_at_time(dt)
    loss_end = loss_func(x_end, y_end)
    if z_end - loss_end > 0:
        # Stayed above surface entire time - no collision
        return None
    
    # Step 2: Newton's method refinement
    t = t_predicted
    gradient_col = None
    
    for iteration in range(max_iters):
        # Evaluate trajectory position at current t
        x_t, y_t, z_t = position_at_time(t)
        
        # Evaluate loss and gradient (ONE expensive call per iteration)
        loss_t = loss_func(x_t, y_t)
        gradient_col = compute_gradient(loss_func, x_t, y_t)
        
        # Function: f(t) = z(t) - L(x(t), y(t))
        f = z_t - loss_t
        
        # Derivative: f'(t) = dz/dt - dL/dt
        # dz/dt = vz0 - gravity*t
        # dL/dt = ∇L · d(xy)/dt = ∇L · [vx0, vy0]
        dz_dt = vz0 - gravity * t
        dL_dt = np.dot(gradient_col, [vx0, vy0])
        f_prime = dz_dt - dL_dt
        
        # Check convergence
        if abs(f) < 1e-6:
            break
        
        # Avoid division by very small numbers
        if abs(f_prime) < 1e-10:
            # Derivative too small - try bisection fallback
            if f > 0:  # Above surface
                t = (t + dt) / 2
            else:  # Below surface
                t = t / 2
            continue
        
        # Newton step
        t_new = t - f / f_prime
        
        # Keep t in valid range [0, dt]
        t = np.clip(t_new, 0, dt)
        
        # If we're converging slowly, we might need more iterations
        # but limit to max_iters for efficiency
    
    # Return collision point and gradient
    x_col, y_col, z_col = position_at_time(t)
    
    # Make sure we actually have a collision (not just ended at dt)
    if t >= dt - 1e-6:
        # Reached end of timestep without clear collision
        # Check if we're actually colliding
        if z_col - loss_func(x_col, y_col) > 0.01:
            return None  # Still above surface
    
    return t, x_col, y_col, z_col, gradient_col


def ballistic_gradient_descent(loss_func, initial_params, drop_height=5.0, 
                                gravity=1.0, elasticity=0.8, bounce_threshold=0.05,
                                dt=0.01, max_iterations=10000, seed=42, bounds=None):
    """
    Ballistic Gradient Descent - physics-based optimization by dropping a ball
    from a height and letting it bounce on the loss landscape.
    
    Args:
        loss_func: Loss function
        initial_params: Starting point [x, y] (horizontal position)
        drop_height: Initial height above the surface to drop from
        gravity: Gravitational acceleration constant
        elasticity: Energy retention coefficient (0-1) after each bounce
        bounce_threshold: Minimum bounce height before stopping
        dt: Time step for physics simulation
        max_iterations: Maximum number of simulation steps
        seed: Random seed
        bounds: Tuple of (min, max) for parameter bounds, or None for no bounds
    
    Returns:
        List of (x, y, z_world) tuples representing the 3D trajectory
        where z_world is the actual 3D height (not loss value)
    """
    np.random.seed(seed)
    
    # Initialize position and velocity
    x, y = initial_params[0], initial_params[1]
    
    # Get initial surface height
    surface_height = loss_func(x, y)
    
    # Start ball at drop_height above the surface
    z = surface_height + drop_height
    
    # Initial velocity (starting from rest)
    vx, vy, vz = 0.0, 0.0, 0.0
    
    trajectory = []
    trajectory.append((float(x), float(y), float(z)))
    
    # Track gradient and loss for Newton prediction
    gradient_hint = None
    loss_hint = None
    
    for iteration in range(max_iterations):
        # Store previous state for collision detection
        x_prev, y_prev, z_prev = x, y, z
        vx_prev, vy_prev, vz_prev = vx, vy, vz
        
        # Update velocity with gravity (before position update)
        vz -= gravity * dt
        
        # Tentatively update position
        x_new = x + vx * dt
        y_new = y + vy * dt
        z_new = z + vz * dt
        
        # Check bounds for horizontal motion
        if bounds is not None:
            bounds_min, bounds_max = bounds
            # Bounce off walls
            if x_new < bounds_min or x_new > bounds_max:
                x_new = np.clip(x_new, bounds_min, bounds_max)
                vx = -vx * elasticity
            if y_new < bounds_min or y_new > bounds_max:
                y_new = np.clip(y_new, bounds_min, bounds_max)
                vy = -vy * elasticity
        
        # Check for collision with surface using Newton's method
        collision_result = find_collision_newton(x_prev, y_prev, z_prev, vx_prev, vy_prev, vz_prev, 
                                                 gravity, dt, loss_func, 
                                                 gradient_hint=gradient_hint, loss_hint=loss_hint)
        
        if collision_result is not None and vz_prev < 0:  # Only bounce when moving downward
            # Collision detected - unpack results (now includes gradient!)
            t_collision, x_col, y_col, z_col, grad_col = collision_result
            
            # Record the collision point in trajectory for smooth arc visualization
            trajectory.append((float(x_col), float(y_col), float(z_col)))
            
            # Calculate velocity at collision time
            vx_col = vx_prev
            vy_col = vy_prev
            vz_col = vz_prev - gravity * t_collision
            
            # Use gradient from Newton's method (already computed!)
            grad = grad_col
            
            # Update hints for next collision prediction
            gradient_hint = grad
            loss_hint = loss_func(x_col, y_col)
            
            # Surface normal (gradient points upward in loss space)
            # Normal vector: (-dL/dx, -dL/dy, 1) then normalize
            normal = np.array([-grad[0], -grad[1], 1.0])
            normal = normal / np.linalg.norm(normal)
            
            # Velocity vector at collision
            velocity = np.array([vx_col, vy_col, vz_col])
            
            # Reflect velocity around normal
            # v_reflected = v - 2 * (v · n) * n
            dot_product = np.dot(velocity, normal)
            
            # Only reflect if moving into the surface (should always be true here)
            if dot_product < 0:
                reflected = velocity - 2 * dot_product * normal
                
                # Apply elasticity to reduce energy
                reflected *= elasticity
                
                vx_after = reflected[0]
                vy_after = reflected[1]
                vz_after = reflected[2]
                
                # Track bounce height for stopping criterion
                last_bounce_height = abs(vz_after) / gravity if gravity > 0 else 0
                
                # Check if bounce is too small to continue
                if last_bounce_height < bounce_threshold:
                    # Settle on surface
                    x, y, z = x_col, y_col, z_col
                    vx, vy, vz = 0.0, 0.0, 0.0
                    trajectory.append((float(x), float(y), float(z)))
                    break
                
                # Continue simulation from collision point with remaining time
                remaining_time = dt - t_collision
                
                # Update position for remaining time after bounce using proper kinematics
                # Position uses initial velocity after bounce plus gravitational acceleration
                x = x_col + vx_after * remaining_time
                y = y_col + vy_after * remaining_time
                z = z_col + vz_after * remaining_time - 0.5 * gravity * remaining_time * remaining_time
                
                # Update velocities to their values at end of remaining time
                vx = vx_after  # No horizontal forces
                vy = vy_after  # No horizontal forces
                vz = vz_after - gravity * remaining_time  # Gravity affects vertical velocity
            else:
                # Shouldn't happen, but use new position if it does
                x, y, z = x_new, y_new, z_new
        else:
            # No collision - use new position
            x, y, z = x_new, y_new, z_new
        
        # Record position
        trajectory.append((float(x), float(y), float(z)))
        
        # Check if settled (very low velocity and on/near surface)
        velocity_magnitude = np.sqrt(vx**2 + vy**2 + vz**2)
        surface_height = loss_func(x, y)
        if velocity_magnitude < 0.001 and abs(z - surface_height) < 0.01:
            break
    
    return trajectory

