"""
Gradient descent optimizer implementations for visualization.
"""

import numpy as np
from loss_functions import DEFAULT_LOSS_FUNCTION, compute_gradient


def stochastic_gradient_descent(loss_func, initial_params, learning_rate, n_iterations, 
                                dataset=None, seed=42, convergence_threshold=1e-6, 
                                max_iterations=10000, bounds=None,
                                step_multiplier=3.0, noise_scale=0.8):
    """
    Stochastic Gradient Descent - simulates mini-batch behavior with larger,
    noisier steps compared to batch gradient descent.
    
    Unlike batch gradient descent which computes the exact gradient over the full
    dataset, SGD approximates the gradient using mini-batches. This results in:
    1. Faster convergence (larger effective step size via step_multiplier)
    2. Noisier gradients that don't always point directly downhill (noise_scale)
    
    The visualization shows SGD "racing ahead" while bouncing around, which
    represents the real-world trade-off: faster convergence but less stable path.
    
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
        noise_scale: Standard deviation of gradient noise (default 0.8)
                    Higher values = more radical "bouncing" behavior
    
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
        sideways_noise = np.random.normal(0, noise_scale * 0.7) * perpendicular
        
        # Magnitude noise (speed variation along gradient)
        magnitude_noise = np.random.normal(0, noise_scale * 0.3) * grad_normalized if grad_magnitude > 1e-8 else np.zeros(2)
        
        # Combined noisy gradient
        noisy_grad = grad + sideways_noise + magnitude_noise
        
        # Effective learning rate is multiplied for faster convergence
        effective_lr = learning_rate * step_multiplier
        
        # Update parameters with noisy gradient and larger step
        params = params - effective_lr * noisy_grad
        
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
                         gradient_hint=None, loss_hint=None, ball_radius=0.05, max_iters=4):
    """
    Find collision using Newton's method with gradient prediction.
    Much faster than bisection: 2-4 evaluations instead of 15.
    
    Args:
        x0, y0, z0: Initial position at start of time step (center of ball)
        vx0, vy0, vz0: Initial velocity at start of time step
        gravity: Gravitational acceleration
        dt: Full time step
        loss_func: Loss function defining the surface
        gradient_hint: Gradient at previous point (for prediction)
        loss_hint: Loss at previous point (for prediction)
        ball_radius: Radius of the ball (collision when bottom touches surface)
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
        # Solve analytically: z_bottom(t) = loss_hint + gradient · (xy(t) - xy0)
        # where z_bottom = z_center - ball_radius
        # (z0 - ball_radius) + vz0*t - 0.5*g*t^2 = loss_hint + grad_x*(vx0*t) + grad_y*(vy0*t)
        # Rearranging: -0.5*g*t^2 + (vz0 - grad·v_xy)*t + (z0 - ball_radius - loss_hint) = 0
        
        a = -0.5 * gravity
        b = vz0 - np.dot(gradient_hint, [vx0, vy0])
        c = (z0 - ball_radius) - loss_hint
        
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
    # Check if BOTTOM of ball (z_center - ball_radius) crosses surface
    x_start, y_start, z_start = position_at_time(0)
    loss_start = loss_func(x_start, y_start)
    if (z_start - ball_radius) - loss_start < 0:
        # Started below surface - shouldn't happen but handle it
        return 0, x0, y0, z0, compute_gradient(loss_func, x0, y0)
    
    x_end, y_end, z_end = position_at_time(dt)
    loss_end = loss_func(x_end, y_end)
    if (z_end - ball_radius) - loss_end > 0:
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
        
        # Function: f(t) = z_bottom(t) - L(x(t), y(t))
        # where z_bottom = z_center - ball_radius
        f = (z_t - ball_radius) - loss_t
        
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
        # Check if we're actually colliding (check bottom of ball)
        if (z_col - ball_radius) - loss_func(x_col, y_col) > 0.01:
            return None  # Still above surface
    
    return t, x_col, y_col, z_col, gradient_col


def ballistic_gradient_descent(loss_func, initial_params, drop_height=5.0, 
                                gravity=1.0, elasticity=0.8, bounce_threshold=0.05,
                                ball_radius=0.05, dt=0.01, max_iterations=10000, seed=42, bounds=None):
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
        ball_radius: Radius of the ball (bottom collides with surface, not center)
        dt: Time step for physics simulation
        max_iterations: Maximum number of simulation steps
        seed: Random seed
        bounds: Tuple of (min, max) for parameter bounds, or None for no bounds
    
    Returns:
        List of (x, y, z_world) tuples representing the 3D trajectory
        where z_world is the actual 3D height of ball center (not loss value)
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
                                                 gradient_hint=gradient_hint, loss_hint=loss_hint,
                                                 ball_radius=ball_radius)
        
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


def ballistic_adam_optimizer(loss_func, initial_params, learning_rate=0.01, momentum=0.9,
                              gravity=0.001, dt=1.0, max_air_steps=20, 
                              max_bisection_iters=10, collision_tol=1e-3,
                              n_iterations=100, dataset=None, seed=42, 
                              convergence_threshold=1e-6, max_iterations=10000, bounds=None):
    """
    Ballistic Adam Optimizer - rolls like SGD with momentum until it gains enough
    speed on a slope to become airborne, then follows ballistic physics.
    
    Physics Analog:
    - "Rolling": SGD Momentum on the surface, computing gradients to update velocity
    - "Takeoff": If predicted height > actual loss (ground drops away), go airborne
    - "Flying": Pure ballistic physics with gravity, no gradient computation
    - "Landing": Bisection search for exact collision point, inelastic collision (v_h = 0)
    
    Args:
        loss_func: Loss function
        initial_params: Starting point [x, y]
        learning_rate: Step size for gradient updates (when on ground)
        momentum: Momentum coefficient (0-1) for horizontal velocity
        gravity: Gravitational acceleration (affects vertical motion)
        dt: Time step for physics simulation
        max_air_steps: Maximum iterations to simulate while airborne
        max_bisection_iters: Maximum bisection iterations for collision detection
        collision_tol: Tolerance for collision detection
        n_iterations: Number of optimization steps (or None for convergence-based)
        dataset: Training data (not used, kept for API compatibility)
        seed: Random seed
        convergence_threshold: Stop if gradient magnitude is below this
        max_iterations: Maximum iterations for convergence mode
        bounds: Tuple of (min, max) for parameter bounds, or None for no bounds
    
    Returns:
        List of (x, y, z_world) tuples representing the 3D trajectory
        where z_world is the actual 3D height (loss value), similar to ballistic optimizer
    """
    np.random.seed(seed)
    
    params = np.array(initial_params, dtype=float)
    velocity = np.zeros_like(params)  # Horizontal velocity (momentum)
    
    # Vertical state
    h = loss_func(params[0], params[1])  # Current height (loss)
    v_h = 0.0  # Vertical velocity
    prev_loss = h
    flying = False
    
    trajectory = []
    trajectory.append((float(params[0]), float(params[1]), float(h)))
    
    # Use convergence mode if n_iterations is None or negative
    use_convergence = n_iterations is None or n_iterations < 0
    iterations = max_iterations if use_convergence else n_iterations
    
    for iteration in range(iterations):
        # Compute gradient only when on ground
        if not flying:
            grad = compute_gradient(loss_func, params[0], params[1])
            
            # Check convergence
            if use_convergence and np.linalg.norm(grad) < convergence_threshold:
                break
            
            # Update horizontal velocity with momentum (SGD momentum style)
            velocity = momentum * velocity - learning_rate * grad
        
        # Physics simulation loop
        air_steps = 0
        while air_steps < max_air_steps:
            air_steps += 1
            
            # Save state before step
            params_old = params.copy()
            h_old = h
            v_h_old = v_h
            
            # Take one physics step (horizontal movement with current velocity)
            params_new = params + velocity
            
            # Check bounds
            if bounds is not None:
                bounds_min, bounds_max = bounds
                if (params_new[0] < bounds_min or params_new[0] > bounds_max or 
                    params_new[1] < bounds_min or params_new[1] > bounds_max):
                    # Hit boundary, stop here
                    break
            
            # Predict height using ballistic physics
            h_pred = h + v_h * dt - 0.5 * gravity * (dt ** 2)
            v_h_pred = v_h - gravity * dt
            
            # Evaluate actual loss at new position
            loss_new = loss_func(params_new[0], params_new[1])
            
            # Check for collision/penetration
            penetration = h_pred - loss_new
            
            if penetration > 0:
                # No collision: we're above the surface
                if not flying:
                    flying = True
                
                # Accept the move
                params = params_new
                h = h_pred
                v_h = v_h_pred
                prev_loss = loss_new
                
                # Record position (use h_pred as z_world for smooth arcs)
                trajectory.append((float(params[0]), float(params[1]), float(h)))
                
                # Continue flying
                continue
            else:
                # Collision detected: h_pred <= loss (penetrated surface)
                if flying:
                    # We were flying and hit ground - use bisection to find exact point
                    t_low = 0.0
                    t_high = 1.0
                    
                    for _ in range(max_bisection_iters):
                        t_mid = (t_low + t_high) / 2.0
                        
                        # Interpolate parameters
                        params_mid = params_old + t_mid * (params_new - params_old)
                        
                        # Predict height at t_mid
                        t_elapsed = t_mid * dt
                        h_pred_mid = h_old + v_h_old * t_elapsed - 0.5 * gravity * (t_elapsed ** 2)
                        
                        # Evaluate actual loss
                        loss_mid = loss_func(params_mid[0], params_mid[1])
                        
                        # Check which side of surface
                        penetration_mid = h_pred_mid - loss_mid
                        
                        if penetration_mid > 0:
                            t_low = t_mid  # Still above, collision is later
                        else:
                            t_high = t_mid  # Below, collision is earlier
                        
                        if abs(penetration_mid) < collision_tol:
                            break
                    
                    # Land at collision point
                    t_collision = t_mid
                    params = params_old + t_collision * (params_new - params_old)
                    loss_collision = loss_func(params[0], params[1])
                    
                    # Inelastic collision: zero vertical velocity, keep horizontal
                    h = loss_collision
                    v_h = 0.0
                    prev_loss = loss_collision
                    flying = False
                    
                    trajectory.append((float(params[0]), float(params[1]), float(h)))
                    
                    # Exit air loop, will compute fresh gradients on next iteration
                    break
                else:
                    # We were on ground and stayed on ground (normal rolling)
                    params = params_new
                    h = loss_new
                    
                    # Update vertical velocity based on slope
                    dh = loss_new - prev_loss
                    v_h = dh / dt
                    prev_loss = loss_new
                    
                    trajectory.append((float(params[0]), float(params[1]), float(h)))
                    
                    # Exit air loop, continue to next iteration
                    break
        
        # Check if we escaped bounds during air simulation
        if bounds is not None:
            bounds_min, bounds_max = bounds
            if (params[0] < bounds_min or params[0] > bounds_max or 
                params[1] < bounds_min or params[1] > bounds_max):
                break
    
    return trajectory

