"""
Flask API server for gradient descent visualization backend.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
import time
import numpy as np
from loss_functions import (
    DEFAULT_LOSS_FUNCTION, 
    DEFAULT_MANIFOLD_ID,
    generate_landscape_mesh,
    get_manifold_function,
    get_manifold_metadata,
    list_all_manifolds
)
from optimizers import (
    stochastic_gradient_descent,
    batch_gradient_descent,
    momentum_gradient_descent,
    adam_optimizer,
    ballistic_gradient_descent,
    ballistic_adam_optimizer
)

app = Flask(__name__)

# Enable CORS for Vercel frontend and local development
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "https://gd.bicrick.com",
            "https://gd-visualizer.vercel.app",
            "https://gd-visualizer-n7t7jrq0e-bicricks-projects.vercel.app",
            "http://localhost:3000",
            "http://localhost:5001",
            "http://127.0.0.1:5001",
            "http://frontend:3000"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Initialize rate limiter - 500 requests per hour per IP
# Increased from 50 since landscape rendering is now client-side
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["500 per hour"],
    storage_uri="memory://"
)

# Rate limit error handler
@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        'error': 'rate_limit_exceeded',
        'message': 'You have reached the demo rate limit (500 requests/hour). Run unlimited locally with Docker!',
        'retry_after': str(e.description)
    }), 429


# API Routes - Backend only, no frontend serving
@app.route('/api/landscape', methods=['GET'])
def get_landscape():
    """
    Generate and return loss landscape mesh data for 3D visualization.
    """
    import json
    manifold_id = request.args.get('manifold', DEFAULT_MANIFOLD_ID)
    resolution = int(request.args.get('resolution', 100))
    
    # Parse manifold parameters if provided
    params_str = request.args.get('params', None)
    func_params = None
    if params_str:
        try:
            func_params = json.loads(params_str)
        except json.JSONDecodeError:
            func_params = None
    
    # Get manifold metadata to use default range if not specified
    manifold_meta = get_manifold_metadata(manifold_id)
    default_range = manifold_meta['default_range']
    
    # Use default range from manifold metadata if not explicitly provided
    x_range_str = request.args.get('x_range', None)
    y_range_str = request.args.get('y_range', None)
    
    if x_range_str:
        x_min, x_max = map(float, x_range_str.split(','))
    else:
        x_min, x_max = default_range
    
    if y_range_str:
        y_min, y_max = map(float, y_range_str.split(','))
    else:
        y_min, y_max = default_range
    
    # Get the appropriate loss function for the selected manifold
    loss_function = get_manifold_function(manifold_id)
    
    mesh_data = generate_landscape_mesh(
        loss_function,
        x_range=(x_min, x_max),
        y_range=(y_min, y_max),
        resolution=resolution,
        func_params=func_params
    )
    
    # Add manifold info and range to response
    mesh_data['manifold_id'] = manifold_id
    mesh_data['x_range'] = [x_min, x_max]
    mesh_data['y_range'] = [y_min, y_max]
    
    return jsonify(mesh_data)


@app.route('/api/optimize', methods=['POST'])
def optimize():
    """
    Run gradient descent optimizers and return trajectories.
    
    Expected JSON body:
    {
        "manifold": string (optional),
        "initial_params": [x, y],
        "learning_rate": float,  # fallback for backwards compatibility
        "momentum": float (0-1),  # fallback for backwards compatibility
        "n_iterations": int,      # fallback for backwards compatibility
        "seed": int (optional),
        "optimizer_params": {     # per-optimizer parameters
            "sgd": {...},
            "batch": {...},
            "momentum": {...}
        }
    }
    """
    data = request.json
    
    manifold_id = data.get('manifold', DEFAULT_MANIFOLD_ID)
    initial_params = data.get('initial_params', [0.0, 0.0])
    seed = int(data.get('seed', 42))
    
    # Get which optimizers to run (default to all if not specified)
    enabled_optimizers = data.get('enabled_optimizers', {
        'sgd': True,
        'batch': True,
        'momentum': True,
        'adam': True,
        'ballistic': True,
        'ballistic_adam': True
    })
    
    # Get per-optimizer parameters, or fall back to global parameters
    optimizer_params = data.get('optimizer_params', {})
    
    # Fallback values for backwards compatibility
    default_learning_rate = float(data.get('learning_rate', 0.01))
    default_momentum = float(data.get('momentum', 0.9))
    default_n_iterations = int(data.get('n_iterations', 100))
    default_use_convergence = data.get('use_convergence', False)
    default_convergence_threshold = float(data.get('convergence_threshold', 1e-6))
    default_max_iterations = int(data.get('max_iterations', 10000))
    
    # Helper function to get optimizer-specific params
    def get_optimizer_params(optimizer_name, has_momentum=False, is_adam=False, is_ballistic=False, is_ballistic_adam=False):
        params = optimizer_params.get(optimizer_name, {})
        
        if is_ballistic:
            # Ballistic uses physics parameters instead of learning rate
            result = {
                'drop_height': float(params.get('dropHeight', 5.0)),
                'gravity': float(params.get('gravity', 1.0)),
                'elasticity': float(params.get('elasticity', 0.8)),
                'bounce_threshold': float(params.get('bounceThreshold', 0.05)),
                'ball_radius': float(params.get('ballRadius', 0.05)),
                'max_iterations': int(params.get('maxIterations', 10000))
            }
            return result
        
        if is_ballistic_adam:
            # Ballistic Adam uses both gradient and physics parameters
            result = {
                'learning_rate': float(params.get('learningRate', 0.01)),
                'momentum': float(params.get('momentum', 0.9)),
                'gravity': float(params.get('gravity', 0.001)),
                'dt': float(params.get('dt', 1.0)),
                'max_air_steps': int(params.get('maxAirSteps', 20)),
                'max_bisection_iters': int(params.get('maxBisectionIters', 10)),
                'collision_tol': float(params.get('collisionTol', 1e-3)),
                'n_iterations': int(params.get('iterations', default_n_iterations)),
                'convergence_threshold': float(params.get('convergenceThreshold', default_convergence_threshold)),
                'max_iterations': int(params.get('maxIterations', default_max_iterations))
            }
            
            # Handle convergence mode
            use_convergence = params.get('useConvergence', default_use_convergence)
            if use_convergence:
                result['n_iterations'] = -1
            
            return result
        
        learning_rate = float(params.get('learningRate', default_learning_rate))
        n_iterations = int(params.get('iterations', default_n_iterations))
        use_convergence = params.get('useConvergence', default_use_convergence)
        convergence_threshold = float(params.get('convergenceThreshold', default_convergence_threshold))
        max_iterations = int(params.get('maxIterations', default_max_iterations))
        
        # If using convergence mode, pass -1 to indicate infinite iterations
        if use_convergence:
            n_iterations = -1
        
        result = {
            'learning_rate': learning_rate,
            'n_iterations': n_iterations,
            'convergence_threshold': convergence_threshold,
            'max_iterations': max_iterations
        }
        
        if has_momentum:
            result['momentum'] = float(params.get('momentum', default_momentum))
            result['lr_decay'] = float(params.get('lrDecay', 0.995))
        
        if is_adam:
            result['beta1'] = float(params.get('beta1', 0.9))
            result['beta2'] = float(params.get('beta2', 0.999))
            result['epsilon'] = float(params.get('epsilon', 1e-8))
        
        return result
    
    # Generate dataset once for consistency
    from synthetic_data import generate_synthetic_dataset
    dataset = generate_synthetic_dataset(n_samples=100, seed=seed)
    
    # Get the appropriate loss function for the selected manifold
    loss_function = get_manifold_function(manifold_id)
    
    # Get manifold parameters if provided
    manifold_params = data.get('manifold_params', {})
    
    # Always wrap for custom_multimodal to ensure parameters are explicitly passed
    def loss_wrapper(x, y):
        return loss_function(x, y, **manifold_params)
    actual_loss_function = loss_wrapper
    
    # Get manifold metadata to get bounds
    manifold_meta = get_manifold_metadata(manifold_id)
    default_range = manifold_meta['default_range']
    bounds = default_range  # Bounds as (min, max) tuple
    
    # Result dictionary
    result = {}
    
    # Timing dictionary
    optimizer_timings = {}
    start_total = time.time()
    
    # Only run enabled optimizers with their specific parameters
    if enabled_optimizers.get('sgd', False):
        params = get_optimizer_params('sgd')
        start_time = time.time()
        result['sgd'] = stochastic_gradient_descent(
            actual_loss_function,
            initial_params,
            params['learning_rate'],
            params['n_iterations'],
            dataset=dataset,
            seed=seed,
            convergence_threshold=params['convergence_threshold'],
            max_iterations=params['max_iterations'],
            bounds=bounds
        )
        optimizer_timings['sgd'] = time.time() - start_time
    
    if enabled_optimizers.get('batch', False):
        params = get_optimizer_params('batch')
        start_time = time.time()
        result['batch'] = batch_gradient_descent(
            actual_loss_function,
            initial_params,
            params['learning_rate'],
            params['n_iterations'],
            dataset=dataset,
            seed=seed,
            convergence_threshold=params['convergence_threshold'],
            max_iterations=params['max_iterations'],
            bounds=bounds
        )
        optimizer_timings['batch'] = time.time() - start_time
    
    if enabled_optimizers.get('momentum', False):
        params = get_optimizer_params('momentum', has_momentum=True)
        start_time = time.time()
        result['momentum'] = momentum_gradient_descent(
            actual_loss_function,
            initial_params,
            params['learning_rate'],
            params['momentum'],
            params['n_iterations'],
            dataset=dataset,
            seed=seed,
            convergence_threshold=params['convergence_threshold'],
            max_iterations=params['max_iterations'],
            lr_decay=params['lr_decay'],
            bounds=bounds
        )
        optimizer_timings['momentum'] = time.time() - start_time
    
    if enabled_optimizers.get('adam', False):
        params = get_optimizer_params('adam', is_adam=True)
        start_time = time.time()
        result['adam'] = adam_optimizer(
            actual_loss_function,
            initial_params,
            params['learning_rate'],
            params['n_iterations'],
            beta1=params['beta1'],
            beta2=params['beta2'],
            epsilon=params['epsilon'],
            dataset=dataset,
            seed=seed,
            convergence_threshold=params['convergence_threshold'],
            max_iterations=params['max_iterations'],
            bounds=bounds
        )
        optimizer_timings['adam'] = time.time() - start_time
    
    if enabled_optimizers.get('ballistic', False):
        params = get_optimizer_params('ballistic', is_ballistic=True)
        start_time = time.time()
        result['ballistic'] = ballistic_gradient_descent(
            actual_loss_function,
            initial_params,
            drop_height=params['drop_height'],
            gravity=params['gravity'],
            elasticity=params['elasticity'],
            bounce_threshold=params['bounce_threshold'],
            ball_radius=params['ball_radius'],
            max_iterations=params['max_iterations'],
            seed=seed,
            bounds=bounds
        )
        optimizer_timings['ballistic'] = time.time() - start_time
    
    if enabled_optimizers.get('ballistic_adam', False):
        params = get_optimizer_params('ballistic_adam', is_ballistic_adam=True)
        start_time = time.time()
        result['ballistic_adam'] = ballistic_adam_optimizer(
            actual_loss_function,
            initial_params,
            learning_rate=params['learning_rate'],
            momentum=params['momentum'],
            gravity=params['gravity'],
            dt=params['dt'],
            max_air_steps=params['max_air_steps'],
            max_bisection_iters=params['max_bisection_iters'],
            collision_tol=params['collision_tol'],
            n_iterations=params['n_iterations'],
            dataset=dataset,
            seed=seed,
            convergence_threshold=params['convergence_threshold'],
            max_iterations=params['max_iterations'],
            bounds=bounds
        )
        optimizer_timings['ballistic_adam'] = time.time() - start_time
    
    # Calculate total time
    total_time = time.time() - start_total
    
    # Add manifold info and timing to response
    result['manifold_id'] = manifold_id
    result['timings'] = {
        'total': total_time,
        'optimizers': optimizer_timings
    }
    
    # Log timing information for debugging
    print(f"[TIMING] Total: {total_time:.3f}s")
    for optimizer, timing in optimizer_timings.items():
        print(f"[TIMING]   {optimizer}: {timing:.3f}s")
    
    return jsonify(result)


@app.route('/api/manifolds', methods=['GET'])
def get_manifolds():
    """
    Get list of all available manifolds.
    """
    manifolds = list_all_manifolds()
    return jsonify({'manifolds': manifolds})


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    # Allow connections from any host when running in Docker
    # Use PORT environment variable for Cloud Run compatibility
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=False, host='0.0.0.0', port=port)

