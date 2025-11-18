"""
Flask API server for gradient descent visualization backend.
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os
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
    adam_optimizer
)

# Determine the frontend path relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, '..', 'frontend')

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')
# Enable CORS for all routes and origins (though not needed when serving from same origin)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Serve frontend files
@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/favicon.ico')
def favicon():
    # Return a 204 No Content response to prevent favicon 404 errors
    return '', 204

@app.route('/<path:path>')
def serve_static(path):
    # Don't serve API routes as static files
    if path.startswith('api/'):
        return None
    # Serve CSS, JS, and other static files
    return send_from_directory(FRONTEND_DIR, path)


@app.route('/api/landscape', methods=['GET'])
def get_landscape():
    """
    Generate and return loss landscape mesh data for 3D visualization.
    """
    manifold_id = request.args.get('manifold', DEFAULT_MANIFOLD_ID)
    resolution = int(request.args.get('resolution', 100))
    
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
        resolution=resolution
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
        'adam': True
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
    def get_optimizer_params(optimizer_name, has_momentum=False, is_adam=False):
        params = optimizer_params.get(optimizer_name, {})
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
    
    # Get manifold metadata to get bounds
    manifold_meta = get_manifold_metadata(manifold_id)
    default_range = manifold_meta['default_range']
    bounds = default_range  # Bounds as (min, max) tuple
    
    # Result dictionary
    result = {}
    
    # Only run enabled optimizers with their specific parameters
    if enabled_optimizers.get('sgd', False):
        params = get_optimizer_params('sgd')
        result['sgd'] = stochastic_gradient_descent(
            loss_function,
            initial_params,
            params['learning_rate'],
            params['n_iterations'],
            dataset=dataset,
            seed=seed,
            convergence_threshold=params['convergence_threshold'],
            max_iterations=params['max_iterations'],
            bounds=bounds
        )
    
    if enabled_optimizers.get('batch', False):
        params = get_optimizer_params('batch')
        result['batch'] = batch_gradient_descent(
            loss_function,
            initial_params,
            params['learning_rate'],
            params['n_iterations'],
            dataset=dataset,
            seed=seed,
            convergence_threshold=params['convergence_threshold'],
            max_iterations=params['max_iterations'],
            bounds=bounds
        )
    
    if enabled_optimizers.get('momentum', False):
        params = get_optimizer_params('momentum', has_momentum=True)
        result['momentum'] = momentum_gradient_descent(
            loss_function,
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
    
    if enabled_optimizers.get('adam', False):
        params = get_optimizer_params('adam', is_adam=True)
        result['adam'] = adam_optimizer(
            loss_function,
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
    
    # Add manifold info to response
    result['manifold_id'] = manifold_id
    
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
    app.run(debug=True, host='0.0.0.0', port=5000)

