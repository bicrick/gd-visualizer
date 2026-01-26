"""
Synthetic loss landscape functions for gradient descent visualization.
Creates 2D parameter spaces with multiple local minima.
"""

import numpy as np


def himmelblau(x, y):
    """
    Himmelblau's function - has 4 local minima at (3, 2), (-2.8, 3.1), (-3.8, -3.3), (3.6, -1.8)
    """
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2


def rastrigin(x, y, A=10):
    """
    Rastrigin function - highly multimodal with many local minima
    """
    return A * 2 + (x**2 - A * np.cos(1.5 * np.pi * x)) + (y**2 - A * np.cos(1.5 * np.pi * y))


def generate_well_positions(num_wells, radius=4.0, base_depth=2.5):
    """
    Generate well positions in a regular polygon pattern.
    
    Args:
        num_wells: Number of wells to generate
        radius: Distance from center to each well
        base_depth: Depth value for each well
    
    Returns:
        List of (x, y, depth) tuples
    """
    wells = []
    n = int(num_wells)
    
    if n == 0:
        return wells
    
    if n == 1:
        # Single well at the center
        wells.append((0, 0, base_depth))
    elif n == 2:
        # Two wells on opposite sides (top and bottom)
        wells.append((0, radius, base_depth))
        wells.append((0, -radius, base_depth))
    else:
        # Regular polygon: n >= 3
        for i in range(n):
            # Start from top (angle = -π/2) and go clockwise
            angle = (2 * np.pi * i) / n - np.pi / 2
            wx = radius * np.cos(angle)
            wy = radius * np.sin(angle)
            wells.append((wx, wy, base_depth))
    
    return wells


def custom_multimodal(x, y, global_scale=0.1, well_width=2.0, well_depth_scale=1.0, num_wells=6):
    """
    Custom function with multiple local minima (wells/valleys) for demonstration.
    Combines Gaussian wells with adjustable parameters.
    
    Args:
        x, y: coordinates in parameter space
        global_scale: multiplier for the quadratic term (default 0.1)
        well_width: controls the Gaussian width (default 2.0)
        well_depth_scale: multiplier for all well depths (default 1.0)
        num_wells: number of wells (valleys) to include (default 6)
    """
    # Global minimum at (0, 0)
    loss = (x**2 + y**2) * global_scale
    
    # Generate well positions dynamically in regular polygon pattern
    wells = generate_well_positions(num_wells, radius=4.0, base_depth=2.5)
    
    # Add wells (subtract to create valleys/local minima)
    for wx, wy, depth in wells:
        dist_sq = (x - wx)**2 + (y - wy)**2
        loss -= depth * well_depth_scale * np.exp(-dist_sq / well_width)
    
    # Add a baseline to ensure mostly positive values
    loss += 15.0
    
    return loss


def ackley(x, y):
    """
    Ackley function - highly multimodal with many local minima surrounding a deep global minimum.
    Global minimum at (0, 0) with value 0.
    """
    a = 20
    b = 0.2
    c = 2 * np.pi
    
    term1 = -a * np.exp(-b * np.sqrt(0.5 * (x**2 + y**2)))
    term2 = -np.exp(0.5 * (np.cos(c * x) + np.cos(c * y)))
    
    return term1 + term2 + a + np.e








def compute_gradient(func, x, y, h=1e-5):
    """
    Compute numerical gradient of a function at point (x, y).
    """
    fx_plus = func(x + h, y)
    fx_minus = func(x - h, y)
    grad_x = (fx_plus - fx_minus) / (2 * h)
    
    fy_plus = func(x, y + h)
    fy_minus = func(x, y - h)
    grad_y = (fy_plus - fy_minus) / (2 * h)
    
    return np.array([grad_x, grad_y])


def generate_landscape_mesh(func, x_range=(-5, 5), y_range=(-5, 5), resolution=100, func_params=None):
    """
    Generate a mesh grid of loss values for 3D visualization.
    
    Args:
        func: Loss function that takes (x, y) and returns loss value
        x_range: Tuple of (min, max) for x-axis
        y_range: Tuple of (min, max) for y-axis
        resolution: Number of points per axis
        func_params: Optional dictionary of additional parameters to pass to the function
    
    Returns:
        Dictionary with 'x', 'y', 'z' (loss) arrays for mesh plotting
    """
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Vectorize the function for efficiency
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if func_params:
                Z[i, j] = func(X[i, j], Y[i, j], **func_params)
            else:
                Z[i, j] = func(X[i, j], Y[i, j])
    
    return {
        'x': X.tolist(),
        'y': Y.tolist(),
        'z': Z.tolist(),
        'x_range': x_range,
        'y_range': y_range
    }


# Manifold registry - maps manifold IDs to function and metadata
MANIFOLD_REGISTRY = {
    'custom_multimodal': {
        'function': custom_multimodal,
        'name': 'Gaussian Wells',
        'description': 'Multiple Gaussian wells with adjustable parameters',
        'default_range': (-5, 5),
        'parameters': [
            {'name': 'global_scale', 'label': 'Global Scale', 'min': 0.0, 'max': 0.5, 'step': 0.01, 'default': 0.1},
            {'name': 'well_width', 'label': 'Well Width', 'min': 0.5, 'max': 5.0, 'step': 0.1, 'default': 2.0},
            {'name': 'well_depth_scale', 'label': 'Well Depth', 'min': 0.1, 'max': 3.0, 'step': 0.1, 'default': 1.0},
            {'name': 'num_wells', 'label': 'Number of Wells', 'min': 1, 'max': 6, 'step': 1, 'default': 6}
        ]
    },
    'himmelblau': {
        'function': himmelblau,
        'name': 'Himmelblau',
        'description': '4 distinct local minima',
        'default_range': (-5, 5),
    },
    'rastrigin': {
        'function': rastrigin,
        'name': 'Rastrigin',
        'description': 'Highly multimodal with regular pattern',
        'default_range': (-5, 5),
    },
    'ackley': {
        'function': ackley,
        'name': 'Ackley',
        'description': 'Corrugated surface with deep central minimum',
        'default_range': (-5, 5),
    },
}


def get_manifold_function(manifold_id):
    """
    Get the loss function for a given manifold ID.
    
    Args:
        manifold_id: String identifier for the manifold
    
    Returns:
        Loss function that takes (x, y) and returns loss value
    """
    if manifold_id not in MANIFOLD_REGISTRY:
        # Default to custom_multimodal if invalid ID
        manifold_id = 'custom_multimodal'
    
    return MANIFOLD_REGISTRY[manifold_id]['function']


def get_manifold_metadata(manifold_id):
    """
    Get metadata for a given manifold ID.
    
    Args:
        manifold_id: String identifier for the manifold
    
    Returns:
        Dictionary with name, description, default_range, and optional parameters
    """
    if manifold_id not in MANIFOLD_REGISTRY:
        manifold_id = 'custom_multimodal'
    
    metadata = {
        'id': manifold_id,
        'name': MANIFOLD_REGISTRY[manifold_id]['name'],
        'description': MANIFOLD_REGISTRY[manifold_id]['description'],
        'default_range': MANIFOLD_REGISTRY[manifold_id]['default_range'],
    }
    
    # Include parameters if they exist
    if 'parameters' in MANIFOLD_REGISTRY[manifold_id]:
        metadata['parameters'] = MANIFOLD_REGISTRY[manifold_id]['parameters']
    
    return metadata


def list_all_manifolds():
    """
    List all available manifolds with their metadata.
    
    Returns:
        List of dictionaries containing manifold information
    """
    return [get_manifold_metadata(manifold_id) for manifold_id in MANIFOLD_REGISTRY.keys()]


# Default function to use
DEFAULT_LOSS_FUNCTION = custom_multimodal
DEFAULT_MANIFOLD_ID = 'custom_multimodal'

