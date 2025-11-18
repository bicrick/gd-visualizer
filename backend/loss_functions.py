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
    return A * 2 + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y))


def custom_multimodal(x, y):
    """
    Custom function with multiple local minima for demonstration.
    Combines multiple Gaussian-like wells.
    """
    # Global minimum at (0, 0)
    loss = (x**2 + y**2) * 0.1
    
    # Local minima at various points
    wells = [
        (-3, -3, 2.0),
        (3, -3, 2.5),
        (-3, 3, 2.2),
        (3, 3, 2.8),
        (0, -4, 1.5),
        (0, 4, 1.8),
    ]
    
    for wx, wy, depth in wells:
        dist_sq = (x - wx)**2 + (y - wy)**2
        loss += depth * np.exp(-dist_sq / 2.0)
    
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


def rosenbrock(x, y):
    """
    Rosenbrock function - the famous 'banana valley' function.
    Global minimum at (1, 1) with value 0.
    Good for showing how optimizers get stuck in narrow valleys.
    """
    return (1 - x)**2 + 100 * (y - x**2)**2


def beale(x, y):
    """
    Beale function - has deep valleys and peaks.
    Global minimum at (3, 0.5) with value 0.
    """
    term1 = (1.5 - x + x*y)**2
    term2 = (2.25 - x + x*y**2)**2
    term3 = (2.625 - x + x*y**3)**2
    
    return term1 + term2 + term3


def eggholder(x, y):
    """
    Eggholder function - very complex landscape with many local minima.
    Global minimum at (512, 404.2319) with value -959.6407.
    Scaled down for better visualization in [-5, 5] range.
    """
    # Scale inputs to explore the interesting region
    x_scaled = x * 100
    y_scaled = y * 100
    
    term1 = -(y_scaled + 47) * np.sin(np.sqrt(abs(x_scaled/2 + (y_scaled + 47))))
    term2 = -x_scaled * np.sin(np.sqrt(abs(x_scaled - (y_scaled + 47))))
    
    # Scale output to be comparable to other functions
    return (term1 + term2) / 100


def three_hump_camel(x, y):
    """
    Three-Hump Camel function - has 3 local minima.
    Global minimum at (0, 0) with value 0.
    """
    return 2*x**2 - 1.05*x**4 + (x**6)/6 + x*y + y**2


def six_hump_camel(x, y):
    """
    Six-Hump Camel function - has 6 local minima.
    Global minima at (0.0898, -0.7126) and (-0.0898, 0.7126) with value -1.0316.
    """
    term1 = (4 - 2.1*x**2 + (x**4)/3) * x**2
    term2 = x*y
    term3 = (-4 + 4*y**2) * y**2
    
    return term1 + term2 + term3


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


def generate_landscape_mesh(func, x_range=(-5, 5), y_range=(-5, 5), resolution=100):
    """
    Generate a mesh grid of loss values for 3D visualization.
    
    Args:
        func: Loss function that takes (x, y) and returns loss value
        x_range: Tuple of (min, max) for x-axis
        y_range: Tuple of (min, max) for y-axis
        resolution: Number of points per axis
    
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
        'name': 'Custom Multimodal',
        'description': 'Multiple Gaussian wells with local minima',
        'default_range': (-5, 5),
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
    'rosenbrock': {
        'function': rosenbrock,
        'name': 'Rosenbrock',
        'description': 'Banana valley - narrow curved minimum',
        'default_range': (-2, 2),
    },
    'beale': {
        'function': beale,
        'name': 'Beale',
        'description': 'Deep valleys and peaks',
        'default_range': (-4.5, 4.5),
    },
    'eggholder': {
        'function': eggholder,
        'name': 'Eggholder',
        'description': 'Very complex organic landscape',
        'default_range': (-5, 5),
    },
    'three_hump_camel': {
        'function': three_hump_camel,
        'name': 'Three-Hump Camel',
        'description': '3 humps creating local minima',
        'default_range': (-2, 2),
    },
    'six_hump_camel': {
        'function': six_hump_camel,
        'name': 'Six-Hump Camel',
        'description': '6 local minima',
        'default_range': (-2, 2),
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
        Dictionary with name, description, and default_range
    """
    if manifold_id not in MANIFOLD_REGISTRY:
        manifold_id = 'custom_multimodal'
    
    return {
        'id': manifold_id,
        'name': MANIFOLD_REGISTRY[manifold_id]['name'],
        'description': MANIFOLD_REGISTRY[manifold_id]['description'],
        'default_range': MANIFOLD_REGISTRY[manifold_id]['default_range'],
    }


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

