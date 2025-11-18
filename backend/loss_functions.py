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


def custom_multimodal(x, y, global_scale=0.1, well_width=2.0, well_depth_scale=1.0, num_wells=6):
    """
    Custom function with multiple local minima (wells/valleys) for demonstration.
    Combines Gaussian wells with adjustable parameters.
    
    Args:
        x, y: coordinates in parameter space
        global_scale: multiplier for the quadratic term (default 0.1)
        well_width: controls the Gaussian width (default 2.0)
        well_depth_scale: multiplier for all well depths (default 1.0)
        num_wells: number of wells (valleys) to include (default 6, max 6)
    """
    # Global minimum at (0, 0)
    loss = (x**2 + y**2) * global_scale
    
    # Well positions (these will be valleys/local minima)
    all_wells = [
        (-3, -3, 2.0),
        (3, -3, 2.5),
        (-3, 3, 2.2),
        (3, 3, 2.8),
        (0, -4, 1.5),
        (0, 4, 1.8),
    ]
    
    # Use only the first num_wells
    wells = all_wells[:int(num_wells)]
    
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


# Circle Classifier Functions
def generate_circle_dataset(n_samples=200, seed=42):
    """
    Generate TWO separate circular clusters for binary classification.
    This creates a non-convex optimization problem with two local minima.
    
    Class 0 (green): Points in TWO circular clusters
        - Cluster A: centered at (-1.5, 0), radius ~0.6
        - Cluster B: centered at (1.5, 0), radius ~0.6
    Class 1 (orange): Points scattered uniformly throughout the space
    
    Returns data points and labels.
    """
    np.random.seed(seed)
    
    # Allocate samples: 25% cluster A, 25% cluster B, 50% scattered  
    # More green points in each cluster to make them more attractive
    n_cluster_a = int(n_samples * 0.25)
    n_cluster_b = int(n_samples * 0.25)
    n_scattered = n_samples - n_cluster_a - n_cluster_b
    
    # Generate Cluster A (left circle) - class 0  
    # Far apart so circle can't capture both
    cluster_a_center = (-1.8, 0.8)
    cluster_a_radius = 0.4
    cluster_a_noise = 0.08
    angles_a = np.random.uniform(0, 2 * np.pi, n_cluster_a)
    radii_a = np.random.normal(cluster_a_radius, cluster_a_noise, n_cluster_a)
    cluster_a_x = cluster_a_center[0] + radii_a * np.cos(angles_a)
    cluster_a_y = cluster_a_center[1] + radii_a * np.sin(angles_a)
    
    # Generate Cluster B (right circle) - class 0
    # Offset vertically and make slightly larger for asymmetry
    cluster_b_center = (1.8, -0.8)
    cluster_b_radius = 0.45  # Slightly larger = slightly better minimum
    cluster_b_noise = 0.08
    angles_b = np.random.uniform(0, 2 * np.pi, n_cluster_b)
    radii_b = np.random.normal(cluster_b_radius, cluster_b_noise, n_cluster_b)
    cluster_b_x = cluster_b_center[0] + radii_b * np.cos(angles_b)
    cluster_b_y = cluster_b_center[1] + radii_b * np.sin(angles_b)
    
    # Generate scattered points (class 1 - orange)
    # Make them denser in a more focused region to penalize middle positions
    scatter_range_x = 2.8
    scatter_range_y = 1.8  # Narrower vertically
    scatter_x = np.random.uniform(-scatter_range_x, scatter_range_x, n_scattered)
    scatter_y = np.random.uniform(-scatter_range_y, scatter_range_y, n_scattered)
    
    # Combine all points
    X = np.vstack([
        np.column_stack([cluster_a_x, cluster_a_y]),
        np.column_stack([cluster_b_x, cluster_b_y]),
        np.column_stack([scatter_x, scatter_y])
    ])
    
    # Labels: 0 for both clusters, 1 for scattered
    y = np.hstack([
        np.zeros(n_cluster_a),
        np.zeros(n_cluster_b),
        np.ones(n_scattered)
    ])
    
    return X, y


# Global dataset for neural network classifier (cached)
_CLASSIFIER_DATASET = None

def get_classifier_dataset():
    """Get or create the classification dataset."""
    global _CLASSIFIER_DATASET
    if _CLASSIFIER_DATASET is None:
        _CLASSIFIER_DATASET = generate_circle_dataset(n_samples=200, seed=42)
    return _CLASSIFIER_DATASET


def sigmoid(x):
    """Sigmoid activation function with numerical stability."""
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )


def neural_net_classifier_loss(center_x, center_y):
    """
    Circle-based binary classifier with 2 learnable parameters.
    
    The classifier is a circle centered at (center_x, center_y) with fixed radius.
    Points inside the circle are classified as class 0 (green - the noisy circle).
    Points outside the circle are classified as class 1 (orange - scattered points).
    
    Parameters to optimize:
        center_x: X-coordinate of the circle center
        center_y: Y-coordinate of the circle center
    
    Loss: Binary cross-entropy based on distance from circle center
    """
    X, y = get_classifier_dataset()
    
    # Fixed classifier radius (smaller so it can only capture one cluster at a time)
    classifier_radius = 1.2
    
    # Compute distance from each point to the circle center
    distances = np.sqrt((X[:, 0] - center_x)**2 + (X[:, 1] - center_y)**2)
    
    # Classification: smooth sigmoid based on distance relative to radius
    # Negative distance_from_boundary means inside (class 0)
    # Positive distance_from_boundary means outside (class 1)
    steepness = 2.0
    distance_from_boundary = (distances - classifier_radius) * steepness
    
    # Sigmoid: inside circle → 0, outside → 1
    predictions = sigmoid(distance_from_boundary)
    
    # Binary cross-entropy loss with clipping for numerical stability
    epsilon = 1e-7
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    
    loss = -np.mean(
        y * np.log(predictions) + (1 - y) * np.log(1 - predictions)
    )
    
    # Add barrier penalty: positions near the middle (saddle region) should have higher loss
    # This creates two distinct basins of attraction
    dist_from_origin = np.sqrt(center_x**2 + center_y**2)
    # Penalty is highest at origin, decreases as you move toward clusters
    barrier_penalty = 1.5 * np.exp(-dist_from_origin)  # Exponential barrier at center
    
    # Small regularization to prevent going too far out
    edge_regularization = 0.005 * (center_x**2 + center_y**2)
    
    # Scale to create more dramatic landscape
    # Add small offset to prevent loss from reaching exactly 0
    baseline = 0.4
    scaled_loss = (loss + barrier_penalty + edge_regularization - baseline) * 5.0
    
    # Ensure always slightly positive to avoid convergence issues
    return max(0.05, scaled_loss)


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
    'neural_net_classifier': {
        'function': neural_net_classifier_loss,
        'name': 'Two-Circle Clustering',
        'description': 'Find best circle position - demonstrates local minima problem!',
        'default_range': (-3, 3),
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

