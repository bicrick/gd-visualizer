/**
 * Landscape function library - client-side landscape generation
 * Ported from backend/loss_functions.py for instant parameter updates
 */

/**
 * Himmelblau's function - has 4 local minima
 */
function himmelblau(x, y) {
    return Math.pow(x * x + y - 11, 2) + Math.pow(x + y * y - 7, 2);
}

/**
 * Rastrigin function - highly multimodal with many local minima
 */
function rastrigin(x, y, A = 10) {
    return A * 2 + 
           (x * x - A * Math.cos(1.5 * Math.PI * x)) + 
           (y * y - A * Math.cos(1.5 * Math.PI * y));
}

/**
 * Generate well positions in a regular polygon pattern
 * @param {number} num_wells - Number of wells to generate
 * @param {number} radius - Distance from center to each well
 * @param {number} base_depth - Depth value for each well
 * @returns {Array} Array of [x, y, depth] well positions
 */
function generateWellPositions(num_wells, radius = 4.0, base_depth = 2.5) {
    const wells = [];
    const n = Math.floor(num_wells);
    
    if (n === 0) {
        return wells;
    }
    
    if (n === 1) {
        // Single well at the center
        wells.push([0, 0, base_depth]);
    } else if (n === 2) {
        // Two wells on opposite sides (top and bottom)
        wells.push([0, radius, base_depth]);
        wells.push([0, -radius, base_depth]);
    } else {
        // Regular polygon: n >= 3
        for (let i = 0; i < n; i++) {
            // Start from top (angle = -π/2) and go clockwise
            const angle = (2 * Math.PI * i) / n - Math.PI / 2;
            const wx = radius * Math.cos(angle);
            const wy = radius * Math.sin(angle);
            wells.push([wx, wy, base_depth]);
        }
    }
    
    return wells;
}

/**
 * Custom function with multiple local minima (wells/valleys)
 * Combines Gaussian wells with adjustable parameters
 */
function custom_multimodal(x, y, global_scale = 0.1, well_width = 2.0, well_depth_scale = 1.0, num_wells = 6) {
    // Global minimum at (0, 0)
    let loss = (x * x + y * y) * global_scale;
    
    // Generate well positions dynamically in regular polygon pattern
    const wells = generateWellPositions(num_wells, 4.0, 2.5);
    
    // Add wells (subtract to create valleys/local minima)
    for (const [wx, wy, depth] of wells) {
        const dist_sq = (x - wx) * (x - wx) + (y - wy) * (y - wy);
        loss -= depth * well_depth_scale * Math.exp(-dist_sq / well_width);
    }
    
    // Add a baseline to ensure mostly positive values
    loss += 15.0;
    
    return loss;
}

/**
 * Ackley function - highly multimodal with deep global minimum
 * Global minimum at (0, 0) with value 0
 */
function ackley(x, y) {
    const a = 20;
    const b = 0.2;
    const c = 2 * Math.PI;
    
    const term1 = -a * Math.exp(-b * Math.sqrt(0.5 * (x * x + y * y)));
    const term2 = -Math.exp(0.5 * (Math.cos(c * x) + Math.cos(c * y)));
    
    return term1 + term2 + a + Math.E;
}

/**
 * Generate a mesh grid of loss values for 3D visualization
 * 
 * @param {Function} func - Loss function that takes (x, y) and returns loss value
 * @param {Array} x_range - [min, max] for x-axis
 * @param {Array} y_range - [min, max] for y-axis
 * @param {number} resolution - Number of points per axis
 * @param {Object} func_params - Optional parameters to pass to the function
 * @returns {Promise<Object>} Dictionary with 'x', 'y', 'z' arrays for mesh plotting
 */
async function generateLandscapeMesh(func, x_range = [-5, 5], y_range = [-5, 5], resolution = 100, func_params = null) {
    const x_min = x_range[0];
    const x_max = x_range[1];
    const y_min = y_range[0];
    const y_max = y_range[1];
    
    // Create linearly spaced arrays
    const x_vals = [];
    const y_vals = [];
    
    for (let i = 0; i < resolution; i++) {
        x_vals.push(x_min + (x_max - x_min) * i / (resolution - 1));
        y_vals.push(y_min + (y_max - y_min) * i / (resolution - 1));
    }
    
    // Create mesh grid
    const X = [];
    const Y = [];
    const Z = [];
    
    for (let i = 0; i < resolution; i++) {
        X.push([]);
        Y.push([]);
        Z.push([]);
        
        for (let j = 0; j < resolution; j++) {
            X[i].push(x_vals[j]);
            Y[i].push(y_vals[i]);
            
            // Compute loss value
            let z_val;
            if (func_params) {
                z_val = await func(x_vals[j], y_vals[i], 
                    func_params.global_scale,
                    func_params.well_width,
                    func_params.well_depth_scale,
                    func_params.num_wells
                );
            } else {
                z_val = await func(x_vals[j], y_vals[i]);
            }
            
            Z[i].push(z_val);
        }
    }
    
    return {
        x: X,
        y: Y,
        z: Z,
        x_range: x_range,
        y_range: y_range
    };
}

/**
 * Manifold registry - maps manifold IDs to functions and metadata
 */
const MANIFOLD_FUNCTIONS = {
    'custom_multimodal': custom_multimodal,
    'himmelblau': himmelblau,
    'rastrigin': rastrigin,
    'ackley': ackley
};

/**
 * Get the landscape function for a given manifold ID
 */
function getManifoldFunction(manifold_id) {
    return MANIFOLD_FUNCTIONS[manifold_id] || custom_multimodal;
}

/**
 * Generate landscape mesh for a specific manifold
 * 
 * @param {string} manifold_id - ID of the manifold to generate
 * @param {number} resolution - Grid resolution (default 80)
 * @param {Array} x_range - X-axis range
 * @param {Array} y_range - Y-axis range
 * @param {Object} params - Optional manifold parameters
 * @returns {Promise<Object>} Mesh data
 */
async function generateManifoldLandscape(manifold_id, resolution = 80, x_range = [-5, 5], y_range = [-5, 5], params = null) {
    const func = getManifoldFunction(manifold_id);
    return await generateLandscapeMesh(func, x_range, y_range, resolution, params);
}

// Export functions to global scope
window.generateManifoldLandscape = generateManifoldLandscape;
window.generateLandscapeMesh = generateLandscapeMesh;
window.getManifoldFunction = getManifoldFunction;
