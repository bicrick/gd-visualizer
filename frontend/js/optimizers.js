/**
 * Optimizer visualization - animated balls and trajectory trails
 */

// Use API_BASE_URL from scene.js (defined globally)
// If not defined, fallback to relative URL
if (typeof API_BASE_URL === 'undefined') {
    window.API_BASE_URL = window.location.origin + '/api';
}

let optimizerBalls = {};
let trajectoryLines = {};
// Make currentTrajectories globally accessible
let currentTrajectories = null;
window.currentTrajectories = currentTrajectories;
let animationState = {
    isPlaying: false,
    currentStep: 0,
    speed: 1.0
};

// Track which optimizers are enabled
let enabledOptimizers = {
    sgd: true,
    batch: true,
    momentum: true
};

// Colors for each optimizer
const OPTIMIZER_COLORS = {
    sgd: 0xff4444,
    batch: 0x4444ff,
    momentum: 0x44ff44
};

// Create ball for an optimizer
function createOptimizerBall(name, color) {
    const geometry = new THREE.SphereGeometry(0.15, 16, 16);
    const material = new THREE.MeshPhongMaterial({ 
        color: color,
        emissive: color,
        emissiveIntensity: 0.3
    });
    const ball = new THREE.Mesh(geometry, material);
    ball.visible = false;
    scene.add(ball);
    return ball;
}

// Create trajectory line for an optimizer
function createTrajectoryLine(name, color) {
    const material = new THREE.LineBasicMaterial({ 
        color: color,
        linewidth: 2,
        opacity: 0.6,
        transparent: true
    });
    const geometry = new THREE.BufferGeometry();
    const line = new THREE.Line(geometry, material);
    line.visible = false;
    scene.add(line);
    return line;
}

// Initialize optimizer visualizations
function initOptimizers() {
    Object.keys(OPTIMIZER_COLORS).forEach(name => {
        optimizerBalls[name] = createOptimizerBall(name, OPTIMIZER_COLORS[name]);
        trajectoryLines[name] = createTrajectoryLine(name, OPTIMIZER_COLORS[name]);
    });
}

// Convert parameter space (x, y) to 3D world coordinates
function paramsToWorldCoords(x, y, loss) {
    // Map from parameter space to world space
    // The mesh is rotated -90° on X axis, so we need to account for that
    const worldX = x;
    const worldZ = y; // y in param space becomes z in world space (after rotation)
    
    // Get height from landscape mesh if available
    let worldY = 0;
    if (landscapeMesh && landscapeGeometry) {
        // Get current manifold range dynamically
        const range = window.getCurrentManifoldRange ? window.getCurrentManifoldRange() : [-5, 5];
        const rangeMin = range[0];
        const rangeMax = range[1];
        const rangeSize = rangeMax - rangeMin;
        
        // Sample height from landscape using bilinear interpolation
        // Map parameter space using actual range to normalized [0, 1]
        const normalizedX = (x - rangeMin) / rangeSize;
        const normalizedY = (y - rangeMin) / rangeSize;
        
        const positions = landscapeGeometry.attributes.position;
        const gridSize = Math.sqrt(positions.count);
        
        // Clamp to valid range
        const u = Math.max(0, Math.min(1, normalizedX));
        const v = Math.max(0, Math.min(1, normalizedY));
        
        // Get grid indices (col corresponds to X, row corresponds to Y in geometry space)
        const col = Math.floor(u * (gridSize - 1));
        const row = Math.floor(v * (gridSize - 1));
        
        // Get the four corners for bilinear interpolation
        const idx00 = row * gridSize + col;
        const idx01 = row * gridSize + Math.min(col + 1, gridSize - 1);
        const idx10 = Math.min(row + 1, gridSize - 1) * gridSize + col;
        const idx11 = Math.min(row + 1, gridSize - 1) * gridSize + Math.min(col + 1, gridSize - 1);
        
        // Interpolation weights
        const fx = u * (gridSize - 1) - col;
        const fy = v * (gridSize - 1) - row;
        
        // CRITICAL FIX: Use getZ() not getY() because Z is the height in the geometry
        // Before rotation: PlaneGeometry has X, Y as the plane, Z as height
        // After -90° rotation on X: X stays X, old Y becomes Z, old Z becomes Y
        const h00 = positions.getZ(idx00);
        const h01 = positions.getZ(idx01);
        const h10 = positions.getZ(idx10);
        const h11 = positions.getZ(idx11);
        
        // Bilinear interpolation
        const h0 = h00 * (1 - fx) + h01 * fx;
        const h1 = h10 * (1 - fx) + h11 * fx;
        const interpolatedHeight = h0 * (1 - fy) + h1 * fy;
        
        // After mesh rotation, the Z coordinate becomes Y coordinate
        worldY = interpolatedHeight + 0.2; // Slight offset above surface
    }
    
    return new THREE.Vector3(worldX, worldY, worldZ);
}

// Update trajectory line geometry
function updateTrajectoryLine(name, trajectory) {
    const line = trajectoryLines[name];
    if (!line || !trajectory || trajectory.length === 0) {
        if (line) {  // Only set visible if line exists
            line.visible = false;
        }
        return;
    }
    
    const points = trajectory.map(([x, y, loss]) => paramsToWorldCoords(x, y, loss));
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    
    line.geometry.dispose();
    line.geometry = geometry;
    
    // Only show if this optimizer is enabled and trails are enabled
    const showTrails = document.getElementById('show-trails')?.checked ?? true;
    line.visible = enabledOptimizers[name] && showTrails;
}

// Set ball position
function setBallPosition(name, x, y, loss) {
    const ball = optimizerBalls[name];
    if (!ball) return;
    
    // Only show if this optimizer is enabled
    if (!enabledOptimizers[name]) {
        ball.visible = false;
        return;
    }
    
    const worldPos = paramsToWorldCoords(x, y, loss);
    ball.position.copy(worldPos);
    ball.visible = true;
}

// Run optimization and get trajectories
async function runOptimization(params) {
    try {
        const apiUrl = typeof API_BASE_URL !== 'undefined' ? API_BASE_URL : window.API_BASE_URL;
        const response = await fetch(`${apiUrl}/optimize`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(params)
        });
        
        const data = await response.json();
        
        // Clear trajectories for disabled optimizers
        const allOptimizers = ['sgd', 'batch', 'momentum'];
        allOptimizers.forEach(name => {
            if (!data[name]) {
                // Clear trajectory for disabled optimizer
                const line = trajectoryLines[name];
                if (line) {
                    line.visible = false;
                }
                const ball = optimizerBalls[name];
                if (ball) {
                    ball.visible = false;
                }
            }
        });
        
        currentTrajectories = data;
        window.currentTrajectories = data; // Make globally accessible
        
        // Update trajectory lines only for actual optimizer names
        const validOptimizers = ['sgd', 'batch', 'momentum'];
        Object.keys(data)
            .filter(name => validOptimizers.includes(name))
            .forEach(name => {
                updateTrajectoryLine(name, data[name]);
            });
        
        // Reset animation
        animationState.currentStep = 0;
        
        return data;
    } catch (error) {
        console.error('Error running optimization:', error);
        alert('Error running optimization. Make sure backend is running.');
        return null;
    }
}
window.runOptimization = runOptimization; // Make globally accessible

// Expose function to get enabled optimizers
function getEnabledOptimizers() {
    return { ...enabledOptimizers };
}
window.getEnabledOptimizers = getEnabledOptimizers;

// Animate optimizers along their trajectories
function animateOptimizers() {
    if (!currentTrajectories || !animationState.isPlaying) {
        return;
    }
    
    // Only consider trajectories that exist (were computed)
    const existingTrajectories = Object.keys(currentTrajectories).filter(name => 
        currentTrajectories[name] && currentTrajectories[name].length > 0
    );
    
    if (existingTrajectories.length === 0) {
        animationState.isPlaying = false;
        return;
    }
    
    const maxSteps = Math.max(
        ...existingTrajectories.map(name => currentTrajectories[name]?.length || 0)
    );
    
    if (animationState.currentStep >= maxSteps) {
        animationState.isPlaying = false;
        return;
    }
    
    // Update each optimizer's position (only for those with trajectories)
    existingTrajectories.forEach(name => {
        const trajectory = currentTrajectories[name];
        if (trajectory && trajectory.length > 0) {
            const step = Math.floor(Math.min(animationState.currentStep, trajectory.length - 1));
            const [x, y, loss] = trajectory[step];
            setBallPosition(name, x, y, loss);
        }
    });
    
    // Advance step based on speed
    animationState.currentStep += animationState.speed;
}

// Start animation
function startAnimation() {
    animationState.isPlaying = true;
}
window.startAnimation = startAnimation; // Make globally accessible

// Pause animation
function pauseAnimation() {
    animationState.isPlaying = false;
}
window.pauseAnimation = pauseAnimation; // Make globally accessible

// Reset animation
function resetAnimation() {
    animationState.currentStep = 0;
    animationState.isPlaying = false;
    
    // Hide balls
    Object.values(optimizerBalls).forEach(ball => {
        ball.visible = false;
    });
}
window.resetAnimation = resetAnimation; // Make globally accessible

// Clear all trajectories (used when changing manifold)
function clearTrajectories() {
    currentTrajectories = null;
    window.currentTrajectories = null;
    animationState.currentStep = 0;
    animationState.isPlaying = false;
    
    // Hide and clear balls
    Object.values(optimizerBalls).forEach(ball => {
        ball.visible = false;
    });
    
    // Hide and clear trajectory lines
    Object.values(trajectoryLines).forEach(line => {
        line.visible = false;
        if (line.geometry) {
            line.geometry.setFromPoints([]);
        }
    });
}
window.clearTrajectories = clearTrajectories; // Make globally accessible

// Set animation speed
function setAnimationSpeed(speed) {
    animationState.speed = speed;
}
window.setAnimationSpeed = setAnimationSpeed; // Make globally accessible

// Toggle trajectory visibility
function setTrajectoryVisibility(visible) {
    Object.keys(trajectoryLines).forEach(name => {
        const line = trajectoryLines[name];
        // Only show if optimizer is enabled and visibility is on
        line.visible = visible && enabledOptimizers[name] && line.geometry.attributes.position?.count > 0;
    });
}
window.setTrajectoryVisibility = setTrajectoryVisibility; // Make globally accessible

// Toggle individual optimizer
function toggleOptimizer(name, enabled) {
    enabledOptimizers[name] = enabled;
    
    // Update ball visibility
    const ball = optimizerBalls[name];
    if (ball && !enabled) {
        ball.visible = false;
    }
    
    // Update trajectory visibility
    const line = trajectoryLines[name];
    if (line) {
        const showTrails = document.getElementById('show-trails')?.checked ?? true;
        line.visible = enabled && showTrails && line.geometry.attributes.position?.count > 0;
    }
}
window.toggleOptimizer = toggleOptimizer; // Make globally accessible

// Animation loop for optimizers
function animateOptimizerLoop() {
    if (animationState.isPlaying) {
        animateOptimizers();
    }
    requestAnimationFrame(animateOptimizerLoop);
}

// Initialize on load
window.addEventListener('DOMContentLoaded', () => {
    // Wait for scene to be ready
    setTimeout(() => {
        initOptimizers();
        // Set initial visibility based on enabled state
        Object.keys(enabledOptimizers).forEach(name => {
            if (!enabledOptimizers[name]) {
                const ball = optimizerBalls[name];
                if (ball) ball.visible = false;
                const line = trajectoryLines[name];
                if (line) line.visible = false;
            }
        });
        animateOptimizerLoop();
    }, 100);
});

