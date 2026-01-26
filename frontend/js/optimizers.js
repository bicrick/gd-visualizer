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

// Segmented ball management for overlapping optimizers
let segmentedBalls = []; // Pool of reusable segmented balls
const OVERLAP_THRESHOLD = 0.01; // Very tight threshold for "directly on top of each other"

// Track which optimizers are enabled
let enabledOptimizers = {
    sgd: false,
    batch: true,
    momentum: true,
    adam: true,
    ballistic: false,
    ballistic_adam: false
};

// Colors for each optimizer
const OPTIMIZER_COLORS = {
    sgd: 0xff4444,
    batch: 0x4444ff,
    momentum: 0x44ff44,
    adam: 0xff8800,
    ballistic: 0xff00ff,
    ballistic_adam: 0x00ffff
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
    // Get current manifold range dynamically
    const range = window.getCurrentManifoldRange ? window.getCurrentManifoldRange() : [-5, 5];
    const rangeMin = range[0];
    const rangeMax = range[1];
    const rangeSize = rangeMax - rangeMin;
    
    // Map from parameter space [rangeMin, rangeMax] to world space [-5, 5]
    // The mesh is rotated -90° on X axis, so we need to account for that
    const worldX = ((x - rangeMin) / rangeSize - 0.5) * 10;
    const worldZ = ((y - rangeMin) / rangeSize - 0.5) * 10;
    
    // Get height from landscape mesh if available
    let worldY = 0;
    if (landscapeMesh && landscapeGeometry) {
        
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

// Convert ballistic trajectory point (x, y, z_world) to 3D world coordinates
// For ballistic optimizer, z is already the 3D world height, not loss
function ballisticToWorldCoords(x, y, z_world) {
    // Get current manifold range dynamically
    const range = window.getCurrentManifoldRange ? window.getCurrentManifoldRange() : [-5, 5];
    const rangeMin = range[0];
    const rangeMax = range[1];
    const rangeSize = rangeMax - rangeMin;
    
    // Map from parameter space [rangeMin, rangeMax] to world space [-5, 5]
    const worldX = ((x - rangeMin) / rangeSize - 0.5) * 10;
    const worldZ = ((y - rangeMin) / rangeSize - 0.5) * 10;
    
    // Get landscape z-range to normalize ballistic coordinates the same way as the landscape
    // The landscape normalizes: (z - zMin) / zRange * scale
    let worldY = 0;
    if (window.getLandscapeZRange) {
        const zRange = window.getLandscapeZRange();
        // Apply same normalization as landscape
        worldY = (z_world - zRange.zMin) / zRange.zRange * zRange.scale;
    } else {
        // Fallback if landscape not loaded yet
        worldY = z_world * 0.2;
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
    
    // Use different coordinate conversion for ballistic optimizers
    let points;
    if (name === 'ballistic' || name === 'ballistic_adam') {
        points = trajectory.map(([x, y, z_world]) => ballisticToWorldCoords(x, y, z_world));
    } else {
        points = trajectory.map(([x, y, loss]) => paramsToWorldCoords(x, y, loss));
    }
    
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    
    line.geometry.dispose();
    line.geometry = geometry;
    
    // Only show if this optimizer is enabled and trails are enabled
    const showTrails = document.getElementById('show-trails')?.checked ?? true;
    line.visible = enabledOptimizers[name] && showTrails;
}

// Set ball position
function setBallPosition(name, x, y, loss_or_z) {
    const ball = optimizerBalls[name];
    if (!ball) return;
    
    // Only show if this optimizer is enabled
    if (!enabledOptimizers[name]) {
        ball.visible = false;
        return;
    }
    
    // Use different coordinate conversion for ballistic optimizers
    let worldPos;
    if (name === 'ballistic' || name === 'ballistic_adam') {
        worldPos = ballisticToWorldCoords(x, y, loss_or_z);
    } else {
        worldPos = paramsToWorldCoords(x, y, loss_or_z);
    }
    
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
        const allOptimizers = ['sgd', 'batch', 'momentum', 'adam', 'ballistic', 'ballistic_adam'];
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
        const validOptimizers = ['sgd', 'batch', 'momentum', 'adam', 'ballistic', 'ballistic_adam'];
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
    
    const currentStepInt = Math.floor(animationState.currentStep);
    
    // Check if we've reached the end
    if (animationState.currentStep >= maxSteps) {
        animationState.isPlaying = false;
        animationState.currentStep = maxSteps;
        
        // Update UI state to stopped (animation finished)
        if (window.setAnimationState) {
            window.setAnimationState('stopped');
        }
        
        // Update timeline to show we're at the end
        if (window.updateTimelineDisplay) {
            window.updateTimelineDisplay(maxSteps, maxSteps);
        }
        return;
    }
    
    // Update each optimizer's position (only for those with trajectories)
    const currentPositions = {};
    existingTrajectories.forEach(name => {
        const trajectory = currentTrajectories[name];
        if (trajectory && trajectory.length > 0) {
            const step = Math.floor(Math.min(animationState.currentStep, trajectory.length - 1));
            const [x, y, z_or_loss] = trajectory[step];
            setBallPosition(name, x, y, z_or_loss);
            currentPositions[name] = { x, y, loss: z_or_loss };
        }
    });
    
    // Check for overlapping balls and show segmented balls if needed
    updateSegmentedBalls();
    
    // Update timeline display with current step
    if (window.updateTimelineDisplay) {
        window.updateTimelineDisplay(currentStepInt, maxSteps);
    }
    
    // Advance step based on speed
    animationState.currentStep += animationState.speed;
}

// Start animation
function startAnimation() {
    // If we're at the end, reset to beginning
    if (currentTrajectories) {
        const existingTrajectories = Object.keys(currentTrajectories).filter(name => 
            currentTrajectories[name] && currentTrajectories[name].length > 0
        );
        
        if (existingTrajectories.length > 0) {
            const maxSteps = Math.max(
                ...existingTrajectories.map(name => currentTrajectories[name]?.length || 0)
            );
            
            // If we're at or past the end, restart from beginning
            if (animationState.currentStep >= maxSteps) {
                animationState.currentStep = 0;
                
                // Update timeline display
                if (window.updateTimelineDisplay) {
                    window.updateTimelineDisplay(0, maxSteps);
                }
            }
        }
    }
    
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
    
    // Hide segmented balls
    segmentedBalls.forEach(ball => {
        ball.visible = false;
    });
    
    // Update timeline display
    if (window.updateTimelineDisplay && currentTrajectories) {
        let maxLength = 0;
        for (const optimizer in currentTrajectories) {
            if (currentTrajectories[optimizer]) {
                maxLength = Math.max(maxLength, currentTrajectories[optimizer].length);
            }
        }
        window.updateTimelineDisplay(0, maxLength);
    }
}
window.resetAnimation = resetAnimation; // Make globally accessible

// Seek to specific step
function seekToStep(step) {
    if (!currentTrajectories) return;
    
    animationState.currentStep = step;
    
    // Update all ball positions to this step
    const currentPositions = {};
    for (const name in currentTrajectories) {
        const trajectory = currentTrajectories[name];
        const ball = optimizerBalls[name];
        
        if (!trajectory || !ball || !enabledOptimizers[name]) continue;
        
        // Show ball if trajectory has data at this step
        if (step < trajectory.length) {
            const [x, y, z_or_loss] = trajectory[step];
            // Use appropriate coordinate conversion
            let pos;
            if (name === 'ballistic' || name === 'ballistic_adam') {
                pos = ballisticToWorldCoords(x, y, z_or_loss);
            } else {
                pos = paramsToWorldCoords(x, y, z_or_loss);
            }
            ball.position.copy(pos);
            ball.visible = true;
            currentPositions[name] = { x, y, loss: z_or_loss };
        } else {
            ball.visible = false;
        }
    }
    
    // Handle overlapping balls at this step
    updateOverlappingBalls(step);
}
window.seekToStep = seekToStep; // Make globally accessible

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
    
    // Hide segmented balls
    segmentedBalls.forEach(ball => {
        ball.visible = false;
    });
    
    // Hide and clear trajectory lines
    Object.values(trajectoryLines).forEach(line => {
        line.visible = false;
        if (line.geometry) {
            line.geometry.setFromPoints([]);
        }
    });
    
    // Reset timeline display
    if (window.updateTimelineDisplay) {
        window.updateTimelineDisplay(0, 0);
    }
    
    // Reset UI state
    if (window.setAnimationState) {
        window.setAnimationState('stopped');
    }
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

// Create a segmented sphere with different colors for each segment
function createSegmentedSphere(colors) {
    const geometry = new THREE.SphereGeometry(0.15, 32, 32);
    const positions = geometry.attributes.position;
    const colorsArray = [];
    
    const numSegments = colors.length;
    
    // Assign colors to vertices based on their angle around the Y-axis
    for (let i = 0; i < positions.count; i++) {
        const x = positions.getX(i);
        const z = positions.getZ(i);
        
        // Calculate angle in radians (0 to 2*PI)
        let angle = Math.atan2(z, x) + Math.PI; // Shift to 0-2PI range
        
        // Determine which segment this vertex belongs to
        const segmentSize = (Math.PI * 2) / numSegments;
        const segmentIndex = Math.floor(angle / segmentSize);
        const clampedIndex = Math.min(segmentIndex, numSegments - 1);
        
        const color = new THREE.Color(colors[clampedIndex]);
        colorsArray.push(color.r, color.g, color.b);
    }
    
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colorsArray, 3));
    
    const material = new THREE.MeshPhongMaterial({
        vertexColors: true,
        emissive: 0x222222,
        emissiveIntensity: 0.2
    });
    
    const mesh = new THREE.Mesh(geometry, material);
    mesh.visible = false;
    scene.add(mesh);
    
    return mesh;
}

// Get or create a segmented ball from the pool
function getSegmentedBall(colors) {
    // Try to find an existing ball with the same number of segments
    for (let ball of segmentedBalls) {
        if (!ball.visible && ball.userData.numSegments === colors.length) {
            // Update colors
            updateSegmentedBallColors(ball, colors);
            return ball;
        }
    }
    
    // Create new segmented ball
    const ball = createSegmentedSphere(colors);
    ball.userData.numSegments = colors.length;
    segmentedBalls.push(ball);
    return ball;
}

// Update colors of an existing segmented ball
function updateSegmentedBallColors(ball, colors) {
    const geometry = ball.geometry;
    const positions = geometry.attributes.position;
    const colorsArray = [];
    
    const numSegments = colors.length;
    
    for (let i = 0; i < positions.count; i++) {
        const x = positions.getX(i);
        const z = positions.getZ(i);
        
        let angle = Math.atan2(z, x) + Math.PI;
        const segmentSize = (Math.PI * 2) / numSegments;
        const segmentIndex = Math.floor(angle / segmentSize);
        const clampedIndex = Math.min(segmentIndex, numSegments - 1);
        
        const color = new THREE.Color(colors[clampedIndex]);
        colorsArray.push(color.r, color.g, color.b);
    }
    
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colorsArray, 3));
}

// Detect overlapping balls and manage segmented ball display
function updateSegmentedBalls() {
    // Hide all segmented balls first
    segmentedBalls.forEach(ball => ball.visible = false);
    
    // Get all visible balls with their positions
    const visibleBalls = [];
    Object.keys(optimizerBalls).forEach(name => {
        const ball = optimizerBalls[name];
        if (ball && ball.visible && enabledOptimizers[name]) {
            visibleBalls.push({
                name: name,
                ball: ball,
                position: ball.position.clone(),
                color: OPTIMIZER_COLORS[name]
            });
        }
    });
    
    // Find groups that are very close together
    const processed = new Set();
    
    for (let i = 0; i < visibleBalls.length; i++) {
        if (processed.has(i)) continue;
        
        const group = [visibleBalls[i]];
        processed.add(i);
        
        // Check for balls that are directly on top of each other
        for (let j = i + 1; j < visibleBalls.length; j++) {
            if (processed.has(j)) continue;
            
            const distance = visibleBalls[i].position.distanceTo(visibleBalls[j].position);
            if (distance < OVERLAP_THRESHOLD) {
                group.push(visibleBalls[j]);
                processed.add(j);
            }
        }
        
        if (group.length > 1) {
            // Multiple balls overlapping - show segmented ball
            // Hide individual balls
            group.forEach(item => item.ball.visible = false);
            
            // Calculate centroid position
            const centroid = new THREE.Vector3();
            group.forEach(item => centroid.add(item.position));
            centroid.divideScalar(group.length);
            
            // Get colors from group
            const colors = group.map(item => item.color);
            
            // Get or create segmented ball
            const segmentedBall = getSegmentedBall(colors);
            segmentedBall.position.copy(centroid);
            segmentedBall.visible = true;
        }
        // If only 1 ball in group, it stays visible (already is)
    }
}

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

