/**
 * UI controls and event handlers
 */

let currentParams = {
    startX: 3,
    startY: 3,
    sgd: {
        learningRate: 0.01,
        iterations: 100,
        useConvergence: true,
        maxIterations: 10000,
        convergenceThreshold: 1e-4
    },
    batch: {
        learningRate: 0.01,
        iterations: 100,
        useConvergence: true,
        maxIterations: 10000,
        convergenceThreshold: 1e-4
    },
    momentum: {
        learningRate: 0.01,
        momentum: 0.9,
        iterations: 100,
        useConvergence: true,
        maxIterations: 10000,
        convergenceThreshold: 1e-4
    }
};

// Initialize UI controls
function initControls() {
    // Starting position inputs
    const startXInput = document.getElementById('start-x');
    const startYInput = document.getElementById('start-y');
    startXInput.addEventListener('change', (e) => {
        currentParams.startX = parseFloat(e.target.value);
        // Update visual marker
        if (window.updateStartPointMarker) {
            window.updateStartPointMarker(currentParams.startX, currentParams.startY);
        }
    });
    startYInput.addEventListener('change', (e) => {
        currentParams.startY = parseFloat(e.target.value);
        // Update visual marker
        if (window.updateStartPointMarker) {
            window.updateStartPointMarker(currentParams.startX, currentParams.startY);
        }
    });
    
    // Random start button
    document.getElementById('random-start').addEventListener('click', () => {
        const x = (Math.random() - 0.5) * 8; // Random between -4 and 4
        const y = (Math.random() - 0.5) * 8;
        startXInput.value = x.toFixed(1);
        startYInput.value = y.toFixed(1);
        currentParams.startX = x;
        currentParams.startY = y;
        // Update visual marker
        if (window.updateStartPointMarker) {
            window.updateStartPointMarker(x, y);
        }
    });
    
    // Pick point button
    document.getElementById('pick-point-btn').addEventListener('click', () => {
        if (window.setPickingMode) {
            window.setPickingMode(true);
        }
    });
    
    // Speed slider
    const speedSlider = document.getElementById('speed');
    const speedValue = document.getElementById('speed-value');
    speedSlider.addEventListener('input', (e) => {
        const value = parseFloat(e.target.value);
        speedValue.textContent = value.toFixed(1);
        window.setAnimationSpeed(value);
    });
    
    // Play button
    document.getElementById('play-btn').addEventListener('click', async () => {
        // If no trajectories exist or params changed, run optimization first
        if (!window.currentTrajectories || hasParamsChanged()) {
            await runOptimizationFromUI();
        }
        window.startAnimation();
    });
    
    // Pause button
    document.getElementById('pause-btn').addEventListener('click', () => {
        window.pauseAnimation();
    });
    
    // Reset button
    document.getElementById('reset-btn').addEventListener('click', () => {
        window.resetAnimation();
    });
    
    // Show trails checkbox
    const showTrailsCheckbox = document.getElementById('show-trails');
    showTrailsCheckbox.addEventListener('change', (e) => {
        window.setTrajectoryVisibility(e.target.checked);
    });
    
    // Initialize optimizer-specific controls
    initOptimizerControls('sgd', false);
    initOptimizerControls('batch', false);
    initOptimizerControls('momentum', true);
    
    // Initialize expand/collapse buttons
    document.querySelectorAll('.expand-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const targetId = btn.getAttribute('data-target');
            const targetEl = document.getElementById(targetId);
            const isCollapsed = targetEl.classList.contains('collapsed');
            
            if (isCollapsed) {
                targetEl.classList.remove('collapsed');
                btn.classList.remove('collapsed');
                btn.textContent = '▼';
            } else {
                targetEl.classList.add('collapsed');
                btn.classList.add('collapsed');
                btn.textContent = '▶';
            }
        });
    });
}

// Initialize controls for a specific optimizer
function initOptimizerControls(optimizerName, hasMomentum) {
    const prefix = optimizerName;
    
    // Toggle checkbox
    document.getElementById(`toggle-${prefix}`).addEventListener('change', (e) => {
        window.toggleOptimizer(prefix, e.target.checked);
    });
    
    // Learning rate slider
    const lrSlider = document.getElementById(`${prefix}-learning-rate`);
    const lrValue = document.getElementById(`${prefix}-learning-rate-value`);
    lrSlider.addEventListener('input', (e) => {
        const value = parseFloat(e.target.value);
        lrValue.textContent = value.toFixed(3);
        currentParams[prefix].learningRate = value;
    });
    
    // Momentum slider (only for momentum optimizer)
    if (hasMomentum) {
        const momentumSlider = document.getElementById(`${prefix}-momentum`);
        const momentumValue = document.getElementById(`${prefix}-momentum-value`);
        momentumSlider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            momentumValue.textContent = value.toFixed(2);
            currentParams[prefix].momentum = value;
        });
    }
    
    // Use convergence checkbox
    const useConvergenceCheckbox = document.getElementById(`${prefix}-use-convergence`);
    const iterationsControl = document.getElementById(`${prefix}-iterations-control`);
    const maxIterationsControl = document.getElementById(`${prefix}-max-iterations-control`);
    const convergenceThresholdControl = document.getElementById(`${prefix}-convergence-threshold-control`);
    
    useConvergenceCheckbox.addEventListener('change', (e) => {
        const useConvergence = e.target.checked;
        currentParams[prefix].useConvergence = useConvergence;
        
        // Toggle visibility of controls
        if (useConvergence) {
            iterationsControl.classList.add('hidden');
            maxIterationsControl.classList.remove('hidden');
            convergenceThresholdControl.classList.remove('hidden');
        } else {
            iterationsControl.classList.remove('hidden');
            maxIterationsControl.classList.add('hidden');
            convergenceThresholdControl.classList.add('hidden');
        }
    });
    
    // Iterations slider
    const iterationsSlider = document.getElementById(`${prefix}-iterations`);
    const iterationsValue = document.getElementById(`${prefix}-iterations-value`);
    iterationsSlider.addEventListener('input', (e) => {
        const value = parseInt(e.target.value);
        iterationsValue.textContent = value;
        currentParams[prefix].iterations = value;
    });
    
    // Max iterations slider
    const maxIterationsSlider = document.getElementById(`${prefix}-max-iterations`);
    const maxIterationsValue = document.getElementById(`${prefix}-max-iterations-value`);
    maxIterationsSlider.addEventListener('input', (e) => {
        const value = parseInt(e.target.value);
        maxIterationsValue.textContent = value;
        currentParams[prefix].maxIterations = value;
    });
    
    // Convergence threshold slider
    const convergenceThresholdSlider = document.getElementById(`${prefix}-convergence-threshold`);
    const convergenceThresholdValue = document.getElementById(`${prefix}-convergence-threshold-value`);
    convergenceThresholdSlider.addEventListener('input', (e) => {
        const value = parseFloat(e.target.value);
        convergenceThresholdValue.textContent = value.toExponential(0);
        currentParams[prefix].convergenceThreshold = value;
    });
}

// Track previous params to detect changes
let previousParams = JSON.parse(JSON.stringify(currentParams));
let previousEnabledOptimizers = {
    sgd: false,  // Will be updated on first check
    batch: false,
    momentum: false
};

function hasParamsChanged() {
    // Check if starting position changed
    const startPosChanged = (
        previousParams.startX !== currentParams.startX ||
        previousParams.startY !== currentParams.startY
    );
    
    // Check if any optimizer params changed
    const paramsChanged = startPosChanged ||
        JSON.stringify(previousParams.sgd) !== JSON.stringify(currentParams.sgd) ||
        JSON.stringify(previousParams.batch) !== JSON.stringify(currentParams.batch) ||
        JSON.stringify(previousParams.momentum) !== JSON.stringify(currentParams.momentum);
    
    // Check if enabled optimizers changed
    const currentEnabled = window.getEnabledOptimizers ? window.getEnabledOptimizers() : {
        sgd: document.getElementById('toggle-sgd')?.checked || false,
        batch: document.getElementById('toggle-batch')?.checked || false,
        momentum: document.getElementById('toggle-momentum')?.checked || false
    };
    
    const enabledOptimizersChanged = (
        previousEnabledOptimizers.sgd !== currentEnabled.sgd ||
        previousEnabledOptimizers.batch !== currentEnabled.batch ||
        previousEnabledOptimizers.momentum !== currentEnabled.momentum
    );
    
    return paramsChanged || enabledOptimizersChanged;
}

// Run optimization with current UI parameters
async function runOptimizationFromUI() {
    const loadingEl = document.getElementById('loading');
    loadingEl.textContent = 'Computing trajectories...';
    loadingEl.classList.remove('hidden');
    
    try {
        // Get currently enabled optimizers from the global state
        const enabledOpts = window.getEnabledOptimizers ? window.getEnabledOptimizers() : {
            sgd: document.getElementById('toggle-sgd')?.checked || false,
            batch: document.getElementById('toggle-batch')?.checked || false,
            momentum: document.getElementById('toggle-momentum')?.checked || false
        };
        
        // Build parameters for each enabled optimizer
        // For now, we'll use a shared set of parameters for all, but with optimizer-specific values
        // In the future, the backend could be modified to accept per-optimizer parameters
        
        // Use the parameters from the first enabled optimizer for the main call
        let mainParams = currentParams.batch;
        if (enabledOpts.sgd) {
            mainParams = currentParams.sgd;
        } else if (enabledOpts.batch) {
            mainParams = currentParams.batch;
        } else if (enabledOpts.momentum) {
            mainParams = currentParams.momentum;
        }
        
        const params = {
            manifold: window.getCurrentManifoldId ? window.getCurrentManifoldId() : 'custom_multimodal',
            initial_params: [currentParams.startX, currentParams.startY],
            learning_rate: mainParams.learningRate,
            momentum: currentParams.momentum.momentum,
            n_iterations: mainParams.iterations,
            seed: 42,
            use_convergence: mainParams.useConvergence,
            max_iterations: mainParams.maxIterations,
            convergence_threshold: mainParams.convergenceThreshold,
            enabled_optimizers: enabledOpts,
            // Pass per-optimizer parameters
            optimizer_params: {
                sgd: currentParams.sgd,
                batch: currentParams.batch,
                momentum: currentParams.momentum
            }
        };
        
        await window.runOptimization(params);
        previousParams = JSON.parse(JSON.stringify(currentParams));
        previousEnabledOptimizers = { ...enabledOpts };
        
        loadingEl.classList.add('hidden');
    } catch (error) {
        console.error('Error:', error);
        loadingEl.textContent = 'Error computing trajectories.';
    }
}

// Function to update start position (called from scene.js when point is picked)
function updateStartPosition(x, y) {
    const startXInput = document.getElementById('start-x');
    const startYInput = document.getElementById('start-y');
    
    startXInput.value = x.toFixed(2);
    startYInput.value = y.toFixed(2);
    currentParams.startX = x;
    currentParams.startY = y;
    
    // Update visual marker on the graph
    if (window.updateStartPointMarker) {
        window.updateStartPointMarker(x, y);
    }
}

// Expose to global scope
window.updateStartPosition = updateStartPosition;
window.currentParams = currentParams;

// Initialize on page load
window.addEventListener('DOMContentLoaded', () => {
    initControls();
});
