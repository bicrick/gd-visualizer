/**
 * UI controls and event handlers
 */

// Animation state management
let animationUIState = 'stopped'; // 'stopped', 'playing', 'paused'

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
        lrDecay: 0.995,
        iterations: 100,
        useConvergence: true,
        maxIterations: 10000,
        convergenceThreshold: 1e-4
    },
    adam: {
        learningRate: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
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
    
    // Play/Pause toggle button
    document.getElementById('play-pause-btn').addEventListener('click', async () => {
        if (animationUIState === 'stopped' || animationUIState === 'paused') {
            // If no trajectories exist or params changed, run optimization first
            if (!window.currentTrajectories || hasParamsChanged()) {
                await runOptimizationFromUI();
            }
            window.startAnimation();
            setAnimationState('playing');
        } else if (animationUIState === 'playing') {
            window.pauseAnimation();
            setAnimationState('paused');
        }
    });
    
    // Stop button
    document.getElementById('stop-btn').addEventListener('click', () => {
        window.resetAnimation();
        setAnimationState('stopped');
    });
    
    // Timeline scrubber
    const timelineScrubber = document.getElementById('timeline-scrubber');
    
    // Update in real-time as user drags
    timelineScrubber.addEventListener('input', (e) => {
        const step = parseInt(e.target.value);
        const totalSteps = parseInt(e.target.max);
        
        // Update the visual display immediately
        updateTimelineDisplay(step, totalSteps);
        
        // Seek to the step
        if (window.seekToStep) {
            window.seekToStep(step);
        }
    });
    
    // Pause when scrubbing starts
    timelineScrubber.addEventListener('mousedown', () => {
        if (animationUIState === 'playing') {
            window.pauseAnimation();
            setAnimationState('paused');
        }
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
    initOptimizerControls('adam', false, true);
    
    // Initialize expand/collapse functionality
    // Make the entire header clickable, but prevent checkbox clicks from triggering expand/collapse
    document.querySelectorAll('.optimizer-header').forEach(header => {
        const btn = header.querySelector('.expand-btn');
        const targetId = btn.getAttribute('data-target');
        const targetEl = document.getElementById(targetId);
        
        // Handle header click (entire row)
        header.addEventListener('click', (e) => {
            // Don't toggle if clicking on checkbox or its label content area
            if (e.target.type === 'checkbox' || e.target.closest('.optimizer-toggle-label input[type="checkbox"]')) {
                return;
            }
            
            e.preventDefault();
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
        
        // Prevent checkbox clicks from bubbling to header
        const checkbox = header.querySelector('input[type="checkbox"]');
        if (checkbox) {
            checkbox.addEventListener('click', (e) => {
                e.stopPropagation();
            });
        }
    });
}

// Initialize controls for a specific optimizer
function initOptimizerControls(optimizerName, hasMomentum, isAdam) {
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
        
        // Learning rate decay slider (only for momentum optimizer)
        const lrDecaySlider = document.getElementById(`${prefix}-lr-decay`);
        const lrDecayValue = document.getElementById(`${prefix}-lr-decay-value`);
        lrDecaySlider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            lrDecayValue.textContent = value.toFixed(3);
            currentParams[prefix].lrDecay = value;
        });
    }
    
    // ADAM-specific parameters
    if (isAdam) {
        // Beta1 slider
        const beta1Slider = document.getElementById(`${prefix}-beta1`);
        const beta1Value = document.getElementById(`${prefix}-beta1-value`);
        beta1Slider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            beta1Value.textContent = value.toFixed(3);
            currentParams[prefix].beta1 = value;
        });
        
        // Beta2 slider
        const beta2Slider = document.getElementById(`${prefix}-beta2`);
        const beta2Value = document.getElementById(`${prefix}-beta2-value`);
        beta2Slider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            beta2Value.textContent = value.toFixed(4);
            currentParams[prefix].beta2 = value;
        });
        
        // Epsilon slider
        const epsilonSlider = document.getElementById(`${prefix}-epsilon`);
        const epsilonValue = document.getElementById(`${prefix}-epsilon-value`);
        epsilonSlider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            epsilonValue.textContent = value.toExponential(0);
            currentParams[prefix].epsilon = value;
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
    momentum: false,
    adam: false
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
        JSON.stringify(previousParams.momentum) !== JSON.stringify(currentParams.momentum) ||
        JSON.stringify(previousParams.adam) !== JSON.stringify(currentParams.adam);
    
    // Check if enabled optimizers changed
    const currentEnabled = window.getEnabledOptimizers ? window.getEnabledOptimizers() : {
        sgd: document.getElementById('toggle-sgd')?.checked || false,
        batch: document.getElementById('toggle-batch')?.checked || false,
        momentum: document.getElementById('toggle-momentum')?.checked || false,
        adam: document.getElementById('toggle-adam')?.checked || false
    };
    
    const enabledOptimizersChanged = (
        previousEnabledOptimizers.sgd !== currentEnabled.sgd ||
        previousEnabledOptimizers.batch !== currentEnabled.batch ||
        previousEnabledOptimizers.momentum !== currentEnabled.momentum ||
        previousEnabledOptimizers.adam !== currentEnabled.adam
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
            momentum: document.getElementById('toggle-momentum')?.checked || false,
            adam: document.getElementById('toggle-adam')?.checked || false
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
                momentum: currentParams.momentum,
                adam: currentParams.adam
            }
        };
        
        await window.runOptimization(params);
        previousParams = JSON.parse(JSON.stringify(currentParams));
        previousEnabledOptimizers = { ...enabledOpts };
        
        loadingEl.classList.add('hidden');
        
        // Initialize timeline after trajectories are loaded
        initializeTimeline();
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

// Function to reset start position to origin (called when manifold changes)
function resetStartPosition() {
    const startXInput = document.getElementById('start-x');
    const startYInput = document.getElementById('start-y');
    
    // Reset to origin (0, 0) - center of most manifolds
    startXInput.value = '0.00';
    startYInput.value = '0.00';
    currentParams.startX = 0;
    currentParams.startY = 0;
    
    // Update visual marker on the graph
    if (window.updateStartPointMarker) {
        window.updateStartPointMarker(0, 0);
    }
}

// Update UI to reflect animation state
function setAnimationState(state) {
    animationUIState = state;
    
    const playPauseBtn = document.getElementById('play-pause-btn');
    const stopBtn = document.getElementById('stop-btn');
    const playIcon = playPauseBtn.querySelector('.play-icon');
    const pauseIcon = playPauseBtn.querySelector('.pause-icon');
    const timelineScrubber = document.getElementById('timeline-scrubber');
    const randomStartBtn = document.getElementById('random-start');
    const pickPointBtn = document.getElementById('pick-point-btn');
    
    switch (state) {
        case 'stopped':
            // Show play icon, disable stop button
            playIcon.classList.remove('hidden');
            pauseIcon.classList.add('hidden');
            playPauseBtn.title = 'Play';
            stopBtn.disabled = true;
            timelineScrubber.disabled = !window.currentTrajectories;
            // Enable start position buttons
            randomStartBtn.disabled = false;
            pickPointBtn.disabled = false;
            break;
            
        case 'playing':
            // Show pause icon, enable stop button
            playIcon.classList.add('hidden');
            pauseIcon.classList.remove('hidden');
            playPauseBtn.title = 'Pause';
            stopBtn.disabled = false;
            timelineScrubber.disabled = false;
            // Disable start position buttons while animation is running
            randomStartBtn.disabled = true;
            pickPointBtn.disabled = true;
            break;
            
        case 'paused':
            // Show play icon, enable stop button
            playIcon.classList.remove('hidden');
            pauseIcon.classList.add('hidden');
            playPauseBtn.title = 'Resume';
            stopBtn.disabled = false;
            timelineScrubber.disabled = false;
            // Keep start position buttons disabled while paused (still in active session)
            randomStartBtn.disabled = true;
            pickPointBtn.disabled = true;
            break;
    }
}

// Update timeline display
function updateTimelineDisplay(currentStep, totalSteps) {
    const currentStepEl = document.getElementById('current-step');
    const totalStepsEl = document.getElementById('total-steps');
    const timelineScrubber = document.getElementById('timeline-scrubber');
    
    if (currentStep !== undefined) {
        currentStepEl.textContent = currentStep;
        timelineScrubber.value = currentStep;
    }
    
    if (totalSteps !== undefined) {
        totalStepsEl.textContent = totalSteps;
        timelineScrubber.max = totalSteps;
        timelineScrubber.disabled = totalSteps === 0;
    }
}

// Initialize timeline when trajectories are loaded
function initializeTimeline() {
    if (window.currentTrajectories) {
        // Find the maximum trajectory length
        let maxLength = 0;
        for (const optimizer in window.currentTrajectories) {
            if (window.currentTrajectories[optimizer]) {
                maxLength = Math.max(maxLength, window.currentTrajectories[optimizer].length);
            }
        }
        updateTimelineDisplay(0, maxLength);
        setAnimationState('stopped');
    }
}

// Expose to global scope
window.updateStartPosition = updateStartPosition;
window.resetStartPosition = resetStartPosition;
window.currentParams = currentParams;
window.setAnimationState = setAnimationState;
window.updateTimelineDisplay = updateTimelineDisplay;
window.initializeTimeline = initializeTimeline;

// Initialize on page load
window.addEventListener('DOMContentLoaded', () => {
    initControls();
    setAnimationState('stopped');
});
