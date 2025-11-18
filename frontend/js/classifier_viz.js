/**
 * 2D Classifier Visualization Module
 * Displays the dataset and decision boundary for the neural network classifier
 */

let classifierCanvas = null;
let classifierCtx = null;
let classifierDataset = null;
let isClassifierActive = false;

// Circle classifier parameters (same as backend)
const CLASSIFIER_RADIUS = 1.2;

// Visualization parameters
const CANVAS_SIZE = 300; // Logical size
const MARGIN = 30;
const DATA_RANGE = [-3, 3]; // Match the manifold range

/**
 * Initialize the classifier visualization
 */
function initClassifierViz() {
    classifierCanvas = document.getElementById('classifier-canvas');
    classifierCtx = classifierCanvas.getContext('2d');
    
    // Set canvas size
    const panel = document.getElementById('classifier-panel');
    const panelWidth = panel.clientWidth;
    const panelHeight = panel.clientHeight - 60; // Subtract header height
    
    classifierCanvas.width = panelWidth;
    classifierCanvas.height = panelHeight;
    
    // Load the dataset
    loadClassifierDataset();
}

/**
 * Load the classification dataset from the backend
 */
async function loadClassifierDataset() {
    try {
        const response = await fetch(`${API_BASE_URL}/classifier_dataset`);
        const data = await response.json();
        classifierDataset = data;
    } catch (error) {
        console.error('Error loading classifier dataset:', error);
    }
}

/**
 * Show or hide the classifier panel
 */
function setClassifierPanelVisible(visible) {
    const panel = document.getElementById('classifier-panel');
    if (visible) {
        panel.classList.remove('hidden');
        isClassifierActive = true;
        if (!classifierCanvas) {
            initClassifierViz();
        }
    } else {
        panel.classList.add('hidden');
        isClassifierActive = false;
    }
}

/**
 * Sigmoid activation function
 */
function sigmoid(x) {
    if (x >= 0) {
        return 1 / (1 + Math.exp(-x));
    } else {
        const expX = Math.exp(x);
        return expX / (1 + expX);
    }
}

/**
 * Compute circle classifier output for given circle center and input point
 * Parameters: center_x and center_y (circle center position)
 */
function computeClassifierOutput(center_x, center_y, x, y) {
    // Compute distance from point to circle center
    const distance = Math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2);
    
    // Distance from boundary (negative = inside, positive = outside)
    const steepness = 2.0;
    const distanceFromBoundary = (distance - CLASSIFIER_RADIUS) * steepness;
    
    // Sigmoid to get probability (inside circle → 0, outside → 1)
    const output = sigmoid(distanceFromBoundary);
    
    return output;
}

/**
 * Map data coordinates to canvas coordinates
 */
function dataToCanvas(x, y) {
    const width = classifierCanvas.width;
    const height = classifierCanvas.height;
    const plotWidth = width - 2 * MARGIN;
    const plotHeight = height - 2 * MARGIN;
    
    const rangeSize = DATA_RANGE[1] - DATA_RANGE[0];
    const canvasX = MARGIN + ((x - DATA_RANGE[0]) / rangeSize) * plotWidth;
    const canvasY = height - MARGIN - ((y - DATA_RANGE[0]) / rangeSize) * plotHeight;
    
    return [canvasX, canvasY];
}

/**
 * Render the 2D classifier visualization
 * Can be called with either (center_x, center_y) or (optimizerPositions)
 */
function renderClassifierViz(center_x_or_positions, center_y) {
    if (!classifierCtx || !classifierDataset || !isClassifierActive) return;
    
    const width = classifierCanvas.width;
    const height = classifierCanvas.height;
    const plotWidth = width - 2 * MARGIN;
    const plotHeight = height - 2 * MARGIN;
    
    // Clear canvas
    classifierCtx.fillStyle = '#1a1a1a';
    classifierCtx.fillRect(0, 0, width, height);
    
    // Determine if we're in multi-optimizer mode or single position mode
    const isMultiMode = typeof center_x_or_positions === 'object' && center_x_or_positions !== null && !Array.isArray(center_x_or_positions);
    const optimizerPositions = isMultiMode ? center_x_or_positions : null;
    const singleCenter = !isMultiMode ? { x: center_x_or_positions, y: center_y } : null;
    
    // Background heatmap removed per user request - show only circles and data points
    const rangeSize = DATA_RANGE[1] - DATA_RANGE[0];
    
    // Draw decision boundary circle(s) - one for each optimizer if available
    const radiusInDataUnits = CLASSIFIER_RADIUS;
    const radiusInCanvasUnits = (radiusInDataUnits / rangeSize) * plotWidth;
    
    // Optimizer colors (matching the ball colors)
    const optimizerColors = {
        sgd: '#ff4444',
        batch: '#4444ff',
        momentum: '#44ff44',
        adam: '#ff8800'
    };
    
    if (isMultiMode && optimizerPositions) {
        // Draw a circle for each active optimizer
        ['sgd', 'batch', 'momentum', 'adam'].forEach(name => {
            if (optimizerPositions[name]) {
                const pos = optimizerPositions[name];
                const [circleCenterCanvasX, circleCenterCanvasY] = dataToCanvas(pos.x, pos.y);
                
                // Draw the circle
                classifierCtx.strokeStyle = optimizerColors[name];
                classifierCtx.lineWidth = 2.5;
                classifierCtx.setLineDash([]);
                classifierCtx.beginPath();
                classifierCtx.arc(circleCenterCanvasX, circleCenterCanvasY, radiusInCanvasUnits, 0, Math.PI * 2);
                classifierCtx.stroke();
                
                // Draw a small dot at the center
                classifierCtx.fillStyle = optimizerColors[name];
                classifierCtx.beginPath();
                classifierCtx.arc(circleCenterCanvasX, circleCenterCanvasY, 3, 0, Math.PI * 2);
                classifierCtx.fill();
            }
        });
    } else if (singleCenter) {
        // Single circle mode (when called directly with coordinates)
        const [circleCenterCanvasX, circleCenterCanvasY] = dataToCanvas(singleCenter.x, singleCenter.y);
        
        // Draw the circle
        classifierCtx.strokeStyle = '#ffdd00'; // Bright yellow for visibility
        classifierCtx.lineWidth = 3;
        classifierCtx.setLineDash([]);
        classifierCtx.beginPath();
        classifierCtx.arc(circleCenterCanvasX, circleCenterCanvasY, radiusInCanvasUnits, 0, Math.PI * 2);
        classifierCtx.stroke();
        
        // Draw a small dot at the center
        classifierCtx.fillStyle = '#ffdd00';
        classifierCtx.beginPath();
        classifierCtx.arc(circleCenterCanvasX, circleCenterCanvasY, 4, 0, Math.PI * 2);
        classifierCtx.fill();
    }
    
    // Draw axes
    classifierCtx.strokeStyle = '#3a3a3a';
    classifierCtx.lineWidth = 1;
    classifierCtx.setLineDash([]);
    
    // X-axis
    const [x0, y0] = dataToCanvas(0, DATA_RANGE[0]);
    const [x1, y1] = dataToCanvas(0, DATA_RANGE[1]);
    classifierCtx.beginPath();
    classifierCtx.moveTo(x0, y0);
    classifierCtx.lineTo(x1, y1);
    classifierCtx.stroke();
    
    // Y-axis
    const [x2, y2] = dataToCanvas(DATA_RANGE[0], 0);
    const [x3, y3] = dataToCanvas(DATA_RANGE[1], 0);
    classifierCtx.beginPath();
    classifierCtx.moveTo(x2, y2);
    classifierCtx.lineTo(x3, y3);
    classifierCtx.stroke();
    
    // Draw border
    classifierCtx.strokeStyle = '#3a3a3a';
    classifierCtx.lineWidth = 1;
    classifierCtx.strokeRect(MARGIN, MARGIN, plotWidth, plotHeight);
    
    // Draw dataset points
    const points = classifierDataset.points;
    const labels = classifierDataset.labels;
    
    for (let i = 0; i < points.length; i++) {
        const [x, y] = points[i];
        const label = labels[i];
        const [canvasX, canvasY] = dataToCanvas(x, y);
        
        // Draw point
        classifierCtx.beginPath();
        classifierCtx.arc(canvasX, canvasY, 4, 0, Math.PI * 2);
        
        if (label === 0) {
            classifierCtx.fillStyle = '#44ff44'; // Green for class 0
            classifierCtx.strokeStyle = '#229922';
        } else {
            classifierCtx.fillStyle = '#ff8800'; // Orange for class 1
            classifierCtx.strokeStyle = '#aa5500';
        }
        
        classifierCtx.fill();
        classifierCtx.lineWidth = 1.5;
        classifierCtx.stroke();
    }
    
    // Draw axis labels
    classifierCtx.fillStyle = '#b0b0b0';
    classifierCtx.font = '12px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
    classifierCtx.textAlign = 'center';
    classifierCtx.textBaseline = 'middle';
    
    // X-axis label
    classifierCtx.fillText('Center X', width / 2, height - 10);
    
    // Y-axis label
    classifierCtx.save();
    classifierCtx.translate(10, height / 2);
    classifierCtx.rotate(-Math.PI / 2);
    classifierCtx.fillText('Center Y', 0, 0);
    classifierCtx.restore();
}

/**
 * Update the classifier visualization with current optimizer positions
 */
function updateClassifierViz(optimizerPositions) {
    if (!isClassifierActive) return;
    
    // Pass the entire optimizerPositions object to render all circles
    renderClassifierViz(optimizerPositions);
}

// Expose functions to global scope
window.initClassifierViz = initClassifierViz;
window.setClassifierPanelVisible = setClassifierPanelVisible;
window.renderClassifierViz = renderClassifierViz;
window.updateClassifierViz = updateClassifierViz;

