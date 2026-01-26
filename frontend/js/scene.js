/**
 * Three.js scene setup for 3D loss landscape visualization
 */

let scene, camera, renderer, controls;
let landscapeMesh = null;
let landscapeGeometry = null;
let landscapeWireframe = null;
let pickingMode = false;
let raycaster = null;
let mouse = new THREE.Vector2();
let currentManifoldId = 'custom_multimodal'; // Default manifold
let availableManifolds = []; // Store manifold list
let currentManifoldRange = [-5, 5]; // Store current manifold range for coordinate mapping
let startPointMarker = null; // Visual marker for the start point
let landscapeZRange = { zMin: 0, zMax: 1, scale: 2.0 }; // Store z-range for ballistic normalization

// Point to Cloud Run backend API, or localhost for development
window.API_BASE_URL = (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
    ? 'http://localhost:5001/api'
    : 'https://gd-experiments-1031734458893.us-central1.run.app/api';
const API_BASE_URL = window.API_BASE_URL; // Also keep as const for this file

// Initialize Three.js scene
function initScene() {
    // Scene
    scene = new THREE.Scene();
    
    // Set initial background color based on saved theme
    const savedTheme = localStorage.getItem('theme') || 'dark';
    const bgColor = savedTheme === 'dark' ? 0x0a0a0a : 0xfafafa;
    scene.background = new THREE.Color(bgColor);
    
    // Export scene globally for theme management
    window.scene = scene;
    
    // Camera
    const container = document.getElementById('canvas-container');
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
    camera.position.set(15, 15, 15);
    camera.lookAt(0, 0, 0);
    
    // Renderer
    renderer = new THREE.WebGLRenderer({ 
        canvas: document.getElementById('canvas'),
        antialias: true 
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    
    // Controls
    if (typeof THREE.OrbitControls !== 'undefined') {
        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.minDistance = 5;
        controls.maxDistance = 50;
    } else {
        // Fallback if OrbitControls not loaded
        console.warn('OrbitControls not available, using basic camera controls');
        controls = null;
    }
    
    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
    scene.add(ambientLight);
    
    const directionalLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight1.position.set(10, 10, 5);
    scene.add(directionalLight1);
    
    const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.4);
    directionalLight2.position.set(-10, -10, -5);
    scene.add(directionalLight2);
    
    // Grid helper with theme-appropriate colors
    const gridColor1 = savedTheme === 'dark' ? 0x444444 : 0xcccccc;
    const gridColor2 = savedTheme === 'dark' ? 0x222222 : 0xe8e8e8;
    const gridHelper = new THREE.GridHelper(20, 20, gridColor1, gridColor2);
    scene.add(gridHelper);
    
    // Axes helper
    const axesHelper = new THREE.AxesHelper(5);
    scene.add(axesHelper);
    
    // Initialize raycaster for point picking
    raycaster = new THREE.Raycaster();
    
    // Handle window resize
    window.addEventListener('resize', onWindowResize);
    
    // Add click handler for point picking
    renderer.domElement.addEventListener('click', onCanvasClick);
    
    // Start render loop
    animate();
}

// Load landscape mesh using client-side generation
async function loadLandscape(manifoldId = null) {
    const loadingEl = document.getElementById('loading');
    loadingEl.classList.remove('hidden');
    loadingEl.textContent = 'Generating landscape...';
    
    // Use provided manifoldId or current one
    const manifold = manifoldId || currentManifoldId;
    
    try {
        // Get manifold metadata to determine range
        const manifoldMeta = availableManifolds.find(m => m.id === manifold);
        const defaultRange = manifoldMeta ? manifoldMeta.default_range : [-5, 5];
        const x_range = [defaultRange[0], defaultRange[1]];
        const y_range = [defaultRange[0], defaultRange[1]];
        
        // Get manifold parameters if available
        let params = null;
        if (window.getManifoldParameters) {
            const manifoldParams = window.getManifoldParameters();
            if (manifoldParams && Object.keys(manifoldParams).length > 0) {
                params = manifoldParams;
            }
        }
        
        // Generate landscape mesh locally (no backend call!)
        const data = await window.generateManifoldLandscape(
            manifold,
            80,  // resolution
            x_range,
            y_range,
            params
        );
        
        createLandscapeMesh(data);
        currentManifoldId = manifold;
        updateManifoldDisplay(manifold);
        loadingEl.classList.add('hidden');
        loadingEl.textContent = 'Loading...';
        
        // Update start point marker after landscape is loaded
        if (window.currentParams) {
            setTimeout(() => {
                if (window.updateStartPointMarker) {
                    window.updateStartPointMarker(window.currentParams.startX, window.currentParams.startY);
                }
            }, 100);
        }
    } catch (error) {
        console.error('Error loading landscape:', error);
        loadingEl.textContent = 'Error generating landscape.';
    }
}

// Load available manifolds from backend
async function loadManifolds() {
    try {
        const response = await fetch(`${API_BASE_URL}/manifolds`);
        const data = await response.json();
        availableManifolds = data.manifolds || [];
        populateManifoldDropdown();
    } catch (error) {
        console.error('Error loading manifolds:', error);
    }
}

// Populate manifold dropdown with available manifolds
function populateManifoldDropdown() {
    const dropdownList = document.getElementById('manifold-dropdown-list');
    const currentBtn = document.getElementById('manifold-current');
    
    if (!dropdownList || !currentBtn) return;
    
    // Clear existing options
    dropdownList.innerHTML = '';
    
    // Add manifold options
    availableManifolds.forEach(manifold => {
        const option = document.createElement('div');
        option.className = 'manifold-option';
        if (manifold.id === currentManifoldId) {
            option.classList.add('selected');
        }
        option.dataset.manifoldId = manifold.id;
        
        const nameDiv = document.createElement('div');
        nameDiv.className = 'manifold-option-name';
        nameDiv.textContent = manifold.name;
        
        const descDiv = document.createElement('div');
        descDiv.className = 'manifold-option-description';
        descDiv.textContent = manifold.description;
        
        option.appendChild(nameDiv);
        option.appendChild(descDiv);
        
        // Add click handler for option
        option.addEventListener('click', () => {
            const newManifoldId = manifold.id;
            changeManifold(newManifoldId);
            closeDropdown();
        });
        
        dropdownList.appendChild(option);
    });
    
    // Add toggle event listener to current button
    currentBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        toggleDropdown();
    });
    
    // Close dropdown when clicking outside
    document.addEventListener('click', (e) => {
        if (!document.getElementById('manifold-selector').contains(e.target)) {
            closeDropdown();
        }
    });
    
    // Update the display with current manifold
    updateManifoldDisplay(currentManifoldId);
    
    // Populate parameters for the initially selected manifold
    const initialManifold = availableManifolds.find(m => m.id === currentManifoldId);
    if (initialManifold && window.populateManifoldParameters) {
        window.populateManifoldParameters(initialManifold);
    }
}

// Toggle dropdown open/closed
function toggleDropdown() {
    const dropdownList = document.getElementById('manifold-dropdown-list');
    const currentBtn = document.getElementById('manifold-current');
    
    if (dropdownList.classList.contains('hidden')) {
        dropdownList.classList.remove('hidden');
        currentBtn.classList.add('active');
    } else {
        dropdownList.classList.add('hidden');
        currentBtn.classList.remove('active');
    }
}

// Close dropdown
function closeDropdown() {
    const dropdownList = document.getElementById('manifold-dropdown-list');
    const currentBtn = document.getElementById('manifold-current');
    
    dropdownList.classList.add('hidden');
    currentBtn.classList.remove('active');
}

// Update the manifold name display
function updateManifoldDisplay(manifoldId) {
    const nameDisplay = document.getElementById('current-manifold-name');
    if (!nameDisplay) return;
    
    const manifold = availableManifolds.find(m => m.id === manifoldId);
    if (manifold) {
        nameDisplay.textContent = manifold.name;
    }
}

// Change the current manifold
async function changeManifold(manifoldId) {
    if (manifoldId === currentManifoldId) return;
    
    // Reset animation and trajectories when changing manifold
    if (window.resetAnimation) {
        window.resetAnimation();
    }
    if (window.clearTrajectories) {
        window.clearTrajectories();
    }
    
    // Reset starting position to origin to avoid out-of-bounds coordinates
    if (window.resetStartPosition) {
        window.resetStartPosition();
    }
    
    // Update selected state in dropdown options
    const dropdownList = document.getElementById('manifold-dropdown-list');
    if (dropdownList) {
        const options = dropdownList.querySelectorAll('.manifold-option');
        options.forEach(option => {
            if (option.dataset.manifoldId === manifoldId) {
                option.classList.add('selected');
            } else {
                option.classList.remove('selected');
            }
        });
    }
    
    // Populate manifold parameters if available
    const manifold = availableManifolds.find(m => m.id === manifoldId);
    if (manifold && window.populateManifoldParameters) {
        window.populateManifoldParameters(manifold);
    }
    
    // Load new landscape
    await loadLandscape(manifoldId);
    
    // Show/hide classifier panel based on manifold
    if (manifoldId === 'neural_net_classifier') {
        if (window.setClassifierPanelVisible) {
            window.setClassifierPanelVisible(true);
            // Initialize and render with start position or default
            setTimeout(() => {
                if (window.renderClassifierViz && window.currentParams) {
                    const startX = window.currentParams.startX || 0;
                    const startY = window.currentParams.startY || 0;
                    window.renderClassifierViz(startX, startY);
                }
            }, 100);
        }
    } else {
        if (window.setClassifierPanelVisible) {
            window.setClassifierPanelVisible(false);
        }
    }
}

// Update wireframe theme colors
function updateWireframeTheme(isDark) {
    if (landscapeWireframe && landscapeWireframe.material) {
        landscapeWireframe.material.color.setHex(isDark ? 0x444444 : 0x999999);
    }
}

// Expose functions to global scope
window.loadLandscape = loadLandscape;
window.getCurrentManifoldId = () => currentManifoldId;
window.getCurrentManifoldRange = () => currentManifoldRange;
window.getLandscapeZRange = () => landscapeZRange;
window.updateWireframeTheme = updateWireframeTheme;

// Create 3D mesh from landscape data
function createLandscapeMesh(landscapeData) {
    // Remove existing mesh if present
    if (landscapeMesh) {
        scene.remove(landscapeMesh);
        landscapeGeometry.dispose();
    }
    
    // Remove existing wireframe if present
    if (landscapeWireframe) {
        scene.remove(landscapeWireframe);
        landscapeWireframe.geometry.dispose();
        landscapeWireframe.material.dispose();
    }
    
    // Store the range from landscape data
    if (landscapeData.x_range && landscapeData.x_range.length === 2) {
        currentManifoldRange = landscapeData.x_range;
    }
    
    const x = landscapeData.x;
    const y = landscapeData.y;
    const z = landscapeData.z;
    
    const rows = z.length;
    const cols = z[0].length;
    
    // Create geometry
    landscapeGeometry = new THREE.PlaneGeometry(10, 10, cols - 1, rows - 1);
    const positions = landscapeGeometry.attributes.position;
    
    // Normalize z values and set heights
    const zMin = Math.min(...z.flat());
    const zMax = Math.max(...z.flat());
    const zRange = zMax - zMin;
    const scale = 2.0; // Height scale factor
    
    // Store z-range globally for ballistic coordinate conversion
    landscapeZRange = { zMin, zMax, zRange, scale };
    
    for (let i = 0; i < positions.count; i++) {
        const row = Math.floor(i / cols);
        const col = i % cols;
        
        if (row < rows && col < cols) {
            const normalizedZ = (z[row][col] - zMin) / zRange;
            positions.setZ(i, normalizedZ * scale);
            
            // Map x, y positions
            const xPos = (col / (cols - 1) - 0.5) * 10;
            const yPos = (row / (rows - 1) - 0.5) * 10;
            positions.setX(i, xPos);
            positions.setY(i, yPos);
        }
    }
    
    positions.needsUpdate = true;
    landscapeGeometry.computeVertexNormals();
    
    // Create material with color gradient based on height
    const material = new THREE.MeshPhongMaterial({
        vertexColors: true,
        side: THREE.DoubleSide,
        flatShading: false
    });
    
    // Add vertex colors based on height
    const colors = [];
    for (let i = 0; i < positions.count; i++) {
        const z = positions.getZ(i);
        const normalizedZ = z / scale;
        
        // Color gradient: blue (low) -> green -> yellow -> red (high)
        let r, g, b;
        if (normalizedZ < 0.25) {
            r = 0;
            g = normalizedZ * 4;
            b = 1;
        } else if (normalizedZ < 0.5) {
            r = 0;
            g = 1;
            b = 1 - (normalizedZ - 0.25) * 4;
        } else if (normalizedZ < 0.75) {
            r = (normalizedZ - 0.5) * 4;
            g = 1;
            b = 0;
        } else {
            r = 1;
            g = 1 - (normalizedZ - 0.75) * 4;
            b = 0;
        }
        
        colors.push(r, g, b);
    }
    
    landscapeGeometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    
    // Create mesh
    landscapeMesh = new THREE.Mesh(landscapeGeometry, material);
    landscapeMesh.rotation.x = -Math.PI / 2;
    scene.add(landscapeMesh);
    
    // Create a wireframe overlay for subtle mesh texture
    const savedTheme = localStorage.getItem('theme') || 'dark';
    const wireframeGeometry = landscapeGeometry.clone();
    const wireframeMaterial = new THREE.MeshBasicMaterial({
        color: savedTheme === 'dark' ? 0x444444 : 0x999999,
        wireframe: true,
        transparent: true,
        opacity: 0.15, // Very subtle
        depthWrite: false // Prevent z-fighting
    });
    
    landscapeWireframe = new THREE.Mesh(wireframeGeometry, wireframeMaterial);
    landscapeWireframe.rotation.x = -Math.PI / 2;
    landscapeWireframe.position.y = 0.01; // Slightly above to prevent z-fighting
    scene.add(landscapeWireframe);
}

// Create or update start point marker
function updateStartPointMarker(x, y) {
    // Remove existing marker if present
    if (startPointMarker) {
        scene.remove(startPointMarker);
        if (startPointMarker.geometry) startPointMarker.geometry.dispose();
        if (startPointMarker.material) startPointMarker.material.dispose();
    }
    
    // Map parameter space coordinates to world space coordinates
    const range = currentManifoldRange;
    const rangeMin = range[0];
    const rangeMax = range[1];
    const rangeSize = rangeMax - rangeMin;
    
    // Map from parameter space [rangeMin, rangeMax] to world space [-5, 5]
    const worldX = ((x - rangeMin) / rangeSize - 0.5) * 10;
    const worldZ = ((y - rangeMin) / rangeSize - 0.5) * 10;
    
    // Get height at the start point using the same logic as in optimizers.js
    let worldY = 0;
    if (landscapeMesh && landscapeGeometry) {
        const normalizedX = (x - rangeMin) / rangeSize;
        const normalizedY = (y - rangeMin) / rangeSize;
        
        const positions = landscapeGeometry.attributes.position;
        const gridSize = Math.sqrt(positions.count);
        
        const u = Math.max(0, Math.min(1, normalizedX));
        const v = Math.max(0, Math.min(1, normalizedY));
        
        const col = Math.floor(u * (gridSize - 1));
        const row = Math.floor(v * (gridSize - 1));
        
        const idx00 = row * gridSize + col;
        const idx01 = row * gridSize + Math.min(col + 1, gridSize - 1);
        const idx10 = Math.min(row + 1, gridSize - 1) * gridSize + col;
        const idx11 = Math.min(row + 1, gridSize - 1) * gridSize + Math.min(col + 1, gridSize - 1);
        
        const fx = u * (gridSize - 1) - col;
        const fy = v * (gridSize - 1) - row;
        
        const h00 = positions.getZ(idx00);
        const h01 = positions.getZ(idx01);
        const h10 = positions.getZ(idx10);
        const h11 = positions.getZ(idx11);
        
        const h0 = h00 * (1 - fx) + h01 * fx;
        const h1 = h10 * (1 - fx) + h11 * fx;
        const interpolatedHeight = h0 * (1 - fy) + h1 * fy;
        
        worldY = interpolatedHeight;
    }
    
    // Create a vertical line from the surface upward (using world coordinates)
    const points = [
        new THREE.Vector3(worldX, worldY, worldZ),  // Bottom of line (at surface)
        new THREE.Vector3(worldX, worldY + 1.5, worldZ)  // Top of line (extends above surface)
    ];
    
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({ 
        color: 0xffaa00,  // Orange color to stand out
        linewidth: 5,
        opacity: 0.9,
        transparent: true
    });
    
    startPointMarker = new THREE.Line(geometry, material);
    scene.add(startPointMarker);
}

// Expose function to global scope
window.updateStartPointMarker = updateStartPointMarker;

// Window resize handler
function onWindowResize() {
    const container = document.getElementById('canvas-container');
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
}

// Animation loop
function animate() {
    requestAnimationFrame(animate);
    if (controls) {
        controls.update();
    }
    renderer.render(scene, camera);
}

// Handle canvas click for point picking
function onCanvasClick(event) {
    if (!pickingMode || !landscapeMesh) return;
    
    // Calculate mouse position in normalized device coordinates (-1 to +1)
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    
    // Update raycaster
    raycaster.setFromCamera(mouse, camera);
    
    // Check for intersections with the landscape mesh
    const intersects = raycaster.intersectObject(landscapeMesh);
    
    if (intersects.length > 0) {
        const intersection = intersects[0];
        const point = intersection.point;
        
        // Convert 3D world coordinates back to 2D parameter space
        // The landscape mesh is rotated -90 degrees around X axis
        // So we need to account for that transformation
        const worldX = point.x;
        const worldZ = point.z; // Z becomes Y due to rotation
        
        // Map from world space [-5, 5] back to parameter space [rangeMin, rangeMax]
        const range = currentManifoldRange;
        const rangeMin = range[0];
        const rangeMax = range[1];
        const rangeSize = rangeMax - rangeMin;
        
        const paramX = (worldX / 10 + 0.5) * rangeSize + rangeMin;
        const paramY = (worldZ / 10 + 0.5) * rangeSize + rangeMin;
        
        // Update the UI inputs and parameters
        if (window.updateStartPosition) {
            window.updateStartPosition(paramX, paramY);
        }
        
        // Disable picking mode after selecting a point
        setPickingMode(false);
    }
}

// Enable or disable point picking mode
function setPickingMode(enabled) {
    pickingMode = enabled;
    const canvas = renderer.domElement;
    const pickBtn = document.getElementById('pick-point-btn');
    
    if (enabled) {
        canvas.classList.add('picking-mode');
        if (pickBtn) pickBtn.classList.add('active');
    } else {
        canvas.classList.remove('picking-mode');
        if (pickBtn) pickBtn.classList.remove('active');
    }
}

// Expose picking mode function to global scope
window.setPickingMode = setPickingMode;

// Initialize on page load
window.addEventListener('DOMContentLoaded', () => {
    initScene();
    loadManifolds(); // Load manifold list first
    loadLandscape(); // Then load default landscape
    
    // Initialize start point marker after a short delay to ensure landscape is loaded
    setTimeout(() => {
        // Get initial start position from controls.js default values (3, 3)
        const startX = 3;
        const startY = 3;
        if (window.updateStartPointMarker) {
            window.updateStartPointMarker(startX, startY);
        }
    }, 500);
});

