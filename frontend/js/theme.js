/**
 * Theme management system
 */

// Initialize theme from localStorage or default to 'dark'
function initTheme() {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    setTheme(savedTheme, false); // Don't animate on initial load
}

// Set theme and update UI
function setTheme(theme, animate = true) {
    const body = document.body;
    const currentTheme = body.getAttribute('data-theme') || 'dark';
    
    // Don't do anything if theme hasn't changed
    if (currentTheme === theme && body.hasAttribute('data-theme')) {
        return;
    }
    
    // Set theme attribute
    body.setAttribute('data-theme', theme);
    
    // Save to localStorage
    localStorage.setItem('theme', theme);
    
    // Update button icons
    updateThemeButton(theme);
    
    // Update Three.js scene background
    updateSceneBackground(theme);
    
    // Update grid colors
    updateSceneGrid(theme);
    
    // Update wireframe colors
    if (window.updateWireframeTheme) {
        window.updateWireframeTheme(theme === 'dark');
    }
}

// Toggle between dark and light themes
function toggleTheme() {
    const current = document.body.getAttribute('data-theme') || 'dark';
    const newTheme = current === 'dark' ? 'light' : 'dark';
    setTheme(newTheme);
}

// Update theme toggle button icons
function updateThemeButton(theme) {
    const sunIcon = document.querySelector('.sun-icon');
    const moonIcon = document.querySelector('.moon-icon');
    
    if (!sunIcon || !moonIcon) return;
    
    if (theme === 'dark') {
        // Show sun icon (to switch to light)
        sunIcon.classList.remove('hidden');
        moonIcon.classList.add('hidden');
    } else {
        // Show moon icon (to switch to dark)
        sunIcon.classList.add('hidden');
        moonIcon.classList.remove('hidden');
    }
}

// Update Three.js scene background color
function updateSceneBackground(theme) {
    if (window.scene) {
        const bgColor = theme === 'dark' ? 0x0a0a0a : 0xfafafa;
        window.scene.background = new THREE.Color(bgColor);
    }
}

// Update Three.js grid colors
function updateSceneGrid(theme) {
    if (!window.scene) return;
    
    // Find and remove old grid
    const oldGrid = window.scene.children.find(child => child.type === 'GridHelper');
    if (oldGrid) {
        window.scene.remove(oldGrid);
        if (oldGrid.geometry) oldGrid.geometry.dispose();
        if (oldGrid.material) oldGrid.material.dispose();
    }
    
    // Create new grid with theme-appropriate colors
    const gridColor1 = theme === 'dark' ? 0x444444 : 0xcccccc;
    const gridColor2 = theme === 'dark' ? 0x222222 : 0xe8e8e8;
    const gridHelper = new THREE.GridHelper(20, 20, gridColor1, gridColor2);
    window.scene.add(gridHelper);
}

// Initialize on DOM load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        initTheme();
        
        // Add event listener to theme toggle button
        const themeToggleBtn = document.getElementById('theme-toggle');
        if (themeToggleBtn) {
            themeToggleBtn.addEventListener('click', toggleTheme);
        }
    });
} else {
    // DOM already loaded
    initTheme();
    
    const themeToggleBtn = document.getElementById('theme-toggle');
    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', toggleTheme);
    }
}

// Export functions for external use
window.toggleTheme = toggleTheme;
window.setTheme = setTheme;
window.getTheme = () => document.body.getAttribute('data-theme') || 'dark';
