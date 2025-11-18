/**
 * Tooltip system - displays tooltips outside of constrained containers
 */

(function() {
    let tooltipElement = null;
    let currentIcon = null;

    // Create tooltip element on page load
    function initTooltips() {
        // Create the tooltip element
        tooltipElement = document.createElement('div');
        tooltipElement.className = 'tooltip-popup';
        document.body.appendChild(tooltipElement);

        // Add event listeners to all info icons
        document.querySelectorAll('.info-icon').forEach(icon => {
            icon.addEventListener('mouseenter', showTooltip);
            icon.addEventListener('mouseleave', hideTooltip);
        });
    }

    function showTooltip(event) {
        const icon = event.currentTarget;
        const text = icon.getAttribute('data-tooltip');
        
        if (!text) return;

        currentIcon = icon;
        tooltipElement.textContent = text;
        
        // Position the tooltip
        positionTooltip(icon);
        
        // Show tooltip
        requestAnimationFrame(() => {
            tooltipElement.classList.add('visible');
        });
    }

    function positionTooltip(icon) {
        const iconRect = icon.getBoundingClientRect();
        const tooltipWidth = 220; // Match CSS width
        const spacing = 8;
        
        // Calculate position (centered above the icon)
        let left = iconRect.left + (iconRect.width / 2) - (tooltipWidth / 2);
        let top = iconRect.top - spacing;
        
        // Make sure tooltip doesn't go off screen horizontally
        const margin = 10;
        if (left < margin) {
            left = margin;
        } else if (left + tooltipWidth > window.innerWidth - margin) {
            left = window.innerWidth - tooltipWidth - margin;
        }
        
        // Position tooltip
        tooltipElement.style.left = left + 'px';
        tooltipElement.style.top = top + 'px';
        tooltipElement.style.transform = 'translateY(-100%)';
    }

    function hideTooltip() {
        currentIcon = null;
        tooltipElement.classList.remove('visible');
    }

    // Handle scroll - update position if tooltip is visible
    let scrollTimeout;
    window.addEventListener('scroll', () => {
        if (currentIcon) {
            clearTimeout(scrollTimeout);
            scrollTimeout = setTimeout(() => {
                if (currentIcon) {
                    positionTooltip(currentIcon);
                }
            }, 10);
        }
    }, true); // Use capture to catch scroll in scrollable divs

    // Initialize on DOM load
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initTooltips);
    } else {
        initTooltips();
    }
})();

