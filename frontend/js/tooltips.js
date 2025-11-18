/**
 * Tooltip system - displays tooltips outside of constrained containers
 */

(function() {
    let tooltipElement = null;
    let currentElement = null;
    let hoverTimeout = null;

    // Create tooltip element on page load
    function initTooltips() {
        // Create the tooltip element
        tooltipElement = document.createElement('div');
        tooltipElement.className = 'tooltip-popup';
        document.body.appendChild(tooltipElement);

        // Add event listeners to all info icons
        document.querySelectorAll('.info-icon').forEach(icon => {
            icon.addEventListener('mouseenter', (e) => showTooltip(e, 0));
            icon.addEventListener('mouseleave', hideTooltip);
        });

        // Add event listeners to all buttons with tooltips
        document.querySelectorAll('button[data-tooltip]').forEach(button => {
            button.addEventListener('mouseenter', (e) => showTooltip(e, 1000));
            button.addEventListener('mouseleave', hideTooltip);
        });
    }

    function showTooltip(event, delay = 0) {
        const element = event.currentTarget;
        const text = element.getAttribute('data-tooltip');
        
        if (!text) return;

        // Clear any existing timeout
        if (hoverTimeout) {
            clearTimeout(hoverTimeout);
        }

        // Set timeout for delayed show
        hoverTimeout = setTimeout(() => {
            currentElement = element;
            tooltipElement.textContent = text;
            
            // Position the tooltip
            positionTooltip(element);
            
            // Show tooltip
            requestAnimationFrame(() => {
                tooltipElement.classList.add('visible');
            });
        }, delay);
    }

    function positionTooltip(element) {
        const elementRect = element.getBoundingClientRect();
        const tooltipWidth = 220; // Match CSS width
        const spacing = 8;
        
        // Calculate position (centered above the element)
        let left = elementRect.left + (elementRect.width / 2) - (tooltipWidth / 2);
        let top = elementRect.top - spacing;
        
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
        // Clear any pending timeout
        if (hoverTimeout) {
            clearTimeout(hoverTimeout);
            hoverTimeout = null;
        }
        
        currentElement = null;
        tooltipElement.classList.remove('visible');
    }

    // Handle scroll - update position if tooltip is visible
    let scrollTimeout;
    window.addEventListener('scroll', () => {
        if (currentElement) {
            clearTimeout(scrollTimeout);
            scrollTimeout = setTimeout(() => {
                if (currentElement) {
                    positionTooltip(currentElement);
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

