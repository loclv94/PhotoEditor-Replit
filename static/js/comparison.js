// Image comparison slider functionality

document.addEventListener('DOMContentLoaded', function() {
    const comparisonContainer = document.getElementById('comparisonContainer');
    const comparisonSlider = document.getElementById('comparisonSlider');
    
    if (comparisonContainer && comparisonSlider) {
        const enhancedImage = comparisonContainer.querySelector('.enhanced');
        let isDragging = false;
        
        // Initialize slider position
        updateSliderPosition(50);
        
        // Mouse events
        comparisonSlider.addEventListener('mousedown', startDrag);
        document.addEventListener('mousemove', drag);
        document.addEventListener('mouseup', stopDrag);
        
        // Touch events for mobile
        comparisonSlider.addEventListener('touchstart', startDrag);
        document.addEventListener('touchmove', drag);
        document.addEventListener('touchend', stopDrag);
        
        // Click on container to move slider
        comparisonContainer.addEventListener('click', function(e) {
            if (e.target === comparisonSlider || comparisonSlider.contains(e.target)) {
                return;
            }
            
            const rect = comparisonContainer.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const percentage = (x / rect.width) * 100;
            updateSliderPosition(Math.max(0, Math.min(100, percentage)));
        });
        
        function startDrag(e) {
            isDragging = true;
            comparisonSlider.style.cursor = 'grabbing';
            e.preventDefault();
        }
        
        function drag(e) {
            if (!isDragging) return;
            
            const clientX = e.clientX || (e.touches && e.touches[0].clientX);
            if (!clientX) return;
            
            const rect = comparisonContainer.getBoundingClientRect();
            const x = clientX - rect.left;
            const percentage = (x / rect.width) * 100;
            
            updateSliderPosition(Math.max(0, Math.min(100, percentage)));
            e.preventDefault();
        }
        
        function stopDrag() {
            isDragging = false;
            comparisonSlider.style.cursor = 'col-resize';
        }
        
        function updateSliderPosition(percentage) {
            const clampedPercentage = Math.max(0, Math.min(100, percentage));
            
            // Update slider position
            comparisonSlider.style.left = clampedPercentage + '%';
            
            // Update enhanced image clip path
            if (enhancedImage) {
                enhancedImage.style.clipPath = `inset(0 ${100 - clampedPercentage}% 0 0)`;
            }
        }
        
        // Keyboard navigation
        comparisonSlider.addEventListener('keydown', function(e) {
            const currentPosition = parseFloat(comparisonSlider.style.left) || 50;
            let newPosition = currentPosition;
            
            switch(e.key) {
                case 'ArrowLeft':
                    newPosition = Math.max(0, currentPosition - 5);
                    break;
                case 'ArrowRight':
                    newPosition = Math.min(100, currentPosition + 5);
                    break;
                case 'Home':
                    newPosition = 0;
                    break;
                case 'End':
                    newPosition = 100;
                    break;
                default:
                    return;
            }
            
            updateSliderPosition(newPosition);
            e.preventDefault();
        });
        
        // Make slider focusable
        comparisonSlider.setAttribute('tabindex', '0');
        comparisonSlider.setAttribute('role', 'slider');
        comparisonSlider.setAttribute('aria-label', 'Image comparison slider');
        comparisonSlider.setAttribute('aria-valuemin', '0');
        comparisonSlider.setAttribute('aria-valuemax', '100');
        comparisonSlider.setAttribute('aria-valuenow', '50');
        
        // Update ARIA attributes when slider moves
        const originalUpdatePosition = updateSliderPosition;
        updateSliderPosition = function(percentage) {
            originalUpdatePosition(percentage);
            comparisonSlider.setAttribute('aria-valuenow', Math.round(percentage));
        };
        
        // Smooth transition on load
        setTimeout(() => {
            comparisonContainer.style.transition = 'opacity 0.3s ease';
            comparisonContainer.style.opacity = '1';
        }, 100);
    }
});

// Image preloading for better performance
function preloadImages() {
    const images = document.querySelectorAll('.comparison-image');
    images.forEach(img => {
        const imageUrl = img.src;
        const preloadImg = new Image();
        preloadImg.src = imageUrl;
    });
}

// Call preload when DOM is ready
document.addEventListener('DOMContentLoaded', preloadImages);
