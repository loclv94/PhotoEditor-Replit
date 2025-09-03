// Main JavaScript functionality for AI Photo Enhancer

document.addEventListener('DOMContentLoaded', function() {
    // File upload functionality
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const uploadForm = document.getElementById('uploadForm');

    if (uploadArea && fileInput) {
        console.log('Upload functionality initialized');
        
        // Handle drag and drop
        uploadArea.addEventListener('dragover', function(e) {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', function(e) {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', function(e) {
            e.preventDefault();
            e.stopPropagation();
            uploadArea.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                console.log('File dropped:', files[0].name);
                fileInput.files = files;
                uploadForm.submit();
            }
        });

        // Handle click to upload
        uploadArea.addEventListener('click', function(e) {
            console.log('Upload area clicked');
            e.preventDefault();
            e.stopPropagation();
            fileInput.click();
        });

        // Handle file selection
        fileInput.addEventListener('change', function(e) {
            console.log('File selected:', this.files[0]?.name);
            e.stopPropagation();
            if (this.files.length > 0) {
                uploadForm.submit();
            }
        });
    } else {
        console.log('Upload elements not found:', { uploadArea, fileInput });
    }

    // Enhancement form functionality
    const enhanceForm = document.getElementById('enhanceForm');
    const enhanceBtn = document.getElementById('enhanceBtn');
    const resetBtn = document.getElementById('resetBtn');

    if (enhanceForm) {
        // Initialize slider value displays
        const sliders = [
            { slider: 'brightnessSlider', value: 'brightnessValue', suffix: '%' },
            { slider: 'contrastSlider', value: 'contrastValue', suffix: '%' },
            { slider: 'saturationSlider', value: 'saturationValue', suffix: '%' },
            { slider: 'heightSlider', value: 'heightValue', suffix: '%' },
            { slider: 'bodySlider', value: 'bodyValue', suffix: '%' },
            { slider: 'postureSlider', value: 'postureValue', suffix: '%' },
            { slider: 'blemishSlider', value: 'blemishValue', suffix: '%' }
        ];

        sliders.forEach(item => {
            const slider = document.getElementById(item.slider);
            const valueDisplay = document.getElementById(item.value);
            
            if (slider && valueDisplay) {
                slider.addEventListener('input', function() {
                    valueDisplay.textContent = this.value + item.suffix;
                });
                
                // Initialize display
                valueDisplay.textContent = slider.value + item.suffix;
            }
        });

        // Handle form submission
        enhanceForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const filename = document.getElementById('filename').value;
            const enhancementPrompt = document.getElementById('enhancementPrompt').value;
            const enhancements = {
                // Basic sliders
                brightness: parseInt(document.getElementById('brightnessSlider').value),
                contrast: parseInt(document.getElementById('contrastSlider').value),
                saturation: parseInt(document.getElementById('saturationSlider').value),
                
                // Body & Posture
                height: parseInt(document.getElementById('heightSlider').value),
                body: parseInt(document.getElementById('bodySlider').value),
                posture: parseInt(document.getElementById('postureSlider').value),
                
                // Skin & Beauty
                blemish: parseInt(document.getElementById('blemishSlider').value),
                
                // Facial Features
                eyeColor: document.getElementById('eyeColor').value,
                eyeShape: document.getElementById('eyeShape').value,
                lipColor: document.getElementById('lipColor').value,
                faceShape: document.getElementById('faceShape').value,
                
                // Hair & Style
                hairColor: document.getElementById('hairColor').value,
                hairStyle: document.getElementById('hairStyle').value,
                makeup: document.getElementById('makeup').value,
                
                // Skin & Beauty
                skinTone: document.getElementById('skinTone').value,
                expression: document.getElementById('expression').value,
                
                // Environment & Style
                background: document.getElementById('background').value,
                lighting: document.getElementById('lighting').value,
                clothing: document.getElementById('clothing').value
            };

            // Show loading state
            const originalText = enhanceBtn.innerHTML;
            enhanceBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Processing with AI...';
            enhanceBtn.disabled = true;

            // Send enhancement request with prompt
            fetch('/api/enhance', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    filename: filename,
                    enhancements: enhancements,
                    enhancement_prompt: enhancementPrompt
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Show success message
                    showAlert('Enhancement completed successfully!', 'success');
                    
                    // Reload the page to show comparison
                    setTimeout(() => {
                        window.location.reload();
                    }, 1000);
                } else {
                    showAlert(data.error || 'Enhancement failed', 'error');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                showAlert('An error occurred during enhancement', 'error');
            })
            .finally(() => {
                // Reset button state
                enhanceBtn.innerHTML = originalText;
                enhanceBtn.disabled = false;
            });
        });

        // Handle reset button
        if (resetBtn) {
            resetBtn.addEventListener('click', function() {
                // Reset sliders
                sliders.forEach(item => {
                    const slider = document.getElementById(item.slider);
                    const valueDisplay = document.getElementById(item.value);
                    
                    if (slider && valueDisplay) {
                        slider.value = item.slider.includes('brightness') || 
                                     item.slider.includes('contrast') || 
                                     item.slider.includes('saturation') ? 0 : 0;
                        valueDisplay.textContent = slider.value + item.suffix;
                    }
                });
                
                // Reset prompt
                const promptField = document.getElementById('enhancementPrompt');
                if (promptField) {
                    promptField.value = '';
                }
            });
        }
    }

    // Auto-dismiss alerts
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        setTimeout(() => {
            if (alert.classList.contains('show')) {
                alert.classList.remove('show');
                setTimeout(() => {
                    alert.remove();
                }, 150);
            }
        }, 5000);
    });
});

// Utility function to show alerts
function showAlert(message, type) {
    const alertContainer = document.createElement('div');
    alertContainer.className = `alert alert-${type === 'error' ? 'danger' : 'success'} alert-dismissible fade show`;
    alertContainer.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Insert at the top of the container
    const container = document.querySelector('.container');
    container.insertBefore(alertContainer, container.firstChild);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        if (alertContainer.classList.contains('show')) {
            alertContainer.classList.remove('show');
            setTimeout(() => {
                alertContainer.remove();
            }, 150);
        }
    }, 5000);
}

// Image lazy loading
document.addEventListener('DOMContentLoaded', function() {
    const images = document.querySelectorAll('img[loading="lazy"]');
    
    if ('IntersectionObserver' in window) {
        const imageObserver = new IntersectionObserver((entries, observer) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target;
                    img.src = img.dataset.src || img.src;
                    img.classList.remove('lazy');
                    observer.unobserve(img);
                }
            });
        });
        
        images.forEach(img => imageObserver.observe(img));
    }
});
