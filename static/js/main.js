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

        // The file input now overlays the upload area, so clicks go directly to it
        // No need for manual click handling

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
            let enhancementPrompt = document.getElementById('enhancementPrompt').value;
            
            // Get all enhancement options
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
            
            // If no conversational prompt, create one from dropdown selections
            if (!enhancementPrompt.trim()) {
                enhancementPrompt = createPromptFromSelections(enhancements);
                if (!enhancementPrompt) {
                    showAlert('Please enter a text prompt or select enhancement options', 'error');
                    return;
                }
            }

            // Show loading state
            const originalText = enhanceBtn.innerHTML;
            enhanceBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Processing with AI...';
            enhanceBtn.disabled = true;

            // Send enhancement request
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
                    showAlert('Enhancement completed successfully!', 'success');
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
                enhanceBtn.innerHTML = originalText;
                enhanceBtn.disabled = false;
            });
        });

        // Handle continue editing for conversational editing
        const continueEditBtn = document.getElementById('continueEditBtn');
        if (continueEditBtn) {
            continueEditBtn.addEventListener('click', function() {
                const filename = this.dataset.filename;
                const enhancementPrompt = document.getElementById('enhancementPrompt').value;
                
                if (!enhancementPrompt.trim()) {
                    showAlert('Please enter what you want to change', 'error');
                    return;
                }
                
                // Show loading state
                const originalText = this.innerHTML;
                this.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Applying Changes...';
                this.disabled = true;
                
                // Send conversational edit request
                fetch('/api/continue-edit', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        filename: filename,
                        enhancement_prompt: enhancementPrompt
                    })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        showAlert('✨ ' + data.message, 'success');
                        // Clear the prompt for next edit
                        document.getElementById('enhancementPrompt').value = '';
                        // Reload to show updated image
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
                    this.innerHTML = originalText;
                    this.disabled = false;
                });
            });
        }

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

// Helper function to create enhancement prompt from dropdown selections
function createPromptFromSelections(enhancements) {
    const changes = [];
    
    // Facial features
    if (enhancements.eyeColor) changes.push(`change eye color to ${enhancements.eyeColor}`);
    if (enhancements.eyeShape) changes.push(`make eyes more ${enhancements.eyeShape}`);
    if (enhancements.lipColor) changes.push(`change lip color to ${enhancements.lipColor}`);
    if (enhancements.faceShape) changes.push(`adjust face shape to be more ${enhancements.faceShape}`);
    
    // Hair
    if (enhancements.hairColor) changes.push(`change hair color to ${enhancements.hairColor}`);
    if (enhancements.hairStyle) changes.push(`style hair to be ${enhancements.hairStyle}`);
    
    // Makeup and expression
    if (enhancements.makeup) changes.push(`apply ${enhancements.makeup} makeup`);
    if (enhancements.expression) changes.push(`change expression to ${enhancements.expression}`);
    
    // Skin and beauty
    if (enhancements.skinTone) changes.push(`improve skin tone to ${enhancements.skinTone}`);
    
    // Environment
    if (enhancements.background) changes.push(`change background to ${enhancements.background}`);
    if (enhancements.lighting) changes.push(`adjust lighting to ${enhancements.lighting}`);
    if (enhancements.clothing) changes.push(`change clothing to ${enhancements.clothing}`);
    
    // Basic adjustments (only include if significantly changed)
    if (Math.abs(enhancements.brightness) > 10) {
        changes.push(`adjust brightness ${enhancements.brightness > 0 ? 'brighter' : 'darker'}`);
    }
    if (Math.abs(enhancements.contrast) > 10) {
        changes.push(`adjust contrast ${enhancements.contrast > 0 ? 'higher' : 'lower'}`);
    }
    if (Math.abs(enhancements.saturation) > 10) {
        changes.push(`adjust saturation ${enhancements.saturation > 0 ? 'more vibrant' : 'more muted'}`);
    }
    
    return changes.length > 0 ? changes.join(', ') : '';
}
