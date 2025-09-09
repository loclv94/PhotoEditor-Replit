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
            
            // Get all enhancement options (safely check if elements exist)
            const enhancements = {
                // Basic sliders
                brightness: document.getElementById('brightnessSlider') ? parseInt(document.getElementById('brightnessSlider').value) : 0,
                contrast: document.getElementById('contrastSlider') ? parseInt(document.getElementById('contrastSlider').value) : 0,
                saturation: document.getElementById('saturationSlider') ? parseInt(document.getElementById('saturationSlider').value) : 0,
                
                // Body & Posture
                height: document.getElementById('heightSlider') ? parseInt(document.getElementById('heightSlider').value) : 0,
                body: document.getElementById('bodySlider') ? parseInt(document.getElementById('bodySlider').value) : 0,
                posture: document.getElementById('postureSlider') ? parseInt(document.getElementById('postureSlider').value) : 0,
                
                // Skin & Beauty
                blemish: document.getElementById('blemishSlider') ? parseInt(document.getElementById('blemishSlider').value) : 0,
                
                // Facial Features
                eyeColor: document.getElementById('eyeColor') ? document.getElementById('eyeColor').value : '',
                eyeShape: document.getElementById('eyeShape') ? document.getElementById('eyeShape').value : '',
                lipColor: document.getElementById('lipColor') ? document.getElementById('lipColor').value : '',
                faceShape: document.getElementById('faceShape') ? document.getElementById('faceShape').value : '',
                
                // Hair & Style
                hairColor: document.getElementById('hairColor') ? document.getElementById('hairColor').value : '',
                hairStyle: document.getElementById('hairStyle') ? document.getElementById('hairStyle').value : '',
                makeup: document.getElementById('makeup') ? document.getElementById('makeup').value : '',
                
                // Environment & Style (check if elements exist)
                background: document.getElementById('background') ? document.getElementById('background').value : '',
                lighting: document.getElementById('lighting') ? document.getElementById('lighting').value : '',
                clothing: document.getElementById('clothing') ? document.getElementById('clothing').value : '',
                skinTone: document.getElementById('skinTone') ? document.getElementById('skinTone').value : '',
                expression: document.getElementById('expression') ? document.getElementById('expression').value : ''
            };
            
            // Check if we have basic adjustments (these work without AI prompt)
            const hasBasicAdjustments = (
                Math.abs(enhancements.brightness) > 0 ||
                Math.abs(enhancements.contrast) > 0 ||
                Math.abs(enhancements.saturation) > 0
            );
            
            // Always combine text prompt with dropdown selections
            const dropdownPrompt = createPromptFromSelections(enhancements);
            
            // Combine both text and dropdown selections
            if (enhancementPrompt.trim() && dropdownPrompt) {
                enhancementPrompt = enhancementPrompt.trim() + ', ' + dropdownPrompt;
            } else if (dropdownPrompt) {
                enhancementPrompt = dropdownPrompt;
            } else if (!enhancementPrompt.trim() && !hasBasicAdjustments) {
                showAlert('Please enter a text prompt, select enhancement options, or adjust basic settings', 'error');
                return;
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
                let enhancementPrompt = document.getElementById('enhancementPrompt').value;
                
                // Get current dropdown selections for continue editing too
                const enhancements = {
                    // Basic sliders
                    brightness: document.getElementById('brightnessSlider') ? parseInt(document.getElementById('brightnessSlider').value) : 0,
                    contrast: document.getElementById('contrastSlider') ? parseInt(document.getElementById('contrastSlider').value) : 0,
                    saturation: document.getElementById('saturationSlider') ? parseInt(document.getElementById('saturationSlider').value) : 0,
                    
                    // Body & Posture
                    height: document.getElementById('heightSlider') ? parseInt(document.getElementById('heightSlider').value) : 0,
                    body: document.getElementById('bodySlider') ? parseInt(document.getElementById('bodySlider').value) : 0,
                    posture: document.getElementById('postureSlider') ? parseInt(document.getElementById('postureSlider').value) : 0,
                    
                    // Skin & Beauty
                    blemish: document.getElementById('blemishSlider') ? parseInt(document.getElementById('blemishSlider').value) : 0,
                    
                    // Facial Features
                    eyeColor: document.getElementById('eyeColor') ? document.getElementById('eyeColor').value : '',
                    eyeShape: document.getElementById('eyeShape') ? document.getElementById('eyeShape').value : '',
                    lipColor: document.getElementById('lipColor') ? document.getElementById('lipColor').value : '',
                    faceShape: document.getElementById('faceShape') ? document.getElementById('faceShape').value : '',
                    
                    // Hair & Style
                    hairColor: document.getElementById('hairColor') ? document.getElementById('hairColor').value : '',
                    hairStyle: document.getElementById('hairStyle') ? document.getElementById('hairStyle').value : '',
                    makeup: document.getElementById('makeup') ? document.getElementById('makeup').value : '',
                    
                    // Environment & Style
                    background: document.getElementById('background') ? document.getElementById('background').value : '',
                    lighting: document.getElementById('lighting') ? document.getElementById('lighting').value : '',
                    clothing: document.getElementById('clothing') ? document.getElementById('clothing').value : '',
                    skinTone: document.getElementById('skinTone') ? document.getElementById('skinTone').value : '',
                    expression: document.getElementById('expression') ? document.getElementById('expression').value : ''
                };
                
                // Check if we have basic adjustments for continue editing too
                const hasBasicAdjustments = (
                    Math.abs(enhancements.brightness) > 0 ||
                    Math.abs(enhancements.contrast) > 0 ||
                    Math.abs(enhancements.saturation) > 0
                );
                
                // Combine text with dropdown selections for continue editing
                const dropdownPrompt = createPromptFromSelections(enhancements);
                
                if (enhancementPrompt.trim() && dropdownPrompt) {
                    enhancementPrompt = enhancementPrompt.trim() + ', ' + dropdownPrompt;
                } else if (dropdownPrompt) {
                    enhancementPrompt = dropdownPrompt;
                } else if (!enhancementPrompt.trim() && !hasBasicAdjustments) {
                    showAlert('Please enter what you want to change, select enhancement options, or adjust basic settings', 'error');
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
                
                // Reset all dropdowns
                const dropdowns = ['eyeColor', 'eyeShape', 'lipColor', 'faceShape', 'hairColor', 
                                 'hairStyle', 'makeup', 'background', 'lighting', 'clothing', 
                                 'skinTone', 'expression'];
                
                dropdowns.forEach(id => {
                    const element = document.getElementById(id);
                    if (element) {
                        element.selectedIndex = 0; // Reset to first option (usually "Original" or "Natural")
                    }
                });
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
    
    console.log('Creating prompt from selections:', enhancements); // Debug log
    
    // Facial features
    if (enhancements.eyeColor && enhancements.eyeColor !== '') {
        changes.push(`change eye color to ${enhancements.eyeColor}`);
    }
    if (enhancements.eyeShape && enhancements.eyeShape !== '') {
        changes.push(`make eyes more ${enhancements.eyeShape}`);
    }
    if (enhancements.lipColor && enhancements.lipColor !== '') {
        changes.push(`change lip color to ${enhancements.lipColor}`);
    }
    if (enhancements.faceShape && enhancements.faceShape !== '') {
        changes.push(`adjust face shape to be more ${enhancements.faceShape}`);
    }
    
    // Hair
    if (enhancements.hairColor && enhancements.hairColor !== '') {
        changes.push(`change hair color to ${enhancements.hairColor}`);
    }
    if (enhancements.hairStyle && enhancements.hairStyle !== '') {
        changes.push(`style hair to be ${enhancements.hairStyle}`);
    }
    
    // Makeup
    if (enhancements.makeup && enhancements.makeup !== '') {
        if (enhancements.makeup === 'remove') {
            changes.push('remove makeup');
        } else {
            changes.push(`apply ${enhancements.makeup} makeup`);
        }
    }
    
    // Environment (if elements exist)
    if (enhancements.background && enhancements.background !== '') {
        changes.push(`change background to ${enhancements.background}`);
    }
    if (enhancements.lighting && enhancements.lighting !== '') {
        changes.push(`adjust lighting to ${enhancements.lighting}`);
    }
    if (enhancements.clothing && enhancements.clothing !== '') {
        changes.push(`change clothing to ${enhancements.clothing}`);
    }
    if (enhancements.skinTone && enhancements.skinTone !== '') {
        changes.push(`improve skin tone to ${enhancements.skinTone}`);
    }
    if (enhancements.expression && enhancements.expression !== '') {
        changes.push(`change expression to ${enhancements.expression}`);
    }
    
    // Basic adjustments are now handled by PIL, not included in AI prompt
    
    // Body adjustments (only include if significantly changed)
    if (Math.abs(enhancements.height) > 5) {
        changes.push(`adjust height ${enhancements.height > 0 ? 'taller' : 'shorter'}`);
    }
    if (Math.abs(enhancements.body) > 5) {
        changes.push(`adjust body shape ${enhancements.body > 0 ? 'fuller' : 'slimmer'}`);
    }
    if (Math.abs(enhancements.posture) > 5) {
        changes.push('improve posture');
    }
    if (Math.abs(enhancements.blemish) > 5) {
        changes.push('remove blemishes and smooth skin');
    }
    
    console.log('Generated changes:', changes); // Debug log
    return changes.length > 0 ? changes.join(', ') : '';
}
