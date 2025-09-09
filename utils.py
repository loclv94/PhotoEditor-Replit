import os
from PIL import Image, ImageEnhance, ImageFilter
from app import app
from gemini_enhancer import GeminiEnhancer

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_all_images():
    """Get list of all uploaded images with their metadata"""
    images = []
    upload_folder = app.config['UPLOAD_FOLDER']
    enhanced_folder = app.config['ENHANCED_FOLDER']
    
    if not os.path.exists(upload_folder):
        return images
    
    for filename in os.listdir(upload_folder):
        if allowed_file(filename):
            filepath = os.path.join(upload_folder, filename)
            enhanced_filename = f"enhanced_{filename}"
            enhanced_filepath = os.path.join(enhanced_folder, enhanced_filename)
            
            # Get file size
            file_size = os.path.getsize(filepath)
            file_size_mb = round(file_size / (1024 * 1024), 2)
            
            # Get image dimensions
            try:
                with Image.open(filepath) as img:
                    width, height = img.size
            except Exception:
                width, height = 0, 0
            
            images.append({
                'filename': filename,
                'size_mb': file_size_mb,
                'dimensions': f"{width}x{height}",
                'has_enhanced': os.path.exists(enhanced_filepath),
                'enhanced_filename': enhanced_filename if os.path.exists(enhanced_filepath) else None
            })
    
    return sorted(images, key=lambda x: x['filename'])

def enhance_image(original_path, enhanced_path, enhancements, enhancement_prompt="", source_image_path=None):
    """
    Apply both PIL basic adjustments and AI-powered enhancements to an image
    
    Args:
        original_path: Path to original image
        enhanced_path: Path to save enhanced image
        enhancements: Dictionary of enhancement parameters
        enhancement_prompt: User's text prompt for AI enhancement
        source_image_path: Path to source image for conversational editing
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Determine input path
        input_path = source_image_path if source_image_path else original_path
        
        # Check if we have basic adjustments (brightness, contrast, saturation)
        has_basic_adjustments = (
            abs(enhancements.get('brightness', 0)) > 0 or
            abs(enhancements.get('contrast', 0)) > 0 or
            abs(enhancements.get('saturation', 0)) > 0
        )
        
        # Check if we have AI features or text prompt
        has_ai_features = (
            enhancement_prompt.strip() or
            any(enhancements.get(key) for key in ['eyeColor', 'hairColor', 'makeup', 'background', 'lighting', 'clothing', 'skinTone', 'expression'])
        )
        
        temp_path = None
        
        # Step 1: Apply basic PIL adjustments first if needed
        if has_basic_adjustments:
            temp_path = enhanced_path.replace('.', '_temp.')
            success = _apply_basic_adjustments(input_path, temp_path, enhancements)
            if not success:
                return False
            input_path = temp_path  # Use adjusted image as input for AI
        
        # Step 2: Apply AI enhancements if needed
        if has_ai_features:
            enhancer = GeminiEnhancer()
            success = enhancer.enhance_image(
                original_path,  # Always pass original for reference
                enhanced_path, 
                enhancements, 
                enhancement_prompt,
                input_path  # This will be temp_path if we did basic adjustments
            )
            if not success:
                return False
        elif has_basic_adjustments and temp_path:
            # Only basic adjustments, move temp file to final destination
            os.rename(temp_path, enhanced_path)
        else:
            # No adjustments requested, just copy original
            Image.open(input_path).save(enhanced_path)
        
        # Clean up temp file if it exists
        if temp_path and os.path.exists(temp_path) and temp_path != enhanced_path:
            os.remove(temp_path)
        
        app.logger.info(f"Enhanced image saved to {enhanced_path}")
        return True
        
    except Exception as e:
        app.logger.error(f"Error enhancing image: {str(e)}")
        return False

def _apply_basic_adjustments(input_path, output_path, enhancements):
    """
    Apply basic PIL adjustments (brightness, contrast, saturation)
    """
    try:
        # Load image
        image = Image.open(input_path)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        # Apply brightness adjustment
        brightness_value = enhancements.get('brightness', 0)
        if brightness_value != 0:
            # Convert from -50/+50 range to 0.5/1.5 range
            brightness_factor = 1.0 + (brightness_value / 100.0)
            brightness_factor = max(0.1, min(2.0, brightness_factor))  # Clamp to reasonable range
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness_factor)
        
        # Apply contrast adjustment  
        contrast_value = enhancements.get('contrast', 0)
        if contrast_value != 0:
            # Convert from -50/+50 range to 0.5/1.5 range
            contrast_factor = 1.0 + (contrast_value / 100.0)
            contrast_factor = max(0.1, min(2.0, contrast_factor))  # Clamp to reasonable range
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast_factor)
            
        # Apply saturation adjustment
        saturation_value = enhancements.get('saturation', 0)  
        if saturation_value != 0:
            # Convert from -50/+50 range to 0.5/1.5 range
            saturation_factor = 1.0 + (saturation_value / 100.0)
            saturation_factor = max(0.0, min(2.0, saturation_factor))  # Clamp to reasonable range
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(saturation_factor)
        
        # Save adjusted image
        image.save(output_path)
        return True
        
    except Exception as e:
        app.logger.error(f"Error applying basic adjustments: {str(e)}")
        return False
