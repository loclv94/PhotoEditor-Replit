import os
from PIL import Image, ImageEnhance, ImageFilter
from app import app

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

def enhance_image(original_path, enhanced_path, enhancements):
    """
    Apply AI-powered enhancements to an image (mocked for MVP)
    
    Args:
        original_path: Path to original image
        enhanced_path: Path to save enhanced image
        enhancements: Dictionary of enhancement parameters
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with Image.open(original_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            enhanced = img.copy()
            
            # Mock AI enhancements based on parameters
            face_enhancement = enhancements.get('face', 0)
            body_enhancement = enhancements.get('body', 0)
            skin_smoothing = enhancements.get('skin', 0)
            brightness = enhancements.get('brightness', 0)
            contrast = enhancements.get('contrast', 0)
            saturation = enhancements.get('saturation', 0)
            
            # Apply brightness adjustment
            if brightness != 0:
                brightness_factor = 1 + (brightness / 100)
                enhancer = ImageEnhance.Brightness(enhanced)
                enhanced = enhancer.enhance(brightness_factor)
            
            # Apply contrast adjustment
            if contrast != 0:
                contrast_factor = 1 + (contrast / 100)
                enhancer = ImageEnhance.Contrast(enhanced)
                enhanced = enhancer.enhance(contrast_factor)
            
            # Apply saturation adjustment
            if saturation != 0:
                saturation_factor = 1 + (saturation / 100)
                enhancer = ImageEnhance.Color(enhanced)
                enhanced = enhancer.enhance(saturation_factor)
            
            # Apply skin smoothing (using blur filter as mock)
            if skin_smoothing > 0:
                blur_radius = skin_smoothing / 20  # Scale to reasonable blur radius
                enhanced = enhanced.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            
            # Apply sharpening for face/body enhancement
            if face_enhancement > 0 or body_enhancement > 0:
                sharpness_factor = 1 + max(face_enhancement, body_enhancement) / 100
                enhancer = ImageEnhance.Sharpness(enhanced)
                enhanced = enhancer.enhance(sharpness_factor)
            
            # Save enhanced image
            enhanced.save(enhanced_path, quality=95)
            
            app.logger.info(f"Enhanced image saved to {enhanced_path}")
            return True
            
    except Exception as e:
        app.logger.error(f"Error enhancing image: {str(e)}")
        return False
