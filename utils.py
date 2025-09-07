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
    Apply AI-powered enhancements to an image using Gemini
    
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
        # Initialize Gemini enhancer
        enhancer = GeminiEnhancer()
        
        # Use Gemini AI enhancement (supports conversational editing)
        success = enhancer.enhance_image(
            original_path, 
            enhanced_path, 
            enhancements, 
            enhancement_prompt,
            source_image_path
        )
        
        if success:
            app.logger.info(f"Enhanced image saved to {enhanced_path}")
            return True
        else:
            app.logger.error("Image enhancement failed")
            return False
            
    except Exception as e:
        app.logger.error(f"Error enhancing image: {str(e)}")
        return False
