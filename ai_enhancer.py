import os
import base64
import requests
from io import BytesIO
from PIL import Image
from openai import OpenAI

# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user

class AIPhotoEnhancer:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    def image_to_base64(self, image_path):
        """Convert image to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def base64_to_image(self, base64_string, output_path):
        """Convert base64 string to image file"""
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))
        image.save(output_path, quality=95)
        return True
    
    def download_image_from_url(self, url, output_path):
        """Download image from URL and save to file"""
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            image = Image.open(BytesIO(response.content))
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(output_path, quality=95)
            return True
        except Exception as e:
            print(f"Error downloading image: {e}")
            return False
    
    def analyze_image(self, image_path):
        """Analyze image using GPT-4o vision to understand content"""
        try:
            base64_image = self.image_to_base64(image_path)
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this image and describe the person, their features, clothing, background, and overall scene. Focus on details that would be useful for photo enhancement."
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            return "Unable to analyze image"
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return "Unable to analyze image"
    
    def enhance_with_prompt(self, image_path, enhancement_prompt, style_prompt=""):
        """
        Enhance image using DALL-E with a descriptive prompt
        
        Args:
            image_path: Path to original image
            enhancement_prompt: User's description of desired changes
            style_prompt: Additional style instructions
            
        Returns:
            tuple: (success: bool, enhanced_image_url: str, error: str)
        """
        try:
            # First analyze the original image
            image_analysis = self.analyze_image(image_path)
            
            # Create a comprehensive prompt for DALL-E
            if style_prompt:
                full_prompt = f"Create an enhanced version of this scene: {image_analysis}. Apply these specific enhancements: {enhancement_prompt}. Style: {style_prompt}. Make it look professional and photorealistic, suitable for social media."
            else:
                full_prompt = f"Create an enhanced version of this scene: {image_analysis}. Apply these specific enhancements: {enhancement_prompt}. Make it look professional and photorealistic, suitable for social media."
            
            # Limit prompt length
            if len(full_prompt) > 1000:
                full_prompt = full_prompt[:1000]
            
            # Generate enhanced image using DALL-E
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=full_prompt,
                n=1,
                size="1024x1024",
                quality="hd"
            )
            
            if hasattr(response, 'data') and response.data and len(response.data) > 0:
                return True, response.data[0].url, None
            else:
                return False, None, "No image generated"
            
        except Exception as e:
            error_msg = str(e)
            print(f"Error enhancing image with AI: {error_msg}")
            return False, None, error_msg
    
    def enhance_with_traditional_methods(self, image_path, enhancements):
        """
        Fallback method using traditional image processing
        """
        try:
            from PIL import ImageEnhance, ImageFilter
            
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                enhanced = img.copy()
                
                # Apply traditional enhancements
                brightness = enhancements.get('brightness', 0)
                contrast = enhancements.get('contrast', 0)
                saturation = enhancements.get('saturation', 0)
                skin_smoothing = enhancements.get('skin', 0)
                
                if brightness != 0:
                    brightness_factor = 1 + (brightness / 100)
                    enhancer = ImageEnhance.Brightness(enhanced)
                    enhanced = enhancer.enhance(brightness_factor)
                
                if contrast != 0:
                    contrast_factor = 1 + (contrast / 100)
                    enhancer = ImageEnhance.Contrast(enhanced)
                    enhanced = enhancer.enhance(contrast_factor)
                
                if saturation != 0:
                    saturation_factor = 1 + (saturation / 100)
                    enhancer = ImageEnhance.Color(enhanced)
                    enhanced = enhancer.enhance(saturation_factor)
                
                if skin_smoothing > 0:
                    blur_radius = skin_smoothing / 20
                    enhanced = enhanced.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                
                return enhanced
                
        except Exception as e:
            print(f"Error with traditional enhancement: {e}")
            return None
    
    def enhance_image(self, original_path, enhanced_path, enhancements, enhancement_prompt=""):
        """
        Main enhancement method that tries AI first, falls back to traditional
        
        Args:
            original_path: Path to original image
            enhanced_path: Path to save enhanced image
            enhancements: Dictionary of enhancement parameters
            enhancement_prompt: User's text prompt for AI enhancement
            
        Returns:
            bool: Success status
        """
        try:
            # If we have a prompt, try AI enhancement
            if enhancement_prompt and enhancement_prompt.strip():
                print(f"Attempting AI enhancement with prompt: {enhancement_prompt}")
                
                success, image_url, error = self.enhance_with_prompt(
                    original_path, 
                    enhancement_prompt,
                    self._create_style_prompt(enhancements)
                )
                
                if success and image_url:
                    # Download the enhanced image
                    if self.download_image_from_url(image_url, enhanced_path):
                        print("AI enhancement successful")
                        return True
                    else:
                        print("Failed to download AI enhanced image")
                else:
                    print(f"AI enhancement failed: {error}")
            
            # Fallback to traditional enhancement
            print("Using traditional image enhancement")
            enhanced_image = self.enhance_with_traditional_methods(original_path, enhancements)
            
            if enhanced_image:
                enhanced_image.save(enhanced_path, quality=95)
                return True
            else:
                return False
                
        except Exception as e:
            print(f"Error in enhance_image: {e}")
            return False
    
    def _create_style_prompt(self, enhancements):
        """Create a style prompt based on enhancement sliders"""
        style_parts = []
        
        face_enhancement = enhancements.get('face', 0)
        body_enhancement = enhancements.get('body', 0)
        skin_smoothing = enhancements.get('skin', 0)
        
        if face_enhancement > 50:
            style_parts.append("enhanced facial features")
        if body_enhancement > 50:
            style_parts.append("improved body proportions")
        if skin_smoothing > 50:
            style_parts.append("smooth, flawless skin")
        
        if style_parts:
            return ", ".join(style_parts)
        return "professional photography style"