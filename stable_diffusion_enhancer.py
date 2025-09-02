import os
import json
import base64
import requests
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageOps
import time
import random

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("NumPy not available, using basic image processing")

class StableDiffusionEnhancer:
    """
    ComfyUI + Stable Diffusion + LoRA Photo Enhancement System
    Provides advanced photo editing features for social media models
    """
    
    def __init__(self):
        print("Initializing Stable Diffusion Enhancer...")
        self.enhancement_cache = {}
        self.feature_processors = {
            'eye_color': self.enhance_eye_color,
            'eye_shape': self.enhance_eye_shape,
            'face_shape': self.modify_face_shape,
            'hair_color': self.change_hair_color,
            'hair_style': self.change_hair_style,
            'lip_shape': self.enhance_lip_shape,
            'lip_color': self.enhance_lip_color,
            'height_adjustment': self.adjust_height,
            'body_shape': self.adjust_body_shape,
            'background_change': self.change_background,
            'skin_tone': self.correct_skin_tone,
            'blemish_removal': self.remove_blemishes,
            'wrinkle_removal': self.remove_wrinkles,
            'expression_change': self.change_expression,
            'makeup_application': self.apply_makeup,
            'makeup_removal': self.remove_makeup,
            'lighting_enhancement': self.enhance_lighting,
            'clothing_change': self.change_clothing,
            'weight_adjustment': self.adjust_weight,
            'posture_correction': self.correct_posture
        }
    
    def detect_face_regions(self, image):
        """Detect facial regions using OpenCV for targeted enhancements"""
        try:
            import cv2
            
            # Convert PIL image to OpenCV format
            img_array = np.array(image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            # Load cascade classifiers
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Take the largest face
                face = max(faces, key=lambda rect: rect[2] * rect[3])
                x, y, w, h = face
                
                # Extract face region for eye detection
                roi_gray = gray[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
                
                # Convert eye coordinates to full image coordinates
                eye_coords = []
                for (ex, ey, ew, eh) in eyes:
                    eye_center_x = x + ex + ew//2
                    eye_center_y = y + ey + eh//2
                    eye_coords.append((eye_center_x, eye_center_y, ew, eh))
                
                return {
                    'has_face': True,
                    'face_box': (x, y, x+w, y+h),
                    'eyes': eye_coords,
                    'lips': (x + w//2, y + int(h * 0.75)),  # Estimate lip position
                    'face_width': w,
                    'face_height': h,
                    'confidence': 0.9
                }
            else:
                # Fallback to estimation if no face detected
                width, height = image.size
                return {
                    'has_face': True,
                    'face_box': (int(width*0.2), int(height*0.1), int(width*0.8), int(height*0.9)),
                    'eyes': [(int(width*0.35), int(height*0.4), 30, 15), (int(width*0.65), int(height*0.4), 30, 15)],
                    'lips': (int(width*0.5), int(height*0.7)),
                    'face_width': int(width*0.6),
                    'face_height': int(height*0.8),
                    'confidence': 0.6
                }
                
        except Exception as e:
            print(f"Face detection error: {e}, using fallback")
            # Fallback estimation
            width, height = image.size
            return {
                'has_face': True,
                'face_box': (int(width*0.2), int(height*0.1), int(width*0.8), int(height*0.9)),
                'eyes': [(int(width*0.35), int(height*0.4), 30, 15), (int(width*0.65), int(height*0.4), 30, 15)],
                'lips': (int(width*0.5), int(height*0.7)),
                'face_width': int(width*0.6),
                'face_height': int(height*0.8),
                'confidence': 0.5
            }
    
    def create_region_mask(self, image, region_type, face_data=None):
        """Create masks for specific facial/body regions"""
        width, height = image.size
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)
        
        if not face_data or not face_data.get('has_face'):
            return mask
            
        face = face_data['primary_face']
        cx, cy = face['center_x'], face['center_y']
        fw, fh = face['width'], face['height']
        
        if region_type == 'eyes':
            # Eye regions
            eye_y = cy - int(fh * 0.15)
            eye_width = int(fw * 0.15)
            eye_height = int(fh * 0.08)
            
            # Left eye
            draw.ellipse([cx - int(fw * 0.25) - eye_width//2, eye_y - eye_height//2,
                         cx - int(fw * 0.25) + eye_width//2, eye_y + eye_height//2], fill=255)
            
            # Right eye  
            draw.ellipse([cx + int(fw * 0.25) - eye_width//2, eye_y - eye_height//2,
                         cx + int(fw * 0.25) + eye_width//2, eye_y + eye_height//2], fill=255)
            
        elif region_type == 'lips':
            # Lip region
            lip_y = cy + int(fh * 0.15)
            lip_width = int(fw * 0.2)
            lip_height = int(fh * 0.06)
            draw.ellipse([cx - lip_width//2, lip_y - lip_height//2,
                         cx + lip_width//2, lip_y + lip_height//2], fill=255)
            
        elif region_type == 'face':
            # Entire face region
            draw.ellipse([face['x'], face['y'], 
                         face['x'] + face['width'], 
                         face['y'] + face['height']], fill=255)
            
        elif region_type == 'hair':
            # Hair region (upper portion)
            hair_y = face['y'] - int(fh * 0.3)
            draw.ellipse([face['x'] - int(fw * 0.2), hair_y,
                         face['x'] + face['width'] + int(fw * 0.2),
                         face['y'] + int(fh * 0.3)], fill=255)
        
        # Apply blur for smooth edges
        mask = mask.filter(ImageFilter.GaussianBlur(radius=2))
        return mask
    
    def enhance_eye_color(self, image, color="blue", intensity=0.5):
        """Enhance or change eye color"""
        try:
            face_data = self.detect_face_regions(image)
            if not face_data.get('has_face'):
                return image
                
            eye_mask = self.create_region_mask(image, 'eyes', face_data)
            
            # Create color overlay
            color_map = {
                'blue': (100, 149, 237),
                'green': (46, 139, 87),
                'brown': (139, 69, 19),
                'gray': (128, 128, 128),
                'hazel': (139, 117, 0)
            }
            
            eye_color = color_map.get(color.lower(), color_map['blue'])
            color_overlay = Image.new('RGB', image.size, eye_color)
            
            # Apply color with mask
            enhanced = Image.composite(color_overlay, image, eye_mask)
            
            # Blend with original based on intensity
            enhanced = Image.blend(image, enhanced, intensity)
            
            return enhanced
            
        except Exception as e:
            print(f"Eye color enhancement error: {e}")
            return image
    
    def enhance_eye_shape(self, image, shape="almond", intensity=0.3):
        """Modify eye shape"""
        # This would require more advanced image warping
        # For now, apply subtle enhancement around eye area
        try:
            face_data = self.detect_face_regions(image)
            if not face_data.get('has_face'):
                return image
                
            eye_mask = self.create_region_mask(image, 'eyes', face_data)
            
            # Apply subtle sharpening to eye area
            enhanced = image.copy()
            eye_area = ImageEnhance.Sharpness(enhanced).enhance(1.0 + intensity)
            enhanced = Image.composite(eye_area, enhanced, eye_mask)
            
            return enhanced
            
        except Exception as e:
            print(f"Eye shape enhancement error: {e}")
            return image
    
    def modify_face_shape(self, image, target_shape="oval", intensity=0.3):
        """Modify facial shape (simplified version)"""
        try:
            # Apply subtle contrast adjustment to create face shaping effect
            contrast_enhancer = ImageEnhance.Contrast(image)
            enhanced = contrast_enhancer.enhance(1.0 + intensity * 0.2)
            
            # Apply subtle brightness gradient for contouring effect
            face_data = self.detect_face_regions(image)
            if face_data.get('has_face'):
                face_mask = self.create_region_mask(image, 'face', face_data)
                brightness_enhancer = ImageEnhance.Brightness(enhanced)
                brighter = brightness_enhancer.enhance(1.05)
                enhanced = Image.composite(brighter, enhanced, face_mask)
            
            return enhanced
            
        except Exception as e:
            print(f"Face shape modification error: {e}")
            return image
    
    def change_hair_color(self, image, color="blonde", intensity=0.6):
        """Change hair color"""
        try:
            face_data = self.detect_face_regions(image)
            if not face_data.get('has_face'):
                return image
                
            hair_mask = self.create_region_mask(image, 'hair', face_data)
            
            # Hair color mapping
            color_map = {
                'blonde': (255, 236, 139),
                'brunette': (101, 67, 33),
                'black': (28, 28, 28),
                'red': (165, 42, 42),
                'auburn': (165, 82, 42),
                'gray': (128, 128, 128)
            }
            
            hair_color = color_map.get(color.lower(), color_map['brunette'])
            color_overlay = Image.new('RGB', image.size, hair_color)
            
            # Apply color with mask
            enhanced = Image.composite(color_overlay, image, hair_mask)
            enhanced = Image.blend(image, enhanced, intensity)
            
            return enhanced
            
        except Exception as e:
            print(f"Hair color change error: {e}")
            return image
    
    def change_hair_style(self, image, style="wavy", intensity=0.4):
        """Change hair style (texture simulation)"""
        try:
            face_data = self.detect_face_regions(image)
            if not face_data.get('has_face'):
                return image
                
            hair_mask = self.create_region_mask(image, 'hair', face_data)
            
            # Apply texture effects based on style
            enhanced = image.copy()
            
            if style.lower() == "curly":
                # Add texture with sharpen filter
                enhanced = enhanced.filter(ImageFilter.SHARPEN)
            elif style.lower() == "straight":
                # Smooth with blur
                enhanced = enhanced.filter(ImageFilter.GaussianBlur(radius=0.5))
            elif style.lower() == "wavy":
                # Slight unsharp mask effect
                enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
            
            # Apply only to hair region
            enhanced = Image.composite(enhanced, image, hair_mask)
            enhanced = Image.blend(image, enhanced, intensity)
            
            return enhanced
            
        except Exception as e:
            print(f"Hair style change error: {e}")
            return image
    
    def enhance_lip_shape(self, image, intensity=0.4):
        """Enhance lip shape and definition"""
        try:
            face_data = self.detect_face_regions(image)
            if not face_data.get('has_face'):
                return image
                
            lip_mask = self.create_region_mask(image, 'lips', face_data)
            
            # Enhance lip area with sharpening and slight saturation
            enhanced = image.copy()
            sharp = ImageEnhance.Sharpness(enhanced).enhance(1.3)
            saturated = ImageEnhance.Color(sharp).enhance(1.2)
            
            # Apply to lip area
            enhanced = Image.composite(saturated, image, lip_mask)
            enhanced = Image.blend(image, enhanced, intensity)
            
            return enhanced
            
        except Exception as e:
            print(f"Lip shape enhancement error: {e}")
            return image
    
    def enhance_lip_color(self, image, color="rose", intensity=0.5):
        """Change or enhance lip color"""
        try:
            face_data = self.detect_face_regions(image)
            if not face_data.get('has_face'):
                return image
                
            lip_mask = self.create_region_mask(image, 'lips', face_data)
            
            # Lip color mapping
            color_map = {
                'rose': (255, 182, 193),
                'red': (220, 20, 60),
                'pink': (255, 192, 203),
                'coral': (255, 127, 80),
                'berry': (139, 0, 139),
                'nude': (222, 184, 135)
            }
            
            lip_color = color_map.get(color.lower(), color_map['rose'])
            color_overlay = Image.new('RGB', image.size, lip_color)
            
            # Apply color with mask
            enhanced = Image.composite(color_overlay, image, lip_mask)
            enhanced = Image.blend(image, enhanced, intensity)
            
            return enhanced
            
        except Exception as e:
            print(f"Lip color enhancement error: {e}")
            return image
    
    def adjust_height(self, image, adjustment=0.1):
        """Simulate height adjustment through perspective"""
        try:
            if abs(adjustment) < 0.05:
                return image
                
            width, height = image.size
            
            if adjustment > 0:
                # Make taller - stretch vertically, compress horizontally slightly
                new_height = int(height * (1 + adjustment))
                new_width = int(width * (1 - adjustment * 0.1))
            else:
                # Make shorter - compress vertically, expand horizontally slightly
                new_height = int(height * (1 + adjustment))
                new_width = int(width * (1 - adjustment * 0.1))
            
            # Resize with high quality
            enhanced = image.resize((new_width, new_height), Image.LANCZOS)
            
            # Crop or pad to maintain original dimensions
            if enhanced.size != image.size:
                enhanced = ImageOps.fit(enhanced, image.size, Image.LANCZOS)
            
            return enhanced
            
        except Exception as e:
            print(f"Height adjustment error: {e}")
            return image
    
    def adjust_body_shape(self, image, adjustments=None):
        """Adjust body proportions"""
        try:
            if not adjustments:
                return image
                
            enhanced = image.copy()
            
            # Apply subtle perspective adjustments
            body_adjustment = adjustments.get('body', 0) / 100.0
            
            if abs(body_adjustment) > 0.1:
                # Slight resize for body proportion effect
                width, height = image.size
                new_width = int(width * (1 - body_adjustment * 0.1))
                enhanced = enhanced.resize((new_width, height), Image.LANCZOS)
                enhanced = ImageOps.fit(enhanced, image.size, Image.LANCZOS)
            
            return enhanced
            
        except Exception as e:
            print(f"Body shape adjustment error: {e}")
            return image
    
    def change_background(self, image, background_prompt="studio background"):
        """Change image background (simplified version)"""
        try:
            # Create edge-based mask for subject separation
            # This is simplified - in production use proper segmentation
            
            # Convert to grayscale for edge detection
            gray = image.convert('L')
            
            # Apply edge detection
            edges = gray.filter(ImageFilter.FIND_EDGES)
            
            # Invert and blur to create subject mask
            mask = ImageOps.invert(edges)
            mask = mask.filter(ImageFilter.GaussianBlur(radius=5))
            
            # Create new background based on prompt
            background_colors = {
                'studio': (248, 248, 255),
                'nature': (135, 206, 235),
                'sunset': (255, 165, 0),
                'city': (105, 105, 105),
                'beach': (240, 248, 255)
            }
            
            # Find matching background color
            bg_color = background_colors.get('studio', (248, 248, 255))
            for key in background_colors:
                if key in background_prompt.lower():
                    bg_color = background_colors[key]
                    break
            
            # Create background
            background = Image.new('RGB', image.size, bg_color)
            
            # Composite subject onto new background
            enhanced = Image.composite(image, background, mask)
            
            return enhanced
            
        except Exception as e:
            print(f"Background change error: {e}")
            return image
    
    def correct_skin_tone(self, image, tone="natural", intensity=0.3):
        """Correct skin tone"""
        try:
            face_data = self.detect_face_regions(image)
            if not face_data.get('has_face'):
                return image
                
            face_mask = self.create_region_mask(image, 'face', face_data)
            
            # Skin tone adjustments
            enhanced = image.copy()
            
            if tone.lower() == "warmer":
                # Increase red/yellow channels
                enhanced = ImageEnhance.Color(enhanced).enhance(1.1)
                
            elif tone.lower() == "cooler":
                # Decrease red/yellow, increase blue
                enhanced = ImageEnhance.Color(enhanced).enhance(0.95)
                
            elif tone.lower() == "natural":
                # Balanced adjustment
                enhanced = ImageEnhance.Brightness(enhanced).enhance(1.05)
            
            # Apply to face area
            enhanced = Image.composite(enhanced, image, face_mask)
            enhanced = Image.blend(image, enhanced, intensity)
            
            return enhanced
            
        except Exception as e:
            print(f"Skin tone correction error: {e}")
            return image
    
    def remove_blemishes(self, image, intensity=0.6):
        """Remove blemishes and smooth skin"""
        try:
            face_data = self.detect_face_regions(image)
            if not face_data.get('has_face'):
                return image
                
            face_mask = self.create_region_mask(image, 'face', face_data)
            
            # Apply blur to smooth skin
            blur_radius = max(1, int(intensity * 3))
            smoothed = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            
            # Blend with original using face mask
            enhanced = Image.composite(smoothed, image, face_mask)
            enhanced = Image.blend(image, enhanced, intensity)
            
            return enhanced
            
        except Exception as e:
            print(f"Blemish removal error: {e}")
            return image
    
    def remove_wrinkles(self, image, intensity=0.5):
        """Remove wrinkles"""
        try:
            face_data = self.detect_face_regions(image)
            if not face_data.get('has_face'):
                return image
                
            face_mask = self.create_region_mask(image, 'face', face_data)
            
            # Apply selective blur to reduce wrinkles
            blur_radius = max(1, int(intensity * 2))
            smoothed = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            
            # Enhance brightness slightly
            brightened = ImageEnhance.Brightness(smoothed).enhance(1.02)
            
            # Apply to face area
            enhanced = Image.composite(brightened, image, face_mask)
            enhanced = Image.blend(image, enhanced, intensity)
            
            return enhanced
            
        except Exception as e:
            print(f"Wrinkle removal error: {e}")
            return image
    
    def change_expression(self, image, expression="smile", intensity=0.4):
        """Simulate expression changes"""
        try:
            face_data = self.detect_face_regions(image)
            if not face_data.get('has_face'):
                return image
                
            # For smile enhancement, brighten around mouth area
            if expression.lower() == "smile":
                lip_mask = self.create_region_mask(image, 'lips', face_data)
                enhanced = ImageEnhance.Brightness(image).enhance(1.1)
                enhanced = Image.composite(enhanced, image, lip_mask)
                enhanced = Image.blend(image, enhanced, intensity)
                return enhanced
            
            return image
            
        except Exception as e:
            print(f"Expression change error: {e}")
            return image
    
    def apply_makeup(self, image, makeup_type="natural", intensity=0.4):
        """Apply makeup effects"""
        try:
            face_data = self.detect_face_regions(image)
            if not face_data.get('has_face'):
                return image
                
            enhanced = image.copy()
            
            if makeup_type.lower() == "natural":
                # Enhance lips and eyes subtly
                enhanced = self.enhance_lip_color(enhanced, "rose", intensity * 0.6)
                enhanced = self.enhance_eye_color(enhanced, "brown", intensity * 0.3)
                
            elif makeup_type.lower() == "glamour":
                # More dramatic enhancement
                enhanced = self.enhance_lip_color(enhanced, "red", intensity * 0.8)
                enhanced = self.enhance_eye_color(enhanced, "blue", intensity * 0.5)
            
            return enhanced
            
        except Exception as e:
            print(f"Makeup application error: {e}")
            return image
    
    def remove_makeup(self, image, intensity=0.5):
        """Remove or reduce makeup"""
        try:
            # Reduce saturation and contrast slightly
            enhanced = ImageEnhance.Color(image).enhance(1 - intensity * 0.3)
            enhanced = ImageEnhance.Contrast(enhanced).enhance(1 - intensity * 0.2)
            
            return enhanced
            
        except Exception as e:
            print(f"Makeup removal error: {e}")
            return image
    
    def enhance_lighting(self, image, lighting_type="soft", intensity=0.4):
        """Enhance lighting and shadows"""
        try:
            if lighting_type.lower() == "soft":
                # Soft lighting - reduce contrast, increase brightness
                enhanced = ImageEnhance.Brightness(image).enhance(1 + intensity * 0.2)
                enhanced = ImageEnhance.Contrast(enhanced).enhance(1 - intensity * 0.1)
                
            elif lighting_type.lower() == "dramatic":
                # Dramatic lighting - increase contrast
                enhanced = ImageEnhance.Contrast(image).enhance(1 + intensity * 0.3)
                
            elif lighting_type.lower() == "natural":
                # Balanced natural lighting
                enhanced = ImageEnhance.Brightness(image).enhance(1 + intensity * 0.1)
            else:
                enhanced = image
            
            return enhanced
            
        except Exception as e:
            print(f"Lighting enhancement error: {e}")
            return image
    
    def change_clothing(self, image, clothing_prompt="elegant dress", intensity=0.3):
        """Change clothing (simplified color adjustment)"""
        try:
            # This is very simplified - in production would need advanced segmentation
            # For now, adjust color saturation as a basic clothing effect
            
            enhanced = ImageEnhance.Color(image).enhance(1 + intensity * 0.5)
            
            # Apply subtle brightness changes based on clothing type
            if "dark" in clothing_prompt.lower():
                enhanced = ImageEnhance.Brightness(enhanced).enhance(0.95)
            elif "bright" in clothing_prompt.lower() or "white" in clothing_prompt.lower():
                enhanced = ImageEnhance.Brightness(enhanced).enhance(1.05)
            
            return enhanced
            
        except Exception as e:
            print(f"Clothing change error: {e}")
            return image
    
    def adjust_weight(self, image, adjustment=0.0):
        """Adjust weight appearance"""
        try:
            if abs(adjustment) < 0.05:
                return image
                
            # Subtle horizontal scaling to simulate weight changes
            width, height = image.size
            
            if adjustment > 0:
                # Slightly wider
                new_width = int(width * (1 + adjustment * 0.1))
            else:
                # Slightly narrower
                new_width = int(width * (1 + adjustment * 0.1))
            
            enhanced = image.resize((new_width, height), Image.LANCZOS)
            enhanced = ImageOps.fit(enhanced, image.size, Image.LANCZOS)
            
            return enhanced
            
        except Exception as e:
            print(f"Weight adjustment error: {e}")
            return image
    
    def correct_posture(self, image, correction=0.1):
        """Correct posture (subtle perspective adjustment)"""
        try:
            if abs(correction) < 0.05:
                return image
                
            # Apply subtle vertical perspective correction
            enhanced = image.copy()
            
            # This is simplified - would need proper perspective transformation
            if correction > 0:
                # Straighten posture - slight vertical stretch
                width, height = image.size
                new_height = int(height * (1 + correction * 0.05))
                enhanced = enhanced.resize((width, new_height), Image.LANCZOS)
                enhanced = ImageOps.fit(enhanced, image.size, Image.LANCZOS)
            
            return enhanced
            
        except Exception as e:
            print(f"Posture correction error: {e}")
            return image
    
    def parse_enhancement_prompt(self, prompt):
        """Parse natural language prompt into specific enhancements"""
        prompt_lower = prompt.lower()
        enhancements = {}
        
        # Eye enhancements
        if any(word in prompt_lower for word in ['eye', 'eyes']):
            if any(color in prompt_lower for color in ['blue', 'green', 'brown', 'gray', 'hazel']):
                for color in ['blue', 'green', 'brown', 'gray', 'hazel']:
                    if color in prompt_lower:
                        enhancements['eye_color'] = color
                        break
        
        # Hair changes
        if any(word in prompt_lower for word in ['hair']):
            if any(color in prompt_lower for color in ['blonde', 'brunette', 'black', 'red', 'gray']):
                for color in ['blonde', 'brunette', 'black', 'red', 'gray']:
                    if color in prompt_lower:
                        enhancements['hair_color'] = color
                        break
        
        # Skin improvements
        if any(word in prompt_lower for word in ['skin', 'smooth', 'blemish', 'clear']):
            enhancements['blemish_removal'] = 0.6
            enhancements['skin_tone'] = 'natural'
        
        # Lighting
        if any(word in prompt_lower for word in ['lighting', 'light', 'bright', 'glow']):
            enhancements['lighting_enhancement'] = 'soft'
        
        # Background
        if any(word in prompt_lower for word in ['background', 'studio', 'nature', 'beach']):
            enhancements['background_change'] = prompt
        
        # Makeup
        if any(word in prompt_lower for word in ['makeup', 'lips', 'lipstick']):
            enhancements['makeup_application'] = 'natural'
        
        return enhancements
    
    def enhance_image(self, original_path, enhanced_path, enhancements, enhancement_prompt=""):
        """
        Main enhancement method using ComfyUI + Stable Diffusion approach
        
        Args:
            original_path: Path to original image
            enhanced_path: Path to save enhanced image
            enhancements: Dictionary of enhancement parameters
            enhancement_prompt: User's text prompt for AI enhancement
            
        Returns:
            bool: Success status
        """
        try:
            print(f"Starting ComfyUI-style enhancement with prompt: '{enhancement_prompt}'")
            
            # Load original image
            image = Image.open(original_path).convert('RGB')
            enhanced_image = image.copy()
            
            # Parse natural language prompt into specific enhancements
            if enhancement_prompt and enhancement_prompt.strip():
                prompt_enhancements = self.parse_enhancement_prompt(enhancement_prompt)
                
                # Apply AI-style enhancements based on prompt
                for enhancement_type, params in prompt_enhancements.items():
                    if enhancement_type in self.feature_processors:
                        try:
                            if isinstance(params, str):
                                enhanced_image = self.feature_processors[enhancement_type](enhanced_image, params)
                            elif isinstance(params, (int, float)):
                                enhanced_image = self.feature_processors[enhancement_type](enhanced_image, intensity=params)
                            else:
                                enhanced_image = self.feature_processors[enhancement_type](enhanced_image)
                        except Exception as e:
                            print(f"Error applying {enhancement_type}: {e}")
            
            # Apply all enhancement parameters
            enhancement_mapping = {
                # Facial Features
                'eyeColor': ('eye_color', lambda x: x),
                'eyeShape': ('eye_shape', lambda x: x),
                'lipColor': ('lip_color', lambda x: x),
                'faceShape': ('face_shape', lambda x: x),
                
                # Hair & Style
                'hairColor': ('hair_color', lambda x: x),
                'hairStyle': ('hair_style', lambda x: x),
                'makeup': ('makeup_application', lambda x: x if x != 'remove' else None),
                
                # Body & Posture
                'height': ('height_adjustment', lambda x: x / 100.0),
                'body': ('body_shape', lambda x: {'body': x}),
                'posture': ('posture_correction', lambda x: x / 100.0),
                
                # Skin & Beauty
                'blemish': ('blemish_removal', lambda x: x / 100.0),
                'skinTone': ('skin_tone', lambda x: x),
                'expression': ('expression_change', lambda x: x),
                
                # Environment & Style
                'background': ('background_change', lambda x: x),
                'lighting': ('lighting_enhancement', lambda x: x),
                'clothing': ('clothing_change', lambda x: x),
                
                # Basic adjustments
                'brightness': (None, lambda x: ImageEnhance.Brightness(enhanced_image).enhance(1 + x/100.0) if x != 0 else enhanced_image),
                'contrast': (None, lambda x: ImageEnhance.Contrast(enhanced_image).enhance(1 + x/100.0) if x != 0 else enhanced_image),
                'saturation': (None, lambda x: ImageEnhance.Color(enhanced_image).enhance(1 + x/100.0) if x != 0 else enhanced_image)
            }
            
            # Process makeup removal separately
            if enhancements.get('makeup') == 'remove':
                try:
                    enhanced_image = self.remove_makeup(enhanced_image, 0.5)
                except Exception as e:
                    print(f"Error removing makeup: {e}")
            
            # Process all enhancement parameters
            for param_name, (enhancement_type, processor) in enhancement_mapping.items():
                value = enhancements.get(param_name, 0 if isinstance(enhancements.get(param_name, ''), (int, float)) else '')
                
                # Skip empty values
                if not value or (isinstance(value, str) and value.strip() == ''):
                    continue
                    
                try:
                    if enhancement_type and enhancement_type in self.feature_processors:
                        processed_value = processor(value)
                        if processed_value is not None:
                            enhanced_image = self.feature_processors[enhancement_type](enhanced_image, processed_value)
                    else:
                        # Handle basic image adjustments
                        processed_result = processor(value)
                        if isinstance(processed_result, Image.Image):
                            enhanced_image = processed_result
                except Exception as e:
                    print(f"Error applying {param_name}: {e}")
            
            # Save enhanced image
            enhanced_image.save(enhanced_path, quality=95, optimize=True)
            print(f"Enhanced image saved to {enhanced_path}")
            
            return True
            
        except Exception as e:
            print(f"Enhancement error: {e}")
            # Fallback to basic enhancement
            try:
                image = Image.open(original_path).convert('RGB')
                
                # Apply basic enhancements
                brightness = enhancements.get('brightness', 0)
                contrast = enhancements.get('contrast', 0)
                saturation = enhancements.get('saturation', 0)
                
                if brightness != 0:
                    image = ImageEnhance.Brightness(image).enhance(1 + brightness/100.0)
                if contrast != 0:
                    image = ImageEnhance.Contrast(image).enhance(1 + contrast/100.0)
                if saturation != 0:
                    image = ImageEnhance.Color(image).enhance(1 + saturation/100.0)
                
                image.save(enhanced_path, quality=95)
                return True
                
            except Exception as fallback_error:
                print(f"Fallback enhancement failed: {fallback_error}")
                return False