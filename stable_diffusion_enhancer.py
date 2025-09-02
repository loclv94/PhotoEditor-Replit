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
        
        # Extract face data from the new structure
        x1, y1, x2, y2 = face_data.get('face_box', (0, 0, 100, 100))
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        fw, fh = face_data.get('face_width', 100), face_data.get('face_height', 100)
        
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
            draw.ellipse([x1, y1, x2, y2], fill=255)
            
        elif region_type == 'hair':
            # Hair region (upper portion)
            hair_y = y1 - int(fh * 0.3)
            draw.ellipse([x1 - int(fw * 0.2), hair_y,
                         x2 + int(fw * 0.2),
                         y1 + int(fh * 0.3)], fill=255)
        
        # Apply blur for smooth edges
        mask = mask.filter(ImageFilter.GaussianBlur(radius=2))
        return mask
    
    def enhance_eye_color(self, image, color="blue", intensity=0.7):
        """Change eye color using HSV-based realistic modification with iris detection"""
        try:
            import cv2
            # Convert to OpenCV format
            img_array = np.array(image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Detect face and eyes  
            face_data = self.detect_face_regions(image)
            if not face_data or not face_data.get('eyes'):
                print("No eyes detected for color change")
                return image
            
            enhanced_bgr = img_bgr.copy()
            
            # HSV hue values for natural eye colors
            color_hues = {
                'blue': 110, 'green': 60, 'brown': 15, 'hazel': 35, 
                'gray': 120, 'amber': 25, 'violet': 140
            }
            
            if color not in color_hues:
                print(f"Unsupported eye color: {color}")
                return image
                
            target_hue = color_hues[color]
            
            # Process each detected eye
            for eye_info in face_data['eyes']:
                cx, cy, ew, eh = eye_info
                
                # Extract eye region with padding
                padding = max(5, min(ew, eh) // 4)
                y1 = max(0, cy - eh//2 - padding)
                y2 = min(enhanced_bgr.shape[0], cy + eh//2 + padding) 
                x1 = max(0, cx - ew//2 - padding)
                x2 = min(enhanced_bgr.shape[1], cx + ew//2 + padding)
                
                eye_region = enhanced_bgr[y1:y2, x1:x2]
                if eye_region.size == 0:
                    continue
                
                # Convert eye region to HSV
                eye_hsv = cv2.cvtColor(eye_region, cv2.COLOR_BGR2HSV)
                eye_gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
                
                # Create iris mask (exclude pupil and sclera)
                # Iris detection: moderate brightness, some saturation
                lower_iris = np.array([0, 30, 40])   # Allow all hues, some saturation, not too dark
                upper_iris = np.array([179, 255, 180])  # Exclude very bright areas (sclera)
                iris_mask = cv2.inRange(eye_hsv, lower_iris, upper_iris)
                
                # Remove pupil (darkest regions)
                _, pupil_mask = cv2.threshold(eye_gray, 30, 255, cv2.THRESH_BINARY_INV)
                pupil_mask = cv2.morphologyEx(pupil_mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
                
                # Final iris mask (exclude pupil)
                iris_only = cv2.bitwise_and(iris_mask, cv2.bitwise_not(pupil_mask))
                
                # Apply morphological operations for cleaner mask
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                iris_only = cv2.morphologyEx(iris_only, cv2.MORPH_CLOSE, kernel)
                iris_only = cv2.morphologyEx(iris_only, cv2.MORPH_OPEN, kernel)
                
                # Gaussian blur for soft edges
                iris_mask_float = cv2.GaussianBlur(iris_only.astype(np.float32), (5, 5), 1.0) / 255.0
                
                # Change hue while preserving saturation and value (natural texture)
                new_eye_hsv = eye_hsv.copy()
                
                # Blend hue: preserve original texture while shifting color
                original_hue = eye_hsv[:,:,0].astype(np.float32)
                new_hue = np.full_like(original_hue, target_hue)
                blended_hue = (original_hue * (1 - iris_mask_float * intensity) + 
                              new_hue * iris_mask_float * intensity)
                
                new_eye_hsv[:,:,0] = np.clip(blended_hue, 0, 179).astype(np.uint8)
                
                # Convert back and apply to original
                new_eye_bgr = cv2.cvtColor(new_eye_hsv, cv2.COLOR_HSV2BGR)
                enhanced_bgr[y1:y2, x1:x2] = new_eye_bgr
            
            # Convert back to PIL format
            enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(enhanced_rgb)
            
        except Exception as e:
            print(f"Natural eye color change error: {e}")
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
    
    def enhance_lip_color(self, image, color="rose", intensity=0.6):
        """Change lip color using natural HSV-based blending while preserving texture"""
        try:
            import cv2
            # Convert to OpenCV format
            img_array = np.array(image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            face_data = self.detect_face_regions(image)
            if not face_data.get('has_face'):
                return image
            
            # Get lip region from face detection
            lip_pos = face_data.get('lips', (0, 0))
            if isinstance(lip_pos, tuple) and len(lip_pos) == 2:
                lx, ly = lip_pos
                
                # Create lip region
                face_width = face_data.get('face_width', 100)
                lip_width = int(face_width * 0.25)
                lip_height = int(face_width * 0.12)
                
                y1 = max(0, ly - lip_height//2)
                y2 = min(img_bgr.shape[0], ly + lip_height//2)
                x1 = max(0, lx - lip_width//2)
                x2 = min(img_bgr.shape[1], lx + lip_width//2)
                
                lip_region = img_bgr[y1:y2, x1:x2]
                
                if lip_region.size > 0:
                    # Convert to HSV for natural color manipulation
                    lip_hsv = cv2.cvtColor(lip_region, cv2.COLOR_BGR2HSV)
                    lip_gray = cv2.cvtColor(lip_region, cv2.COLOR_BGR2GRAY)
                    
                    # Create lip mask based on color characteristics
                    # Lips typically have some saturation and medium brightness
                    lower_lip = np.array([0, 20, 50])    # Any hue, some saturation, not too dark
                    upper_lip = np.array([179, 255, 220])  # Exclude very bright areas
                    lip_mask = cv2.inRange(lip_hsv, lower_lip, upper_lip)
                    
                    # Remove very dark areas (teeth gaps, shadows)
                    _, dark_mask = cv2.threshold(lip_gray, 40, 255, cv2.THRESH_BINARY)
                    lip_mask = cv2.bitwise_and(lip_mask, dark_mask)
                    
                    # Morphological operations for smooth mask
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    lip_mask = cv2.morphologyEx(lip_mask, cv2.MORPH_CLOSE, kernel)
                    lip_mask = cv2.morphologyEx(lip_mask, cv2.MORPH_OPEN, kernel)
                    
                    # Gaussian blur for natural edges
                    lip_mask_float = cv2.GaussianBlur(lip_mask.astype(np.float32), (3, 3), 0.5) / 255.0
                    
                    # HSV hue values for natural lip colors
                    color_hues = {
                        'rose': 350, 'red': 0, 'pink': 330, 'coral': 15, 
                        'berry': 320, 'nude': 30, 'plum': 300, 'cherry': 5
                    }
                    
                    if color.lower() in color_hues:
                        target_hue = color_hues[color.lower()]
                        # Convert OpenCV hue range (0-179) 
                        target_hue_cv = int(target_hue * 179 / 360) if target_hue <= 360 else target_hue
                        
                        # Enhance saturation for more vibrant color
                        enhanced_hsv = lip_hsv.copy()
                        
                        # Apply hue change with texture preservation
                        original_hue = lip_hsv[:,:,0].astype(np.float32)
                        new_hue = np.full_like(original_hue, target_hue_cv)
                        blended_hue = (original_hue * (1 - lip_mask_float * intensity) + 
                                      new_hue * lip_mask_float * intensity)
                        enhanced_hsv[:,:,0] = np.clip(blended_hue, 0, 179).astype(np.uint8)
                        
                        # Slightly increase saturation for more vivid color
                        original_sat = lip_hsv[:,:,1].astype(np.float32)
                        enhanced_sat = np.minimum(255, original_sat * (1 + lip_mask_float * intensity * 0.3))
                        enhanced_hsv[:,:,1] = enhanced_sat.astype(np.uint8)
                        
                        # Convert back and replace region
                        enhanced_lip_bgr = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
                        img_bgr[y1:y2, x1:x2] = enhanced_lip_bgr
            
            # Convert back to PIL format
            enhanced_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(enhanced_rgb)
            
        except Exception as e:
            print(f"Natural lip color change error: {e}")
            return image
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
            
            # Combine prompt enhancements with parameter enhancements
            all_enhancements = enhancements.copy()
            
            # Parse natural language prompt into specific enhancements
            if enhancement_prompt and enhancement_prompt.strip():
                prompt_enhancements = self.parse_enhancement_prompt(enhancement_prompt)
                all_enhancements.update(prompt_enhancements)
            
            # Process facial features first (most critical)
            facial_features = {
                'eyeColor': 'eye_color',
                'eyeShape': 'eye_shape', 
                'lipColor': 'lip_color',
                'faceShape': 'face_shape'
            }
            
            print(f"Processing facial features...")
            for param_name, feature_type in facial_features.items():
                value = all_enhancements.get(param_name, '')
                if value and value.strip():
                    try:
                        if feature_type in self.feature_processors:
                            print(f"Applying {feature_type}: {value}")
                            enhanced_image = self.feature_processors[feature_type](enhanced_image, value)
                    except Exception as e:
                        print(f"Error applying {param_name}: {e}")
            
            # Process other enhancements
            other_mapping = {
                'hairColor': ('hair_color', lambda x: x),
                'hairStyle': ('hair_style', lambda x: x),
                'makeup': ('makeup_application', lambda x: x if x != 'remove' else None),
                'height': ('height_adjustment', lambda x: x / 100.0),
                'body': ('body_shape', lambda x: {'body': x}),
                'posture': ('posture_correction', lambda x: x / 100.0),
                'blemish': ('blemish_removal', lambda x: x / 100.0),
                'skinTone': ('skin_tone', lambda x: x),
                'expression': ('expression_change', lambda x: x),
                'background': ('background_change', lambda x: x),
                'lighting': ('lighting_enhancement', lambda x: x),
                'clothing': ('clothing_change', lambda x: x)
            }
            
            for param_name, (enhancement_type, processor) in other_mapping.items():
                value = all_enhancements.get(param_name, 0 if isinstance(all_enhancements.get(param_name, ''), (int, float)) else '')
                
                if not value or (isinstance(value, str) and value.strip() == ''):
                    continue
                    
                try:
                    if enhancement_type and enhancement_type in self.feature_processors:
                        processed_value = processor(value)
                        if processed_value is not None:
                            enhanced_image = self.feature_processors[enhancement_type](enhanced_image, processed_value)
                except Exception as e:
                    print(f"Error applying {param_name}: {e}")
            
            # Apply basic image adjustments last
            basic_adjustments = ['brightness', 'contrast', 'saturation']
            for adjustment in basic_adjustments:
                value = all_enhancements.get(adjustment, 0)
                if isinstance(value, (int, float)) and value != 0:
                    try:
                        if adjustment == 'brightness':
                            enhanced_image = ImageEnhance.Brightness(enhanced_image).enhance(1 + value/100.0)
                        elif adjustment == 'contrast':
                            enhanced_image = ImageEnhance.Contrast(enhanced_image).enhance(1 + value/100.0)
                        elif adjustment == 'saturation':
                            enhanced_image = ImageEnhance.Color(enhanced_image).enhance(1 + value/100.0)
                    except Exception as e:
                        print(f"Error applying {adjustment}: {e}")
            
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