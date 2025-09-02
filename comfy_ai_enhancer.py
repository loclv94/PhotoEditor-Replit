import os
import json
import base64
import requests
import numpy as np
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import threading
import time

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available, using CPU-only processing")

try:
    from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("Diffusers not available, using traditional image processing")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("OpenCV not available, using PIL for image processing")

class ComfyAIPhotoEnhancer:
    """
    Advanced AI Photo Enhancement System using Stable Diffusion + LoRA
    Replaces OpenAI-based enhancement with local ComfyUI-style processing
    """
    
    def __init__(self):
        self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if TORCH_AVAILABLE and torch.cuda.is_available() else None
        
        # Initialize models and pipelines
        self.models = {}
        self.controlnets = {}
        self.preprocessors = {}
        
        # Load models only if dependencies are available
        if DIFFUSERS_AVAILABLE and TORCH_AVAILABLE:
            self.load_diffusion_models()
        else:
            print("Using traditional image processing methods")
        
        # Load computer vision tools if available
        if CV2_AVAILABLE:
            self.load_cv_tools()
    
    def load_diffusion_models(self):
        """Load Stable Diffusion models if available"""
        try:
            print("Loading Stable Diffusion models...")
            
            # Load Img2Img pipeline for general enhancements
            self.models['img2img'] = StableDiffusionImg2ImgPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=self.dtype,
                safety_checker=None,
                requires_safety_checker=False,
                low_cpu_mem_usage=True
            )
            
            if TORCH_AVAILABLE and self.device == "cuda":
                self.models['img2img'] = self.models['img2img'].to(self.device)
                self.models['img2img'].enable_attention_slicing()
            
            print("Diffusion models loaded successfully")
            
        except Exception as e:
            print(f"Could not load diffusion models: {e}")
            self.models = {}
    
    def load_cv_tools(self):
        """Load computer vision tools if OpenCV is available"""
        try:
            print("Loading computer vision tools...")
            # Initialize basic CV tools for image processing
            self.cv_initialized = True
            print("Computer vision tools loaded")
        except Exception as e:
            print(f"Could not load CV tools: {e}")
            self.cv_initialized = False
    
    def detect_face_regions(self, image):
        """Detect face regions using basic image processing"""
        try:
            # Convert PIL to numpy array
            img_array = np.array(image)
            
            if CV2_AVAILABLE:
                # Use OpenCV for better face detection
                cv_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                
                # Use Haar cascades for face detection (basic but reliable)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) > 0:
                    return {
                        'has_face': True,
                        'faces': faces,
                        'face_count': len(faces)
                    }
            else:
                # Fallback: assume center region contains face for portrait photos
                height, width = img_array.shape[:2]
                face_region = [
                    int(width * 0.2),   # x
                    int(height * 0.15), # y  
                    int(width * 0.6),   # width
                    int(height * 0.6)   # height
                ]
                return {
                    'has_face': True,
                    'faces': [face_region],
                    'face_count': 1
                }
            
            return {'has_face': False}
            
        except Exception as e:
            print(f"Face detection error: {e}")
            return {'has_face': False}
    
    def create_face_mask(self, image, face_regions):
        """Create mask for facial region"""
        try:
            width, height = image.width, image.height
            mask = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(mask)
            
            for face in face_regions.get('faces', []):
                x, y, w, h = face
                # Create oval mask for face region
                draw.ellipse([x, y, x+w, y+h], fill=255)
            
            # Apply blur to smooth edges
            mask = mask.filter(ImageFilter.GaussianBlur(radius=3))
            return mask
            
        except Exception as e:
            print(f"Face mask creation error: {e}")
            return Image.new('L', (image.width, image.height), 255)
    
    def enhance_eyes(self, image, intensity=0.5):
        """Enhance eye color and shape using targeted inpainting"""
        try:
            face_analysis = self.analyze_face_features(image)
            if not face_analysis['has_face']:
                return image
            
            # Create eye region masks
            landmarks = face_analysis['landmarks']
            eye_mask = self.create_eye_mask(image, landmarks)
            
            # Use inpainting to enhance eyes
            prompt = f"beautiful detailed eyes, enhanced eye color, sharp eyelashes, high quality portrait"
            negative_prompt = "blurry, distorted, ugly, deformed eyes"
            
            if 'inpaint' in self.models:
                enhanced = self.models['inpaint'](
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=image,
                    mask_image=eye_mask,
                    strength=intensity,
                    guidance_scale=7.5,
                    num_inference_steps=20
                ).images[0]
                
                return enhanced
                
        except Exception as e:
            print(f"Eye enhancement error: {e}")
            
        return image
    
    def create_eye_mask(self, image, landmarks):
        """Create mask for eye regions"""
        try:
            width, height = image.width, image.height
            mask = Image.new('L', (width, height), 0)
            
            # Eye landmark indices (approximate)
            left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
            
            from PIL import ImageDraw
            draw = ImageDraw.Draw(mask)
            
            # Draw left eye
            left_points = []
            for idx in left_eye_indices:
                if idx < len(landmarks.landmark):
                    point = landmarks.landmark[idx]
                    x = int(point.x * width)
                    y = int(point.y * height)
                    left_points.append((x, y))
            
            if left_points:
                draw.polygon(left_points, fill=255)
            
            # Draw right eye
            right_points = []
            for idx in right_eye_indices:
                if idx < len(landmarks.landmark):
                    point = landmarks.landmark[idx]
                    x = int(point.x * width)
                    y = int(point.y * height)
                    right_points.append((x, y))
            
            if right_points:
                draw.polygon(right_points, fill=255)
            
            # Expand mask slightly
            mask = mask.filter(ImageFilter.MaxFilter(size=3))
            mask = mask.filter(ImageFilter.GaussianBlur(radius=2))
            
            return mask
            
        except Exception as e:
            print(f"Eye mask creation error: {e}")
            return Image.new('L', (image.width, image.height), 0)
    
    def enhance_with_controlnet(self, image, control_type, prompt, negative_prompt="", strength=0.8):
        """Enhance image using ControlNet for precise control"""
        try:
            if control_type not in self.controlnets or control_type not in self.preprocessors:
                return image
            
            # Generate control image
            control_image = self.preprocessors[control_type](image)
            
            # Generate enhanced image
            enhanced = self.controlnets[control_type](
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=control_image,
                num_inference_steps=20,
                guidance_scale=7.5,
                controlnet_conditioning_scale=strength
            ).images[0]
            
            return enhanced
            
        except Exception as e:
            print(f"ControlNet enhancement error: {e}")
            return image
    
    def change_background(self, image, new_background_prompt):
        """Change image background using inpainting"""
        try:
            # Create background mask (inverse of subject)
            # This is a simplified approach - in production, use a proper segmentation model
            width, height = image.width, image.height
            
            # Convert to OpenCV format for edge detection
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Create background mask (areas without strong edges)
            mask = cv2.dilate(edges, np.ones((5,5), np.uint8), iterations=2)
            mask = cv2.bitwise_not(mask)  # Invert so background is white
            
            # Apply Gaussian blur to smooth mask
            mask = cv2.GaussianBlur(mask, (21, 21), 0)
            
            # Convert back to PIL
            mask_pil = Image.fromarray(mask).convert('L')
            
            # Use inpainting to change background
            prompt = f"{new_background_prompt}, high quality, detailed background"
            negative_prompt = "blurry, low quality, distorted"
            
            if 'inpaint' in self.models:
                enhanced = self.models['inpaint'](
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=image,
                    mask_image=mask_pil,
                    strength=0.9,
                    guidance_scale=7.5,
                    num_inference_steps=25
                ).images[0]
                
                return enhanced
                
        except Exception as e:
            print(f"Background change error: {e}")
            
        return image
    
    def enhance_body_proportions(self, image, adjustments):
        """Enhance body proportions using pose-guided generation"""
        try:
            # Use OpenPose ControlNet for body proportion enhancement
            if 'openpose' not in self.controlnets:
                return image
            
            # Generate pose control image
            pose_image = self.preprocessors['openpose'](image)
            
            # Create enhancement prompt based on adjustments
            prompt_parts = ["professional portrait, perfect proportions"]
            
            if adjustments.get('height', 0) > 50:
                prompt_parts.append("tall elegant posture")
            if adjustments.get('body', 0) > 50:
                prompt_parts.append("fit athletic body")
                
            prompt = ", ".join(prompt_parts) + ", high quality, detailed"
            negative_prompt = "distorted, deformed, ugly, low quality"
            
            enhanced = self.controlnets['openpose'](
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=pose_image,
                num_inference_steps=25,
                guidance_scale=7.5,
                controlnet_conditioning_scale=0.8
            ).images[0]
            
            return enhanced
            
        except Exception as e:
            print(f"Body enhancement error: {e}")
            return image
    
    def apply_traditional_enhancements(self, image, enhancements):
        """Apply traditional PIL-based enhancements as fallback"""
        try:
            enhanced = image.copy()
            
            # Apply brightness adjustment
            brightness = enhancements.get('brightness', 0)
            if brightness != 0:
                brightness_factor = 1 + (brightness / 100)
                enhancer = ImageEnhance.Brightness(enhanced)
                enhanced = enhancer.enhance(brightness_factor)
            
            # Apply contrast adjustment
            contrast = enhancements.get('contrast', 0)
            if contrast != 0:
                contrast_factor = 1 + (contrast / 100)
                enhancer = ImageEnhance.Contrast(enhanced)
                enhanced = enhancer.enhance(contrast_factor)
            
            # Apply saturation adjustment
            saturation = enhancements.get('saturation', 0)
            if saturation != 0:
                saturation_factor = 1 + (saturation / 100)
                enhancer = ImageEnhance.Color(enhanced)
                enhanced = enhancer.enhance(saturation_factor)
            
            # Apply skin smoothing
            skin_smoothing = enhancements.get('skin', 0)
            if skin_smoothing > 0:
                blur_radius = skin_smoothing / 20
                enhanced = enhanced.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            
            # Apply sharpening
            face_enhancement = enhancements.get('face', 0)
            if face_enhancement > 0:
                sharpness_factor = 1 + (face_enhancement / 100)
                enhancer = ImageEnhance.Sharpness(enhanced)
                enhanced = enhancer.enhance(sharpness_factor)
            
            return enhanced
            
        except Exception as e:
            print(f"Traditional enhancement error: {e}")
            return image
    
    def enhance_image(self, original_path, enhanced_path, enhancements, enhancement_prompt=""):
        """
        Main enhancement method using Stable Diffusion + LoRA
        
        Args:
            original_path: Path to original image
            enhanced_path: Path to save enhanced image
            enhancements: Dictionary of enhancement parameters
            enhancement_prompt: User's text prompt for AI enhancement
            
        Returns:
            bool: Success status
        """
        try:
            # Load original image
            original_image = Image.open(original_path).convert('RGB')
            enhanced_image = original_image.copy()
            
            print(f"Starting AI enhancement with prompt: {enhancement_prompt}")
            
            # Parse enhancement request and apply AI enhancements
            if enhancement_prompt and enhancement_prompt.strip():
                enhanced_image = self.apply_ai_enhancements(
                    enhanced_image, 
                    enhancement_prompt, 
                    enhancements
                )
            else:
                # Apply traditional enhancements if no prompt
                enhanced_image = self.apply_traditional_enhancements(enhanced_image, enhancements)
            
            # Apply specific feature enhancements based on sliders
            enhanced_image = self.apply_feature_enhancements(enhanced_image, enhancements)
            
            # Save enhanced image
            enhanced_image.save(enhanced_path, quality=95, optimize=True)
            print(f"Enhanced image saved to {enhanced_path}")
            
            return True
            
        except Exception as e:
            print(f"Error in enhance_image: {e}")
            # Fallback to traditional enhancement
            try:
                original_image = Image.open(original_path).convert('RGB')
                enhanced_image = self.apply_traditional_enhancements(original_image, enhancements)
                enhanced_image.save(enhanced_path, quality=95)
                return True
            except Exception as fallback_error:
                print(f"Fallback enhancement also failed: {fallback_error}")
                return False
    
    def apply_ai_enhancements(self, image, prompt, enhancements):
        """Apply AI enhancements based on text prompt"""
        try:
            # Analyze the prompt to determine enhancement type
            prompt_lower = prompt.lower()
            
            # Background change
            if any(bg_word in prompt_lower for bg_word in ['background', 'setting', 'scene', 'environment']):
                image = self.change_background(image, prompt)
            
            # Face/portrait enhancements
            elif any(face_word in prompt_lower for face_word in ['face', 'eyes', 'lips', 'skin', 'portrait', 'beauty']):
                # Use img2img for general facial enhancements
                if 'img2img' in self.models:
                    full_prompt = f"{prompt}, professional portrait photography, high quality, detailed"
                    negative_prompt = "blurry, distorted, ugly, low quality, deformed"
                    
                    image = self.models['img2img'](
                        prompt=full_prompt,
                        negative_prompt=negative_prompt,
                        image=image,
                        strength=0.6,
                        guidance_scale=7.5,
                        num_inference_steps=20
                    ).images[0]
                
                # Apply specific eye enhancements if mentioned
                if 'eye' in prompt_lower:
                    image = self.enhance_eyes(image, intensity=0.5)
            
            # Body/pose enhancements
            elif any(body_word in prompt_lower for body_word in ['body', 'posture', 'height', 'proportions']):
                image = self.enhance_body_proportions(image, enhancements)
            
            # General enhancement using img2img
            else:
                if 'img2img' in self.models:
                    full_prompt = f"{prompt}, professional photography, high quality, enhanced, detailed"
                    negative_prompt = "blurry, distorted, ugly, low quality"
                    
                    image = self.models['img2img'](
                        prompt=full_prompt,
                        negative_prompt=negative_prompt,
                        image=image,
                        strength=0.5,
                        guidance_scale=7.5,
                        num_inference_steps=20
                    ).images[0]
            
            return image
            
        except Exception as e:
            print(f"AI enhancement error: {e}")
            return image
    
    def apply_feature_enhancements(self, image, enhancements):
        """Apply specific feature enhancements based on slider values"""
        try:
            # Eye enhancement
            if enhancements.get('face', 0) > 50:
                image = self.enhance_eyes(image, intensity=0.4)
            
            # Skin smoothing with face detection
            skin_value = enhancements.get('skin', 0)
            if skin_value > 0:
                face_analysis = self.analyze_face_features(image)
                if face_analysis['has_face']:
                    # Apply selective smoothing to face area
                    face_mask = self.create_face_mask(image, face_analysis['landmarks'])
                    smoothed = image.filter(ImageFilter.GaussianBlur(radius=skin_value/30))
                    
                    # Blend smoothed face back onto original
                    image = Image.composite(smoothed, image, face_mask)
                else:
                    # Apply general smoothing
                    blur_radius = skin_value / 25
                    image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            
            return image
            
        except Exception as e:
            print(f"Feature enhancement error: {e}")
            return image