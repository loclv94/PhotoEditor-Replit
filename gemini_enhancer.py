import os
import io
import json
import logging
from PIL import Image
from google import genai

# IMPORTANT: KEEP THIS COMMENT
# Follow these instructions when using this blueprint:
# - Note that the newest Gemini model series is "gemini-2.5-flash" or gemini-2.5-pro"
#   - do not change this unless explicitly requested by the user
# - Sometimes the google genai SDK has occasional type errors. You might need to run to validate, at time.  
# The SDK was recently renamed from google-generativeai to google-genai. This file reflects the new name and the new APIs.

class GeminiEnhancer:
    """
    Gemini AI Photo Enhancement System
    Provides AI-powered photo enhancement using Google's Gemini models
    """
    
    def __init__(self):
        print("Initializing Gemini Enhancer...")
        # This API key is from Gemini Developer API Key, not vertex AI API Key
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        # The model name for image generation
        self.model_name = "gemini-2.5-flash-image-preview"
        print(f"Gemini Enhancer ready with model: {self.model_name}")
    
    def enhance_image(self, original_path, enhanced_path, enhancements, enhancement_prompt=""):
        """
        Apply AI-powered enhancements to an image using Gemini
        
        Args:
            original_path: Path to original image
            enhanced_path: Path to save enhanced image
            enhancements: Dictionary of enhancement parameters
            enhancement_prompt: User's text prompt for AI enhancement
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"Enhancing image with Gemini: {original_path}")
            
            # Load the original image
            with open(original_path, 'rb') as f:
                image_data = f.read()
            
            # Create enhancement prompt based on parameters and user input
            prompt = self._create_enhancement_prompt(enhancements, enhancement_prompt)
            print(f"Enhancement prompt: {prompt}")
            
            # Send the prompt to Gemini for image generation
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            
            # Process the response and save enhanced image
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        # Check if the part contains image data
                        if part.inline_data is not None and part.inline_data.data is not None:
                            # Use PIL to open the image from bytes
                            enhanced_image = Image.open(io.BytesIO(part.inline_data.data))
                            # Save the enhanced image
                            enhanced_image.save(enhanced_path)
                            print(f"Enhanced image saved to: {enhanced_path}")
                            return True
                        elif part.text is not None:
                            # If the response includes text, log it
                            print("Gemini response:", part.text)
            
            print("No image data found in Gemini response")
            return False
            
        except Exception as e:
            print(f"Gemini enhancement error: {e}")
            logging.error(f"Gemini enhancement error: {str(e)}")
            return False
    
    def _create_enhancement_prompt(self, enhancements, user_prompt=""):
        """
        Create a detailed prompt for Gemini based on enhancement parameters and user input
        """
        # Start with base prompt for photo enhancement
        base_prompt = "Create a high-quality, professional enhanced version of this photo with natural-looking improvements. "
        
        # Add specific enhancements based on parameters
        enhancement_descriptions = []
        
        if enhancements.get('eye_color'):
            enhancement_descriptions.append(f"change eye color to {enhancements['eye_color']}")
        
        if enhancements.get('eye_shape'):
            enhancement_descriptions.append(f"enhance eye shape to be more {enhancements['eye_shape']}")
        
        if enhancements.get('face_shape'):
            enhancement_descriptions.append(f"adjust face shape to be more {enhancements['face_shape']}")
        
        if enhancements.get('hair_color'):
            enhancement_descriptions.append(f"change hair color to {enhancements['hair_color']}")
        
        if enhancements.get('hair_style'):
            enhancement_descriptions.append(f"style hair to be {enhancements['hair_style']}")
        
        if enhancements.get('lip_color'):
            enhancement_descriptions.append(f"enhance lip color to {enhancements['lip_color']}")
        
        if enhancements.get('skin_tone'):
            enhancement_descriptions.append("improve skin tone and texture")
        
        if enhancements.get('blemish_removal'):
            enhancement_descriptions.append("remove blemishes and imperfections")
        
        if enhancements.get('makeup_application'):
            enhancement_descriptions.append("apply natural-looking makeup")
        
        if enhancements.get('lighting_enhancement'):
            enhancement_descriptions.append("enhance lighting and overall brightness")
        
        if enhancements.get('background_change'):
            enhancement_descriptions.append(f"change background to {enhancements.get('background_type', 'a professional setting')}")
        
        # Combine all enhancements
        if enhancement_descriptions:
            enhancement_text = "Apply these specific enhancements: " + ", ".join(enhancement_descriptions) + ". "
        else:
            enhancement_text = "Apply general photo enhancement including skin smoothing, color correction, and lighting improvements. "
        
        # Add user's custom prompt if provided
        if user_prompt and user_prompt.strip():
            custom_prompt = f"Additionally, {user_prompt.strip()}. "
        else:
            custom_prompt = ""
        
        # Quality and style instructions
        quality_prompt = ("Maintain high image quality, natural skin tones, and realistic proportions. "
                         "Ensure the enhanced photo looks professional and suitable for social media. "
                         "Keep facial features recognizable and avoid over-processing.")
        
        # Combine all parts
        full_prompt = base_prompt + enhancement_text + custom_prompt + quality_prompt
        
        return full_prompt
    
    def analyze_image(self, image_path):
        """
        Analyze an image to understand its content and suggest enhancements
        """
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
                response = self.client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=[
                        genai.types.Part.from_bytes(
                            data=image_bytes,
                            mime_type="image/jpeg",
                        ),
                        "Analyze this portrait photo and suggest specific enhancements that would make it more suitable for social media. "
                        "Focus on facial features, lighting, skin quality, and overall composition. "
                        "Provide specific suggestions for improvements.",
                    ],
                )
            
            return response.text if response.text else "No analysis available"
            
        except Exception as e:
            print(f"Image analysis error: {e}")
            return f"Analysis failed: {str(e)}"